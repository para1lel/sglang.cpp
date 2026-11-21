#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
#include "sglang/text/decoder_logprobs.h"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_prefill.h"
#include "sglang/decoder_sampling.h"
#include "sglang/decoder_verify.h"
#include "sglang/decoder_prefix_cache.h"
#include "sglang/prefill_fp8.h"
#include "sglang/text/prefill_attn_sm100.h"

#include <algorithm>
#include <utility>

namespace sglang {
namespace {

using namespace decoder;

static_assert(decoder_batch_max_mtp_steps + 1 == target_verify_max_rows);

struct Lane {
    std::unique_ptr<Request> request;
    std::vector<int> prompt;
    DecoderOptions options;
    RequestProgress progress{};
    std::unique_ptr<MtpDraftRequest> draft = nullptr;
    std::unique_ptr<TargetVerification> verification = nullptr;
    std::optional<SamplingParameters> sampling = std::nullopt;
    std::optional<MtpConstraintRound> constraint_round{};
    int checkpoint = 0;
    int counted_draft_rows = 0;
    bool restored_prompt = false;
    std::shared_ptr<const MultimodalInput> media = nullptr;
    MediaPrefill media_prefill{};
    std::size_t resident_bytes = 0;

    int remaining_prompt() const { return progress.remaining_prompt(int(prompt.size())); }
    int prefill_limit() const {
        return checkpoint > progress.position ? checkpoint - progress.position
                                              : remaining_prompt();
    }
    int remaining_output() const { return progress.remaining_output(options); }
    int capacity() const { return RequestProgress::capacity(int(prompt.size()), options); }
    int proposals() const { return std::min(options.mtp_steps, remaining_output() - 1); }

    ForwardLane input() const {
        const int position = progress.position;
        if (remaining_prompt()) {
            return {request.get(), prompt[position], position, TextGdnMode::prefill, false,
                    media ? media_prefill.feature(position) : nullptr,
                    media ? media_prefill.positions + position : nullptr};
        }
        const int rotary = media ? position + (media->prompt.next_position - int(prompt.size()))
                                 : -1;
        return {request.get(), progress.last_token, position, TextGdnMode::decode, false,
                nullptr, nullptr, rotary};
    }

    void finish_round(MtpRound& round, DecoderBatchStep& result, cudaStream_t stream,
                      SerialOutput* output, const MtpRound::Cancel& cancel) {
        result.mtp_drafted_tokens = round.drafted;
        result.mtp_verified_rows = round.rows();
        try {
            const bool completed = round.commit([&](int token, const Bf16* logits, bool accepted) {
                if (!result.token) {
                    result.token = token;
                } else {
                    result.extra_tokens[result.extra_token_count++] = token;
                }
                result.mtp_accepted_tokens += accepted;
                result.finish_reason = progress.commit(options, token, logits, output);
                return !result.finish_reason;
            }, stream, cancel);
            if (!completed) {
                result.finish_reason = DecoderFinishReason::cancelled;
            }
        } catch (const DecoderSamplingError&) {
            result.sampling_failed = true;
            return;
        } catch (const DecoderConstraintError&) {
            result.constraint_failed = true;
            return;
        }
        if (result.extra_token_count != 0) {
            result.extra_token = result.extra_tokens.front();
        }
        result.mtp_replayed_rows = round.consumed < round.rows() ? round.consumed : 0;
        progress.advance(round.consumed);
    }
};

struct BatchStorage {
    std::shared_ptr<const Allocations> weights;
    std::array<std::optional<Lane>, decoder_batch_max_lanes> slots;
    DecodeGraphs decode_graphs;
    std::optional<VerificationGraph> verification_graph;
    std::optional<PrefixCache> prefix_cache;
};

void retire(std::optional<Lane>& lane) {
    if (lane && lane->request) {
        lane->request->memory.close();
    }
    lane.reset();
}

std::size_t request_bytes(const BatchStorage& storage) {
    std::size_t bytes = 0;
    for (const auto& lane : storage.slots) {
        if (lane && lane->request) {
            bytes += lane->request->memory.bytes();
            if (lane->draft) {
                bytes += lane->draft->device_allocation_bytes();
            }
        }
    }
    return bytes;
}

using Selection = std::array<bool, decoder_batch_max_lanes>;
using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;
using PrefillRows = std::array<int, decoder_batch_max_lanes>;
constexpr int batch_prefill_tile_rows = 1024;
constexpr std::size_t decoder_shared_reserve = std::size_t{2} * 1024 * 1024 * 1024;
static_assert(decoder_batch_max_lanes <= prefill_chunk_rows);
static_assert(prefill_chunk_rows == mxfp8_page_size);

PrefillRows plan_prefill(const BatchStorage& storage, const Selection& selected, bool chunked) {
    PrefillRows rows{};
    PrefillRows remaining{};
    int unused = prefill_chunk_rows;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        const auto& lane = storage.slots[slot];
        if (selected[slot] && lane && lane->request && !lane->restored_prompt) {
            --unused;
            remaining[slot] = lane->prefill_limit();
            rows[slot] = std::min(1, remaining[slot]);
        }
    }
    while (chunked && unused > 0) {
        const int before = unused;
        for (int slot = 0; slot < decoder_batch_max_lanes && unused > 0; ++slot) {
            if (rows[slot] < remaining[slot]) {
                ++rows[slot];
                --unused;
            }
        }
        if (unused == before) {
            break;
        }
    }
    return rows;
}

PrefillRows plan_wide_prefill(const BatchStorage& storage, const Selection& selected,
                              int lanes, int& cursor) {
    PrefillRows rows{};
    int unused = prefill_chunk_rows;
    for (int slot = 0; slot < lanes; ++slot) {
        const auto& lane = storage.slots[slot];
        if (selected[slot] && lane && lane->request && !lane->restored_prompt &&
            lane->remaining_prompt() == 0) {
            --unused;
        }
    }
    const int start = cursor;
    for (int offset = 0; offset < lanes && unused > 0; ++offset) {
        const int slot = (start + offset) % lanes;
        const auto& lane = storage.slots[slot];
        if (selected[slot] && lane && lane->request && lane->remaining_prompt() > 0) {
            rows[slot] = std::min(unused, lane->prefill_limit());
            unused -= rows[slot];
            cursor = (slot + 1) % lanes;
        }
    }
    return rows;
}

PrefillRows plan_tensor_cache_prefill(const BatchStorage& storage, const Selection& selected,
                                      int lanes, int& cursor) {
    PrefillRows rows{};
    int unused = prefill_chunk_rows;
    const int start = cursor;
    for (int offset = 0; offset < lanes && unused > 0; ++offset) {
        const int slot = (start + offset) % lanes;
        const auto& lane = storage.slots[slot];
        if (selected[slot] && lane && lane->request) {
            const int tile = std::min(prefill_chunk_rows, lane->prefill_limit());
            if (tile > 0 && tile <= unused) {
                rows[slot] = tile;
                unused -= tile;
                cursor = (slot + 1) % lanes;
            }
        }
    }
    return rows;
}

struct StepPlan {
    PrefillRows prefill{};
    PrefillRows decode{};
};

int cache_prefill_tile(int budget) {
    int tile = prefill_chunk_rows;
    while (tile > budget) {
        tile /= 2;
    }
    return tile;
}

StepPlan plan_cached_step(const BatchStorage& storage, const Selection& selected, int lanes,
                          int unused, int& cursor) {
    StepPlan plan;
    PrefillRows limits{};
    const int tile = cache_prefill_tile(unused);
    const int start = cursor;
    cursor = (start + 1) % lanes;
    bool first = true;
    for (int offset = 0; offset < lanes && unused > 0; ++offset) {
        const int slot = (start + offset) % lanes;
        const auto& lane = storage.slots[slot];
        if (!selected[slot] || !lane || !lane->request || lane->restored_prompt) {
            continue;
        }
        const int rows = limits[slot] = lane->prefill_limit();
        if (rows == 0) {
            plan.decode[slot] = 1;
            --unused;
        } else if (first) {
            plan.prefill[slot] = std::min(tile, rows);
            unused -= plan.prefill[slot];
        }
        first = false;
    }
    for (bool progress = true; progress && unused > 0;) {
        progress = false;
        for (int offset = 0; offset < lanes && unused > 0; ++offset) {
            const int slot = (start + offset) % lanes;
            const int rows = std::min(tile, limits[slot] - plan.prefill[slot]);
            if (rows > 0 && rows <= unused) {
                plan.prefill[slot] += rows;
                unused -= rows;
                progress = true;
            }
        }
    }
    return plan;
}

StepPlan plan_step(const BatchStorage& storage, const Selection& selected, int lanes,
                   int unused, int& cursor, bool chunked) {
    StepPlan plan;
    PrefillRows limits{};
    const int start = cursor;
    for (int offset = 0; offset < lanes && unused > 0; ++offset) {
        const int slot = (start + offset) % lanes;
        const auto& lane = storage.slots[slot];
        if (!selected[slot] || !lane || !lane->request || lane->restored_prompt) {
            continue;
        }
        if (lane->remaining_prompt() == 0) {
            plan.decode[slot] = 1;
        } else {
            limits[slot] = chunked ? lane->prefill_limit() : 1;
            plan.prefill[slot] = 1;
        }
        --unused;
        cursor = (slot + 1) % lanes;
    }
    if (unused > 0) {
        cursor = (start + 1) % lanes;
    }
    for (int offset = 0; offset < lanes && unused > 0; ++offset) {
        const int slot = (start + offset) % lanes;
        if (plan.decode[slot]) {
            const int extra = std::min(unused, storage.slots[slot]->proposals());
            plan.decode[slot] += extra;
            unused -= extra;
        }
    }
    while (unused > 0) {
        int unfinished = 0;
        for (int slot = 0; slot < lanes; ++slot) {
            unfinished += plan.prefill[slot] < limits[slot];
        }
        if (unfinished == 0) {
            break;
        }
        const int share = std::max(1, unused / unfinished);
        for (int offset = 0; offset < lanes && unused > 0; ++offset) {
            const int slot = (start + offset) % lanes;
            const int extra = std::min({unused, share, limits[slot] - plan.prefill[slot]});
            plan.prefill[slot] += extra;
            unused -= extra;
        }
    }
    return plan;
}

}

struct DecoderBatch::State {
    State(const Decoder::Model& owner, DecoderBatchOptions options)
        : model(owner), options(options), storage(std::make_shared<BatchStorage>()),
          tensor_cache(options.prefix_cache_entries > 0 &&
                       model.fp8_prefill() == Fp8PrefillPolicy::tensor_core),
          execution(model.device) {
        if (options.max_lanes < 1 || options.max_lanes > decoder_batch_max_lanes) {
            throw std::invalid_argument("Invalid decoder batch capacity");
        }
        if (options.prefix_cache_entries < 0 ||
            options.prefix_cache_entries > decoder_batch_prefix_max_entries) {
            throw std::invalid_argument("Invalid prefix cache capacity");
        }
        if (options.max_num_batched_tokens < 0 ||
            options.max_num_batched_tokens > decoder_batch_max_step_tokens) {
            throw std::invalid_argument("Invalid decoder batch step budget");
        }
        DeviceScope device(model.device);
        storage->weights = model.memory;
        if (options.prefix_cache_entries != 0) {
            storage->prefix_cache.emplace(model.device, options.prefix_cache_entries,
                                          !tensor_cache && model.mtp_weights != nullptr);
        }
        if (options.cuda_graphs) {
            storage->verification_graph.emplace(execution);
        }
        execution.open_stream(storage);
        if (options.resident_memory_limit) {
            const auto free = available_device_bytes();
            resident_capacity = std::min(*options.resident_memory_limit,
                                         free - std::min(free, decoder_shared_reserve));
        }
        device.restore();
    }

    Steps advance(const Selection& selected, bool chunked, DecoderBatchStepStats* stats);
    void submit(const Selection& selected, const PrefillRows& decode, Steps& results,
                DecoderBatchStepStats& stats, bool& contains_mtp);
    void prefill(const PrefillRows& rows, int max_rows, DecoderBatchStepStats& stats);
    void score(int slot, Lane& lane, Workspace& work, bool logits_ready);
    void speculate_group(const PrefillRows& verification_rows, int& next_slot, Steps& results,
                         DecoderBatchStepStats& stats);
    void collect_scores(const std::array<Request*, decoder_batch_max_lanes>& scoring,
                        const std::array<int, decoder_batch_max_lanes>& packed_slots,
                        const DecodeRun& run, Steps& results);
    std::size_t reserve(DecoderOptions& request, int prompt, std::size_t media_bytes) const;
    void initialize(Lane& lane);
    void drain(const char* operation, std::optional<Lane>* lane = nullptr) {
        try {
            DeviceScope device(model.device);
            check(cudaStreamSynchronize(execution.stream()), operation);
            if (lane) retire(*lane);
        } catch (...) {
            disable();
            throw;
        }
    }
    void fail_step(bool recoverable) {
        if (!recoverable || execution.verification_graph_failed()) {
            disable();
            return;
        }
        drain("Drain failed decoder batch step");
        storage->slots = {};
    }
    std::optional<Lane>& slot_at(int slot) {
        if (slot < 0 || slot >= options.max_lanes) {
            throw std::out_of_range("Invalid decoder batch slot");
        }
        return storage->slots[slot];
    }

    void require_ready() const {
        if (failed) {
            throw std::runtime_error("Decoder batch is unavailable");
        }
    }

    void disable() noexcept {
        failed = true;
        try {
            DeviceScope device(model.device);
            execution.close();
        } catch (...) {
            return;
        }
        storage->slots = {};
        storage->decode_graphs = {};
        storage->verification_graph.reset();
        storage->prefix_cache.reset();
    }

    const Decoder::Model& model;
    const DecoderBatchOptions options;
    std::shared_ptr<BatchStorage> storage;
    const bool tensor_cache;
    Allocations execution;
    std::byte* sampling_workspace = nullptr;
    std::byte* logprob_workspace = nullptr;
    GeneratedLogprobReadback* logprob_readback = nullptr;
    ConstraintWorkspace constraint_workspace;
    BiasedWorkspace biased_workspace;
    std::size_t resident_capacity = 0;
    int prefill_cursor = 0;
    bool failed = false;
    SerialOutput* output = nullptr;
    MtpRound::Cancel no_cancel;
    const auto& cancel_probe() const { return output ? output->cancel_probe() : no_cancel; }
    bool cancelled() const { return output && output->cancelled(); }
};

DecoderBatch::DecoderBatch(const Decoder& decoder)
    : DecoderBatch(decoder, {}) {}

DecoderBatch::DecoderBatch(const Decoder& decoder, DecoderBatchOptions options)
    : state_(std::make_unique<State>(*decoder.model_, options)) {}

DecoderBatch::~DecoderBatch() = default;

int DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    return admit_impl(slot, std::move(prompt), options, nullptr);
}

int DecoderBatch::admit_multimodal(int slot, std::vector<int> prompt, const DecoderOptions& options,
                                  const std::vector<DecoderMedia>& media) {
    return admit_impl(slot, std::move(prompt), options, &media);
}

std::size_t DecoderBatch::State::reserve(DecoderOptions& request, int prompt,
                                         std::size_t media_bytes) const {
    if (!options.resident_memory_limit) {
        return 0;
    }
    std::size_t charged = 0;
    for (const auto& lane : storage->slots) {
        charged += lane ? lane->resident_bytes : 0;
    }
    const auto free = available_device_bytes();
    const auto unreserved = free - std::min(free, decoder_shared_reserve);
    request.max_new_tokens = limit_resident_output(
        prompt, request, resident_capacity, std::min(unreserved, resident_capacity - charged),
        media_bytes);
    request.incremental_kv = false;
    return media_bytes + request_resident_bytes(RequestProgress::capacity(prompt, request),
        reserved_verification_rows(request.max_new_tokens, request.mtp_steps),
        request.mtp_steps != 0);
}

void DecoderBatch::State::initialize(Lane& lane) {
    const auto& opts = lane.options;
    const auto stream = execution.stream();
    const int prompt = static_cast<int>(lane.prompt.size());
    if (opts.mtp_steps != 0) {
        lane.draft = std::make_unique<MtpDraftRequest>(model.mtp_weights, lane.capacity());
    }
    lane.request->initialize(opts.incremental_kv ? prompt : lane.capacity(), stream,
                             opts.incremental_kv ? lane.capacity() : 0);
    if (opts.requires_target_only()) {
        biased_workspace.prepare(execution, *lane.request, opts, stream);
    }
    if (lane.media) {
        lane.media_prefill = upload_media_positions(*lane.request, *lane.media, stream);
    } else if (auto& cache = storage->prefix_cache) {
        lane.checkpoint = PrefixCache::checkpoint(prompt);
        lane.progress.advance(cache->restore(*lane.request, lane.prompt, opts.prefix_cache_salt,
                                             stream, lane.draft.get()));
        lane.counted_draft_rows = lane.draft ? lane.draft->valid_prefix() : 0;
    }
    if (lane.progress.position == 0) {
        lane.request->reset(stream);
    }
    lane.restored_prompt = lane.remaining_prompt() == 0;
    if (lane.resident_bytes) {
        if (const int rows = reserved_verification_rows(opts.max_new_tokens, opts.mtp_steps)) {
            lane.verification = std::make_unique<TargetVerification>(
                *lane.request, lane.capacity(), prompt, rows, false, stream);
        }
        if (lane.sampling && !opts.requires_target_only() && !sampling_workspace) {
            sampling_workspace = execution.allocate<std::byte>(sampling_workspace_bytes());
        }
        if (opts.constraint && !opts.requires_target_only()) {
            constraint_workspace.allocate(execution, opts.constraint->uses_wide_mask());
        }
    }
    check(cudaStreamSynchronize(stream), "Initialize decoder batch slot");
}

int DecoderBatch::admit_impl(int slot, std::vector<int> prompt, const DecoderOptions& requested,
                             const std::vector<DecoderMedia>* media) {
    state_->require_ready();
    auto& lane = state_->slot_at(slot);
    auto options = requested;
    if (lane || (options.top_logits != 0 && !state_->output)) {
        throw std::invalid_argument("Decoder batch requires an empty slot and no top logits");
    }
    auto input = media ? prepare_input(prompt, *media, state_->model.memory) : nullptr;
    if (input) {
        prompt = std::move(input->prompt.token_ids);
    }
    validate_request(prompt, options, !media);
    if (options.requires_target_only()) options.mtp_steps = 0;
    const int top = options.generated_top_logprobs;
    if (top < -1 || top > decoder_max_generated_top_logprobs ||
        (top >= 0 && (media || options.samples() || options.constraint || options.mtp_steps ||
                      options.top_logits || state_->output))) {
        throw std::invalid_argument("Generated logprobs require unconstrained greedy batch K=0");
    }
    if (options.prefix_cache_salt.size() > decoder_prefix_max_salt_bytes) {
        throw std::invalid_argument("Prefix cache salt exceeds 256 bytes");
    }
    if (state_->tensor_cache && options.mtp_steps != 0) {
        throw std::invalid_argument("Tensor-core prefix cache requires MTP off");
    }
    if (options.mtp_steps != 0 && !state_->model.mtp_weights) {
        throw std::invalid_argument("Decoder batch requires draft weights for MTP");
    }
    if (options.max_new_tokens == 0) {
        lane.emplace(Lane{nullptr, std::move(prompt), options});
        return 0;
    }
    try {
        DeviceScope device(state_->model.device);
        if (top >= 0 && !state_->logprob_workspace) {
            state_->logprob_readback = state_->execution.prepare_generated_logprobs();
            state_->logprob_workspace =
                state_->execution.allocate<std::byte>(generated_logprob_workspace_bytes());
        }
        if (input) {
            validate_feature_device(*input, state_->model.device);
        }
        reclaim_prefill_workspaces(state_->model.device);
        const auto resident = state_->reserve(options, static_cast<int>(prompt.size()),
            input ? input->prompt.positions.size() * sizeof(MropePosition) : 0);
        if (options.max_new_tokens != requested.max_new_tokens) {
            validate_request(prompt, options, !media);
        }
        auto request = std::make_unique<Request>(state_->model.device);
        lane.emplace(Lane{std::move(request), std::move(prompt), options});
        if (options.top_logits > 0) lane->request->memory.transfers().logits.resize(vocab_size);
        if (options.samples()) lane->sampling = request_sampling(options);
        lane->media = std::move(input);
        lane->resident_bytes = resident;
        state_->initialize(*lane);
        device.restore();
        return lane->progress.position;
    } catch (const CudaError& error) {
        const bool out_of_memory = error.status == cudaErrorMemoryAllocation;
        const auto pending = out_of_memory ? cudaGetLastError() : error.status;
        if (CudaError::terminal(pending)) {
            state_->disable();
            if (!out_of_memory) throw;
            check(pending, "Check CUDA error after failed decoder admission");
        }
        state_->drain("Drain failed decoder admission", &lane);
        if (!out_of_memory) throw;
        if (pending != cudaErrorMemoryAllocation) {
            check(pending, "Check CUDA error after failed decoder admission");
        }
        throw DecoderCapacityError("CUDA memory became unavailable during admission. Retry later.",
                                   true);
    } catch (...) {
        if (lane) state_->drain("Drain failed decoder admission", &lane);
        throw;
    }
}

void DecoderBatch::cancel(int slot) {
    state_->require_ready();
    auto& lane = state_->slot_at(slot);
    try {
        retire(lane);
    } catch (...) {
        state_->disable();
        throw;
    }
}

std::size_t DecoderBatch::request_device_allocation_bytes() const {
    return request_bytes(*state_->storage);
}

DecoderPrefixCacheStats DecoderBatch::prefix_cache_stats() const {
    const auto& cache = state_->storage->prefix_cache;
    return cache ? cache->stats() : DecoderPrefixCacheStats{};
}

void DecoderBatch::reset_prefix_cache() {
    state_->require_ready();
    auto& storage = *state_->storage;
    if (std::any_of(storage.slots.begin(), storage.slots.end(),
                    [](const auto& slot) { return slot.has_value(); })) {
        throw std::logic_error("Cannot reset prefix cache with active decoder slots");
    }
    if (storage.prefix_cache) {
        storage.prefix_cache->reset();
    }
}

bool DecoderBatch::available() const { return !state_->failed; }

void DecoderBatch::State::score(int slot, Lane& lane, Workspace& work, bool logits_ready) {
    const auto stream = execution.stream();
    if (!logits_ready) model.compute_logits(work, stream);
    auto* biased = lane.options.requires_target_only() ? &biased_workspace : nullptr;
    if (lane.options.generated_top_logprobs >= 0) {
        if (biased) biased->select(*lane.request, work, lane.options, nullptr, nullptr, 0, stream);
        check(enqueue_generated_logprobs(work.logits, vocab_size, work.argmax,
                                        lane.options.generated_top_logprobs, logprob_workspace,
                                        logprob_readback + slot, stream,
                                        biased ? biased->logits : nullptr), "Score token");
        return;
    }
    if (!biased && lane.sampling && !sampling_workspace) {
        sampling_workspace = execution.allocate<std::byte>(sampling_workspace_bytes());
    }
    if (!biased && lane.options.constraint) {
        constraint_workspace.allocate(execution, lane.options.constraint->uses_wide_mask());
    }
    lane.progress.select(*lane.request, work, lane.options, constraint_workspace,
                         lane.sampling ? &*lane.sampling : nullptr, sampling_workspace, stream,
                         biased);
}

void DecoderBatch::State::submit(const Selection& selected, const PrefillRows& decode,
                                 Steps& results, DecoderBatchStepStats& stats, bool& contains_mtp) {
    const auto stream = execution.stream();
    const auto& rows = stats.prefill_rows;
    auto& slots = storage->slots;
    std::array<ForwardLane, decoder_batch_max_lanes> work;
    std::array<Request*, decoder_batch_max_lanes> scoring{};
    std::array<int, decoder_batch_max_lanes> packed_slots;
    packed_slots.fill(-1);
    PrefillRows speculative{};
    const int max_rows = *std::max_element(rows.begin(), rows.end());
    bool text_decode = max_rows == 0;
    std::uint64_t greedy_winners = 0;
    int count = 0;
    for (int slot = 0; slot < options.max_lanes; ++slot) {
        if (!selected[slot] || !slots[slot] || !slots[slot]->request) {
            continue;
        }
        auto& lane = *slots[slot];
        if (lane.restored_prompt) {
            scoring[slot] = lane.request.get();
            continue;
        }
        const int remaining = lane.remaining_prompt();
        const int decode_rows = options.max_num_batched_tokens || remaining
            ? decode[slot] : lane.proposals() + 1;
        if (rows[slot] == 0 && decode_rows == 0) {
            continue;
        }
        if (lane.options.incremental_kv) {
            lane.request->grow_kv(lane.progress.position + std::max(rows[slot], decode_rows),
                                  lane.progress.position, stream);
        }
        results[slot].decode = remaining == 0;
        contains_mtp = contains_mtp || lane.draft != nullptr;
        const int position = lane.progress.position;
        if (rows[slot] && lane.draft && position > lane.draft->valid_prefix()) {
            model.seed_draft(lane.request->work, *lane.draft, position - 1,
                             lane.prompt[position], stream);
        }
        text_decode = text_decode && !lane.media;
        if (results[slot].decode && lane.draft && lane.remaining_output() > 1) {
            speculative[slot] = decode_rows;
            continue;
        }
        results[slot].mtp_boundary = results[slot].decode && lane.draft != nullptr;
        if (rows[slot] <= 1) {
            packed_slots[slot] = count;
            work[count++] = lane.input();
            const bool plain = results[slot].decode && !lane.draft && !lane.media;
            work[count - 1].packed = plain;
            work[count - 1].long_gqa =
                plain && model.fp8_prefill() == Fp8PrefillPolicy::tensor_core;
            if (plain && !lane.sampling && !lane.options.requires_target_only() &&
                !lane.options.constraint && lane.request->memory.transfers().logits.empty()) {
                greedy_winners |= std::uint64_t{1} << packed_slots[slot];
            }
        }
        if (rows[slot] == remaining) {
            scoring[slot] = lane.request.get();
        }
    }
    const auto run = run_decode_lanes(model, execution, storage->decode_graphs, work.data(), count,
                                      {options.cuda_graphs, text_decode && !contains_mtp,
                                       !contains_mtp && max_rows == 0,
                                       !contains_mtp &&
                                           model.fp8_prefill() == Fp8PrefillPolicy::tensor_core,
                                       greedy_winners});
    prefill(rows, max_rows, stats);
    bool submitted = count != 0 || max_rows > 1;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (auto* request = scoring[slot]) {
            auto& lane = *slots[slot];
            const int index = lane.restored_prompt ? -1 : packed_slots[slot];
            auto* ready = index < 0 ? nullptr : run.ready[index];
            if (index < 0 || !run.winners[index] || lane.options.generated_top_logprobs >= 0) {
                score(slot, lane, ready ? *ready : request->work, ready != nullptr);
            }
            submitted = true;
            if (index >= 0 && (run.graphed >> index & 1) != 0) {
                stats.graph_slots |= std::uint32_t{1} << slot;
            }
        }
    }
    stats.constraint_workspace_bytes = constraint_workspace.bytes();
    if (contains_mtp) {
        int next_slot = 0;
        while (next_slot < options.max_lanes) {
            speculate_group(speculative, next_slot, results, stats);
        }
    }
    stats.sampling_workspace_bytes = sampling_workspace ? sampling_workspace_bytes() : 0;
    if (stats.verification_lanes == 0 && submitted) {
        check(cudaStreamSynchronize(stream), "Complete decoder batch step");
    }
    collect_scores(scoring, packed_slots, run, results);
}

void DecoderBatch::State::prefill(const PrefillRows& rows, int max_rows,
                                  DecoderBatchStepStats& stats) {
    if (max_rows == 0) {
        return;
    }
    const auto stream = execution.stream();
    const int budget = options.max_num_batched_tokens;
    const int cache_tile = tensor_cache && budget ? cache_prefill_tile(budget) : 0;
    const int tile_limit = cache_tile ? cache_tile : batch_prefill_tile_rows;
    const bool wide = !cache_tile && model.is_b200() &&
        model.fp8_prefill() == Fp8PrefillPolicy::tensor_core && sm100_prefill_enabled();
    const auto tile_rows = [&](const Lane& lane, int count) {
        int limit = lane.draft || lane.media ? prefill_chunk_rows : tile_limit;
        if (wide && !lane.draft && !lane.media && lane.prompt.size() <= sm100_prefill_max_visible) {
            limit = serial_prefill_max_rows;
        }
        return std::min(count, limit);
    };
    int capacity = 1;
    for (int slot = 0; slot < options.max_lanes; ++slot) {
        if (rows[slot] > 1) {
            capacity = std::max(capacity, tile_rows(*storage->slots[slot], rows[slot]));
        }
    }
    PrefillWorkspace workspace(stream, capacity);
    if (tensor_cache && max_rows > 1 && !workspace.data()) {
        throw std::runtime_error("Tensor-core prefix cache requires a prefill workspace");
    }
    stats.prefill_workspace_bytes = workspace.bytes();
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (rows[slot] == 0) {
            continue;
        }
        auto& lane = *storage->slots[slot];
        if (rows[slot] == 1 && lane.draft && lane.prefill_limit() > 1) {
            const int position = lane.progress.position;
            model.seed_draft(lane.request->work, *lane.draft, position,
                             lane.prompt[position + 1], stream);
        } else if (rows[slot] > 1) {
            const auto* media = lane.media ? &lane.media_prefill : nullptr;
            for (int offset = 0; offset < rows[slot];) {
                const int count = tile_rows(lane, rows[slot] - offset);
                const int position = lane.progress.position + offset;
                model.prefill_chunk(*lane.request, lane.prompt.data() + position, count,
                                    position, workspace, lane.draft.get(),
                                    lane.prefill_limit() - offset, media);
                offset += count;
            }
        }
    }
    workspace.close();
}

void DecoderBatch::State::collect_scores(
    const std::array<Request*, decoder_batch_max_lanes>& scoring,
    const std::array<int, decoder_batch_max_lanes>& packed_slots, const DecodeRun& run,
    Steps& results) {
    const auto& slots = storage->slots;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (scoring[slot]) {
            const auto& lane = *slots[slot];
            const int index = packed_slots[slot];
            const auto* winner = index < 0 ? nullptr : run.winners[index];
            try {
                const int top = lane.options.generated_top_logprobs;
                if (top >= 0) {
                    const auto& readback = logprob_readback[slot];
                    results[slot].generated_logprobs = collect_generated_logprobs(readback, top);
                    results[slot].token = readback.selected.token_id;
                } else {
                    results[slot].token = winner
                        ? validate_winner(winner->token_id, false, false)
                        : read_token(*scoring[slot], lane.sampling.has_value(),
                                     lane.options.constraint != nullptr);
                }
            } catch (const DecoderSamplingError&) {
                results[slot].sampling_failed = true;
            } catch (const DecoderConstraintError&) {
                results[slot].constraint_failed = true;
            }
        }
    }
}

void DecoderBatch::State::speculate_group(const PrefillRows& verification_rows, int& next_slot,
                                          Steps& results, DecoderBatchStepStats& stats) {
    const auto stream = execution.stream();
    std::array<MtpRound, decoder_batch_max_mtp_lanes> rounds;
    std::array<int, decoder_batch_max_mtp_lanes> slots;
    int count = 0;
    while (next_slot < options.max_lanes && count < decoder_batch_max_mtp_lanes) {
        const int slot = next_slot++;
        const int rows = verification_rows[slot];
        if (rows == 0) {
            continue;
        }
        auto& lane = *storage->slots[slot];
        const int capacity = lane.proposals() + 1;
        if (!lane.verification) {
            lane.verification = std::make_unique<TargetVerification>(
                *lane.request, lane.capacity(), lane.progress.position, capacity,
                lane.options.top_logits > 0, stream);
        }
        if (lane.options.constraint) {
            if (!lane.constraint_round) {
                lane.constraint_round.emplace(*lane.request, *lane.options.constraint, capacity);
            }
            lane.constraint_round->begin(lane.progress.constraint, lane.remaining_output());
        }
        rounds[count] = {lane.draft.get(), lane.verification.get(), lane.request.get(),
                         lane.constraint_round ? &*lane.constraint_round : nullptr,
                         lane.sampling ? &*lane.sampling : nullptr, {lane.progress.last_token},
                         lane.progress.position, lane.progress.generated,
                         lane.remaining_output(), rows - 1};
        slots[count++] = slot;
    }
    if (count == 0) {
        return;
    }
    const auto& cancel = cancel_probe();
    if (!MtpRound::propose_all(rounds.data(), count, constraint_workspace, stream, cancel)) {
        check(cudaStreamSynchronize(stream), "Complete cancelled MTP draft wave");
        for (int index = 0; index < count; ++index) {
            auto& result = results[slots[index]];
            result.mtp_drafted_tokens = rounds[index].drafted;
            result.finish_reason = DecoderFinishReason::cancelled;
        }
        return;
    }
    auto* graph = storage->verification_graph ? &*storage->verification_graph : nullptr;
    if (model.verify_mtp(rounds.data(), count, constraint_workspace, sampling_workspace,
                         stream, graph)) {
        stats.verification_graph_slots |= std::uint32_t{1} << slots[0];
    }
    for (int index = 0; index < count; ++index) {
        const auto slot = slots[index];
        stats.verification_rows += rounds[index].rows();
        storage->slots[slot]->finish_round(rounds[index], results[slot], stream, output, cancel);
    }
    check(cudaStreamSynchronize(stream), "Complete batched MTP commits");
    stats.verification_lanes += count;
}

Steps DecoderBatch::State::advance(const Selection& selected, bool chunked,
                                   DecoderBatchStepStats* stats) {
    require_ready();
    bool contains_mtp = false;
    bool publishing_prefix = false;
    try {
        DeviceScope device(model.device);
        auto& slots = storage->slots;
        Steps results{};
        DecoderBatchStepStats current;
        PrefillRows decode{};
        if (options.max_num_batched_tokens) {
            const int budget = options.max_num_batched_tokens;
            const auto plan = tensor_cache
                ? plan_cached_step(*storage, selected, options.max_lanes, budget, prefill_cursor)
                : plan_step(*storage, selected, options.max_lanes, budget, prefill_cursor, chunked);
            current.prefill_rows = plan.prefill;
            decode = plan.decode;
        } else if (tensor_cache) {
            current.prefill_rows =
                plan_tensor_cache_prefill(*storage, selected, options.max_lanes, prefill_cursor);
        } else if (chunked && options.max_lanes > decoder_batch_max_mtp_lanes) {
            current.prefill_rows =
                plan_wide_prefill(*storage, selected, options.max_lanes, prefill_cursor);
        } else {
            current.prefill_rows = plan_prefill(*storage, selected, chunked);
        }
        const auto live_before = stats ? request_bytes(*storage) : 0;
        submit(selected, decode, results, current, contains_mtp);
        if (stats) {
            current.peak_request_bytes = std::max(live_before + current.prefill_workspace_bytes,
                                                 request_bytes(*storage));
        }
        for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
            auto& lane = slots[slot];
            auto& result = results[slot];
            if (!selected[slot] || !lane) {
                continue;
            }
            if (result.sampling_failed || result.constraint_failed) {
                retire(lane);
                continue;
            }
            if (!lane->request) {
                result.finish_reason = DecoderFinishReason::length;
            } else if (lane->restored_prompt) {
                result.finish_reason = lane->progress.commit(lane->options, *result.token);
                lane->restored_prompt = false;
            } else if (result.mtp_verified_rows == 0 &&
                       (result.decode || current.prefill_rows[slot] != 0)) {
                const int rows = current.prefill_rows[slot];
                if (auto& cache = storage->prefix_cache; cache && rows != 0 && !lane->media) {
                    const int draft_rows = lane->draft ? lane->draft->valid_prefix() : 0;
                    cache->count_prefill(rows, draft_rows - lane->counted_draft_rows);
                    lane->counted_draft_rows = draft_rows;
                    if (lane->progress.position + rows == lane->checkpoint) {
                        publishing_prefix = true;
                        cache->store(*lane->request, lane->prompt, lane->options.prefix_cache_salt,
                                     lane->checkpoint, execution.stream(), lane->draft.get());
                        publishing_prefix = false;
                    }
                }
                if (result.token) {
                    if (cancelled()) {
                        result.finish_reason = DecoderFinishReason::cancelled;
                    } else {
                        const auto& logits = lane->request->memory.transfers().logits;
                        result.finish_reason = lane->progress.commit(
                            lane->options, *result.token, logits.data(), output);
                    }
                }
                lane->progress.advance(std::max(1, rows));
            }
            if (result.finish_reason) {
                retire(lane);
            }
        }
        device.restore();
        if (stats) {
            for (int slot = 0; slot < options.max_lanes; ++slot) {
                current.target_rows += current.prefill_rows[slot];
                if (results[slot].decode) {
                    current.target_rows += std::max(1, results[slot].mtp_verified_rows);
                }
            }
            *stats = current;
        }
        return results;
    } catch (const CudaError& error) {
        fail_step(!CudaError::terminal(error.status) && (contains_mtp || publishing_prefix));
        throw;
    } catch (...) {
        fail_step(contains_mtp && !publishing_prefix);
        throw;
    }
}

void DecoderBatch::generate_mtp(const std::vector<int>& prompt, const DecoderOptions& options,
                                SerialOutput& output, DecoderResult& result) {
    state_->output = &output;
    admit(0, prompt, options);
    const auto model_bytes = result.peak_device_allocation_bytes;
    while (!state_->cancelled()) {
        DecoderBatchStepStats stats;
        const auto step = advance_chunked({true}, &stats).front();
        if (step.sampling_failed) throw DecoderSamplingError();
        if (step.constraint_failed) throw DecoderConstraintError();
        result.peak_device_allocation_bytes = std::max(result.peak_device_allocation_bytes,
            model_bytes + stats.peak_request_bytes + stats.sampling_workspace_bytes +
                stats.constraint_workspace_bytes);
        result.mtp_rounds += step.mtp_drafted_tokens > 0;
        result.mtp_accepted_tokens += step.mtp_accepted_tokens;
        result.mtp_drafted_tokens += step.mtp_drafted_tokens;
        if (step.finish_reason) {
            result.finish_reason = *step.finish_reason;
            return;
        }
    }
    cancel(0);
    result.finish_reason = DecoderFinishReason::cancelled;
}

Steps DecoderBatch::advance(const Selection& selected) {
    return state_->advance(selected, false, nullptr);
}

Steps DecoderBatch::advance_chunked(const Selection& selected, DecoderBatchStepStats* stats) {
    return state_->advance(selected, true, stats);
}

}
