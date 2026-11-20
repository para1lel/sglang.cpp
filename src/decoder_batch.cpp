#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_prefill.h"
#include "sglang/decoder_sampling.h"
#include "sglang/decoder_verify.h"
#include "sglang/decoder_prefix_cache.h"

#include <algorithm>
#include <utility>

namespace sglang {
namespace {

using namespace decoder;

using VerificationTokens = std::array<int, target_verify_max_rows>;
static_assert(decoder_batch_max_mtp_steps + 1 == target_verify_max_rows);

struct Lane {
    std::unique_ptr<Request> request;
    std::vector<int> prompt;
    DecoderOptions options;
    int position = 0;
    int generated = 0;
    int last_token = 0;
    std::unique_ptr<MtpDraftRequest> draft = nullptr;
    std::unique_ptr<TargetVerification> verification = nullptr;
    std::optional<SamplingParameters> sampling = std::nullopt;
    ConstraintCursor constraint{};
    int checkpoint = 0;
    bool restored_prompt = false;
    std::shared_ptr<const MultimodalInput> media = nullptr;
    MediaPrefill media_prefill{};
    std::size_t resident_bytes = 0;

    int remaining_prompt() const { return std::max(0, static_cast<int>(prompt.size()) - position); }
    int prefill_limit() const {
        return checkpoint > position ? checkpoint - position : remaining_prompt();
    }
    int remaining_output() const { return options.max_new_tokens - generated; }
    int capacity() const { return static_cast<int>(prompt.size()) + options.max_new_tokens - 1; }
    int proposals() const { return std::min(options.mtp_steps, remaining_output() - 1); }

    ForwardLane input() const {
        if (remaining_prompt()) {
            return {request.get(), prompt[position], position, TextGdnMode::prefill, false,
                    media ? media_prefill.feature(position) : nullptr,
                    media ? media_prefill.positions + position : nullptr};
        }
        const int rotary = media ? position + (media->prompt.next_position - int(prompt.size()))
                                 : -1;
        return {request.get(), last_token, position, TextGdnMode::decode, false, nullptr, nullptr,
                rotary};
    }

    void select_token(int token, DecoderBatchStep& result) {
        last_token = token;
        ++generated;
        if (options.constraint) {
            if (constraint.commit(*options.constraint, token, remaining_output())) {
                result.finish_reason = DecoderFinishReason::constraint;
            }
        } else if (options.stops_at(token)) {
            result.finish_reason = DecoderFinishReason::stop;
        } else if (options.max_new_tokens == generated) {
            result.finish_reason = DecoderFinishReason::length;
        }
    }

    void finish_round(const VerificationTokens& tokens, int rows, DecoderBatchStep& result,
                      cudaStream_t stream) {
        result.mtp_drafted_tokens = rows - 1;
        result.mtp_verified_rows = rows;
        draft->rewind(position);
        int consumed = 0;
        for (int index = 0; index < rows; ++index) {
            int token;
            try {
                token = verification->winner(index, sampling.has_value());
            } catch (const DecoderSamplingError&) {
                result.sampling_failed = true;
                break;
            }
            if (index == 0) {
                result.token = token;
            } else {
                result.extra_tokens[result.extra_token_count++] = token;
            }
            ++consumed;
            const bool accepted = index < rows - 1 && token == tokens[index + 1];
            result.mtp_accepted_tokens += accepted;
            select_token(token, result);
            if (!accepted || result.finish_reason) {
                break;
            }
            const auto row = verification->row(index);
            draft->enqueue(row.position, token, row.work->normalized, false, stream);
        }
        if (result.extra_token_count != 0) {
            result.extra_token = result.extra_tokens.front();
        }
        if (!result.finish_reason && !result.sampling_failed) {
            const auto* hidden = verification->row(consumed - 1).work->normalized;
            check(cudaMemcpyAsync(request->work.normalized, hidden, hidden_size * sizeof(Bf16),
                                  cudaMemcpyDeviceToDevice, stream),
                  "Carry batched target hidden state into next draft");
        }
        result.mtp_replayed_rows = consumed < rows ? consumed : 0;
        verification->commit(consumed);
        position += consumed;
    }
};

struct BatchStorage {
    std::shared_ptr<const Allocations> weights;
    std::array<std::optional<Lane>, decoder_batch_max_lanes> slots;
    std::unique_ptr<BatchGraphScratch> graph_scratch;
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

struct DraftRound {
    Lane* lane;
    VerificationTokens tokens;
    int count;
    const Bf16* hidden;
    int slot;
};

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
            storage->prefix_cache.emplace(model.device, options.prefix_cache_entries);
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
    bool replay(const ForwardLane* lanes, int count);
    void submit(const Selection& selected, const PrefillRows& decode, Steps& results,
                DecoderBatchStepStats& stats, bool& contains_mtp);
    void prefill(const PrefillRows& rows, int max_rows, DecoderBatchStepStats& stats);
    void score(Lane& lane, Request& request, Workspace& work, bool graph);
    void speculate_group(const PrefillRows& verification_rows, int& next_slot, Steps& results,
                         DecoderBatchStepStats& stats);
    void draft_proposals(DraftRound* rounds, int count, int longest);
    void collect_scores(const std::array<Request*, decoder_batch_max_lanes>& scoring, Steps&);
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
        storage->graph_scratch.reset();
        storage->verification_graph.reset();
        storage->prefix_cache.reset();
    }

    const Decoder::Model& model;
    const DecoderBatchOptions options;
    std::shared_ptr<BatchStorage> storage;
    const bool tensor_cache;
    Allocations execution;
    std::byte* sampling_workspace = nullptr;
    ConstraintWorkspace constraint_workspace;
    std::size_t resident_capacity = 0;
    std::array<bool, decoder_batch_max_graph_lanes> graph_warmed{};
    int prefill_cursor = 0;
    bool failed = false;
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
    return media_bytes + request_resident_bytes(prompt + request.max_new_tokens - 1,
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
    if (lane.media) {
        lane.media_prefill = upload_media_positions(*lane.request, *lane.media, stream);
    } else if (auto& cache = storage->prefix_cache; cache && !lane.draft) {
        lane.checkpoint = PrefixCache::checkpoint(prompt);
        lane.position = cache->restore(*lane.request, lane.prompt, opts.prefix_cache_salt, stream);
    }
    if (lane.position == 0) {
        lane.request->reset(stream);
    }
    lane.restored_prompt = lane.remaining_prompt() == 0;
    if (lane.resident_bytes) {
        if (const int rows = reserved_verification_rows(opts.max_new_tokens, opts.mtp_steps)) {
            lane.verification = std::make_unique<TargetVerification>(
                *lane.request, lane.capacity(), prompt, rows, false, stream);
        }
        if (lane.sampling && !sampling_workspace) {
            sampling_workspace = execution.allocate<std::byte>(sampling_workspace_bytes());
        }
        if (opts.constraint) {
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
    if (lane || options.top_logits != 0) {
        throw std::invalid_argument("Decoder batch requires an empty slot and no top logits");
    }
    auto input = media ? prepare_input(prompt, *media, state_->model.memory) : nullptr;
    if (input) {
        prompt = std::move(input->prompt.token_ids);
    }
    validate_request(prompt, options, !media);
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
        if (options.samples()) lane->sampling = request_sampling(options);
        lane->media = std::move(input);
        lane->resident_bytes = resident;
        state_->initialize(*lane);
        device.restore();
        return lane->position;
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

bool DecoderBatch::State::replay(const ForwardLane* lanes, int count) {
    if (count == 1 && lanes[0].position >=
        lanes[0].request->work.gqa_scores(text_attention_max_tokens).capacity) {
        return false;
    }
    auto& warmed = graph_warmed[count - 1];
    if (!std::exchange(warmed, true)) {
        return false;
    }
    return execution.replay_batch_graph(count, [&] {
        if (!storage->graph_scratch) {
            storage->graph_scratch = std::make_unique<BatchGraphScratch>(model.device);
        }
    }, [&] {
        auto& scratch = *storage->graph_scratch;
        std::array<BatchGraphLane, decoder_batch_max_graph_lanes> capture;
        for (int index = 0; index < count; ++index) {
            capture[index] = {&*scratch.work[index], scratch.device_bindings + index};
        }
        model.capture_batch_decode(capture.data(), count, execution.stream());
    }, [&] {
        auto& scratch = *storage->graph_scratch;
        for (int index = 0; index < count; ++index) {
            const auto& lane = lanes[index];
            auto& request = *lane.request;
            auto scores = scratch.work[index]->gqa_scores(text_attention_max_tokens);
            if (paged_decode_position(lane.position)) {
                scores = {};
            } else if (lane.position >= scores.capacity) {
                scores = request.batch_graph_scores();
                if (lane.position >= scores.capacity) {
                    return false;
                }
            }
            scratch.bindings[index] = {{0.0f, lane.token_id}, lane.position, scores,
                                       request.pages, request.caches, request.states};
        }
        check(cudaMemcpyAsync(scratch.device_bindings, scratch.bindings.data(),
                              count * sizeof(BatchDecodeBinding), cudaMemcpyHostToDevice,
                              execution.stream()), "Upload batch decode bindings");
        return true;
    });
}

void DecoderBatch::State::score(Lane& lane, Request& request, Workspace& work, bool graph) {
    const auto stream = execution.stream();
    if (!graph) {
        model.compute_logits(work, stream);
    }
    if (lane.sampling && !sampling_workspace) {
        sampling_workspace = execution.allocate<std::byte>(sampling_workspace_bytes());
    }
    if (lane.options.constraint) {
        constraint_workspace.allocate(execution, lane.options.constraint->uses_wide_mask());
        lane.constraint.candidates(*lane.options.constraint, lane.remaining_output());
        select_constrained(work, constraint_workspace, lane.constraint,
                            lane.sampling ? &*lane.sampling : nullptr, lane.generated,
                            sampling_workspace, stream);
    } else if (lane.sampling) {
        check(sample_logits(work.logits, vocab_size, &work.argmax->token_id,
                            *lane.sampling, lane.generated, sampling_workspace, stream),
              "Sample batched decoder logits");
    }
    copy_token(request, work, stream);
}

void DecoderBatch::State::submit(const Selection& selected, const PrefillRows& decode,
                                 Steps& results, DecoderBatchStepStats& stats, bool& contains_mtp) {
    const auto stream = execution.stream();
    const auto& rows = stats.prefill_rows;
    auto& slots = storage->slots;
    std::array<ForwardLane, decoder_batch_max_lanes> work;
    std::array<Request*, decoder_batch_max_lanes> scoring{};
    std::array<int, decoder_batch_max_lanes> packed_slots{};
    PrefillRows speculative{};
    const int max_rows = *std::max_element(rows.begin(), rows.end());
    bool graph_eligible = options.cuda_graphs && max_rows == 0;
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
            lane.request->grow_kv(lane.position + std::max(rows[slot], decode_rows),
                                  lane.position, stream);
        }
        results[slot].decode = remaining == 0;
        contains_mtp = contains_mtp || lane.draft != nullptr;
        graph_eligible = graph_eligible && !lane.media;
        if (results[slot].decode && lane.draft && lane.remaining_output() > 1) {
            speculative[slot] = decode_rows;
            continue;
        }
        results[slot].mtp_boundary = results[slot].decode && lane.draft != nullptr;
        if (rows[slot] <= 1) {
            packed_slots[slot] = count;
            work[count++] = lane.input();
            work[count - 1].long_gqa = results[slot].decode && !lane.draft && !lane.media &&
                model.fp8_prefill() == Fp8PrefillPolicy::tensor_core;
        }
        if (rows[slot] == remaining) {
            scoring[slot] = lane.request.get();
        }
    }
    const bool graph = graph_eligible && !contains_mtp && count > 0 &&
        count <= decoder_batch_max_graph_lanes && replay(work.data(), count);
    if (count != 0 && !graph) {
        work[0].narrow_gqa = count == 1 && !contains_mtp && max_rows == 0 &&
            work[0].rotary_position == -1;
        model.forward(work.data(), count, stream);
    }
    prefill(rows, max_rows, stats);
    bool submitted = count != 0 || max_rows > 1;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (auto* request = scoring[slot]) {
            auto& lane = *slots[slot];
            const bool packed = graph && !lane.restored_prompt;
            auto& workspace = packed ? *storage->graph_scratch->work[packed_slots[slot]]
                                     : request->work;
            score(lane, *request, workspace, packed);
            submitted = true;
            if (packed) {
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
    collect_scores(scoring, results);
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
    const auto tile_rows = [&](const Lane& lane, int count) {
        const int limit = lane.draft || lane.media ? prefill_chunk_rows : tile_limit;
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
        if (rows[slot] == 1 && lane.draft && lane.remaining_prompt() > 1) {
            model.normalize_final(lane.request->work, stream);
            lane.draft->enqueue(lane.position, lane.prompt[lane.position + 1],
                                lane.request->work.normalized, false, stream);
        } else if (rows[slot] > 1) {
            const auto* media = lane.media ? &lane.media_prefill : nullptr;
            for (int offset = 0; offset < rows[slot];) {
                const int count = tile_rows(lane, rows[slot] - offset);
                const int position = lane.position + offset;
                model.prefill_chunk(*lane.request, lane.prompt.data() + position, count,
                                    position, workspace, lane.draft.get(),
                                    lane.remaining_prompt() - offset, media);
                offset += count;
            }
        }
    }
    workspace.close();
}

void DecoderBatch::State::collect_scores(
    const std::array<Request*, decoder_batch_max_lanes>& scoring, Steps& results) {
    const auto& slots = storage->slots;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (scoring[slot]) {
            const auto& lane = *slots[slot];
            try {
                results[slot].token = read_token(*scoring[slot], lane.sampling.has_value(),
                                                  lane.options.constraint != nullptr);
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
    std::array<DraftRound, decoder_batch_max_mtp_lanes> rounds;
    std::array<VerificationSequence, decoder_batch_max_mtp_lanes> sequences;
    std::array<TargetVerification*, decoder_batch_max_mtp_lanes> verifiers;
    int count = 0;
    int longest = 0;
    while (next_slot < options.max_lanes && count < decoder_batch_max_mtp_lanes) {
        const int slot = next_slot++;
        const int rows = verification_rows[slot];
        if (rows) {
            auto& lane = *storage->slots[slot];
            const int proposals = rows - 1;
            if (!lane.verification) {
                lane.verification = std::make_unique<TargetVerification>(
                    *lane.request, lane.capacity(), lane.position, lane.proposals() + 1,
                    false, stream);
            }
            if (proposals == 0) {
                lane.draft->enqueue(lane.position - 1, lane.last_token,
                                    lane.request->work.normalized, false, stream);
            }
            rounds[count] = {&lane, {lane.last_token}, proposals,
                             lane.request->work.normalized, slot};
            verifiers[count] = lane.verification.get();
            sequences[count] = {verifiers[count], lane.position, rounds[count].tokens.data(),
                                proposals + 1};
            longest = std::max(longest, proposals);
            stats.verification_rows += proposals + 1;
            ++count;
        }
    }
    if (count == 0) {
        return;
    }
    draft_proposals(rounds.data(), count, longest);
    const auto& lane = *rounds[0].lane;
    auto* graph = storage->verification_graph && count == 1 && lane.options.mtp_steps == 1 &&
        !lane.media && !lane.options.constraint ? &*storage->verification_graph : nullptr;
    if (model.forward_verify(sequences.data(), count, stream, graph)) {
        stats.verification_graph_slots |= std::uint32_t{1} << rounds[0].slot;
    }
    for (int index = 0; index < count; ++index) {
        auto& lane = *rounds[index].lane;
        if (lane.sampling) {
            if (!sampling_workspace) {
                sampling_workspace = execution.allocate<std::byte>(sampling_workspace_bytes());
            }
            lane.verification->sample(*lane.sampling, lane.generated, sampling_workspace);
        }
    }
    TargetVerification::read_all(verifiers.data(), count, stream);
    for (int index = 0; index < count; ++index) {
        const auto& round = rounds[index];
        round.lane->finish_round(round.tokens, round.count + 1, results[round.slot], stream);
    }
    check(cudaStreamSynchronize(stream), "Complete batched MTP commits");
    stats.verification_lanes += count;
}

void DecoderBatch::State::draft_proposals(DraftRound* rounds, int count, int longest) {
    const auto stream = execution.stream();
    for (int wave = 0; wave < longest; ++wave) {
        for (int index = 0; index < count; ++index) {
            auto& round = rounds[index];
            if (wave < round.count) {
                round.hidden = round.lane->draft->enqueue(round.lane->position - 1 + wave,
                    round.tokens[wave], round.hidden, true, stream);
            }
        }
        check(cudaStreamSynchronize(stream), "Complete batch MTP draft wave");
        for (int index = 0; index < count; ++index) {
            auto& round = rounds[index];
            if (wave < round.count) {
                round.tokens[wave + 1] = round.lane->draft->read_token();
            }
        }
    }
}

Steps DecoderBatch::State::advance(const Selection& selected, bool chunked,
                                   DecoderBatchStepStats* stats) {
    require_ready();
    bool contains_mtp = false;
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
                lane->select_token(*result.token, result);
                lane->restored_prompt = false;
            } else if (result.mtp_verified_rows == 0 &&
                       (result.decode || current.prefill_rows[slot] != 0)) {
                const int rows = current.prefill_rows[slot];
                if (auto& cache = storage->prefix_cache; cache && rows != 0 && !lane->media &&
                    lane->options.mtp_steps == 0) {
                    cache->count_prefill(rows);
                    if (lane->position + rows == lane->checkpoint) {
                        cache->store(*lane->request, lane->prompt, lane->options.prefix_cache_salt,
                                     lane->checkpoint, execution.stream());
                    }
                }
                if (result.token) {
                    lane->select_token(*result.token, result);
                }
                lane->position += std::max(1, rows);
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
    } catch (...) {
        if (!contains_mtp || execution.verification_graph_failed()) {
            disable();
        } else {
            drain("Drain failed decoder batch step");
            storage->slots = {};
        }
        throw;
    }
}

Steps DecoderBatch::advance(const Selection& selected) {
    return state_->advance(selected, false, nullptr);
}

Steps DecoderBatch::advance_chunked(const Selection& selected, DecoderBatchStepStats* stats) {
    return state_->advance(selected, true, stats);
}

}
