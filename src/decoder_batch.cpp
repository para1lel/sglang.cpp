#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
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

    int remaining_prompt() const { return std::max(0, static_cast<int>(prompt.size()) - position); }
    int prefill_limit() const {
        return checkpoint > position ? checkpoint - position : remaining_prompt();
    }
    int remaining_output() const { return options.max_new_tokens - generated; }
    int capacity() const { return static_cast<int>(prompt.size()) + options.max_new_tokens - 1; }

    ForwardLane input() const {
        if (remaining_prompt()) {
            return {request.get(), prompt[position], position, TextGdnMode::prefill};
        }
        return {request.get(), last_token, position, TextGdnMode::decode};
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

struct BatchGraphScratch {
    explicit BatchGraphScratch(int device)
        : memory(device), device_bindings(memory.allocate<BatchDecodeBinding>(work.size())) {
        for (auto& workspace : work) {
            workspace.emplace(memory);
        }
    }

    Allocations memory;
    std::array<std::optional<Workspace>, decoder_batch_max_graph_lanes> work;
    BatchDecodeBinding* device_bindings;
    std::array<BatchDecodeBinding, decoder_batch_max_graph_lanes> bindings{};
};

struct BatchStorage {
    std::shared_ptr<const Allocations> weights;
    std::array<std::optional<Lane>, decoder_batch_max_lanes> slots;
    std::unique_ptr<BatchGraphScratch> graph_scratch;
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
        DeviceScope device(model.device);
        storage->weights = model.memory;
        if (options.prefix_cache_entries != 0) {
            storage->prefix_cache.emplace(model.device, options.prefix_cache_entries);
        }
        execution.open_stream(storage);
        device.restore();
    }

    Steps advance(const Selection& selected, bool chunked, DecoderBatchStepStats* stats);
    bool replay(const ForwardLane* lanes, int count);
    void submit(const Selection& selected, Steps& results, DecoderBatchStepStats& stats,
                bool& contains_mtp);
    void score(Lane& lane, Request& request, Workspace& work, bool graph);
    void speculate(const Selection& selected, Steps& results, DecoderBatchStepStats& stats);
    void draft_proposals(DraftRound* rounds, int count, int longest);
    void collect_scores(const std::array<Request*, decoder_batch_max_lanes>& scoring, Steps&);
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
        storage->prefix_cache.reset();
    }

    bool drain_after_error() noexcept {
        try {
            DeviceScope device(model.device);
            check(cudaStreamSynchronize(execution.stream()), "Drain failed decoder batch step");
            return true;
        } catch (...) {
            disable();
            return false;
        }
    }

    const Decoder::Model& model;
    const DecoderBatchOptions options;
    std::shared_ptr<BatchStorage> storage;
    const bool tensor_cache;
    Allocations execution;
    std::byte* sampling_workspace = nullptr;
    ConstraintWorkspace constraint_workspace;
    std::array<bool, decoder_batch_max_graph_lanes - 1> graph_warmed{};
    int prefill_cursor = 0;
    bool failed = false;
};

DecoderBatch::DecoderBatch(const Decoder& decoder)
    : DecoderBatch(decoder, {}) {}

DecoderBatch::DecoderBatch(const Decoder& decoder, DecoderBatchOptions options)
    : state_(std::make_unique<State>(*decoder.model_, options)) {}

DecoderBatch::~DecoderBatch() = default;

int DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    state_->require_ready();
    auto& lane = state_->slot_at(slot);
    if (lane || options.top_logits != 0) {
        throw std::invalid_argument("Decoder batch requires an empty slot and no top logits");
    }
    validate_request(prompt, options, true);
    if (options.prefix_cache_salt.size() > decoder_prefix_max_salt_bytes) {
        throw std::invalid_argument("Prefix cache salt exceeds 256 bytes");
    }
    if (state_->tensor_cache && options.mtp_steps != 0) {
        throw std::invalid_argument("Tensor-core prefix cache requires MTP off");
    }
    if (options.mtp_steps != 0 && state_->options.max_lanes > decoder_batch_max_mtp_lanes) {
        throw std::invalid_argument("Decoder batch capacity above 4 requires MTP off");
    }
    if (options.mtp_steps != 0 && !state_->model.mtp_weights) {
        throw std::invalid_argument("Decoder batch requires draft weights for MTP");
    }
    if (options.max_new_tokens == 0) {
        lane.emplace(Lane{nullptr, std::move(prompt), options});
        return 0;
    }
    const auto sampling = options.samples() ? std::optional{request_sampling(options)}
                                            : std::nullopt;
    try {
        DeviceScope device(state_->model.device);
        reclaim_prefill_workspaces(state_->model.device);
        const int capacity = static_cast<int>(prompt.size()) + options.max_new_tokens - 1;
        auto request = std::make_unique<Request>(state_->model.device);
        lane.emplace(Lane{std::move(request), std::move(prompt), options});
        lane->sampling = sampling;
        const auto stream = state_->execution.stream();
        if (options.mtp_steps != 0) {
            lane->draft = std::make_unique<MtpDraftRequest>(state_->model.mtp_weights, capacity);
        }
        lane->request->initialize(capacity, stream);
        if (auto& cache = state_->storage->prefix_cache; cache && options.mtp_steps == 0) {
            lane->checkpoint = PrefixCache::checkpoint(static_cast<int>(lane->prompt.size()));
            lane->position = cache->restore(*lane->request, lane->prompt, options.prefix_cache_salt,
                                            stream);
        }
        if (lane->position == 0) {
            lane->request->reset(stream);
        }
        lane->restored_prompt = lane->remaining_prompt() == 0;
        check(cudaStreamSynchronize(stream), "Initialize decoder batch slot");
        device.restore();
        return lane->position;
    } catch (...) {
        if (options.mtp_steps == 0) {
            state_->disable();
        } else if (state_->drain_after_error()) {
            lane.reset();
        }
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
    auto& warmed = graph_warmed[count - 2];
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

void DecoderBatch::State::submit(const Selection& selected, Steps& results,
                                 DecoderBatchStepStats& stats, bool& contains_mtp) {
    const auto stream = execution.stream();
    const auto& rows = stats.prefill_rows;
    auto& slots = storage->slots;
    std::array<ForwardLane, decoder_batch_max_lanes> work;
    std::array<Request*, decoder_batch_max_lanes> scoring{};
    std::array<int, decoder_batch_max_lanes> packed_slots{};
    Selection speculative{};
    const int max_rows = *std::max_element(rows.begin(), rows.end());
    const bool graph_eligible = options.cuda_graphs && max_rows == 0;
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
        if (rows[slot] == 0 && remaining > 0) {
            continue;
        }
        results[slot].decode = remaining == 0;
        contains_mtp = contains_mtp || lane.draft != nullptr;
        if (results[slot].decode && lane.draft && lane.remaining_output() > 1) {
            speculative[slot] = true;
            continue;
        }
        results[slot].mtp_boundary = results[slot].decode && lane.draft != nullptr;
        if (rows[slot] <= 1) {
            packed_slots[slot] = count;
            work[count++] = lane.input();
        }
        if (rows[slot] == remaining) {
            scoring[slot] = lane.request.get();
        }
    }
    const bool graph = graph_eligible && !contains_mtp &&
        count >= 2 && count <= decoder_batch_max_graph_lanes && replay(work.data(), count);
    if (count != 0 && !graph) {
        work[0].narrow_gqa = count == 1 && !contains_mtp && max_rows == 0;
        model.forward(work.data(), count, stream);
    }
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        auto& lane = slots[slot];
        if (rows[slot] == 1 && lane->draft && lane->remaining_prompt() > 1) {
            model.normalize_final(lane->request->work, stream);
            lane->draft->enqueue(lane->position, lane->prompt[lane->position + 1],
                                 lane->request->work.normalized, false, stream);
        }
    }
    if (max_rows > 1) {
        PrefillWorkspace workspace(stream, max_rows);
        if (tensor_cache && !workspace.data()) {
            throw std::runtime_error("Tensor-core prefix cache requires a prefill workspace");
        }
        stats.prefill_workspace_bytes = workspace.bytes();
        for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
            if (rows[slot] > 1) {
                auto& lane = *slots[slot];
                model.prefill_chunk(*lane.request, lane.prompt.data() + lane.position,
                                     rows[slot], lane.position, workspace, lane.draft.get(),
                                     lane.remaining_prompt());
            }
        }
        workspace.close();
    }
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
        speculate(speculative, results, stats);
    }
    stats.sampling_workspace_bytes = sampling_workspace ? sampling_workspace_bytes() : 0;
    if (stats.verification_lanes == 0 && submitted) {
        check(cudaStreamSynchronize(stream), "Complete decoder batch step");
    }
    collect_scores(scoring, results);
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

void DecoderBatch::State::speculate(const Selection& selected, Steps& results,
                                    DecoderBatchStepStats& stats) {
    const auto stream = execution.stream();
    std::array<DraftRound, decoder_batch_max_mtp_lanes> rounds;
    std::array<VerificationSequence, decoder_batch_max_mtp_lanes> sequences;
    std::array<TargetVerification*, decoder_batch_max_mtp_lanes> verifiers;
    int count = 0;
    int longest = 0;
    for (int slot = 0; slot < options.max_lanes; ++slot) {
        if (selected[slot]) {
            auto& lane = *storage->slots[slot];
            const int proposals = std::min(lane.options.mtp_steps, lane.remaining_output() - 1);
            if (!lane.verification) {
                lane.verification = std::make_unique<TargetVerification>(
                    *lane.request, lane.capacity(), lane.position, proposals + 1, false, stream);
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
    model.forward_verify(sequences.data(), count, stream);
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
    stats.verification_lanes = count;
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
        if (tensor_cache) {
            current.prefill_rows =
                plan_tensor_cache_prefill(*storage, selected, options.max_lanes, prefill_cursor);
        } else if (chunked && options.max_lanes > decoder_batch_max_mtp_lanes) {
            current.prefill_rows =
                plan_wide_prefill(*storage, selected, options.max_lanes, prefill_cursor);
        } else {
            current.prefill_rows = plan_prefill(*storage, selected, chunked);
        }
        const auto live_before = stats ? request_bytes(*storage) : 0;
        submit(selected, results, current, contains_mtp);
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
            } else if (result.mtp_drafted_tokens == 0 &&
                       (result.decode || current.prefill_rows[slot] != 0)) {
                const int rows = current.prefill_rows[slot];
                if (auto& cache = storage->prefix_cache; cache && rows != 0 &&
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
            *stats = current;
        }
        return results;
    } catch (...) {
        if (!contains_mtp) {
            disable();
        } else if (drain_after_error()) {
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
