#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_prefill.h"
#include "sglang/decoder_verify.h"

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

    int remaining_prompt() const { return std::max(0, static_cast<int>(prompt.size()) - position); }
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
        if (options.stops_at(token)) {
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
            const int token = verification->winner(index);
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
        if (!result.finish_reason) {
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
    std::array<std::optional<Workspace>, decoder_batch_max_lanes> work;
    BatchDecodeBinding* device_bindings;
    std::array<BatchDecodeBinding, decoder_batch_max_lanes> bindings{};
};

struct BatchStorage {
    std::shared_ptr<const Allocations> weights;
    std::array<std::optional<Lane>, decoder_batch_max_lanes> slots;
    std::unique_ptr<BatchGraphScratch> graph_scratch;
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
        if (selected[slot] && lane && lane->request) {
            --unused;
            remaining[slot] = lane->remaining_prompt();
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

}

struct DecoderBatch::State {
    State(const Decoder::Model& owner, DecoderBatchOptions options)
        : model(owner), options(options), storage(std::make_shared<BatchStorage>()),
          execution(model.device) {
        DeviceScope device(model.device);
        storage->weights = model.memory;
        execution.open_stream(storage);
        device.restore();
    }

    Steps advance(const Selection& selected, bool chunked, DecoderBatchStepStats* stats);
    bool replay(const ForwardLane* lanes, int count);
    void submit(const Selection& selected, Steps& results, DecoderBatchStepStats& stats,
                bool& contains_mtp);
    void speculate(const Selection& selected, Steps& results, DecoderBatchStepStats& stats);

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
    Allocations execution;
    std::array<bool, decoder_batch_max_lanes - 1> graph_warmed{};
    bool failed = false;
};

DecoderBatch::DecoderBatch(const Decoder& decoder)
    : DecoderBatch(decoder, {}) {}

DecoderBatch::DecoderBatch(const Decoder& decoder, DecoderBatchOptions options)
    : state_(std::make_unique<State>(*decoder.model_, options)) {}

DecoderBatch::~DecoderBatch() = default;

void DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    state_->require_ready();
    auto& lane = state_->storage->slots.at(slot);
    if (lane || options.top_logits != 0) {
        throw std::invalid_argument("Decoder batch requires an empty slot and no top logits");
    }
    validate_request(prompt, options, true);
    if (options.mtp_steps != 0 && !state_->model.mtp_weights) {
        throw std::invalid_argument("Decoder batch requires draft weights for MTP");
    }
    if (options.max_new_tokens == 0) {
        lane.emplace(Lane{nullptr, std::move(prompt), options});
        return;
    }
    try {
        DeviceScope device(state_->model.device);
        reclaim_prefill_workspaces(state_->model.device);
        const int capacity = static_cast<int>(prompt.size()) + options.max_new_tokens - 1;
        auto request = std::make_unique<Request>(state_->model.device);
        lane.emplace(Lane{std::move(request), std::move(prompt), options});
        const auto stream = state_->execution.stream();
        if (options.mtp_steps != 0) {
            lane->draft = std::make_unique<MtpDraftRequest>(state_->model.mtp_weights, capacity);
        }
        lane->request->initialize(capacity, stream);
        lane->request->reset(stream);
        check(cudaStreamSynchronize(stream), "Initialize decoder batch slot");
        device.restore();
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
    auto& lane = state_->storage->slots.at(slot);
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
        std::array<BatchGraphLane, decoder_batch_max_lanes> capture;
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
            if (lane.position >= scores.capacity) {
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
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (selected[slot] && slots[slot] && slots[slot]->request) {
            auto& lane = *slots[slot];
            results[slot].decode = rows[slot] == 0;
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
            if (rows[slot] == lane.remaining_prompt()) {
                scoring[slot] = lane.request.get();
            }
        }
    }
    const bool graph = graph_eligible && !contains_mtp && count >= 2 &&
        replay(work.data(), count);
    if (count != 0 && !graph) {
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
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (auto* request = scoring[slot]) {
            if (graph) {
                auto& scratch = *storage->graph_scratch;
                auto& winner = request->memory.transfers().winner;
                const auto* output = &scratch.work[packed_slots[slot]]->argmax->token_id;
                check(cudaMemcpyAsync(&winner, output, sizeof(winner), cudaMemcpyDeviceToHost,
                                      stream), "Read batch graph greedy token");
                stats.graph_slots |= 1u << slot;
                continue;
            }
            model.compute_logits(request->work, stream);
            copy_greedy_token(*request, stream);
        }
    }
    speculate(speculative, results, stats);
    if (stats.verification_lanes == 0 && (count != 0 || max_rows > 1)) {
        check(cudaStreamSynchronize(stream), "Complete decoder batch step");
    }
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (scoring[slot]) {
            results[slot].token = read_greedy_token(*scoring[slot]);
        }
    }
}

void DecoderBatch::State::speculate(const Selection& selected, Steps& results,
                                    DecoderBatchStepStats& stats) {
    const auto stream = execution.stream();
    std::array<DraftRound, decoder_batch_max_lanes> rounds;
    std::array<VerificationSequence, decoder_batch_max_lanes> sequences;
    std::array<TargetVerification*, decoder_batch_max_lanes> verifiers;
    int count = 0;
    int longest = 0;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
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
    model.forward_verify(sequences.data(), count, stream);
    TargetVerification::read_all(verifiers.data(), count, stream);
    for (int index = 0; index < count; ++index) {
        const auto& round = rounds[index];
        round.lane->finish_round(round.tokens, round.count + 1, results[round.slot], stream);
    }
    check(cudaStreamSynchronize(stream), "Complete batched MTP commits");
    stats.verification_lanes = count;
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
        current.prefill_rows = plan_prefill(*storage, selected, chunked);
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
            if (!lane->request) {
                result.finish_reason = DecoderFinishReason::length;
            } else if (result.mtp_drafted_tokens == 0) {
                if (result.token) {
                    lane->select_token(*result.token, result);
                }
                lane->position += std::max(1, current.prefill_rows[slot]);
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
