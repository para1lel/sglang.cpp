#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_prefill.h"

#include <algorithm>
#include <utility>

namespace sglang {
namespace {

using namespace decoder;

struct Lane {
    std::unique_ptr<Request> request;
    std::vector<int> prompt;
    DecoderOptions options;
    int position = 0;
    int generated = 0;
    int last_token = 0;

    int remaining_prompt() const { return std::max(0, static_cast<int>(prompt.size()) - position); }

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

using Selection = std::array<bool, decoder_batch_max_lanes>;
using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;
using PrefillRows = std::array<int, decoder_batch_max_lanes>;

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
    void submit(const Selection& selected, const PrefillRows& rows, Steps& results,
                DecoderBatchStepStats& stats);
    bool replay(const ForwardLane* lanes, int count);

    void require_ready() const {
        if (failed) {
            throw std::runtime_error("Decoder batch is unavailable");
        }
    }

    void disable() noexcept {
        failed = true;
        try {
            execution.close();
        } catch (...) {
            return;
        }
        storage->slots = {};
        storage->graph_scratch.reset();
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
    validate_request(prompt, options);
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
        lane->request->initialize(capacity, stream);
        lane->request->reset(stream);
        check(cudaStreamSynchronize(stream), "Initialize decoder batch slot");
        device.restore();
    } catch (...) {
        state_->disable();
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
    std::size_t bytes = 0;
    for (const auto& lane : state_->storage->slots) {
        if (lane && lane->request) {
            bytes += lane->request->memory.bytes();
        }
    }
    return bytes;
}

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
            const auto& request = *lane.request;
            scratch.bindings[index] = {{0.0f, lane.token_id}, lane.position, request.pages,
                                       request.caches, request.states};
        }
        check(cudaMemcpyAsync(scratch.device_bindings, scratch.bindings.data(),
                              count * sizeof(BatchDecodeBinding), cudaMemcpyHostToDevice,
                              execution.stream()), "Upload batch decode bindings");
    });
}

void DecoderBatch::State::submit(const Selection& selected, const PrefillRows& rows,
                                Steps& results, DecoderBatchStepStats& stats) {
    const auto stream = execution.stream();
    auto& slots = storage->slots;
    std::array<ForwardLane, decoder_batch_max_lanes> work;
    std::array<Request*, decoder_batch_max_lanes> scoring{};
    std::array<int, decoder_batch_max_lanes> packed_slots{};
    const int max_rows = *std::max_element(rows.begin(), rows.end());
    int count = 0;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (selected[slot] && slots[slot] && slots[slot]->request) {
            auto& lane = *slots[slot];
            if (rows[slot] <= 1) {
                packed_slots[slot] = count;
                work[count++] = lane.input();
            }
            results[slot].decode = rows[slot] == 0;
            if (rows[slot] == lane.remaining_prompt()) {
                scoring[slot] = lane.request.get();
            }
        }
    }
    const bool graph = options.cuda_graphs && max_rows == 0 && count >= 2 &&
        std::all_of(work.begin(), work.begin() + count, [](const auto& lane) {
            const auto scores = lane.request->work.gqa_scores(text_attention_max_tokens);
            return lane.position < scores.capacity;
        }) && replay(work.data(), count);
    if (count != 0 && !graph) {
        model.forward(work.data(), count, stream);
    }
    if (max_rows > 1) {
        PrefillWorkspace workspace(stream, max_rows);
        stats.prefill_workspace_bytes = workspace.bytes();
        for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
            if (rows[slot] > 1) {
                auto& lane = *slots[slot];
                model.prefill_chunk(*lane.request, lane.prompt.data() + lane.position,
                                     rows[slot], lane.position, workspace);
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
    if (count != 0 || max_rows > 1) {
        check(cudaStreamSynchronize(stream), "Complete decoder batch step");
    }
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (scoring[slot]) {
            results[slot].token = read_greedy_token(*scoring[slot]);
        }
    }
}

Steps DecoderBatch::State::advance(const Selection& selected, bool chunked,
                                   DecoderBatchStepStats* stats) {
    require_ready();
    try {
        DeviceScope device(model.device);
        auto& slots = storage->slots;
        Steps results{};
        DecoderBatchStepStats submitted;
        submitted.prefill_rows = plan_prefill(*storage, selected, chunked);
        const auto& rows = submitted.prefill_rows;
        submit(selected, rows, results, submitted);
        for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
            auto& lane = slots[slot];
            auto& result = results[slot];
            if (!selected[slot] || !lane) {
                continue;
            }
            if (!lane->request) {
                result.finish_reason = DecoderFinishReason::length;
            } else {
                if (result.token) {
                    lane->select_token(*result.token, result);
                }
                lane->position += std::max(1, rows[slot]);
            }
            if (result.finish_reason) {
                retire(lane);
            }
        }
        device.restore();
        if (stats) {
            *stats = submitted;
        }
        return results;
    } catch (...) {
        disable();
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
