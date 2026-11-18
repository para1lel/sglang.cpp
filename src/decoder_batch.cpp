#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"

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

    ForwardLane input() const {
        if (static_cast<std::size_t>(position) < prompt.size()) {
            return {request.get(), prompt[position], position, TextGdnMode::prefill};
        }
        return {request.get(), last_token, position, TextGdnMode::decode};
    }

    bool produces_token() const { return static_cast<std::size_t>(position) + 1 >= prompt.size(); }

    void select_token(int token, DecoderBatchStep& result) {
        last_token = token;
        ++generated;
        if (options.stop_token_id == token) {
            result.finish_reason = DecoderFinishReason::stop;
        } else if (options.max_new_tokens == generated) {
            result.finish_reason = DecoderFinishReason::length;
        }
    }
};

struct BatchStorage {
    std::shared_ptr<const Allocations> weights;
    std::array<std::optional<Lane>, decoder_batch_max_lanes> slots;
};

void retire(std::optional<Lane>& lane) {
    if (lane && lane->request) {
        lane->request->memory.close();
    }
    lane.reset();
}

using Selection = std::array<bool, decoder_batch_max_lanes>;
using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;

}

struct DecoderBatch::State {
    explicit State(const Decoder::Model& owner)
        : model(owner), storage(std::make_shared<BatchStorage>()), execution(model.device) {
        DeviceScope device(model.device);
        storage->weights = model.memory;
        execution.open_stream(storage);
        device.restore();
    }

    Steps advance(const Selection& selected);

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
    }

    const Decoder::Model& model;
    std::shared_ptr<BatchStorage> storage;
    Allocations execution;
    bool failed = false;
};

DecoderBatch::DecoderBatch(const Decoder& decoder)
    : state_(std::make_unique<State>(*decoder.model_)) {}

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

Steps DecoderBatch::State::advance(const Selection& selected) {
    DeviceScope device(model.device);
    const auto stream = execution.stream();
    auto& slots = storage->slots;
    Steps results{};
    std::array<ForwardLane, decoder_batch_max_lanes> work;
    std::array<Request*, decoder_batch_max_lanes> scoring{};
    int count = 0;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (selected[slot] && slots[slot] && slots[slot]->request) {
            auto& lane = *slots[slot];
            const auto input = lane.input();
            work[count++] = input;
            results[slot].decode = input.mode == TextGdnMode::decode;
            if (lane.produces_token()) {
                scoring[slot] = input.request;
            }
        }
    }
    if (count != 0) {
        model.forward(work.data(), count, stream);
        for (auto* request : scoring) {
            if (request) {
                model.compute_logits(request->work, stream);
                copy_greedy_token(*request, stream);
            }
        }
        check(cudaStreamSynchronize(stream), "Complete decoder batch step");
    }
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (scoring[slot]) {
            results[slot].token = read_greedy_token(*scoring[slot]);
        }
    }
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
            ++lane->position;
        }
        if (result.finish_reason) {
            retire(lane);
        }
    }
    device.restore();
    return results;
}

Steps DecoderBatch::advance(const Selection& selected) {
    state_->require_ready();
    try {
        return state_->advance(selected);
    } catch (...) {
        state_->disable();
        throw;
    }
}

}
