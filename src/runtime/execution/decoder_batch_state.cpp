#include "sglang/runtime/execution/decoder_batch_state.h"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace sglang {

namespace decoder::batch_detail {

BatchStorage::BatchStorage(int device, std::shared_ptr<const Allocations> memory,
                           bool resident, int lanes)
    : weights(std::move(memory)), retired(device, resident), admission(lanes) {}

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

StepPlan plan_step(const BatchStorage& storage, const Selection& selected, int lanes,
                          int budget, bool chunked, WideMtpCandidates* wide,
                          bool tensor_parallel) {
    BudgetLanes work{};
    for (int slot = 0; slot < lanes; ++slot) {
        const auto& lane = storage.slots[slot];
        if (!selected[slot] || !lane || !lane->request || lane->restored_prompt) continue;
        work[slot] = lane->remaining_prompt()
            ? BudgetLane{chunked ? lane->prefill_rows(lane->checkpoints()) : 1}
            : BudgetLane{0, std::max(0, lane->proposals(lane->checkpoints())), true};
        if (wide && work[slot].decode && lane->draft && lane->remaining_output() > 1) {
            const auto bit = std::uint32_t{1} << slot;
            wide->speculative |= bit;
            if (work[slot].proposals == decoder_batch_max_mtp_steps &&
                lane->remaining_output() >= target_verify_max_rows &&
                !lane->options.constraint && !lane->sampling && !lane->options.top_logits &&
                (!tensor_parallel || lane->options.generated_top_logprobs < 0)) {
                wide->eligible |= bit;
                ++wide->eligible_count;
            }
        }
    }
    return plan_budget_step(work, storage.admission, budget ? budget : prefill_chunk_rows,
                            budget != 0);
}

}

DecoderBatch::State::State(const Decoder::Model& owner, DecoderBatchOptions options,
      Decoder::TpRuntime* parallel)
    : model(owner), options(options),
      residency(options),
      storage(std::make_shared<BatchStorage>(owner.device, owner.memory,
                                             residency.resident(), options.max_lanes)),
      execution(model.device), tp(parallel),
      tp_driver(parallel ? std::make_unique<TpDriver>(owner, *parallel) : nullptr) {
    validate_batch_options(options);
    DeviceScope device(model.device);

    if (options.cuda_graphs && !tp) storage->verification_graph.emplace(execution);
    if (residency.resident() && model.mtp_weights)
        storage->verification_winners = prepare_verification_winners(options.max_lanes);
    execution.open_stream(storage);
    storage->tc_vocab = model.tc_vocab
        ? prepare_tc_vocab_scratch(execution, options.max_lanes >= 4, residency.resident(),
                                   tp && options.max_lanes >= 2)
        : nullptr;
    if (options.max_lanes >= wide_mtp_lanes &&
        options.max_num_batched_tokens >= wide_mtp_rows &&
        model.mtp_weights && storage->tc_vocab) {

        wide_mtp_limit = tp
            ? std::min(model.backend.mtp_verification_lanes, wide_mtp_lanes * 2)
            : model.backend.mtp_verification_lanes;

        if (tp && std::getenv("SGLANG_DISABLE_TP2_WIDE_VERIFY")) {
            wide_mtp_limit = 0;
        }
    }
    if (options.resident_memory_limit) {
        residency.measure_available(available_device_bytes());
    }
    device.restore();
}

void DecoderBatch::State::drain(const char* operation, std::optional<Lane>* lane) {
    try {
        DeviceScope device(model.device);
        check(cudaStreamSynchronize(execution.stream()), operation);
        if (lane) retire(*lane, tp_driver.get());
    } catch (...) {
        disable();
        throw;
    }
}

void DecoderBatch::State::fail_step(bool recoverable) {
    if (!recoverable || execution.verification_graph_failed()) {
        disable();
        return;
    }
    drain("Drain failed decoder batch step");
    for (auto& lane : storage->slots) {
        if (lane && lane->prefix && lane->prefix->completion.valid()) {
            lane->prefix->discard = true;
        } else {
            lane.reset();
        }
    }
}

std::optional<Lane>& DecoderBatch::State::slot_at(int slot) {
    if (slot < 0 || slot >= options.max_lanes) {
        throw std::out_of_range("Invalid decoder batch slot");
    }
    return storage->slots[slot];
}

void DecoderBatch::State::require_ready() const {
    if (failed) {
        throw std::runtime_error("Decoder batch is unavailable");
    }
}

void DecoderBatch::State::disable() noexcept {
    failed = true;
    if (tp_driver) tp_driver->sync_peer_stream();
    try {
        DeviceScope device(model.device);
        execution.close();
    } catch (...) {
        return;
    }
    storage->slots = {};
    storage->decode_graphs = {};
    storage->verification_graph.reset();
}

TextGdnReplayBatch* DecoderBatch::State::prepare_gdn_replays() {
    static const bool batch_disabled =
        std::getenv("SGLANG_DISABLE_GDN_REPLAY_BATCH") != nullptr;
    if (batch_disabled || gdn_replays_broken) return nullptr;
    if (gdn_replays.capacity() == 0) {
        DeviceScope device(model.device);
        if (!gdn_replays.reserve(decoder::gdn_layers * decoder::max_wide_mtp_lanes)) {
            gdn_replays_broken = true;
            return nullptr;
        }
    }
    gdn_replays.reset();
    return &gdn_replays;
}

const MtpRound::Cancel& DecoderBatch::State::cancel_probe() const {
    return output ? output->cancel_probe() : no_cancel;
}

bool DecoderBatch::State::cancelled() const { return output && output->cancelled(); }

}
