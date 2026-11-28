#include "sglang/runtime/execution/decoder_batch_state.h"
#include "sglang/runtime/backend/device_arena.h"
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
        const bool prefill = lane->remaining_prompt() != 0;
        work[slot] = prefill ? BudgetLane{chunked ? lane->prefill_rows() : 1}
                             : BudgetLane{0, std::max(0, lane->proposals()), true};
        if (wide && work[slot].decode && lane->draft && lane->remaining_output() > 1) {
            const auto bit = std::uint32_t{1} << slot;
            wide->speculative |= bit;
            if (tensor_parallel && lane->sampling && !lane->options.constraint)
                wide->sampled |= bit;
            if (storage.verification_graph && !lane->options.eager)
                wide->replay |= bit;
            if (work[slot].proposals == decoder_batch_max_mtp_steps &&
                lane->remaining_output() >= target_verify_max_rows && !lane->options.constraint &&
                !lane->sampling && lane->options.top_logits == 0 &&
                lane->progress.generated >= lane->options.min_new_tokens &&
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
    if (residency.resident() || options.resident_prefix_cache || options.prefix_gqa_pages != 0 ||
        options.prefix_gdn_slots != 0) {
        std::vector<int> devices{model.device};
        if (tp != nullptr)
            devices.push_back(tp->peer_device);
        prefix_pool = std::make_unique<PrefixResidency>(
            devices, options.resident_prefix_cache, model.mtp_weights != nullptr, options.max_lanes,
            options.prefix_gqa_pages, options.prefix_gdn_slots);
    }

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
        if (tp != nullptr) {
            peer_residency.emplace(options);
            const auto peer_arena = DeviceArena::current(tp->peer_device);
            peer_residency->measure_available(peer_arena->available_bytes());
        }
    }
    device.restore();
}

std::array<std::size_t, 2> DecoderBatch::State::reserve(int prompt, DecoderOptions& request,
                                                        std::size_t extra_bytes) const {
    if (!residency.resident())
        return {};
    std::array<std::size_t, 2> charged{};
    for (const auto& lane : storage->slots) {
        if (lane) {
            charged[0] += lane->resident_bytes[0];
            charged[1] += lane->resident_bytes[1];
        }
    }
    const int heads = tp != nullptr ? gqa_rank_geometry.kv_heads : mxfp8_kv_heads;
    std::array<RequestMemoryBudget, 2> ranks{
        RequestMemoryBudget{
            &storage->retired,
            residency.capacity(),
            charged[0],
            extra_bytes,
            model.device,
            heads,
        },
        {},
    };
    if (peer_residency) {
        auto* retired = storage->peer_retired ? &*storage->peer_retired : nullptr;
        ranks[1] = {retired, peer_residency->capacity(), charged[1], 0, tp->peer_device, heads};
    }
    return reserve_request_memory(request, prompt, ranks, tp != nullptr ? 2 : 1,
                                  prefix_pool != nullptr);
}

DecoderBatch::State::~State() {
    if (prefix_pool)
        disable();
}

RequestBlockCache* DecoderBatch::State::request_cache(const DecoderOptions& request) {
    if (prefix_pool)
        return residency.resident() ? &storage->retired : nullptr;
    if (request.publish_prefix_checkpoints && request.store_prefix) {
        storage->retired.enable_detached_reclamation();
        return &storage->retired;
    }
    return residency.resident() ? &storage->retired : nullptr;
}

RequestBlockCache* DecoderBatch::State::peer_request_cache(const Lane& lane) {
    if (!lane.checkpoints())
        return nullptr;
    auto& retired = storage->peer_retired;
    if (!retired)
        retired.emplace(tp->peer_device, false, true);
    return &*retired;
}

void DecoderBatch::State::drain(const char* operation, std::optional<Lane>* lane) {
    try {
        DeviceScope device(model.device);
        check(cudaStreamSynchronize(execution.stream()), operation);
        if (lane != nullptr)
            retire(*lane);
    } catch (...) {
        disable();
        throw;
    }
}

void DecoderBatch::State::fail_step(bool recoverable) {
    if (failed)
        return;
    if (!recoverable || execution.verification_graph_failed()) {
        disable();
        return;
    }
    drain("Drain failed decoder batch step");
    for (auto& lane : storage->slots) {
        if (lane && lane->prefix && lane->prefix->completion.valid()) {
            lane->prefix->discard = true;
        } else {
            retire(lane);
        }
    }
    if (prefix_pool)
        prefix_pool->notify_released();
}

std::optional<Lane>& DecoderBatch::State::slot_at(int slot) {
    if (slot < 0 || slot >= options.max_lanes) {
        throw std::out_of_range("Invalid decoder batch slot");
    }
    return storage->slots[slot];
}

void DecoderBatch::State::require_ready() {
    if (failed)
        throw std::runtime_error("Decoder batch is unavailable");
    try {
        storage->retired.check_releases();
        if (storage->peer_retired)
            storage->peer_retired->check_releases();
    } catch (const CudaError&) {
        disable();
        throw;
    }
}

void DecoderBatch::State::disable() noexcept {
    if (failed)
        return;
    failed = true;
    const bool peer_fenced = !tp_driver || tp_driver->sync_peer_stream();
    if (!peer_fenced && prefix_pool)
        prefix_pool->quarantine();
    try {
        DeviceScope device(model.device);
        execution.close();
    } catch (...) {
        if (prefix_pool)
            prefix_pool->quarantine();
        return;
    }
    storage->slots = {};
    storage->decode_graphs = {};
    storage->verification_graph.reset();
}

namespace {

Lane& require_lane(std::optional<Lane>& entry) {
    if (!entry.has_value())
        throw std::bad_optional_access{};
    return *entry;
}

}

int DecoderBatch::State::collect_rounds(const PrefillRows& verification_rows, int& next_slot,
                                        std::array<MtpRound, max_wide_mtp_lanes>& rounds,
                                        std::array<int, max_wide_mtp_lanes>& slots,
                                        std::array<MtpRound, max_wide_mtp_lanes>& peer_rounds,
                                        const WideMtpGroups& wide_groups) {
    auto* const stream = execution.stream();
    constexpr int packet = decoder_batch_max_mtp_lanes;
    int limit = packet;
    int count = 0;
    while (next_slot < options.max_lanes && count < limit) {
        const int slot = next_slot++;
        const int rows = verification_rows[slot];
        if (rows == 0)
            continue;
        if (count == 0)
            limit = std::max(packet, static_cast<int>(wide_groups.width[slot]));
        auto& lane = require_lane(storage->slots[slot]);
        const int capacity = lane.proposals() + 1;
        if (!lane.verification) {
            lane.verification = std::make_unique<TargetVerification>(
                *lane.request, lane.capacity(), lane.progress.position, capacity,
                lane.options.top_logits > 0, stream, nullptr, lane.options.generated_top_logprobs,
                tp != nullptr);
        }
        if (tp_driver)
            tp_driver->prepare_peer_verification(lane, lane.progress.position, capacity);
        if (lane.options.constraint) {
            if (!lane.constraint_round) {
                lane.constraint_round.emplace(*lane.request, *lane.options.constraint, capacity);
            }
            lane.constraint_round->begin(lane.progress.constraint, lane.remaining_output());
        }
        rounds[count] = {
            lane.draft.get(),
            lane.verification.get(),
            lane.request.get(),
            lane.constraint_round ? &*lane.constraint_round : nullptr,
            lane.sampling ? &*lane.sampling : nullptr,
            {lane.progress.last_token},
            lane.progress.position,
            lane.progress.generated,
            lane.remaining_output(),
            rows - 1,
        };
        rounds[count].media = lane.media ? &lane.media_prefill : nullptr;
        if (lane.progress.generated < lane.options.min_new_tokens) {
            rounds[count].minimum_options = &lane.options;
            rounds[count].adjustment = &adjusted_workspace;
        }
        if (tp_driver)
            peer_rounds[count] = TpDriver::peer_round(lane, rows - 1);
        slots[count++] = slot;
    }
    return count;
}

void DecoderBatch::State::commit_verification(MtpRound* rounds, const int* slots,
                                              MtpRound* peer_rounds, int size, Steps& results,
                                              DecoderBatchStepStats& stats,
                                              CheckpointRows* checkpoint_rows, cudaStream_t stream,
                                              const MtpRound::Cancel& cancel) {
    std::array<DraftRepairs, draft_projection_rows> repairs{};
    for (int index = 0; index < size; ++index) {
        auto& round = rounds[index];
        const int slot = slots[index];
        stats.verification_rows += round.rows();
        auto* repair = tp_driver || (size > 1 && round.constraint == nullptr)
                           ? repairs[index].data()
                           : nullptr;
        auto& lane = require_lane(storage->slots[slot]);
        auto* saved =
            checkpoint_rows != nullptr && lane.checkpoints() ? &(*checkpoint_rows)[slot] : nullptr;
        lane.finish_round(round, results[slot], stream, output, cancel, repair, saved);
        if (tp_driver) {
            tp_driver->commit_peer_round(lane, round, saved);
        }
    }
    if (tp_driver) {
        tp_driver->repair(rounds, repairs.data(), peer_rounds, size, stream);
    } else {
        repair_verification_group(repairs.data(), size, stream);
    }
    stats.verification_lanes += size;
}

const MtpRound::Cancel& DecoderBatch::State::cancel_probe() const {
    return output ? output->cancel_probe() : no_cancel;
}

bool DecoderBatch::State::cancelled() const { return output && output->cancelled(); }

}
