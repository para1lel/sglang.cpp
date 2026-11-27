#include "sglang/runtime/execution/decoder_batch_state.h"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_prefill.h"
#include "sglang/decoder_sampling.h"
#include "sglang/prefill_fp8.h"
#include "sglang/runtime/backend/tensor_parallel.h"
#include "sglang/text/decoder_logprobs.h"
#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>
namespace sglang {
namespace {

using namespace decoder;
using namespace decoder::batch_detail;

static_assert(decoder_batch_max_mtp_steps + 1 == target_verify_max_rows);

}

DecoderBatch::DecoderBatch(const Decoder& decoder) : DecoderBatch(decoder, {}) {}

DecoderBatch::DecoderBatch(const Decoder& decoder, DecoderBatchOptions options)
    : state_(std::make_unique<State>(*decoder.model_, options, decoder.tp_.get())) {}
void DecoderBatch::State::retire(std::optional<Lane>& lane, const TpDriver* tp_driver) {
    if (lane) lane->prefix.reset();
    if (lane) lane->peer_prefix.reset();
    if (lane && lane->request) lane->request->memory.recycle_after_completion();
    if (tp_driver) tp_driver->drain_peer(*lane);
    if (lane && lane->draft && lane->resident_bytes) lane->draft->recycle_after_completion();
    lane.reset();
}

DecoderBatch::~DecoderBatch() {
    auto* tp = state_->tp;
    const auto stream = state_->execution.stream();
    if (!tp || !stream || tp->verification_capture_stream != stream) return;
    if (tp->verification_capture) {
        try {
            DeviceScope rank0(tp->device);
            synchronize_cuda(stream, "Fence rank-0 TP verifier graph");
            DeviceScope rank1(tp->peer_device);
            synchronize_cuda(tp->peer_stream, "Fence rank-1 TP verifier graph");
        } catch (const CudaError&) {
            tp->group->fail_fused_graph_replay();
            return;
        }
        tp->verification_capture.reset();
    }
    tp->verification_capture_stream = nullptr;
    tp->verification_capture_disabled = false;
}
int DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    return admit_impl(slot, std::move(prompt), options, nullptr);
}

int DecoderBatch::admit_multimodal(int slot, std::vector<int> prompt, const DecoderOptions& options,
                                  const std::vector<DecoderMedia>& media) {
    return admit_impl(slot, std::move(prompt), options, &media);
}

std::size_t DecoderBatch::State::reserve(DecoderOptions& request, int prompt,
                                         std::size_t extra_bytes) const {
    if (!residency.resident()) return 0;
    std::size_t charged = 0;
    for (const auto& lane : storage->slots) {
        charged += lane ? lane->resident_bytes : 0;
    }
    return reserve_request_memory(storage->retired, request, prompt, residency.capacity(),
                                  charged, extra_bytes);
}

void DecoderBatch::State::initialize(Lane& lane, int slot) {
    const auto& opts = lane.options;
    const auto stream = execution.stream();
    const int prompt = static_cast<int>(lane.prompt.size());
    if (opts.mtp_steps != 0) lane.draft = std::make_unique<MtpDraftRequest>(
        model.mtp_weights, lane.capacity(), lane.resident_bytes ? &storage->retired : nullptr);
    const auto [request_capacity, resident_limit] =
        residency.request_storage(prompt, lane.capacity(), opts.incremental_kv);
    lane.request->initialize(request_capacity, stream, resident_limit,
                             tp ? gqa_rank_geometry.kv_heads : mxfp8_kv_heads);
    if (opts.requires_target_only()) {
        adjusted_workspace.prepare(execution, *lane.request, opts, stream);
    }
    if (lane.media) {
        lane.media_prefill = upload_media_positions(*lane.request, *lane.media, stream);
    }
    lane.request->prefix_checkpoint_outputs = lane.checkpoints();
    lane.request->reset(stream);
    if (tp_driver) {
        tp_driver->initialize_peer(lane, request_capacity, resident_limit);
    }
    auto read = std::exchange(lane.options.prefix_cache_read, {});
    if (lane.resident_bytes) {
        if (const int rows = reserved_verification_rows(opts.max_new_tokens, opts.mtp_steps)) {
            const auto& winners = storage->verification_winners;
            const auto readback = winners ? winners.get() + slot * target_verify_max_rows : nullptr;
            lane.verification = std::make_unique<TargetVerification>(
                *lane.request, lane.capacity(), prompt, rows, false, stream,
                std::shared_ptr<TextLogitCandidate>(winners, readback),
                opts.generated_top_logprobs, tp != nullptr);
        }
        if (lane.sampling && !opts.requires_target_only() && !sampling_workspace)
            sampling_workspace = execution.allocate<std::byte>(sampling_workspace_bytes());
        if (opts.constraint && !opts.requires_target_only())
            constraint_workspace.allocate(execution, opts.constraint->uses_wide_mask());
    }
    check(cudaStreamSynchronize(stream), "Initialize decoder batch slot");
    if (read && read->usable_gqa && opts.restore_prefix) {
        if (tp_driver) {
            tp_driver->register_prefix_restore(lane, read, stream);
        } else {
            lane.prefix = std::make_unique<PendingBatchPrefix>(
                restore_prefix_view(*lane.request, *read, stream, lane.draft.get(), nullptr));
            lane.prefix->completion = opts.restore_prefix(lane.prefix->command);
        }
    }
}

void DecoderBatch::State::finish_restore(Lane& lane, int restored) {
    lane.progress.advance(restored);
    lane.checkpoint = restored;
    lane.restored_prompt = lane.remaining_prompt() == 0;
    const int gqa = lane.request->cached_gqa_tokens;
    const int draft = lane.draft ? lane.draft->valid_prefix() : 0;
    prefix_stats.hits += restored != 0;
    prefix_stats.cached_tokens += restored;
    prefix_stats.gqa_restored_tokens += gqa;
    prefix_stats.gqa_read_only_tokens += gqa - restored;
    prefix_stats.draft_cached_rows += draft;
    prefix_stats.draft_misses += lane.draft && restored == 0;
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
    validate_request(prompt, options, true);
    if (options.requires_target_only() || options.label_token_ids) options.mtp_steps = 0;
    const int top = options.generated_top_logprobs;
    if (const auto& labels = options.label_token_ids;
        labels && (options.max_new_tokens != 0 || options.score_prompt || top != -1 ||
                   options.top_logits || labels->empty() ||
                   std::any_of(labels->begin(), labels->end(), [](int label) {
                       return label < 0 || label >= vocab_size;
                   }))) {
        throw std::invalid_argument("Label scoring requires prefill and valid label IDs");
    }
    if ((options.score_prompt && top < 0) ||
        top < -1 || top > decoder_max_generated_top_logprobs ||
        (top >= 0 && (options.samples() || options.constraint || options.top_logits ||
                      state_->output))) {
        throw std::invalid_argument(
            "Generated logprobs require unconstrained greedy batch decoding");
    }
    if (options.prefix_cache_salt.size() > decoder_prefix_max_salt_bytes) {
        throw std::invalid_argument("Prefix cache salt exceeds 256 bytes");
    }
    if (options.mtp_steps != 0 && !state_->model.mtp_weights) {
        throw std::invalid_argument("Decoder batch requires draft weights for MTP");
    }
    if (media && state_->tp) {
        throw std::invalid_argument(
            "Multimodal requests are not supported under tensor parallelism");
    }
    if (options.max_new_tokens == 0 && !options.score_prompt && !options.label_token_ids) {
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
        if (input) validate_feature_device(*input, state_->model.device);
        reclaim_prefill_workspaces(state_->model.device);
        const auto resident = state_->reserve(options, static_cast<int>(prompt.size()),
            logit_adjustment_bytes(options) +
            (input ? input->prompt.positions.size() * sizeof(MropePosition) : 0));
        if (options.max_new_tokens != requested.max_new_tokens) {
            validate_request(prompt, options, true);
        }
        auto request = std::make_unique<Request>(
            state_->model.device, false, resident ? &state_->storage->retired : nullptr);
        lane.emplace(Lane{std::move(request), std::move(prompt), options});
        if (state_->tp_driver) state_->tp_driver->create_peer_request(*lane);
        if (options.top_logits > 0 || options.label_token_ids) {
            lane->request->memory.transfers().logits.resize(vocab_size);
        }
        if (options.samples()) lane->sampling = request_sampling(options);
        lane->media = std::move(input);
        lane->resident_bytes = resident;
        state_->initialize(*lane, slot);
        device.restore();
        state_->storage->admission.admitted(slot);
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
        State::retire(lane, state_->tp_driver.get());
    } catch (...) {
        state_->disable();
        throw;
    }
}

std::size_t DecoderBatch::request_device_allocation_bytes() const {
    return request_bytes(*state_->storage);
}

bool DecoderBatch::available() const { return !state_->failed; }

void DecoderBatch::State::score(int slot, Lane& lane, Workspace& work, bool logits_ready,
                                bool decode) {
    const auto stream = execution.stream();
    if (!logits_ready) model.compute_logits(work, stream, decode);
    auto* adjusted = lane.options.adjusts_logits(lane.progress.generated)
        ? &adjusted_workspace : nullptr;
    if (lane.options.generated_top_logprobs >= 0) {
        const bool repetition = lane.options.repetition_penalty != 1;
        auto* readback = logprob_readback + slot;
        if (adjusted) {
            adjusted->select(*lane.request, work, lane.options, nullptr, nullptr,
                             lane.progress.generated, lane.progress.last_token, stream,
                             repetition ? readback : nullptr);
        }
        if (repetition) return;
        check(enqueue_generated_logprobs(work.logits, vocab_size, work.argmax,
                                        lane.options.generated_top_logprobs, logprob_workspace,
                                        readback, stream,
                                        adjusted ? adjusted->logits : nullptr), "Score token");
        return;
    }
    if (!adjusted && lane.sampling && !sampling_workspace) {
        sampling_workspace = execution.allocate<std::byte>(sampling_workspace_bytes());
    }
    if (!adjusted && lane.options.constraint) {
        constraint_workspace.allocate(execution, lane.options.constraint->uses_wide_mask());
    }
    lane.progress.select(*lane.request, work, lane.options, constraint_workspace,
                         lane.sampling ? &*lane.sampling : nullptr, sampling_workspace, stream,
                         adjusted);
}

void DecoderBatch::State::submit(const Selection& selected, const PrefillRows& decode,
                                 Steps& results, DecoderBatchStepStats& stats, bool& contains_mtp,
                                 CheckpointRows* checkpoint_rows,
                                 const WideMtpGroups& wide_groups) {
    StepLanes lanes;
    const int max_rows =
        *std::max_element(stats.prefill_rows.begin(), stats.prefill_rows.end());
    collect_step_lanes(selected, stats.prefill_rows, decode, results, checkpoint_rows, lanes);
    dispatch_step_lanes(lanes, max_rows, results, stats, checkpoint_rows, wide_groups);
    contains_mtp = lanes.contains_mtp;
}

void DecoderBatch::State::collect_step_lanes(const Selection& selected, const PrefillRows& rows,
                                             const PrefillRows& decode, Steps& results,
                                             CheckpointRows* checkpoint_rows, StepLanes& lanes) {
    const auto stream = execution.stream();
    auto& slots = storage->slots;
    lanes.packed_slots.fill(-1);
    for (int slot = 0; slot < options.max_lanes; ++slot) {
        if (!selected[slot] || !slots[slot] || !slots[slot]->request) {
            continue;
        }
        auto& lane = *slots[slot];
        if (lane.restored_prompt) {
            lanes.scoring[slot] = lane.request.get();
            continue;
        }
        const int remaining = lane.remaining_prompt();
        const int decode_rows = options.max_num_batched_tokens || remaining
            ? decode[slot] : lane.proposals(lane.checkpoints()) + 1;
        if (rows[slot] == 0 && decode_rows == 0) {
            continue;
        }
        if (lane.options.incremental_kv) {
            lane.request->grow_kv(lane.progress.position + std::max(rows[slot], decode_rows),
                                  lane.progress.position, stream);
            if (tp_driver && lane.peer_request) {
                tp_driver->grow_peer_kv(
                    lane, lane.progress.position + std::max(rows[slot], decode_rows),
                    lane.progress.position);
            }
        }
        results[slot].decode = remaining == 0;
        lanes.contains_mtp = lanes.contains_mtp || lane.draft != nullptr;
        const int position = lane.progress.position;
        if (rows[slot] && lane.draft && position > lane.draft->valid_prefix()) {
            model.seed_draft(lane.request->work, *lane.draft, position - 1, lane.prompt[position],
                             stream, lane.media ? &lane.media_prefill : nullptr);
        }
        if (results[slot].decode && lane.draft && lane.remaining_output() > 1) {
            lanes.speculative[slot] = decode_rows;
            continue;
        }
        results[slot].mtp_boundary = results[slot].decode && lane.draft != nullptr;
        if (results[slot].decode) {
            if (lane.checkpoints()) lane.record_decode_input(stream);
            lanes.packed_slots[slot] = lanes.count;
            if (lane.checkpoints()) {
                (*checkpoint_rows)[slot] = &lane.request->work;
                if (tp_driver) lane.peer_checkpoint_row = &lane.peer_request->work;
            }
            lanes.work[lanes.count++] = lane.input();
            if (tp_driver)
                lanes.peer_work[lanes.count - 1] =
                    TpDriver::mirror_lane(lanes.work[lanes.count - 1],
                                          lane.peer_request.get());
        } else if (rows[slot] == 1) {
            lanes.single_prefill[lanes.single_count++] = lane.input();
            if (tp_driver)
                lanes.peer_single[lanes.single_count - 1] =
                    TpDriver::mirror_lane(lanes.single_prefill[lanes.single_count - 1],
                                          lane.peer_request.get());
        }
        if (rows[slot] == remaining &&
            (lane.options.max_new_tokens > 0 || lane.options.label_token_ids)) {
            lanes.scoring[slot] = lane.request.get();
        }
    }
}

void DecoderBatch::State::dispatch_step_lanes(StepLanes& lanes, int max_rows, Steps& results,
                                              DecoderBatchStepStats& stats,
                                              CheckpointRows* checkpoint_rows,
                                              const WideMtpGroups& wide_groups) {
    const auto stream = execution.stream();
    auto& slots = storage->slots;
    const auto& policy = model.execution;
    const bool regular_graph = policy.regular_decode_graph(lanes.count);
    DecodeRun run{};
    if (tp_driver) {
        tp_driver->forward_decode(lanes.work.data(), lanes.peer_work.data(), lanes.count, stream,
                                  regular_graph, run);
        tp_driver->forward_single(lanes.single_prefill.data(), lanes.peer_single.data(),
                                  lanes.single_count, stream);
    } else {
        run = run_decode_lanes(model, execution, storage->decode_graphs, lanes.work.data(),
                               lanes.count,
                               {options.cuda_graphs, regular_graph,
                                policy.narrow_eager_decode});
        if (lanes.single_count)
            model.forward(lanes.single_prefill.data(), lanes.single_count, stream);
    }
    if (max_rows) prefill(stats.prefill_rows, stats);
    bool submitted = lanes.count != 0 || max_rows != 0;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (auto* request = lanes.scoring[slot]) {
            auto& lane = *slots[slot];
            const int index = lane.restored_prompt ? -1 : lanes.packed_slots[slot];
            auto* ready = index < 0 ? nullptr : run.ready[index];
            if (lane.options.label_token_ids) {
                model.enqueue_label_logits(*request, stream);
            } else {
                score(slot, lane, ready ? *ready : request->work, ready != nullptr,
                      results[slot].decode);
            }
            submitted = true;
            if (index >= 0 && (run.graphed >> index & 1) != 0) {
                stats.graph_slots |= std::uint32_t{1} << slot;
            }
        }
    }
    stats.constraint_workspace_bytes = constraint_workspace.bytes();
    for (int next_slot = 0; lanes.contains_mtp && next_slot < options.max_lanes;) {
        speculate_group(lanes.speculative, next_slot, results, stats, wide_groups,
                        checkpoint_rows);
    }
    stats.sampling_workspace_bytes = sampling_workspace ? sampling_workspace_bytes() : 0;
    if (stats.verification_lanes == 0 && submitted) {
        check(cudaStreamSynchronize(stream), "Complete decoder batch step");
    }
    collect_scores(lanes.scoring, results);
}

void DecoderBatch::State::prefill(const PrefillRows& rows, DecoderBatchStepStats& stats) {
    const auto stream = execution.stream();
    const int limit = model.execution.prefill_limit();
    std::array<PrefillGrant, decoder_batch_max_lanes> grants;
    std::array<PrefillGrant, decoder_batch_max_lanes> peer_grants;
    std::array<PromptLogprobCapture, decoder_batch_max_lanes> captures;
    int count = 0;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (rows[slot] == 0) continue;
        auto& lane = *storage->slots[slot];
        const int position = lane.progress.position;
        if (lane.options.score_prompt) {
            captures[count] = {&lane.prompt, &lane.prompt_logprobs, logprob_readback,
                               logprob_workspace, lane.options.generated_top_logprobs,
                               lane.options.prompt_logprob_start};
        }
        const int tile_rows = std::min(rows[slot], limit);
        grants[count] = {lane.request.get(), lane.prompt.data() + position, position, rows[slot],
                         tile_rows, lane.remaining_prompt(), lane.draft.get(),
                         lane.media ? &lane.media_prefill : nullptr,
                         lane.options.score_prompt ? &captures[count] : nullptr};
        if (tp_driver) {
            peer_grants[count] = {lane.peer_request.get(), lane.prompt.data() + position,
                                  position, rows[slot], tile_rows, lane.remaining_prompt(),
                                  nullptr, nullptr, nullptr};
        }
        ++count;
    }
    if (tp_driver) {
        tp_driver->prefill(grants.data(), peer_grants.data(), count, stream, stats);
    } else {
        model.prefill_grants(grants.data(), count, stream, stats);
    }
}

void DecoderBatch::State::collect_scores(
    const std::array<Request*, decoder_batch_max_lanes>& scoring, Steps& results) {
    const auto& slots = storage->slots;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (scoring[slot]) {
            const auto& lane = *slots[slot];
            if (lane.options.label_token_ids) {
                collect_label_score(lane.request->memory.transfers().logits,
                                    *lane.options.label_token_ids, results[slot]);
                continue;
            }
            try {
                const int top = lane.options.generated_top_logprobs;
                if (top >= 0) {
                    const auto& readback = logprob_readback[slot];
                    results[slot].generated_logprobs = collect_generated_logprobs(readback, top);
                    results[slot].token = readback.selected.token_id;
                } else {
                    results[slot].token =
                        read_token(*scoring[slot], lane.sampling || lane.progress.generated <
                                   lane.options.min_new_tokens,
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

bool DecoderBatch::State::prepare_wide_mtp(int lanes) {
    if (wide_mtp_capacity >= lanes) return true;
    constexpr std::size_t bytes_per_lane = target_verify_max_rows * intermediate_size;
    const auto bytes = lanes * bytes_per_lane;
    const auto current_bytes = wide_mtp_capacity * bytes_per_lane;
    if (residency.resident() &&
        available_device_bytes() < decoder_shared_reserve + bytes - current_bytes) {
        return false;
    }
    check(cudaPeekAtLastError(), "Check CUDA status before allocating verification scratch");
    if (wide_mtp_activation) {
        execution.free_block(wide_mtp_activation, current_bytes);
        draft_depth.activation = nullptr;
        wide_mtp_activation = nullptr;
        wide_mtp_capacity = 0;
    }
    try {
        wide_mtp_activation = execution.allocate<std::uint8_t>(bytes);
        wide_mtp_capacity = lanes;
        return true;
    } catch (const CudaError& error) {
        if (error.status != cudaErrorMemoryAllocation) throw;
        const auto pending = cudaGetLastError();
        if (pending != cudaErrorMemoryAllocation) {
            check(pending, "Check CUDA status before smaller verification fallback");
        }
        return false;
    }
}

WideMtpGroups DecoderBatch::State::prepare_wide_groups(const PrefillRows& decode,
        const WideMtpCandidates& candidates, bool& contains_mtp) {
    auto groups = wide_mtp_groups(decode, candidates, wide_mtp_limit);
    
    
    
    
    
    
    if (tp) {
        contains_mtp = groups.largest > 0;
        if (groups.largest) prepare_wide_mtp(groups.largest);
        return groups;
    }
    while (groups.largest) {
        contains_mtp = true;
        if (prepare_wide_mtp(groups.largest)) break;
        wide_mtp_limit = groups.largest / 2;
        groups = wide_mtp_groups(decode, candidates, wide_mtp_limit);
    }
    return groups;
}

static bool full_verification(const MtpRound* rounds, int count) {
    return std::all_of(rounds, rounds + count,
                       [](const auto& round) { return round.rows() == target_verify_max_rows; });
}

static int verification_width(const MtpRound* rounds, int count, int first, bool full) {
    if (full) return count;
    if (first % wide_mtp_lanes == 0 && count - first >= wide_mtp_lanes &&
        full_verification(rounds + first, wide_mtp_lanes)) {
        return wide_mtp_lanes;
    }
    return std::min(count - first, decoder_batch_max_mtp_lanes);
}

int DecoderBatch::State::collect_rounds(const PrefillRows& verification_rows, int& next_slot,
                                        const WideMtpGroups& wide_groups,
                                        std::array<MtpRound, max_wide_mtp_lanes>& rounds,
                                        std::array<MtpRound, max_wide_mtp_lanes>& peer_rounds,
                                        std::array<int, max_wide_mtp_lanes>& slots) {
    const auto stream = execution.stream();
    constexpr int packet = decoder_batch_max_mtp_lanes;
    int limit = packet;
    int count = 0;
    while (next_slot < options.max_lanes && count < limit) {
        const int slot = next_slot++;
        const int rows = verification_rows[slot];
        if (rows == 0) continue;
        if (count == 0 && wide_groups.width[slot]) limit = wide_groups.width[slot];
        auto& lane = *storage->slots[slot];
        const int capacity = lane.proposals() + 1;
        if (!lane.verification) {
            lane.verification = std::make_unique<TargetVerification>(
                *lane.request, lane.capacity(), lane.progress.position, capacity,
                lane.options.top_logits > 0, stream, nullptr,
                lane.options.generated_top_logprobs, tp != nullptr);
        }
        if (tp_driver) tp_driver->prepare_peer_verification(lane, capacity);
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
        rounds[count].media = lane.media ? &lane.media_prefill : nullptr;
        if (tp_driver) peer_rounds[count] = TpDriver::peer_round(lane, rows - 1);
        slots[count++] = slot;
    }
    return count;
}

static void repair_rounds(const DraftRepairs* repairs, int size, std::uint8_t* packed, bool exact,
                          cudaStream_t stream) {
    if (packed) {
        MtpDraftRequest::repair_wave(repairs, size, reinterpret_cast<Bf16*>(packed), stream,
                                     exact);
        return;
    }
    constexpr int packet = decoder_batch_max_mtp_lanes;
    for (int lane = 0; lane < size; lane += packet) {
        MtpDraftRequest::repair_wave(repairs + lane, std::min(size - lane, packet), nullptr,
                                     stream);
    }
}

bool DecoderBatch::State::prepare_peer_repair(const MtpRound* rounds, int count) {
    return tp_driver && output == nullptr && residency.resident() &&
           wide_mtp_activation != nullptr && wide_mtp_capacity >= count &&
           tp_driver->prepare_peer_repair(rounds, count);
}

void DecoderBatch::State::repair_verification_group(const DraftRepairs* repairs, int count,
                                                    cudaStream_t stream) {
    const bool exact = tp && count == draft_projection_rows && wide_mtp_capacity >= count;
    if (exact && tp_driver->peer_repair && tp_driver->peer_repair->apply(repairs, count, stream)) {
        return;
    }
    auto* packed = count > decoder_batch_max_mtp_lanes && (!tp || exact)
        ? wide_mtp_activation : nullptr;
    repair_rounds(repairs, count, packed, exact, stream);
}

void DecoderBatch::State::speculate_group(const PrefillRows& verification_rows, int& next_slot,
                                          Steps& results, DecoderBatchStepStats& stats,
                                          const WideMtpGroups& wide_groups,
                                          CheckpointRows* checkpoint_rows) {
    const auto stream = execution.stream();
    constexpr int packet = decoder_batch_max_mtp_lanes;
    std::array<MtpRound, max_wide_mtp_lanes> rounds;
    std::array<MtpRound, max_wide_mtp_lanes> peer_rounds;
    std::array<int, max_wide_mtp_lanes> slots;
    const int count =
        collect_rounds(verification_rows, next_slot, wide_groups, rounds, peer_rounds, slots);
    if (count == 0) return;
    const bool allow_peer_repair = prepare_peer_repair(rounds.data(), count);
    const auto& cancel = cancel_probe();
    const bool score_together = count >= draft_projection_rows ||
        (count == wide_mtp_lanes && wide_mtp_activation != nullptr);
    const bool depth_scoring = score_together && !output && MtpDraftRequest::prepare_depth(
        *rounds.front().draft, draft_depth, wide_mtp_activation, tp, stream);
    if (depth_scoring) {
        MtpRound::propose_depth(rounds.data(), count, draft_depth, stream);
    } else if (!MtpRound::propose_packets(rounds.data(), count, constraint_workspace, stream,
                                          cancel)) {
        check(cudaStreamSynchronize(stream), "Complete cancelled MTP draft wave");
        for (int index = 0; index < count; ++index) {
            auto& result = results[slots[index]];
            result.mtp_drafted_tokens = rounds[index].drafted;
            result.finish_reason = DecoderFinishReason::cancelled;
        }
        return;
    }
    auto* graph = storage->verification_graph ? &*storage->verification_graph : nullptr;
    const auto verify = [&](int first, int size, std::uint8_t* activation = nullptr) {
        if (tp_driver) {
            const bool capture = options.tp_verification_graphs && size == 2 &&
                !storage->slots[slots[first]]->options.eager &&
                !storage->slots[slots[first + 1]]->options.eager;
            tp_driver->verify(rounds.data() + first, peer_rounds.data() + first, size, stream,
                              constraint_workspace, sampling_workspace, logprob_workspace,
                              storage->tc_vocab.get(), capture, allow_peer_repair);
        } else if (model.verify_mtp(rounds.data() + first, size, constraint_workspace,
                                    sampling_workspace, stream, graph,
                                    storage->tc_vocab.get(), activation,
                             logprob_workspace)) {
            stats.verification_graph_slots |= std::uint32_t{1} << slots[first];
        }
    };
    const auto commit = [&](int first, int size, DraftRepairs* deferred) {
        std::array<DraftRepairs, draft_projection_rows> packet_repairs;
        auto* repairs = deferred ? deferred : packet_repairs.data();
        if (!deferred) std::fill_n(repairs, size, DraftRepairs{});
        TextGdnReplayBatch* gdn_batch = prepare_gdn_replays();
        TextGdnReplayBatch* peer_gdn_batch =
            tp_driver ? tp_driver->prepare_peer_gdn_replays() : nullptr;
        for (int index = 0; index < size; ++index) {
            auto& round = rounds[first + index];
            const int slot = slots[first + index];
            stats.verification_rows += round.rows();
            auto* repair = size > 1 && !round.constraint ? repairs[index].data() : nullptr;
            auto& lane = *storage->slots[slot];
            const auto saved = checkpoint_rows && lane.checkpoints() ? &(*checkpoint_rows)[slot]
                                                                     : nullptr;
            lane.finish_round(round, results[slot], stream, output, cancel, repair, saved,
                              gdn_batch);
            if (tp_driver) tp_driver->commit_peer_round(lane, round, saved, peer_gdn_batch);
        }
        if (!deferred) repair_verification_group(repairs, size, stream);
        if (gdn_batch) {
            check(gdn_batch->launch(tp != nullptr, stream), "Batched GDN prefix replay");
        }
        if (tp_driver) tp_driver->finish_peer_gdn_replays();
        stats.verification_lanes += size;
    };
    const bool full = count > packet &&
        count <= (tp ? wide_mtp_lanes * 2 : max_wide_mtp_lanes) &&
        (!tp || count == wide_mtp_lanes || count == wide_mtp_lanes * 2) &&
        full_verification(rounds.data(), count);
    for (int first = 0; first < count;) {
        const int width = verification_width(rounds.data(), count, first, full);
        verify(first, width, width > packet ? wide_mtp_activation : nullptr);
        const bool group_repair = tp && !output && width == wide_mtp_lanes &&
            wide_mtp_capacity >= width && std::none_of(rounds.data() + first,
                rounds.data() + first + width, [](const auto& round) {
                    return round.media || round.sampling || round.constraint;
                });
        std::array<DraftRepairs, wide_mtp_lanes> repairs;
        if (group_repair) repairs = {};
        const int commit_width = !output && width >= draft_projection_rows
            ? draft_projection_rows
            : (depth_scoring && !tp && width == wide_mtp_lanes ? width : packet);
        for (int next = 0; next < width; next += commit_width) {
            commit(first + next, std::min(width - next, commit_width),
                   group_repair ? repairs.data() + next : nullptr);
        }
        if (group_repair && (!tp_driver->peer_repair ||
                            !tp_driver->peer_repair->apply(repairs.data(), width, stream))) {
            repair_rounds(repairs.data(), width, wide_mtp_activation, true, stream);
        }
        first += width;
    }
}
Steps DecoderBatch::State::advance(const Selection& requested, bool chunked,
                                   DecoderBatchStepStats* stats) {
    require_ready();
    bool contains_mtp = false, publishing_prefix = false;
    try {
        DeviceScope device(model.device);
        auto& slots = storage->slots;
        auto selected = requested;
        for (int slot = 0; slot < options.max_lanes; ++slot) {
            const auto& lane = slots[slot];
            if (lane && (lane->prefix || lane->prefix_store)) selected[slot] = false;
        }
        CheckpointRows checkpoint_rows{};
        Steps results{};
        DecoderBatchStepStats current;
        WideMtpCandidates wide;
        const auto plan = plan_step(*storage, selected, options.max_lanes,
                                    options.max_num_batched_tokens, chunked,
                                    wide_mtp_limit >= wide_mtp_lanes && !output ? &wide : nullptr,
                                    tp != nullptr);
        current.prefill_rows = plan.prefill;
        const auto live_before = stats ? request_bytes(*storage) : 0;
        const auto wide_groups = prepare_wide_groups(plan.decode, wide, contains_mtp);
        submit(selected, plan.decode, results, current, contains_mtp,
               &checkpoint_rows, wide_groups);
        if (stats) current.peak_request_bytes = std::max(
            live_before + current.prefill_workspace_bytes, request_bytes(*storage));
        
        for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
            auto& lane = slots[slot];
            auto& result = results[slot];
            if (!selected[slot] || !lane) {
                continue;
            }
            if (result.sampling_failed || result.constraint_failed) {
                retire(lane, tp_driver.get());
                continue;
            }
            if (!lane->request) {
                result.finish_reason = DecoderFinishReason::length;
            } else if (result.mtp_verified_rows == 0 &&
                       result.finish_reason != DecoderFinishReason::cancelled &&
                       (lane->restored_prompt || result.decode || current.prefill_rows[slot])) {
                const int rows = current.prefill_rows[slot];
                prefix_stats.prefill_tokens += rows;
                prefix_stats.gqa_computed_tokens += std::clamp(
                    lane->progress.position + rows - lane->request->cached_gqa_tokens, 0, rows);
                if (result.token) {
                    if (cancelled()) {
                        result.finish_reason = DecoderFinishReason::cancelled;
                    } else {
                        const auto& logits = lane->request->memory.transfers().logits;
                        result.finish_reason = lane->progress.commit_single(
                            lane->options, *result.token, logits.data(), output);
                    }
                }
                if (!std::exchange(lane->restored_prompt, false)) {
                    lane->progress.advance(std::max(1, rows));
                }
                if (lane->options.score_prompt && !lane->remaining_prompt() && !result.decode) {
                    finish_prompt_logprobs(lane->options, lane->prompt_logprobs, result);
                }
            }
        }
        for (int slot = 0; slot < options.max_lanes; ++slot) {
            auto& lane = slots[slot];
            auto& result = results[slot];
            if (!selected[slot] || !lane) continue;
            if (result.sampling_failed || result.constraint_failed) {
                retire(lane, tp_driver.get());
                continue;
            }
            if (lane->checkpoints()) publish_checkpoint(*lane, result, checkpoint_rows[slot],
                                                         publishing_prefix);
            if (result.finish_reason && !result.prefix_pending) retire(lane, tp_driver.get());
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
        if (tp_driver && tp_driver->peer_repair) tp_driver->peer_repair->drain();
        fail_step(!CudaError::terminal(error.status) && (contains_mtp || publishing_prefix));
        throw;
    } catch (...) {
        if (tp_driver && tp_driver->peer_repair) tp_driver->peer_repair->drain();
        fail_step(contains_mtp && !publishing_prefix);
        throw;
    }
}

void DecoderBatch::generate_mtp(const std::vector<int>& prompt, const DecoderOptions& options,
                                SerialOutput& output, DecoderResult& result,
                                const std::vector<DecoderMedia>* media) {
    state_->output = &output;
    if (media) {
        admit_multimodal(0, prompt, options, *media);
    } else {
        admit(0, prompt, options);
    }
    result.output_token_ids.reserve(options.max_new_tokens);
    const auto model_bytes = result.peak_device_allocation_bytes;
    const auto& lane = state_->storage->slots[0];
    bool prefix_pending = bool(lane->prefix);
    bool completion_claimed = false;
    while (completion_claimed || !state_->cancelled()) {
        if (prefix_pending) {
            const auto update = poll_prefix(0);
            if (update.pending) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            if (update.restored) result.cached_prompt_tokens = *update.restored;
            if (!lane) return;  
        }
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
        prefix_pending = step.prefix_pending;
        if (step.finish_reason) {
            result.finish_reason = *step.finish_reason;
            if (!prefix_pending) return;
            completion_claimed = bool(options.claim_completion);
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
