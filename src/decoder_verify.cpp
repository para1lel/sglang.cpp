#include "sglang/decoder_verify.h"

#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_sampling.h"
#include "sglang/decoder_mtp.h"
#include "sglang/text/gdn_local.h"
#include "sglang/runtime/execution/verification_row_copy.h"
#include "sglang/text/gqa_verify_batch.h"
#include "sglang/text/decoder_logprobs.h"
#include "sglang/text/mtp/tc_vocab.h"
#include "sglang/text/bf16_head_projection.h"
#include "sglang/runtime/execution/tp_verify_graph.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <utility>
#include <new>
#include <tuple>

namespace sglang::decoder {
namespace {

void free_pinned(void* results) {
    const auto status = cudaFreeHost(results);
    if (status != cudaSuccess) {
        std::cerr << "sglang decoder: Free verifier pinned storage: " << cudaGetErrorString(status)
                  << '\n';
    }
}
}
bool TargetVerification::can_begin(int position, int count, cudaStream_t stream) const {
    return stream == stream_ && pending_rows_ == 0 && position == valid_prefix_ &&
           count >= 1 && count <= static_cast<int>(eager_workspace_->work.size()) &&
           count <= capacity_ - position &&
           count <= request_.pages.page_count * mxfp8_page_size - position;
}

template <>
VerificationGraphBinding* Allocations::allocate<VerificationGraphBinding>(std::size_t count) {
    return static_cast<VerificationGraphBinding*>(
        allocate_elements(count, sizeof(VerificationGraphBinding)));
}

VerificationGraphScratch::VerificationGraphScratch(int device)
    : memory(device), device_binding(memory.allocate<VerificationGraphBinding>(1)) {
    void* pinned = nullptr;
    check(cudaMallocHost(&pinned, sizeof(VerificationGraphBinding)),
          "Allocate verifier graph bindings");
    auto* binding = ::new (pinned) VerificationGraphBinding{};
    host_binding = std::shared_ptr<VerificationGraphBinding>(binding, free_pinned);
}

MtpConstraintRound::MtpConstraintRound(Request& request, const TokenConstraint& constraint,
                                       int rows)
    : constraint_(&constraint), stride_(constraint.uses_wide_mask()
          ? constraint_mask_words * sizeof(std::uint32_t)
          : constraint_max_candidates * sizeof(int)) {
    auto& owner = request.memory.transfers().mtp_constraint_inputs;
    if (owner || rows < 1 || rows > target_verify_max_rows) {
        throw std::invalid_argument("Invalid MTP constraint storage");
    }
    void* pinned = nullptr;
    check(cudaMallocHost(&pinned, stride_ * rows), "Allocate MTP constraint inputs");
    owner = std::shared_ptr<void>(pinned, free_pinned);
    staging_ = static_cast<std::byte*>(pinned);
}

void MtpConstraintRound::begin(const ConstraintCursor& committed, int remaining) {
    cursor_.reset_to(committed);
    remaining_ = remaining;
    prepare(0, remaining);
}

bool MtpConstraintRound::advance(int row, int token) {
    const int remaining = remaining_ - row - 1;
    if (cursor_.commit(*constraint_, token, remaining)) {
        return false;
    }
    prepare(row + 1, remaining);
    return true;
}

void MtpConstraintRound::prepare(int row, int remaining) {
    cursor_.candidates(*constraint_, remaining);
    prepared_rows_ = row + 1;
    auto* destination = staging_ + row * stride_;
    const auto& ids = cursor_.ids();
    const auto& mask = cursor_.mask();
    rows_[row] = {static_cast<int>(ids.size()), nullptr, nullptr};
    if (mask.empty()) {
        std::memcpy(destination, ids.data(), ids.size() * sizeof(int));
        rows_[row].ids = reinterpret_cast<const int*>(destination);
    } else {
        std::memcpy(destination, mask.data(), mask.size() * sizeof(std::uint32_t));
        rows_[row].mask = reinterpret_cast<const std::uint32_t*>(destination);
    }
}

int MtpRound::rows() const {
    return constraint ? constraint->rows() : std::min(drafted + 1, remaining);
}

int MtpRound::rotary_delta() const {
    if (!media) {
        return 0;
    }
    const auto& prompt = media->input->prompt;
    return prompt.next_position - static_cast<int>(prompt.positions.size());
}

MtpDraftInput MtpRound::draft_input(int row_position, int token, const Bf16* hidden) const {
    MtpDraftInput input{draft, row_position, token, hidden};
    if (media) {
        const int prompt_rows = static_cast<int>(media->input->prompt.positions.size());
        if (row_position < prompt_rows) {
            input.mrope = media->positions + row_position;
        } else {
            input.rotary_position = row_position + rotary_delta();
        }
    }
    return input;
}

namespace {

bool enqueue_proposal_wave(MtpRound* rounds, int count, int wave,
                           const ConstraintWorkspace& scratch, cudaStream_t stream) {
    std::array<MtpDraftInput, decoder_batch_max_mtp_lanes> rows;
    std::array<MtpRound*, decoder_batch_max_mtp_lanes> batched;
    int size = 0;
    bool submitted = false;
    for (int index = 0; index < count; ++index) {
        auto& round = rounds[index];
        if (wave >= round.proposal_limit) {
            continue;
        }
        const auto input = round.draft_input(round.position - 1 + wave, round.tokens[wave],
                                             round.draft_hidden);
        if (round.constraint) {
            round.draft_hidden = round.draft->enqueue(
                input.position, input.token, input.hidden, true, stream, &scratch,
                &round.constraint->candidates(wave), input.mrope, input.rotary_position);
            submitted = true;
        } else {
            rows[size] = input;
            batched[size++] = &round;
        }
    }
    MtpDraftRequest::enqueue_wave(rows.data(), size, true, stream);
    for (int index = 0; index < size; ++index) {
        batched[index]->draft_hidden = rows[index].hidden;
    }
    return submitted || size != 0;
}

}

bool MtpRound::propose_all(MtpRound* rounds, int count, const ConstraintWorkspace& scratch,
                           cudaStream_t stream, const Cancel& cancel) {
    std::array<MtpDraftInput, decoder_batch_max_mtp_lanes> unscored;
    int unscored_count = 0;
    int longest = 0;
    for (int index = 0; index < count; ++index) {
        auto& round = rounds[index];
        round.draft_hidden = round.request->work.normalized;
        longest = std::max(longest, round.proposal_limit);
        if (round.proposal_limit == 0) {
            unscored[unscored_count++] = round.draft_input(
                round.position - 1, round.tokens.front(), round.draft_hidden);
        }
    }
    MtpDraftRequest::enqueue_wave(unscored.data(), unscored_count, false, stream);
    for (int wave = 0; wave < longest; ++wave) {
        if (cancel && cancel()) {
            return false;
        }
        if (!enqueue_proposal_wave(rounds, count, wave, scratch, stream)) {
            break;
        }
        synchronize_cuda(stream, "Complete MTP draft wave");
        bool retried = false;
        for (int index = 0; index < count; ++index) {
            auto& round = rounds[index];
            if (wave < round.proposal_limit && round.draft->retry_invalid_int8(stream)) {
                retried = true;
            }
        }
        if (retried) {
            synchronize_cuda(stream, "Complete BF16 MTP draft score retries");
        }
        for (int index = 0; index < count; ++index) {
            auto& round = rounds[index];
            if (wave >= round.proposal_limit) {
                continue;
            }
            const int token = round.draft->read_token();
            if (token == -1) {
                round.proposal_limit = wave;
            } else {
                round.tokens[++round.drafted] = token;
                if (round.constraint && !round.constraint->advance(wave, token)) {
                    round.proposal_limit = wave + 1;
                }
            }
        }
    }
    return !cancel || !cancel();
}

bool MtpRound::propose_packets(MtpRound* rounds, int count, const ConstraintWorkspace& scratch,
                               cudaStream_t stream, const Cancel& cancel) {
    constexpr int packet = decoder_batch_max_mtp_lanes;
    for (int first = 0; first < count; first += packet) {
        if (!propose_all(rounds + first, std::min(count - first, packet), scratch, stream,
                         cancel)) {
            return false;
        }
    }
    return true;
}

void MtpRound::propose_depth(MtpRound* rounds, int count, const DraftDepthScratch& scratch,
                             cudaStream_t stream) {
    std::array<MtpDraftInput, decoder_batch_max_lanes> inputs;
    for (int index = 0; index < count; ++index) {
        rounds[index].draft_hidden = rounds[index].request->work.normalized;
    }
    for (int depth = 0; depth < decoder_batch_max_mtp_steps; ++depth) {
        int active = 0;
        for (int index = 0; index < count; ++index) {
            auto& round = rounds[index];
            inputs[index].request = nullptr;
            if (depth >= round.proposal_limit) continue;
            inputs[index] = round.draft_input(round.position - 1 + depth, round.tokens[depth],
                                              round.draft_hidden);
            ++active;
        }
        if (active == 0) break;
        MtpDraftRequest::enqueue_depth(inputs.data(), count, scratch, stream);
        synchronize_cuda(stream, "Complete depth-major MTP draft wave");
        bool retried = false;
        for (int index = 0; index < count; ++index) {
            if (!inputs[index].request) continue;
            auto& round = rounds[index];
            round.draft_hidden = inputs[index].hidden;
            retried |= round.draft->retry_invalid_int8(stream);
        }
        if (retried) {
            synchronize_cuda(stream, "Complete BF16 MTP draft score retries");
        }
        for (int index = 0; index < count; ++index) {
            if (!inputs[index].request) continue;
            auto& round = rounds[index];
            const int token = round.draft->read_token();
            if (token == -1) round.proposal_limit = depth;
            else round.tokens[++round.drafted] = token;
        }
    }
}

bool MtpRound::masks_stops(int row) const {
    return minimum_options != nullptr && first_output + row < minimum_options->min_new_tokens;
}

int MtpRound::winner(int row) const {
    return target->winner(row, sampling != nullptr || masks_stops(row), constraint != nullptr);
}

int MtpRound::text_stop_after(const DecoderOptions& options,
                              const RequestProgress& progress) const {
    ConstraintCursor preview;
    preview.reset_to(progress.constraint);
    std::array<int, target_verify_max_rows> winners;
    int count = 0;
    bool terminal = false;
    const int pending = rows();
    while (count < pending) {
        const int token = winner(count);
        winners[count++] = token;
        terminal = RequestProgress::finish_token(options, preview, token,
                                                  remaining - count).has_value();
        if (terminal || count > drafted || token != tokens[count]) break;
    }
    const int stop = options.text_stop(winners.data(), count, terminal);
    return stop >= 1 && stop <= count ? stop : 0;
}

bool MtpRound::commit(const Emit& emit, cudaStream_t stream, const Cancel& cancel,
                      MtpDraftInput* repair, const Workspace** committed_row) {
    if (draft->valid_prefix() > position) {
        draft->rewind(position);
    }
    const int count = rows();
    bool continuing = true;
    for (int index = 0; index < count; ++index) {
        if (cancel && cancel()) {
            if (consumed > 0 && committed_row != nullptr) {
                *committed_row = target->checkpoint_row(consumed - 1);
            }
            target->commit(consumed);
            return false;
        }
        const int token = winner(index);
        const bool accepted = index < drafted && token == tokens[index + 1];
        ++consumed;
        continuing = emit(token, target->logits(index), accepted, index);
        if (!accepted || !continuing) {
            break;
        }
        const auto row = target->row(index);
        const auto input = draft_input(row.position, token, row.work->normalized);
        if (!repair) {
            draft->enqueue(input.position, input.token, input.hidden, false, stream, nullptr,
                           nullptr, input.mrope, input.rotary_position);
        } else {
            repair[index] = input;
        }
    }
    if (committed_row != nullptr && !continuing) {
        *committed_row = target->checkpoint_row(consumed - 1);
    }
    if (continuing) {
        const auto* hidden = target->row(consumed - 1).work->normalized;
        copy_cuda_async(request->work.normalized, hidden, hidden_size * sizeof(Bf16),
                        cudaMemcpyDeviceToDevice, stream, "Carry committed MTP hidden row");
    }
    target->commit(consumed);
    return true;
}

TargetVerification::TargetVerification(Request& request, int capacity, int prefix, int rows,
                                       bool logits, cudaStream_t stream,
                                       std::shared_ptr<TextLogitCandidate> winners,
                                       int generated_top_logprobs, bool rank_local_gdn)
    : request_(request), stream_(stream ? stream : request.memory.stream()),
      capacity_(capacity), valid_prefix_(prefix),
      generated_top_logprobs_(generated_top_logprobs),
      rank_local_gdn_(rank_local_gdn) {
    auto& memory = request.memory;
    auto& host_owner = memory.transfers().verification_results;
    const int reserved = std::max(request.kv_limit, request.pages.page_count * mxfp8_page_size);
    if (!stream_ || (memory.stream() && memory.stream() != stream_) ||
        memory.graph() || host_owner || rows < 1 ||
        rows > target_verify_max_rows || prefix < 0 || prefix > capacity ||
        capacity > reserved || generated_top_logprobs < -1 ||
        generated_top_logprobs > decoder_max_generated_top_logprobs) {
        throw std::invalid_argument("Invalid target verification storage");
    }
    eager_workspace_.emplace(memory, rows, rank_local_gdn_);

    const bool score = generated_top_logprobs >= 0;
    if (winners && !logits && !score) {
        host_winners_ = winners.get();
        host_owner = std::move(winners);
        host_logits_ = nullptr;
        return;
    }
    const auto winner_bytes = rows * sizeof(TextLogitCandidate);
    static_assert(sizeof(TextLogitCandidate) % alignof(GeneratedLogprobReadback) == 0);
    const auto score_bytes = score ? rows * sizeof(GeneratedLogprobReadback) : 0;
    const auto logits_bytes = logits ? rows * vocab_size * sizeof(Bf16) : 0;
    void* host = nullptr;
    check(cudaMallocHost(&host, winner_bytes + score_bytes + logits_bytes),
          "Allocate verifier results");
    host_owner = std::shared_ptr<void>(host, free_pinned);
    host_winners_ = static_cast<TextLogitCandidate*>(host);
    auto* const scores = static_cast<std::byte*>(host) + winner_bytes;
    host_generated_logprobs_ = score ? reinterpret_cast<GeneratedLogprobReadback*>(scores)
                                      : nullptr;
    host_logits_ = logits ? reinterpret_cast<Bf16*>(scores + score_bytes) : nullptr;
}

void TargetVerification::begin(int position, int count, cudaStream_t stream) {
    if (!can_begin(position, count, stream)) {
        throw std::invalid_argument("Invalid target verification interval");
    }
    pending_rows_ = count;
    active_workspace_ = &*eager_workspace_;
}

void TargetVerification::cancel_unlaunched() {
    pending_rows_ = 0;
    active_workspace_ = nullptr;
}

VerificationRow TargetVerification::row(int index) {
    return {&request_, &workspace().work.at(index), valid_prefix_ + index, this, index == 0};
}

const Workspace* TargetVerification::checkpoint_row(int index) {
    assert(eager_workspace_.has_value());
    const auto& destination = eager_workspace_->work.at(index);
    if (active_workspace_ != nullptr && active_workspace_ != &*eager_workspace_) {
        const auto& source = active_workspace_->work[index];
        constexpr auto bytes = hidden_size * sizeof(Bf16);
        copy_cuda_async(destination.hidden, source.hidden, bytes, cudaMemcpyDeviceToDevice, stream_,
                        "Copy committed verification hidden row");
        copy_cuda_async(destination.residual, source.residual, bytes, cudaMemcpyDeviceToDevice,
                        stream_, "Copy committed verification residual row");
    }
    return &destination;
}

bool TargetVerification::graph_eligible(const VerificationSequence& sequence,
                                        bool split_attention) const {
    assert(eager_workspace_.has_value());
    const int capacity = request_.pages.page_count * mxfp8_page_size;
    const auto rotary = static_cast<std::int64_t>(sequence.position) + sequence.rotary_delta;
    return rotary >= 0 && rotary + sequence.count <= text_attention_max_tokens &&
           (split_attention ||
            sequence.position <=
                eager_workspace_->work.front().gqa_scores(capacity).capacity - sequence.count) &&
           std::all_of(sequence.tokens, sequence.tokens + sequence.count,
                       [](int token) { return token >= 0 && token < vocab_size; });
}

void TargetVerification::use_graph_workspace(VerificationWorkspace& graph_workspace) {
    active_workspace_ = &graph_workspace;
}

gdn_replay_ssm::Lane TargetVerification::gdn_lane(int layer) {
    auto& scratch = workspace();
    const auto& work = scratch.work.front();
    return {
        request_.gdn_replay[layer], scratch.gdn,    work.gdn_z,
        scratch.observable,         work.attention, pending_rows_,
    };
}

void TargetVerification::constrain(const MtpConstraintRound& constraint,
                                  const ConstraintWorkspace& scratch,
                                  const SamplingParameters* sampling, int first_output,
                                  void* sampling_scratch) {
    for (int index = 0; index < pending_rows_; ++index) {
        const auto& work = workspace().work[index];
        select_constrained({work.logits, work.argmax_scratch, work.argmax}, scratch,
                           constraint.candidates(index), sampling, first_output + index,
                           sampling_scratch, stream_);
    }
}

void TargetVerification::enqueue_generated_logprobs(int row, void* scratch, const float* adjusted) {
    if (!needs_generated_logprobs()) return;
    const auto& work = workspace().work[row];
    check(decoder::enqueue_generated_logprobs(work.logits, vocab_size, work.argmax,
                                              generated_top_logprobs_, scratch,
                                              host_generated_logprobs_ + row, stream_, adjusted),
          "Score target verification row");
}

void TargetVerification::read() {
    auto* verifier = this;
    read_all(&verifier, 1, stream_);
}

void TargetVerification::read_all(TargetVerification* const* verifiers, int count,
                                   cudaStream_t stream) {
    for (int index = 0; index < count; ++index) {
        const auto* verifier = verifiers[index];
        if (!verifier || verifier->stream_ != stream ||
            verifier->pending_rows_ == 0 || verifier->ready_) {
            throw std::logic_error("No target verification results pending on this stream");
        }
    }
    for (int index = 0; index < count; ++index) {
        verifiers[index]->enqueue_results();
    }
    synchronize_cuda(stream, "Complete target verification round");
    for (int index = 0; index < count; ++index) {
        verifiers[index]->ready_ = true;
    }
}

void TargetVerification::enqueue_results() {
    const auto& scratch = workspace();
    copy_cuda_async(host_winners_, scratch.winners, pending_rows_ * sizeof(TextLogitCandidate),
                    cudaMemcpyDeviceToHost, stream_, "Read target verification winners");
    if (host_logits_) {
        for (int index = 0; index < pending_rows_; ++index) {
            copy_cuda_async(host_logits_ + index * vocab_size, scratch.work[index].logits,
                            vocab_size * sizeof(Bf16), cudaMemcpyDeviceToHost, stream_,
                            "Read target verification logits");
        }
    }
}

int TargetVerification::winner(int index, bool sampled, bool constrained) const {
    return validate_winner(host_winners_[index].token_id, sampled, constrained);
}

const Bf16* TargetVerification::logits(int index) const {
    return host_logits_ ? host_logits_ + index * vocab_size : nullptr;
}

std::shared_ptr<const DecoderTokenLogprobs>
TargetVerification::generated_logprobs(int index) const {
    if (!needs_generated_logprobs()) return nullptr;
    return collect_generated_logprobs(host_generated_logprobs_[index], generated_top_logprobs_);
}

std::shared_ptr<TextLogitCandidate> prepare_verification_winners(int slots) {
    check(cudaPeekAtLastError(), "Check CUDA status before allocating batch verifier results");
    void* pinned = nullptr;
    const auto bytes = slots * target_verify_max_rows * sizeof(TextLogitCandidate);
    const auto status = cudaMallocHost(&pinned, bytes);
    if (status == cudaErrorMemoryAllocation) {
        const auto pending = cudaGetLastError();
        if (pending != status) check(pending, "Check CUDA status before verifier fallback");
        return {};
    }
    check(status, "Allocate batch verifier results");
    return {static_cast<TextLogitCandidate*>(pinned), free_pinned};
}

}
namespace sglang {

using namespace decoder;

void decoder::project_verification_bf16(Bf16Weight weight, const VerificationRow* packet,
                                        int count, Bf16* Workspace::*output, cudaStream_t stream) {
    std::array<TextBf16Projection, text_fp8_max_batch> projections;
    const int groups = (count + text_fp8_max_batch - 1) / text_fp8_max_batch;
    for (int first = 0, group = 0; first < count; ++group) {
        const int rows = count / groups + (group < count % groups);
        for (int index = 0; index < rows; ++index) {
            const auto& work = *packet[first + index].work;
            projections[index] = {work.normalized, work.*output};
        }
        check(text_bf16_gemm(weight.values, weight.rows, weight.columns, projections.data(),
                             rows, stream), "Run batched BF16 projection");
        first += rows;
    }
}
namespace {
static_assert(tc_vocab_short_rows == 2 * target_verify_max_rows);

void score_verification_packet(const DecoderModel& model, const VerificationRow* first,
                               int size, const TcVocabScratch* vocabulary,
                               cudaStream_t stream, bool short_packet = false) {
    for (int index = 0; index < size; ++index)
        model.normalize_final(*first[index].work, stream);
    const bool short_rows = short_packet && size == tc_vocab_short_rows;
    if (vocabulary && (short_rows || size == tc_vocab_rows || size <= tc_vocab_narrow_rows)) {
        std::array<VerificationRow, tc_vocab_rows> scored;
        const auto end = std::copy_if(first, first + size, scored.begin(),
            [](const VerificationRow& row) {
                return row.verification && row.verification->needs_generated_logprobs();
            });
        const int scored_rows = static_cast<int>(end - scored.begin());
        if (scored_rows == 0 || (size == tc_vocab_rows && scored_rows < size)) {
            std::array<TcVocabRow, tc_vocab_rows> projection;
            for (int index = 0; index < size; ++index) {
                const auto& work = *first[index].work;
                projection[index] = {work.normalized, work.argmax};
            }
            const bool certified = short_rows
                ? try_tc_vocab_short(*model.tc_vocab, *vocabulary, projection.data(), stream)
                : try_tc_vocab(*model.tc_vocab, *vocabulary, projection.data(), size, stream);
            if (certified) {
                if (scored_rows != 0) {
                    project_verification_bf16(model.lm_head, scored.data(), scored_rows,
                                              &Workspace::logits, stream);
                }
                return;
            }
        }
    }
    bool projected = false;
    if (size >= 2 && size <= verification_packet_rows) {
        const auto* verification = first->verification;
        auto* staging = verification != nullptr ? verification->bf16_head_staging() : nullptr;
        auto* staging_logits =
            verification != nullptr ? verification->bf16_head_logits_staging() : nullptr;
        if (staging != nullptr && staging_logits != nullptr) {
            std::array<Bf16HeadRow, verification_packet_rows> rows;
            for (int index = 0; index < size; ++index) {
                const auto& work = *first[index].work;
                rows[index] = {work.normalized, work.logits};
            }
            const auto status = project_bf16_head(
                model.lm_head.values, model.lm_head.rows, model.lm_head.columns, rows.data(),
                size, staging, staging_logits, stream);
            if (status) {
                check(*status, "Project verification packet through the BF16 head");
                projected = true;
            }
        }
    }
    if (!projected) {
        project_verification_bf16(model.lm_head, first, size, &Workspace::logits, stream);
    }
    for (int index = 0; index < size; ++index) {
        const auto& row = first[index];
        const auto& work = *row.work;
        if (!row.constrained) {
            check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
                  "Select target verification winner");
        }
    }
}

void score_target_rows(const DecoderModel& model, const VerificationRow* rows, int count,
                       const TcVocabScratch* vocabulary, cudaStream_t stream, bool packed);

void read_round_results(const MtpRound* rounds, int count, void* sampling_scratch,
                        const ConstraintWorkspace& scratch, void* logprob_workspace,
                        TargetVerification* const* verifiers, cudaStream_t stream) {
    std::array<SamplingRow, max_wide_mtp_rows> sampling;
    int sampled = 0;
    for (int index = 0; index < count; ++index) {
        const auto& round = rounds[index];
        if (round.constraint != nullptr) {
            round.target->constrain(*round.constraint, scratch, round.sampling, round.first_output,
                                    sampling_scratch);
        }
        for (int row = 0; row < round.target->pending_rows(); ++row) {
            auto& work = *round.target->row(row).work;
            const int position = round.first_output + row;
            const float* adjusted = nullptr;
            if (round.masks_stops(row)) {
                round.adjustment->select(*round.request, work, *round.minimum_options, nullptr,
                                         round.sampling, position, round.tokens[row], stream);
                adjusted = round.adjustment->logits;
            } else if (round.sampling != nullptr && round.constraint == nullptr) {
                sampling[sampled++] = {
                    work.logits,
                    &work.argmax->token_id,
                    *round.sampling,
                    position,
                };
            }
            round.target->enqueue_generated_logprobs(row, logprob_workspace, adjusted);
        }
    }
    if (sampled != 0) {
        constexpr auto batch_bytes = static_cast<std::size_t>(verification_packet_rows) *
                                     (hidden_size + vocab_size) * sizeof(Bf16);
        check(sample_logits_rows(sampling.data(), sampled, vocab_size, sampling_scratch,
                                 batch_bytes, rounds[0].target->bf16_head_staging(), stream),
              "Sample target verification packet");
    }
    TargetVerification::read_all(verifiers, count, stream);
}

bool needs_exact_head(const MtpRound& round, int packet_width, int rows) {
    return round.constraint != nullptr || round.sampling != nullptr ||
           round.minimum_options != nullptr || round.target->needs_logits() ||
           (packet_width != 1 && rows != target_verify_max_rows);
}

VerificationSequence make_sequence(const MtpRound& round, int rows, bool exact_head = false) {
    return {
        round.target,
        round.position,
        round.tokens.data(),
        rows,
        round.constraint != nullptr,
        round.rotary_delta(),
        exact_head,
    };
}

VerificationRow packet_row(const VerificationSequence& sequence, TargetVerification& target,
                           int depth) {
    auto row = target.row(depth);
    row.constrained = sequence.constrained;
    row.rotary_position = row.position + sequence.rotary_delta;
    return row;
}

struct PacketExtent {
    int lanes;
    int depth;
};
struct PacketIndex {
    int lane;
    int depth;
    int index;
};

template <typename Gather>
int gather_packet_rows(const VerificationSequence* sequences, PacketExtent extent,
                       const Gather& gather) {
    int next = 0;
    for (int first = 0; first < extent.lanes; first += decoder_batch_max_mtp_lanes) {
        const int end = std::min(extent.lanes, first + decoder_batch_max_mtp_lanes);
        for (int depth = 0; depth < extent.depth; ++depth) {
            for (int lane = first; lane < end; ++lane) {
                const auto& sequence = sequences[lane];
                if (depth < sequence.count) {
                    gather(sequence, PacketIndex{lane, depth, next++});
                }
            }
        }
    }
    return next;
}
}

bool Decoder::Model::verify_mtp(MtpRound* rounds, int count, const ConstraintWorkspace& scratch,
                                void* sampling_scratch, cudaStream_t stream,
                                VerificationGraph* graph, const TcVocabScratch* vocabulary,
                                std::uint8_t* packed_activation, void* logprob_workspace) const {
    std::array<VerificationSequence, max_wide_mtp_lanes> sequences;
    std::array<TargetVerification*, max_wide_mtp_lanes> verifiers{};
    const bool joined = count > decoder_batch_max_mtp_lanes && packed_activation == nullptr;
    if (count < 1 || count > max_wide_mtp_lanes || (joined && graph == nullptr) ||
        (packed_activation != nullptr && (!wide_mtp_width(count) || vocabulary == nullptr))) {
        throw std::invalid_argument("Invalid number of MTP rounds");
    }
    for (int index = 0; index < count; ++index) {
        const auto& round = rounds[index];
        const int rows = round.rows();
        verifiers[index] = round.target;
        const int packet_first = index - (index % decoder_batch_max_mtp_lanes);
        const int packet_width =
            joined ? std::min(count - packet_first, decoder_batch_max_mtp_lanes) : count;
        const bool exact = needs_exact_head(round, packet_width, rows);
        sequences[index] = make_sequence(round, rows, exact);
        if (exact && !joined)
            vocabulary = nullptr;
    }
    if (packed_activation != nullptr && vocabulary == nullptr) {
        throw std::invalid_argument("Wide MTP requires full greedy verification rows");
    }
    const bool replayed =
        forward_verify(sequences.data(), count, stream, graph, vocabulary, packed_activation);
    if (joined && !replayed)
        return false;
    read_round_results(rounds, count, sampling_scratch, scratch, logprob_workspace,
                       verifiers.data(), stream);
    return replayed;
}

namespace {
void gather_vocab_rows(const VerificationRow* input, TcVocabRow* output, TcVocabRmsRows& rms) {
    for (int index = 0; index < tc_vocab_wide_rows; ++index) {
        const auto& work = *input[index].work;
        output[index] = {work.normalized, work.argmax};
        rms.rows[index] = {work.hidden, work.residual, work.normalized};
    }
}

bool score_paired_target_rows(const DecoderModel& model, const VerificationRow* rows, int count,
                              const TcVocabScratch* vocabulary, cudaStream_t stream) {
    constexpr int paired_rows = 2 * tc_vocab_wide_rows;
    if (vocabulary == nullptr || count != paired_rows)
        return false;
    std::array<TcVocabRow, paired_rows> projection{};
    TcVocabRmsRows rms[2];
    for (int half = 0; half < 2; ++half) {
        const int start = half * tc_vocab_wide_rows;
        gather_vocab_rows(rows + start, projection.data() + start, rms[half]);
    }
    return try_tc_vocab_final_pair(*model.tc_vocab, *vocabulary, projection.data(), rms,
                                   model.final_norm, stream);
}

void score_target_rows(const DecoderModel& model, const VerificationRow* rows, int total,
                       const TcVocabScratch* vocabulary, cudaStream_t stream, bool packed) {
    if (!packed) {
        score_verification_packet(model, rows, total, vocabulary, stream);
        return;
    }
    static_assert(tc_vocab_wide_rows == 2 * verification_packet_rows);
    for (int first = 0; first < total;) {
        const auto* packet = rows + first;
        if (vocabulary && total - first >= tc_vocab_wide_rows &&
            std::none_of(packet, packet + tc_vocab_wide_rows, [](const VerificationRow& row) {
                return row.verification && row.verification->needs_generated_logprobs();
            })) {
            std::array<TcVocabRow, tc_vocab_wide_rows> projection;
            TcVocabRmsRows rms;
            gather_vocab_rows(packet, projection.data(), rms);
            if (try_tc_vocab_final(*model.tc_vocab, *vocabulary, projection.data(), rms,
                                   model.final_norm, stream)) {
                first += tc_vocab_wide_rows;
                continue;
            }
        }
        const int size = tp_verify_graph::packet_rows(packet, total - first);
        score_verification_packet(model, packet, size, vocabulary, stream);
        first += size;
    }
}
}

bool Decoder::Model::verify_mtp_tp(
    const Model& peer, MtpRound* rounds, const MtpRound* peer_rounds, int count,
    const ConstraintWorkspace& scratch, void* sampling_scratch, cudaStream_t stream,
    cudaStream_t peer_stream, TensorParallel& group, const TpPartials& primary_partials,
    const TpPartials& peer_partials, const TcVocabScratch* vocabulary, std::uint8_t* primary_packed,
    std::uint8_t* peer_packed, void* logprob_workspace, bool capture) const {
    std::array<VerificationSequence, max_wide_mtp_lanes> sequences;
    std::array<TargetVerification*, max_wide_mtp_lanes> verifiers{};
    std::array<TargetVerification*, max_wide_mtp_lanes> peer_verifiers;
    if (count < 1 || count > tp_verify_graph::max_lanes || rounds == nullptr ||
        peer_rounds == nullptr) {
        throw std::invalid_argument("Invalid number of tensor-parallel MTP rounds");
    }
    if (count > decoder_batch_max_mtp_lanes && (!primary_packed || !peer_packed)) {
        throw std::invalid_argument("Wide tensor-parallel MTP requires packed activation staging");
    }
    const bool sampled_wide = count > decoder_batch_max_mtp_lanes &&
                              std::all_of(rounds, rounds + count, [](const MtpRound& round) {
                                  return round.sampling != nullptr && round.constraint == nullptr;
                              });
    int total = 0;
    int longest = 0;
    for (int index = 0; index < count; ++index) {
        const auto& round = rounds[index];
        const auto& peer_round = peer_rounds[index];
        const int rows = round.rows();
        if (!round.target || !peer_round.target || peer_round.constraint ||
            peer_round.sampling || peer_round.position != round.position ||
            !round.target->can_begin(round.position, rows, stream) ||
            !peer_round.target->can_begin(round.position, rows, peer_stream)) {
            throw std::invalid_argument("Invalid tensor-parallel MTP round");
        }
        if (needs_exact_head(round, count, rows) || round.target->needs_generated_logprobs()) {
            vocabulary = nullptr;
        }
        total += rows;
        longest = std::max(longest, rows);
    }
    if (count > decoder_batch_max_mtp_lanes && !sampled_wide &&
        (!wide_mtp_width(count) || vocabulary == nullptr)) {
        throw std::invalid_argument("Wide MTP requires full greedy verification rows");
    }
    for (int index = 0; index < count; ++index) {
        const auto& round = rounds[index];
        const auto& peer_round = peer_rounds[index];
        sequences[index] = make_sequence(round, round.rows());
        verifiers[index] = round.target;
        peer_verifiers[index] = peer_round.target;
        round.target->begin(round.position, round.rows(), stream);
        peer_round.target->begin(round.position, round.rows(), peer_stream);
    }
    std::array<VerificationRow, max_wide_mtp_rows> rows;
    std::array<VerificationRow, max_wide_mtp_rows> peer_rows;
    static_assert(max_wide_mtp_rows <= VerificationCopyRows::capacity);
    VerificationEmbeddingTokens tokens{};
    VerificationCopyRows inputs{};
    VerificationCopyRows peer_inputs{};
    const int next =
        gather_packet_rows(sequences.data(), {count, longest},
                           [&](const VerificationSequence& sequence, PacketIndex coordinate) {
                               const auto [lane, depth, index] = coordinate;
                               rows[index] = packet_row(sequence, *sequence.verification, depth);
                               peer_rows[index] =
                                   packet_row(sequence, *peer_verifiers[lane], depth);
                               rows[index].binding_lane = lane;
                               peer_rows[index].binding_lane = lane;
                               tokens.id[index] = sequence.tokens[depth];
                               inputs.destination[index] = rows[index].work->residual;
                               peer_inputs.destination[index] = peer_rows[index].work->residual;
                           });
    check(gather_verification_embeddings(embedding, vocab_size, tokens, inputs, next, stream),
          "Look up verification inputs");
    {
        DeviceScope peer_scope(peer.device);
        check(gather_verification_embeddings(peer.embedding, vocab_size, tokens, peer_inputs, next,
                                             peer_stream),
              "Look up peer verification inputs");
        peer_scope.restore();
    }
    bool eager_replicated = false;
    auto* const captured =
        forward_verification_tp(peer, rows.data(), peer_rows.data(), count, total, stream,
                                peer_stream, group, primary_partials, peer_partials, primary_packed,
                                peer_packed, vocabulary != nullptr, capture, eager_replicated);
    const tp_verify_graph::Submission submitted(group, captured);
    if (count <= decoder_batch_max_mtp_lanes) {
        score_verification_packet(*this, rows.data(), total, vocabulary, stream, count == 2);
    } else if (!score_paired_target_rows(*this, rows.data(), total, vocabulary, stream)) {
        score_target_rows(*this, rows.data(), total, vocabulary, stream, true);
    }
    read_round_results(rounds, count, sampling_scratch, scratch, logprob_workspace,
                       verifiers.data(), stream);
    const DeviceScope peer_scope(peer.device);
    TargetVerification::read_all(peer_verifiers.data(), count, peer_stream);
    return captured != nullptr;
}

bool Decoder::Model::forward_verify(const VerificationSequence* sequences, int count,
                                    cudaStream_t stream, VerificationGraph* graph,
                                    const TcVocabScratch* vocabulary,
                                    std::uint8_t* packed_activation) const {
    const bool wide = packed_activation != nullptr;
    const bool joined = count > decoder_batch_max_mtp_lanes && !wide;
    if (sequences == nullptr || count < 1 || count > max_wide_mtp_lanes ||
        (joined && graph == nullptr)) {
        throw std::invalid_argument("Invalid number of verification sequences");
    }
    int total = 0;
    int longest = 0;
    for (int index = 0; index < count; ++index) {
        const auto& sequence = sequences[index];
        if (!sequence.verification || !sequence.tokens ||
            sequence.count < 1 || sequence.count > target_verify_max_rows) {
            throw std::invalid_argument("Invalid number of verification rows");
        }
        if (wide && (sequence.count != target_verify_max_rows || sequence.constrained)) {
            throw std::invalid_argument("Invalid wide verification packet");
        }
        total += sequence.count;
        longest = std::max(longest, sequence.count);
        sequence.verification->begin(sequence.position, sequence.count, stream);
    }
    std::array<VerificationRow, max_wide_mtp_rows> rows;
    gather_packet_rows(sequences, {count, longest},
                       [&](const VerificationSequence& sequence, PacketIndex coordinate) {
                           const auto [lane, depth, index] = coordinate;
                           auto row = packet_row(sequence, *sequence.verification, depth);
                           row.binding_lane = lane;
                           row.binding_row = depth;
                           row.binding_index = index;
                           rows[index] = row;
                       });
    if (graph != nullptr &&
        graph->replay(*this, sequences, count, rows.data(), total, wide, stream)) {
        if (joined) {
            for (int first = 0; first < total;) {
                const int packet = rows[first].binding_lane / decoder_batch_max_mtp_lanes;
                int end = first + 1;
                bool exact = sequences[rows[first].binding_lane].exact_head;
                for (;
                     end < total && rows[end].binding_lane / decoder_batch_max_mtp_lanes == packet;
                     ++end) {
                    exact |= sequences[rows[end].binding_lane].exact_head;
                }
                score_verification_packet(*this, rows.data() + first, end - first,
                                          exact ? nullptr : vocabulary, stream);
                first = end;
            }
        } else {
            score_target_rows(*this, rows.data(), total, vocabulary, stream, wide);
        }
        return true;
    }
    if (joined) {
        for (int lane = 0; lane < count; ++lane)
            sequences[lane].verification->cancel_unlaunched();
        return false;
    }
    for (int index = 0; index < total; ++index) {
        const auto& row = rows[index];
        check(text_embedding(embedding, vocab_size, hidden_size,
                             sequences[row.binding_lane].tokens[row.binding_row],
                             row.work->residual, stream),
              "Look up verification input");
    }
    forward_verification_layers(rows.data(), total, stream, packed_activation);
    score_target_rows(*this, rows.data(), total, vocabulary, stream, wide);
    return false;
}
}
