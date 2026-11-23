#include "sglang/decoder_verify.h"

#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_sampling.h"
#include "sglang/text/decoder_logprobs.h"
#include "sglang/text/mtp/tc_vocab.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <utility>

namespace sglang::decoder {
namespace {

TextGdnReplayRows replay_rows(Bf16* storage, int count) {
    const int qkv = count * text_gdn_qkv_width;
    const int gates = count * text_gdn_value_heads;
    return {storage, storage + qkv, storage + 2 * qkv, storage + 2 * qkv + gates};
}

void free_pinned(void* results) {
    const auto status = cudaFreeHost(results);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "sglang decoder: Free verifier pinned storage: %s\n",
                     cudaGetErrorString(status));
    }
}

}

VerificationWorkspace::VerificationWorkspace(Allocations& memory, int rows) {
    const int layer_elements = 2 * rows * (text_gdn_qkv_width + text_gdn_value_heads);
    auto* storage = memory.allocate<Bf16>(gdn_layers * layer_elements);
    for (auto& layer : replay) {
        layer.inputs = replay_rows(storage, rows);
        storage += layer_elements;
    }
    auto* z = memory.allocate<Bf16>(rows * text_gdn_value_width);
    auto* attention = memory.allocate<Bf16>(rows * text_gdn_value_width);
    observable = memory.allocate<Bf16>(rows * text_gdn_value_width);
    winners = memory.allocate<TextLogitCandidate>(rows);
    work.reserve(rows);
    for (int index = 0; index < rows; ++index) {
        auto& row = work.emplace_back(memory, attention + index * text_gdn_value_width,
                                      winners + index);
        row.gdn_z = z + index * text_gdn_value_width;
    }
}

void VerificationWorkspace::bind_gdn(int layer, int rows) {
    const auto inputs = replay[layer].inputs;
    for (int index = 0; index < rows; ++index) {
        work[index].gdn_qkv = inputs.qkv + index * text_gdn_qkv_width;
        work[index].gdn_b = inputs.b + index * text_gdn_value_heads;
        work[index].gdn_a = inputs.a + index * text_gdn_value_heads;
    }
}

VerificationGraphScratch::VerificationGraphScratch(int device)
    : memory(device), workspace(memory, k1_verification_rows),
      device_binding(memory.allocate<VerificationGraphBinding>(1)) {
    void* pinned = nullptr;
    check(cudaMallocHost(&pinned, sizeof(VerificationGraphBinding)),
          "Allocate K1 verifier graph bindings");
    host_binding = std::shared_ptr<VerificationGraphBinding>(
        static_cast<VerificationGraphBinding*>(pinned), free_pinned);
}

void VerificationGraphScratch::upload(const VerificationSequence& sequence, cudaStream_t stream) {
    auto& binding = *host_binding;
    sequence.verification->bind_graph(binding);
    const int capacity = binding.pages.page_count * mxfp8_page_size;
    for (int row = 0; row < k1_verification_rows; ++row) {
        binding.inputs[row] = {0.0f, sequence.tokens[row]};
        binding.positions[row] = sequence.position + row;
        binding.rotary_positions[row] = binding.positions[row] + sequence.rotary_delta;
        binding.scores[row] = workspace.work[row].gqa_scores(capacity);
    }
    check(cudaMemcpyAsync(device_binding, host_binding.get(), sizeof(binding),
                          cudaMemcpyHostToDevice, stream), "Upload K1 verifier graph bindings");
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
        check(cudaStreamSynchronize(stream), "Complete MTP draft wave");
        bool retried = false;
        for (int index = 0; index < count; ++index) {
            auto& round = rounds[index];
            if (wave < round.proposal_limit && round.draft->retry_invalid_int8(stream)) {
                retried = true;
            }
        }
        if (retried) {
            check(cudaStreamSynchronize(stream), "Complete BF16 MTP draft score retries");
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
        check(cudaStreamSynchronize(stream), "Complete depth-major MTP draft wave");
        bool retried = false;
        for (int index = 0; index < count; ++index) {
            if (!inputs[index].request) continue;
            auto& round = rounds[index];
            round.draft_hidden = inputs[index].hidden;
            retried |= round.draft->retry_invalid_int8(stream);
        }
        if (retried) {
            check(cudaStreamSynchronize(stream), "Complete BF16 MTP draft score retries");
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

bool MtpRound::commit(const Emit& emit, cudaStream_t stream, const Cancel& cancel,
                      MtpDraftInput* repair) {
    if (draft->valid_prefix() > position) {
        draft->rewind(position);
    }
    const int count = rows();
    bool continuing = true;
    for (int index = 0; index < count; ++index) {
        if (cancel && cancel()) {
            target->commit(consumed);
            return false;
        }
        const int token = target->winner(index, sampling != nullptr, constraint != nullptr);
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
    if (continuing) {
        const auto* hidden = target->row(consumed - 1).work->normalized;
        check(cudaMemcpyAsync(request->work.normalized, hidden, hidden_size * sizeof(Bf16),
                              cudaMemcpyDeviceToDevice, stream), "Carry committed MTP hidden row");
    }
    target->commit(consumed);
    return true;
}

TargetVerification::TargetVerification(Request& request, int capacity, int prefix, int rows,
                                       bool logits, cudaStream_t stream,
                                       std::shared_ptr<TextLogitCandidate> winners,
                                       int generated_top_logprobs)
    : request_(request), stream_(stream ? stream : request.memory.stream()),
      capacity_(capacity), valid_prefix_(prefix), generated_top_logprobs_(generated_top_logprobs) {
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
    allocate_gdn(memory, tail_);
    eager_workspace_.emplace(memory, rows);

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
    if (stream != stream_ || pending_rows_ != 0 ||
        position != valid_prefix_ || count < 1 ||
        count > static_cast<int>(eager_workspace_->work.size()) || count > capacity_ - position ||
        count > request_.pages.page_count * mxfp8_page_size - position) {
        throw std::invalid_argument("Invalid target verification interval");
    }
    pending_rows_ = count;
    active_workspace_ = &*eager_workspace_;
}

VerificationRow TargetVerification::row(int index) {
    return {&request_, &workspace().work.at(index), valid_prefix_ + index, this, index == 0};
}

bool TargetVerification::graph_eligible(const VerificationSequence& sequence) const {
    const int capacity = request_.pages.page_count * mxfp8_page_size;
    return sequence.count == k1_verification_rows &&
        sequence.position <= eager_workspace_->work.front().gqa_scores(capacity).capacity -
            k1_verification_rows &&
        std::all_of(sequence.tokens, sequence.tokens + k1_verification_rows,
                    [](int token) { return token >= 0 && token < vocab_size; });
}

void TargetVerification::bind_graph(VerificationGraphBinding& binding) const {
    binding.pages = request_.pages;
    binding.caches = request_.caches;
    binding.initial = request_.states;
    binding.tail = tail_;
}

void TargetVerification::use_graph_workspace(VerificationWorkspace& graph_workspace) {
    active_workspace_ = &graph_workspace;
}

void TargetVerification::bind_gdn(int layer) {
    workspace().bind_gdn(layer, pending_rows_);
}

GdnVerificationLane TargetVerification::gdn_lane(int layer, TextGdnWeights weights) {
    auto& scratch = workspace();
    auto& saved = scratch.replay[layer];
    saved.weights = weights;
    const auto& work = scratch.work.front();
    return {saved.inputs, work.gdn_z, request_.states[layer], tail_[layer],
            scratch.observable, work.attention};
}

void TargetVerification::advance_gdn(int layer, TextGdnWeights weights) {
    const auto lane = gdn_lane(layer, weights);
    check(text_gdn_verify_tail(pending_rows_, lane.inputs, lane.z, weights, lane.initial,
                               lane.tail, lane.observable, lane.output, stream_),
          "Advance target GDN and retain its tail");
}

void TargetVerification::sample(const SamplingParameters& parameters, int first_output,
                                void* scratch) {
    for (int index = 0; index < pending_rows_; ++index) {
        const auto& work = workspace().work[index];
        check(sample_logits(work.logits, vocab_size, &work.argmax->token_id, parameters,
                            first_output + index, scratch, stream_),
              "Sample target verification logits");
    }
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

void TargetVerification::enqueue_generated_logprobs(void* scratch) {
    if (!needs_generated_logprobs()) return;
    for (int index = 0; index < pending_rows_; ++index) {
        const auto& work = workspace().work[index];
        check(decoder::enqueue_generated_logprobs(work.logits, vocab_size, work.argmax,
              generated_top_logprobs_, scratch, host_generated_logprobs_ + index, stream_),
              "Score target verification row");
    }
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
    check(cudaStreamSynchronize(stream), "Complete target verification round");
    for (int index = 0; index < count; ++index) {
        verifiers[index]->ready_ = true;
    }
}

void TargetVerification::enqueue_results() {
    const auto& scratch = workspace();
    check(cudaMemcpyAsync(host_winners_, scratch.winners,
                          pending_rows_ * sizeof(TextLogitCandidate), cudaMemcpyDeviceToHost,
                          stream_), "Read target verification winners");
    if (host_logits_) {
        for (int index = 0; index < pending_rows_; ++index) {
            check(cudaMemcpyAsync(host_logits_ + index * vocab_size, scratch.work[index].logits,
                                  vocab_size * sizeof(Bf16), cudaMemcpyDeviceToHost, stream_),
                  "Read target verification logits");
        }
    }
}

void TargetVerification::commit(int count) {
    if (!ready_ || count < 0 || count > pending_rows_) {
        throw std::logic_error("Invalid target verification commit");
    }
    if (count > 0 && count < pending_rows_) {
        for (int layer = 0; layer < gdn_layers; ++layer) {
            const auto& saved = workspace().replay[layer];
            check(text_gdn_replay_prefix(count, saved.inputs, saved.weights,
                                         request_.states[layer], tail_[layer],
                                         stream_),
                  "Replay committed target GDN prefix");
        }
    }
    if (count > 0) {
        request_.states.swap(tail_);
        valid_prefix_ += count;
    }
    ready_ = false;
    pending_rows_ = 0;
    active_workspace_ = nullptr;
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

bool Decoder::Model::verify_mtp(MtpRound* rounds, int count, const ConstraintWorkspace& scratch,
                                void* sampling_scratch, cudaStream_t stream,
                                VerificationGraph* graph, const TcVocabScratch* vocabulary,
                                std::uint8_t* packed_activation, void* logprob_workspace) const {
    std::array<VerificationSequence, max_wide_mtp_lanes> sequences;
    std::array<TargetVerification*, max_wide_mtp_lanes> verifiers;
    if (count < 1 || (count > decoder_batch_max_mtp_lanes &&
        (!wide_mtp_width(count) || !packed_activation || !vocabulary))) {
        throw std::invalid_argument("Invalid number of MTP rounds");
    }
    for (int index = 0; index < count; ++index) {
        const auto& round = rounds[index];
        const int rows = round.rows();
        verifiers[index] = round.target;
        sequences[index] = {round.target, round.position, round.tokens.data(), rows,
                            round.constraint != nullptr, round.rotary_delta()};
        if (round.constraint || round.sampling || round.target->needs_logits() ||
            rows != target_verify_max_rows) {
            vocabulary = nullptr;
        }
    }
    if (count > decoder_batch_max_mtp_lanes && !vocabulary) {
        throw std::invalid_argument("Wide MTP requires full greedy verification rows");
    }
    const bool replayed = forward_verify(sequences.data(), count, stream, graph, vocabulary,
                                         packed_activation);
    for (int index = 0; index < count; ++index) {
        const auto& round = rounds[index];
        if (round.constraint) {
            round.target->constrain(*round.constraint, scratch, round.sampling, round.first_output,
                                    sampling_scratch);
        } else if (round.sampling) {
            round.target->sample(*round.sampling, round.first_output, sampling_scratch);
        }
        round.target->enqueue_generated_logprobs(logprob_workspace);
    }
    TargetVerification::read_all(verifiers.data(), count, stream);
    return replayed;
}

bool Decoder::Model::forward_verify(const VerificationSequence* sequences, int count,
                                    cudaStream_t stream, VerificationGraph* graph,
                                    const TcVocabScratch* vocabulary,
                                    std::uint8_t* packed_activation) const {
    const bool wide = wide_mtp_width(count) && packed_activation && vocabulary;
    if (!sequences || count < 1 || (count > decoder_batch_max_mtp_lanes && !wide)) {
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
    const auto& first = sequences[0];
    if (graph && count == 1 && !first.constrained && first.verification->graph_eligible(first) &&
        std::exchange(graph->warmed, true)) {
        const bool replayed = graph->execution.replay_batch_graph(1, [&] {
            graph->scratch = std::make_unique<VerificationGraphScratch>(device);
        }, [&] {
            auto& scratch = *graph->scratch;
            auto* binding = scratch.device_binding;
            std::array<VerificationRow, k1_verification_rows> rows;
            for (int index = 0; index < k1_verification_rows; ++index) {
                auto& work = scratch.workspace.work[index];
                rows[index] = {nullptr, &work, 0, nullptr, index == 0, binding, &scratch.workspace};
                check(decode_embedding(embedding, vocab_size, hidden_size, &binding->inputs[index],
                                       work.residual, stream), "Look up bound verification input");
            }
            enqueue_verification(rows.data(), k1_verification_rows, stream);
        }, [&] {
            graph->scratch->upload(first, stream);
            return true;
        }, true);
        if (replayed) {
            first.verification->use_graph_workspace(graph->scratch->workspace);
            return true;
        }
        graph->scratch.reset();
    }
    std::array<VerificationRow, max_wide_mtp_rows> rows;
    int next = 0;
    const auto prepare_packet = [&](int first_lane, int end_lane) {
        for (int index = 0; index < longest; ++index) {
            for (int lane = first_lane; lane < end_lane; ++lane) {
                const auto& sequence = sequences[lane];
                if (index < sequence.count) {
                    auto row = sequence.verification->row(index);
                    row.constrained = sequence.constrained;
                    row.rotary_position = row.position + sequence.rotary_delta;
                    rows[next++] = row;
                    check(text_embedding(embedding, vocab_size, hidden_size,
                                         sequence.tokens[index], row.work->residual, stream),
                          "Look up verification input");
                }
            }
        }
    };
    if (wide) {
        for (int first_lane = 0; first_lane < count; first_lane += decoder_batch_max_mtp_lanes) {
            prepare_packet(first_lane, first_lane + decoder_batch_max_mtp_lanes);
        }
    } else {
        prepare_packet(0, count);
    }
    enqueue_verification(rows.data(), total, stream, vocabulary, packed_activation);
    return false;
}

void Decoder::Model::enqueue_verification(const VerificationRow* rows, int count,
                                          cudaStream_t stream,
                                          const TcVocabScratch* vocabulary,
                                          std::uint8_t* packed_activation) const {
    forward_verification_layers(rows, count, stream, packed_activation);
    const auto score_packet = [&](const VerificationRow* first, int size) {
        for (int index = 0; index < size; ++index) {
            normalize_final(*first[index].work, stream);
        }
        if (vocabulary && size == tc_vocab_rows) {
            std::array<VerificationRow, tc_vocab_rows> scored;
            const auto end = std::copy_if(first, first + size, scored.begin(),
                [](const VerificationRow& row) {
                    return row.verification && row.verification->needs_generated_logprobs();
                });
            const int scored_rows = static_cast<int>(end - scored.begin());
            if (scored_rows < size) {
                std::array<TcVocabRow, tc_vocab_rows> projection;
                for (int index = 0; index < size; ++index) {
                    const auto& work = *first[index].work;
                    projection[index] = {work.normalized, work.argmax};
                }
                if (try_tc_vocab(*tc_vocab, *vocabulary, projection, stream)) {
                    if (scored_rows != 0) {
                        project_verification_bf16(lm_head, scored.data(), scored_rows,
                                                  &Workspace::logits, stream);
                    }
                    return;
                }
            }
        }
        project_verification_bf16(lm_head, first, size, &Workspace::logits, stream);
        for (int index = 0; index < size; ++index) {
            const auto& row = first[index];
            const auto& work = *row.work;
            if (!row.constrained) {
                check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax,
                                  stream),
                      "Select target verification winner");
            }
        }
    };
    if (packed_activation) {
        for (int first = 0; first < count; first += verification_packet_rows) {
            score_packet(rows + first, verification_packet_rows);
        }
    } else {
        score_packet(rows, count);
    }
}

}
