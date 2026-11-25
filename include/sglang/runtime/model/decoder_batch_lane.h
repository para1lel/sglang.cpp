#pragma once

#include "sglang/runtime/model/decoder_batch_checkpoint.h"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_sampling.h"
#include "sglang/decoder_verify.h"

namespace sglang::decoder {

struct Lane {
    std::unique_ptr<Request> request;
    std::vector<int> prompt;
    DecoderOptions options;
    int prompt_size = static_cast<int>(prompt.size());
    int checkpoint = 0;
    RequestProgress progress{};
    std::unique_ptr<MtpDraftRequest> draft = nullptr;
    std::unique_ptr<TargetVerification> verification = nullptr;
    // Tensor-parallel mirrors on rank 1 (rank 0 owns decisions and the draft).
    std::unique_ptr<Request> peer_request = nullptr;
    std::unique_ptr<TargetVerification> peer_verification = nullptr;
    std::unique_ptr<PendingBatchPrefix> peer_prefix = nullptr;
    std::optional<SamplingParameters> sampling = std::nullopt;
    std::optional<MtpConstraintRound> constraint_round{};
    bool checkpoints() const {
        return options.publish_prefix_checkpoints && bool(options.store_prefix);
    }
    // Rank-1 mirror of the rank-0 checkpoint row (the committed verification
    // row during MTP, else the peer request workspace) used by prefix exports.
    const Workspace* peer_checkpoint_row = nullptr;
    bool restored_prompt = false;
    std::shared_ptr<const MultimodalInput> media = nullptr;
    DecoderPromptLogprobs prompt_logprobs = {};
    MediaPrefill media_prefill{};
    std::size_t resident_bytes = 0;
    std::unique_ptr<PendingBatchPrefix> prefix = nullptr;
    std::optional<DeferredPrefixStore> prefix_store = std::nullopt;

    int remaining_prompt() const { return progress.remaining_prompt(prompt_size); }
    int remaining_output() const { return progress.remaining_output(options); }
    int capacity() const { return RequestProgress::capacity(prompt_size, options); }
    int checkpoint_room() const {
        constexpr int interval = decoder_batch_prefix_checkpoint_interval;
        return interval - progress.position % interval;
    }
    int prefill_rows(bool checkpoints) const {
        return checkpoints ? std::min(remaining_prompt(), checkpoint_room()) : remaining_prompt();
    }
    int proposals(bool checkpoints = false) const {
        const int available = checkpoints ? std::min(remaining_output(), checkpoint_room())
                                          : remaining_output();
        return std::min(options.mtp_steps, available - 1);
    }

    ForwardLane input() const {
        const int position = progress.position;
        if (remaining_prompt()) {
            return {request.get(), prompt[position], position, TextGdnMode::prefill, false,
                    media ? media_prefill.feature(position) : nullptr,
                    media ? media_prefill.positions + position : nullptr};
        }
        const int rotary = media ? position + media->prompt.next_position - prompt_size : -1;
        return {request.get(), progress.last_token, position, TextGdnMode::decode, false,
                nullptr, nullptr, rotary};
    }

    void record_decode_input(cudaStream_t stream) {
        prompt.push_back(progress.last_token);
        if (!draft || draft->valid_prefix() >= progress.position) return;
        // The last ordinary target step does not enter the verifier's draft preparation.
        MtpRound tail{};
        tail.draft = draft.get();
        tail.media = media ? &media_prefill : nullptr;
        auto input = tail.draft_input(progress.position - 1, progress.last_token,
                                      request->work.normalized);
        MtpDraftRequest::enqueue_wave(&input, 1, false, stream);
    }

    void finish_round(MtpRound& round, DecoderBatchStep& result, cudaStream_t stream,
                      SerialOutput* output, const MtpRound::Cancel& cancel,
                      MtpDraftInput* repair, const Workspace** committed_row) {
        result.mtp_drafted_tokens = round.drafted;
        result.mtp_verified_rows = round.rows();
        try {
            const bool scored = round.target->needs_generated_logprobs();
            const int stop = options.text_stop && (!cancel || !cancel())
                ? round.text_stop_after(options, progress) : 0;
            const bool completed = round.commit([&](int token, const Bf16* logits, bool accepted,
                                                    int row) {
                if (!result.token) {
                    result.token = token;
                    if (scored) result.generated_logprobs = round.target->generated_logprobs(row);
                } else {
                    const int index = result.extra_token_count++;
                    result.extra_tokens[index] = token;
                    if (scored) {
                        result.extra_generated_logprobs[index] =
                            round.target->generated_logprobs(row);
                    }
                }
                result.mtp_accepted_tokens += accepted;
                result.finish_reason = progress.commit(options, token, logits, output);
                if (row + 1 == stop && result.finish_reason != DecoderFinishReason::cancelled)
                    result.finish_reason = DecoderFinishReason::stop;
                return !result.finish_reason;
            }, stream, cancel, repair, committed_row);
            if (!completed) {
                result.finish_reason = DecoderFinishReason::cancelled;
            }
        } catch (const DecoderSamplingError&) {
            result.sampling_failed = true;
            return;
        } catch (const DecoderConstraintError&) {
            result.constraint_failed = true;
            return;
        }
        if (result.extra_token_count != 0) {
            result.extra_token = result.extra_tokens.front();
        }
        result.mtp_replayed_rows = round.consumed < round.rows() ? round.consumed : 0;
        progress.advance(round.consumed);
        if (committed_row) {
            // The final sampled ID is the unprocessed input of the next turn.
            prompt.insert(prompt.end(), round.tokens.begin(),
                          round.tokens.begin() + round.consumed);
        }
    }
};

}
