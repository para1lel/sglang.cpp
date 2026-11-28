#include "sglang/runtime/model/decoder_batch_lane.h"
#include "sglang/decoder_mtp.h"
#include <algorithm>
#include <utility>

namespace sglang::decoder {

bool Lane::checkpoints() const {
    return resident_prefix ||
           (options.publish_prefix_checkpoints && options.store_prefix != nullptr);
}

int Lane::remaining_prompt() const { return progress.remaining_prompt(prompt_size); }

int Lane::remaining_output() const { return progress.remaining_output(options); }

int Lane::capacity() const { return RequestProgress::capacity(prompt_size, options); }

int Lane::prefill_rows() const {
    if (!checkpoints())
        return remaining_prompt();
    constexpr int grant_rows = 8192;
    return std::min(remaining_prompt(), grant_rows - (progress.position % grant_rows));
}

int Lane::proposals() const {
    return std::min(options.mtp_steps, remaining_output() - 1);
}

ForwardLane Lane::input() const {
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

void Lane::record_decode_input(cudaStream_t stream) {
    prompt.push_back(progress.last_token);
    if (!draft || peer_draft || draft->valid_prefix() >= progress.position)
        return;
    MtpRound tail{};
    tail.draft = draft.get();
    tail.media = media ? &media_prefill : nullptr;
    auto input = tail.draft_input(progress.position - 1, progress.last_token,
                                  request->work.normalized);
    MtpDraftRequest::enqueue_wave(&input, 1, false, stream);
}

void Lane::finish_round(MtpRound& round, DecoderBatchStep& result, cudaStream_t stream,
                        SerialOutput* output, const MtpRound::Cancel& cancel, MtpDraftInput* repair,
                        const Workspace** committed_row) {
    result.mtp_drafted_tokens = round.drafted;
    result.mtp_verified_rows = round.rows();
    try {
        const bool scored = round.target->needs_generated_logprobs();
        const int stop = options.text_stop && (!cancel || !cancel())
            ? round.text_stop_after(options, progress) : 0;
        const bool completed = round.commit(
            [&](int token, const Bf16* logits, bool accepted, int row) {
                if (!result.token) {
                    result.token = token;
                    if (scored)
                        result.generated_logprobs = round.target->generated_logprobs(row);
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
            },
            stream, cancel, repair, committed_row);
        if (!completed) {
            result.finish_reason = DecoderFinishReason::cancelled;
        }
    } catch (const DecoderSamplingError&) {
        round.target->commit(round.consumed);
        result.sampling_failed = true;
        return;
    } catch (const DecoderConstraintError&) {
        round.target->commit(round.consumed);
        result.constraint_failed = true;
        return;
    }
    if (result.extra_token_count != 0) {
        result.extra_token = result.extra_tokens.front();
    }
    progress.advance(round.consumed);
    if (committed_row) {
        prompt.insert(prompt.end(), round.tokens.begin(),
                      round.tokens.begin() + round.consumed);
    }
}

PendingBatchPrefix::PendingBatchPrefix(PrefixDeviceTransfer&& view) : command(std::move(view)) {}

PendingBatchPrefix::~PendingBatchPrefix() {
    if (completion.valid()) completion.wait();
}

bool PendingBatchPrefix::ready() const {
    return completion.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

}
