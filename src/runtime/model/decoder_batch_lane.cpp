#include "sglang/runtime/model/decoder_batch_lane.h"
#include "sglang/decoder_mtp.h"
#include <algorithm>
#include <utility>

namespace sglang::decoder {

bool Lane::checkpoints() const {
    return options.publish_prefix_checkpoints && bool(options.store_prefix);
}

int Lane::remaining_prompt() const { return progress.remaining_prompt(prompt_size); }

int Lane::remaining_output() const { return progress.remaining_output(options); }

int Lane::capacity() const { return RequestProgress::capacity(prompt_size, options); }

int Lane::checkpoint_room() const {
    constexpr int interval = decoder_batch_prefix_checkpoint_interval;
    return interval - progress.position % interval;
}

int Lane::prefill_rows(bool checkpoints) const {
    return checkpoints ? std::min(remaining_prompt(), checkpoint_room()) : remaining_prompt();
}

int Lane::proposals(bool checkpoints) const {
    const int available = checkpoints ? std::min(remaining_output(), checkpoint_room())
                                      : remaining_output();
    return std::min(options.mtp_steps, available - 1);
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
    if (!draft || draft->valid_prefix() >= progress.position) return;
    MtpRound tail{};
    tail.draft = draft.get();
    tail.media = media ? &media_prefill : nullptr;
    auto input = tail.draft_input(progress.position - 1, progress.last_token,
                                  request->work.normalized);
    MtpDraftRequest::enqueue_wave(&input, 1, false, stream);
}

void Lane::finish_round(MtpRound& round, DecoderBatchStep& result, cudaStream_t stream,
                  SerialOutput* output, const MtpRound::Cancel& cancel,
                  MtpDraftInput* repair, const Workspace** committed_row,
                  TextGdnReplayBatch* gdn_batch) {
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
        }, stream, cancel, repair, committed_row, gdn_batch);
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
