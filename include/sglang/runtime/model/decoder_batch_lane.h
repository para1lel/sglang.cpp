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
    std::unique_ptr<Request> peer_request = nullptr;
    std::unique_ptr<TargetVerification> peer_verification = nullptr;
    std::unique_ptr<PendingBatchPrefix> peer_prefix = nullptr;
    std::optional<SamplingParameters> sampling = std::nullopt;
    std::optional<MtpConstraintRound> constraint_round{};
    bool checkpoints() const;
    const Workspace* peer_checkpoint_row = nullptr;
    bool restored_prompt = false;
    std::shared_ptr<const MultimodalInput> media = nullptr;
    DecoderPromptLogprobs prompt_logprobs = {};
    MediaPrefill media_prefill{};
    std::size_t resident_bytes = 0;
    std::unique_ptr<PendingBatchPrefix> prefix = nullptr;
    std::optional<DeferredPrefixStore> prefix_store = std::nullopt;

    int remaining_prompt() const;
    int remaining_output() const;
    int capacity() const;
    int checkpoint_room() const;
    int prefill_rows(bool checkpoints) const;
    int proposals(bool checkpoints = false) const;

    ForwardLane input() const;

    void record_decode_input(cudaStream_t stream);

    void finish_round(MtpRound& round, DecoderBatchStep& result, cudaStream_t stream,
                      SerialOutput* output, const MtpRound::Cancel& cancel,
                      MtpDraftInput* repair, const Workspace** committed_row,
                      TextGdnReplayBatch* gdn_batch = nullptr);
};

}
