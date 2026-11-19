#pragma once

#include "sglang/decoder.h"

#include <array>
#include <cstdint>

namespace sglang {

inline constexpr int decoder_batch_max_lanes = 32;
inline constexpr int decoder_batch_max_graph_lanes = 4;
inline constexpr int decoder_batch_max_mtp_lanes = 4;
inline constexpr int decoder_batch_max_mtp_steps = decoder_max_mtp_steps;

struct DecoderBatchStep {
    std::optional<int> token;
    std::optional<DecoderFinishReason> finish_reason;
    bool decode = false;
    std::optional<int> extra_token = std::nullopt;
    int mtp_drafted_tokens = 0;
    int mtp_accepted_tokens = 0;
    int mtp_replayed_rows = 0;
    bool mtp_boundary = false;
    std::array<int, decoder_batch_max_mtp_steps> extra_tokens{};
    int extra_token_count = 0;
    bool sampling_failed = false;
};

struct DecoderBatchStepStats {
    std::array<int, decoder_batch_max_lanes> prefill_rows{};
    std::size_t prefill_workspace_bytes = 0;
    std::uint32_t graph_slots = 0;
    std::size_t peak_request_bytes = 0;
    int verification_lanes = 0;
    int verification_rows = 0;
    std::size_t sampling_workspace_bytes = 0;
};

struct DecoderBatchOptions {
    bool cuda_graphs = false;
    int max_lanes = decoder_batch_max_mtp_lanes;
};

/// The Decoder must outlive the batch. Use one host thread to operate a batch.
/// Each slot owns its request state until cancel() or a finishing step.
class DecoderBatch {
public:
    explicit DecoderBatch(const Decoder& decoder);
    DecoderBatch(const Decoder& decoder, DecoderBatchOptions options);
    ~DecoderBatch();

    void admit(int slot, std::vector<int> prompt, const DecoderOptions& options);
    void cancel(int slot);

    /// Advance each selected slot by one position or one speculative round.
    /// Consume token and the first extra_token_count entries of extra_tokens in order.
    /// extra_token mirrors the first extra token for callers using depth one.
    /// A zero-budget slot finishes without work. A sampling_failed step retires its slot.
    /// On a failed advance, all slots are retired
    /// if the stream can be drained. A stream that cannot be drained disables the batch.
    std::array<DecoderBatchStep, decoder_batch_max_lanes>
    advance(const std::array<bool, decoder_batch_max_lanes>& selected);

    /// Share one bounded prompt position budget across selected slots. Each ready
    /// decode slot advances by one position or speculative round.
    std::array<DecoderBatchStep, decoder_batch_max_lanes>
    advance_chunked(const std::array<bool, decoder_batch_max_lanes>& selected,
                    DecoderBatchStepStats* stats = nullptr);

    std::size_t request_device_allocation_bytes() const;
    bool available() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}
