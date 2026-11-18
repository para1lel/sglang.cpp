#pragma once

#include "sglang/decoder.h"

#include <array>

namespace sglang {

inline constexpr int decoder_batch_max_lanes = 4;

struct DecoderBatchStep {
    std::optional<int> token;
    std::optional<DecoderFinishReason> finish_reason;
    bool decode = false;
};

/// The Decoder must outlive the batch. Use one host thread to operate a batch.
/// Each slot owns its request state until cancel() or a finishing step.
class DecoderBatch {
public:
    explicit DecoderBatch(const Decoder& decoder);
    ~DecoderBatch();

    void admit(int slot, std::vector<int> prompt, const DecoderOptions& options);
    void cancel(int slot);

    /// Advance each selected slot by one model position. A zero-budget slot
    /// finishes without a model position. A CUDA failure disables the batch.
    std::array<DecoderBatchStep, decoder_batch_max_lanes>
    advance(const std::array<bool, decoder_batch_max_lanes>& selected);

    std::size_t request_device_allocation_bytes() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}
