#pragma once

#include "sglang/decoder_mtp.h"

#include <initializer_list>
#include <utility>

namespace sglang::decoder {

void check_draft_submission(cudaError_t direct, const char* operation);

struct RepairProjectionRow {
    struct Work {
        Bf16* attention{};
        Bf16* residual{};
        Bf16* concatenated{};
        Bf16* normalized{};
        TextFp8Activation activation{};
        Bf16* keys{};
        Bf16* values{};
    } work;

    void embed(int, const Bf16*, cudaStream_t);
};

template <typename State, typename Work>
class DraftProjectionBatch {
public:
    using Output = decltype(&Work::attention);
    using Projection = std::pair<Fp8Weight, Output>;

    DraftProjectionBatch(State* const* states, int count, Bf16* packed, cudaStream_t stream,
                         bool exact, bool streaming);

    void prepare_fc(const MtpDraftInput* input, const Bf16* embedding,
                    const Bf16* embedding_norm, const Bf16* hidden_norm, Bf16Weight fc);

    void project(std::initializer_list<Projection> stage) const;

    void rms_activation(Output input, const Bf16* weight, bool add_residual) const;

    void quantize_activation(Output input, Output up, int width) const;

    void append(const MtpDraftInput* rows, const Bf16* q_norm, const Bf16* k_norm,
                bool score) const;

    void read_attention(const MtpDraftInput* rows, bool online) const;

    void gate_attention() const;

    void normalize_output(const Bf16* weight) const;

private:
    template <int Width>
    void read_attention_packets(const MtpDraftInput* rows, bool online) const;

    void submit_joined_kv(const TextFp8DecodeTarget* targets, const TextFp8Projection* rows,
                          int first, int columns) const;

    Bf16* residual(int lane) const;

    State* const* states_;
    int count_;
    Bf16* packed_;
    Bf16* dense_residual_ = nullptr;
    cudaStream_t stream_;
    bool exact_;
    bool streaming_;
};

}
