#pragma once

#include "sglang/decoder_mtp.h"

#include <initializer_list>
#include <utility>

namespace sglang::decoder {

void check_draft_submission(cudaError_t direct, const char* operation);

struct DraftFcWeights {
    const Bf16* embedding = nullptr;
    const Bf16* embedding_norm = nullptr;
    const Bf16* hidden_norm = nullptr;
    Bf16Weight projection;
};

template <typename State, typename Work>
class DraftProjectionBatch {
public:
    using Output = decltype(&Work::attention);
    using Projection = std::pair<Fp8Weight, Output>;

    DraftProjectionBatch(State* const* states, int count, Bf16* packed, cudaStream_t stream,
                         bool exact, bool streaming);

    void prepare_fc(const MtpDraftInput* input, const DraftFcWeights& weights,
                    Bf16* output = nullptr);
    void bind_partials(TpPartials partials);

    void project(std::initializer_list<Projection> stage) const;

    void rms_activation(Output input, const Bf16* weight, bool add_residual) const;

    void quantize_activation(Output input, Output up, int width) const;

    void prepare_attention(const MtpDraftInput* rows, int count, const GqaLayer& gqa, bool score,
                           GqaGeometry geometry, bool grouped_rope) const;

    void read_attention(const MtpDraftInput* rows, bool online) const;

    void gate_attention(GqaGeometry geometry = gqa_full_geometry) const;

    void normalize_output(const Bf16* weight) const;

private:
  void append(const MtpDraftInput* rows, int count, const Bf16* q_norm, const Bf16* k_norm,
              bool score, GqaGeometry geometry) const;
  template <int Width> void read_attention_packets(const MtpDraftInput* rows, bool online) const;

  void submit_joined_kv(const TextFp8DecodeTarget* targets, const TextFp8Projection* rows,
                        int first, int columns) const;

  Bf16* output(Output member, int lane) const;

  State* const* states_;
  int count_;
  Bf16* packed_;
  Bf16* dense_residual_ = nullptr;
  Bf16* dense_branch_ = nullptr;
  cudaStream_t stream_;
  bool exact_;
  bool streaming_;
};

}
