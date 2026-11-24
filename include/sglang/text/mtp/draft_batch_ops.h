#pragma once

#include "sglang/decoder_mtp.h"
#include "sglang/decoder_verify.h"
#include "sglang/text/gqa_verify_batch.h"

namespace sglang::decoder {

struct DraftRmsRow {
    const Bf16* branch;
    const Bf16* residual;
    Bf16* output;
};

struct DraftRmsRows {
    DraftRmsRow rows[draft_projection_rows];
};

cudaError_t normalize_draft_inputs(const DraftRmsRows& embeddings, const DraftRmsRows& hidden,
                                   const Bf16* embedding_weight, const Bf16* hidden_weight,
                                   cudaStream_t stream);
cudaError_t normalize_draft_output(const DraftRmsRows& rows, const Bf16* weight,
                                   cudaStream_t stream);

cudaError_t draft_verify_preprocess_keys(const GqaVerifyRows<GqaVerifyPreprocessRow>& batch,
                                        int count, const Bf16* k_norm, cudaStream_t stream);

inline constexpr int draft_fc_input_values = draft_projection_rows * 2 * hidden_size;
// The FP8 pack reuses the former FC inputs while the FC residual remains live.
static_assert(draft_fc_input_values * sizeof(Bf16) >= draft_projection_rows * intermediate_size);
static_assert((draft_fc_input_values + draft_projection_rows * hidden_size) * sizeof(Bf16) <=
              draft_projection_rows * target_verify_max_rows * intermediate_size);

}
