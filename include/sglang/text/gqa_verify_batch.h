#pragma once

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_verify.h"
#include "sglang/text/gqa/gqa_decode_long.cuh"

#include <array>
#include <cstdint>

namespace sglang::decoder {

struct GqaVerifyPreprocessRow {
    const __nv_bfloat16* q_gate;
    const __nv_bfloat16* keys;
    __nv_bfloat16* rope_queries;
    __nv_bfloat16* rope_keys;
    int position;
};

struct GqaVerifyAppendRow {
    Mxfp8KvCache cache;
    Mxfp8PageTable pages;
    const __nv_bfloat16* keys;
    const __nv_bfloat16* values;
    int position;
};

struct GqaVerifyGateRow {
    const __nv_bfloat16* q_gate;
    __nv_bfloat16* attention;
};

template <typename Row, int Capacity = target_split_eager_max_rows>
struct GqaVerifyRows {
    Row rows[Capacity];
};

template <int Capacity>
cudaError_t gqa_verify_preprocess(const GqaVerifyRows<GqaVerifyPreprocessRow, Capacity>& batch,
                                  int count, const __nv_bfloat16* q_norm,
                                  const __nv_bfloat16* k_norm, cudaStream_t stream,
                                  GqaGeometry geometry = gqa_full_geometry);

template <int Capacity>
cudaError_t gqa_verify_append(const GqaVerifyRows<GqaVerifyAppendRow, Capacity>& batch,
                              int count, cudaStream_t stream);

template <int Capacity>
cudaError_t gqa_verify_postgate(const GqaVerifyRows<GqaVerifyGateRow, Capacity>& batch,
                                int count, cudaStream_t stream,
                                GqaGeometry geometry = gqa_full_geometry);

bool cache_covers(Mxfp8KvCache cache, Mxfp8PageTable pages, std::int64_t token_count);
PagedDecodeBuffers paged_gqa_buffers(Workspace& work);

bool try_gqa_verify_batch(const GqaLayer& layer, const VerificationRow* rows, int count,
                          int cache_index, bool vector_attention, int split_end,
                          cudaStream_t stream);

bool try_gqa_verify_short(const GqaLayer& layer, const VerificationRow* rows, int count,
                          int cache_index, bool vector_attention, cudaStream_t stream);

using GqaVerifySelection = std::array<bool, max_wide_mtp_rows>;

struct GqaVerifyRemainder {
    const void* context;
    void (*attend)(const void*, const GqaVerifySelection&);
};

bool try_gqa_verify_long(const GqaLayer& layer, const VerificationRow* rows, int count,
                         int cache_index, bool vector_attention, int split_end,
                         cudaStream_t stream, GqaVerifyRemainder remainder);

}
