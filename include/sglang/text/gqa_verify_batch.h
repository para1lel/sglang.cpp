#pragma once

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_verify.h"

#include <array>
#include <cstdint>
#include <limits>

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

template <typename Row>
struct GqaVerifyRows {
    Row rows[target_split_eager_max_rows];
};

cudaError_t gqa_verify_preprocess(const GqaVerifyRows<GqaVerifyPreprocessRow>& batch,
                                  int count, const __nv_bfloat16* q_norm,
                                  const __nv_bfloat16* k_norm, cudaStream_t stream);

cudaError_t gqa_verify_append(const GqaVerifyRows<GqaVerifyAppendRow>& batch,
                              int count, cudaStream_t stream);

cudaError_t gqa_verify_postgate(const GqaVerifyRows<GqaVerifyGateRow>& batch,
                                int count, cudaStream_t stream);

inline bool cache_covers(Mxfp8KvCache cache, Mxfp8PageTable pages, std::int64_t token_count) {
    if (token_count <= 0 || token_count > std::numeric_limits<int>::max()) {
        return false;
    }
    const auto last_page = (token_count - 1) / mxfp8_page_size;
    return last_page < cache.page_count && last_page < pages.page_count && pages.device_pages
        && cache.key_payload && cache.value_payload && cache.key_scales && cache.value_scales;
}

inline PagedDecodeBuffers paged_gqa_buffers(Workspace& work) {
    static_assert(vocab_size * sizeof(Bf16) >= paged_decode_scratch_bytes);
    return {work.rope_queries, work.attention, work.logits};
}

inline bool try_gqa_verify_batch(const GqaLayer& layer, const VerificationRow* rows, int count,
                                 int cache_index, bool vector_attention, int split_end,
                                 cudaStream_t stream) {
    if (!vector_attention || split_end != text_attention_max_tokens ||
        count <= 1 || count > target_split_eager_max_rows) {
        return false;
    }
    GqaVerifyRows<GqaVerifyPreprocessRow> preprocess{};
    GqaVerifyRows<GqaVerifyAppendRow> append{};
    GqaVerifyRows<GqaVerifyGateRow> gate{};
    std::array<PagedDecodeLane, target_split_eager_max_rows> reads;
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        if (!target_split_position(row.position, split_end)) {
            return false;
        }
        auto& work = *row.work;
        const auto cache = row.request->caches[cache_index];
        const auto pages = row.request->pages;
        preprocess.rows[index] = {work.q_gate, work.keys, work.rope_queries, work.rope_keys,
                                  row.position};
        append.rows[index] = {cache, pages, work.rope_keys, work.values, row.position};
        reads[index] = {cache, pages, row.position, paged_gqa_buffers(work)};
        gate.rows[index] = {work.q_gate, work.attention};
    }
    check(gqa_verify_preprocess(preprocess, count, layer.query_norm, layer.key_norm, stream),
          "Normalize and rotate grouped target Q/K");
    check(gqa_verify_append(append, count, stream), "Append grouped target MXFP8 KV");
    check(target_split_gqa(reads.data(), count, split_end, stream),
          "Read grouped target MXFP8 attention");
    check(gqa_verify_postgate(gate, count, stream), "Gate grouped target attention");
    return true;
}

}
