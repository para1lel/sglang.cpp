#pragma once

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_verify.h"
#include "sglang/text/gqa/gqa_decode_long.cuh"

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

template <typename Row, int Capacity = target_split_eager_max_rows>
struct GqaVerifyRows {
    Row rows[Capacity];
};

template <int Capacity>
cudaError_t gqa_verify_preprocess(const GqaVerifyRows<GqaVerifyPreprocessRow, Capacity>& batch,
                                  int count, const __nv_bfloat16* q_norm,
                                  const __nv_bfloat16* k_norm, cudaStream_t stream);

template <int Capacity>
cudaError_t gqa_verify_append(const GqaVerifyRows<GqaVerifyAppendRow, Capacity>& batch,
                              int count, cudaStream_t stream);

template <int Capacity>
cudaError_t gqa_verify_postgate(const GqaVerifyRows<GqaVerifyGateRow, Capacity>& batch,
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
                                  row.rotary_position < 0 ? row.position : row.rotary_position};
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

inline bool try_gqa_verify_long(const GqaLayer& layer, const VerificationRow* rows, int count,
                                int cache_index, bool vector_attention, int split_end,
                                cudaStream_t stream) {
    using namespace gqa_decode;
    static_assert(verification_packet_rows == decoder_batch_max_mtp_lanes * long_rows);
    static_assert(max_wide_mtp_rows == long_requests * long_rows);
    if (!vector_attention || split_end != LongBatch::end || count < verification_packet_rows ||
        count > max_wide_mtp_rows || count % verification_packet_rows) {
        return false;
    }

    LongBatch read{};
    GqaVerifyRows<GqaVerifyPreprocessRow, max_wide_mtp_rows> preprocess{};
    GqaVerifyRows<GqaVerifyAppendRow, max_wide_mtp_rows> append{};
    GqaVerifyRows<GqaVerifyGateRow, max_wide_mtp_rows> gate{};
    constexpr int width = decoder_batch_max_mtp_lanes;
    for (int index = 0; index < count; ++index) {
        const int packet = index / verification_packet_rows;
        const int lane = index % width;
        const int depth = index % verification_packet_rows / width;
        const auto& row = rows[index];
        const auto& seed = rows[packet * verification_packet_rows + lane];
        if (row.request != seed.request || row.binding || row.constrained ||
            row.position != seed.position + depth || row.position < long_min_position ||
            row.position >= split_end ||
            row.position / split_chunk_tokens != seed.position / split_chunk_tokens) {
            return false;
        }
        auto& request = read.requests[packet * width + lane];
        auto& work = *row.work;
        if (depth == 0) {
            request.cache = row.request->caches[cache_index];
            request.pages = row.request->pages;
            request.first = row.position;
            for (int prior = 0; prior < packet * width + lane; ++prior) {
                if (read.requests[prior].pages.device_pages == request.pages.device_pages) {
                    return false;
                }
            }
        }
        request.rows[depth] = paged_gqa_buffers(work);
        const auto input = request.lane(depth);
        if (!readable(input) || !tensor_aligned(input) || !valid_buffers(input.buffers)) {
            return false;
        }
        const int rotary = row.rotary_position < 0 ? row.position : row.rotary_position;
        preprocess.rows[index] = {work.q_gate, work.keys, work.rope_queries, work.rope_keys,
                                  rotary};
        append.rows[index] = {input.cache, input.pages, work.rope_keys, work.values, row.position};
        gate.rows[index] = {work.q_gate, work.attention};
    }
    check(cudaPeekAtLastError(), "Prepare long grouped target attention");
    check(long_decode_status(gqa_verify_preprocess(preprocess, count, layer.query_norm,
                                                  layer.key_norm, stream)),
          "Normalize and rotate long grouped target Q/K");
    check(long_decode_status(gqa_verify_append(append, count, stream)),
          "Append long grouped target MXFP8 KV");
    if (const auto status = launch_long_decode(read, count / long_rows, stream)) {
        check(*status, "Read long grouped target MXFP8 attention");
    } else {
        std::array<PagedDecodeLane, verification_packet_rows> fallback;
        for (int first = 0; first < count / long_rows; first += width) {
            for (int depth = 0; depth < long_rows; ++depth) {
                for (int lane = 0; lane < width; ++lane) {
                    fallback[depth * width + lane] = read.requests[first + lane].lane(depth);
                }
            }
            check(long_decode_status(target_split_gqa(fallback.data(), fallback.size(),
                                                      split_end, stream)),
                  "Read grouped target MXFP8 attention");
        }
    }
    check(long_decode_status(gqa_verify_postgate(gate, count, stream)),
          "Gate long grouped target attention");
    return true;
}

}
