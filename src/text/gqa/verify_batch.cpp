#include "sglang/text/gqa_verify_batch.h"

#include <algorithm>
#include <limits>

namespace sglang::decoder {

PagedDecodeBuffers paged_gqa_buffers(Workspace& work) {
    static_assert(vocab_size * sizeof(Bf16) >= paged_decode_scratch_bytes);
    return {work.rope_queries, work.attention, work.logits};
}

bool try_gqa_verify_batch(const GqaLayer& layer, const VerificationRow* rows, int count,
                                 int cache_index, bool vector_attention, int split_end,
                                 cudaStream_t stream) {
    if (!vector_attention || split_end != text_attention_max_tokens ||
        count <= 1 || count > max_wide_mtp_rows) {
        return false;
    }
    GqaVerifyRows<GqaVerifyPreprocessRow, max_wide_mtp_rows> preprocess{};
    GqaVerifyRows<GqaVerifyAppendRow, max_wide_mtp_rows> append{};
    GqaVerifyRows<GqaVerifyGateRow, max_wide_mtp_rows> gate{};
    std::array<PagedDecodeLane, max_wide_mtp_rows> reads;
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
    const auto& geometry_cache = rows[0].request->caches[cache_index];
    const GqaGeometry geometry{gqa_query_heads(geometry_cache.kv_heads),
                               geometry_cache.kv_heads};
    check(gqa_verify_preprocess(preprocess, count, layer.query_norm, layer.key_norm, stream,
                               geometry),
          "Normalize and rotate grouped target Q/K");
    check(gqa_verify_append(append, count, stream), "Append grouped target MXFP8 KV");
    for (int first = 0; first < count; first += target_split_eager_max_rows) {
        const int size = std::min(target_split_eager_max_rows, count - first);
        check(target_split_gqa(reads.data() + first, size, split_end, stream),
              "Read grouped target MXFP8 attention");
    }
    check(gqa_verify_postgate(gate, count, stream, geometry), "Gate grouped target attention");
    return true;
}

bool try_gqa_verify_short(const GqaLayer& layer, const VerificationRow* rows, int count,
                                 int cache_index, bool vector_attention, cudaStream_t stream) {
    if (!vector_attention || count <= 1 || count > target_split_eager_max_rows ||
        rows[0].binding) {
        return false;
    }
    const auto kv_heads = rows[0].request->caches[cache_index].kv_heads;
    std::array<PagedDecodeLane, target_split_eager_max_rows> reads;
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        if (row.constrained || row.binding ||
            row.request->caches[cache_index].kv_heads != kv_heads) {
            return false;
        }
        auto& work = *row.work;
        reads[index] = {row.request->caches[cache_index], row.request->pages, row.position,
                        paged_gqa_buffers(work)};
    }
    if (!decoder::short_verify_supported(reads.data(), count)) {
        return false;
    }
    GqaVerifyRows<GqaVerifyPreprocessRow> preprocess{};
    GqaVerifyRows<GqaVerifyAppendRow> append{};
    GqaVerifyRows<GqaVerifyGateRow> gate{};
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        auto& work = *row.work;
        preprocess.rows[index] = {work.q_gate, work.keys, work.rope_queries, work.rope_keys,
                                  row.rotary_position < 0 ? row.position : row.rotary_position};
        append.rows[index] = {row.request->caches[cache_index], row.request->pages, work.rope_keys,
                              work.values, row.position};
        gate.rows[index] = {work.q_gate, work.attention};
    }
    const GqaGeometry geometry{gqa_query_heads(kv_heads), kv_heads};
    check(gqa_verify_preprocess(preprocess, count, layer.query_norm, layer.key_norm, stream,
                               geometry),
          "Normalize and rotate grouped target Q/K");
    check(gqa_verify_append(append, count, stream), "Append grouped target MXFP8 KV");
    check(decoder::short_verify_gqa(reads.data(), count, stream),
          "Read grouped target MXFP8 attention");
    check(gqa_verify_postgate(gate, count, stream, geometry), "Gate grouped target attention");
    return true;
}

namespace {

struct LongVerifySequence {
    int indices[gqa_decode::long_rows];
    int count = 0;
};

int collect_long_verify_sequences(const VerificationRow* rows, int count,
    std::array<LongVerifySequence, gqa_decode::long_requests>& sequences) {
    int sequence_count = 0;
    int packet_first = 0;
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        if (row.binding) return 0;
        if (row.first) {
            if (index && !rows[index - 1].first) packet_first = sequence_count;
            if (sequence_count == gqa_decode::long_requests) return 0;
            for (int prior = 0; prior < sequence_count; ++prior) {
                if (rows[sequences[prior].indices[0]].request->pages.device_pages ==
                    row.request->pages.device_pages) return 0;
            }
            auto& sequence = sequences[sequence_count++];
            sequence.indices[sequence.count++] = index;
            continue;
        }
        int lane = packet_first;
        while (lane < sequence_count &&
               rows[sequences[lane].indices[0]].request != row.request) ++lane;
        if (lane == sequence_count) return 0;
        auto& sequence = sequences[lane];
        if (sequence.count == gqa_decode::long_rows ||
            row.position != rows[sequence.indices[0]].position + sequence.count) return 0;
        sequence.indices[sequence.count++] = index;
    }

    return sequence_count;
}

constexpr int rank_batch_min_position = 8191;

bool is_rank_long_verify_batch(const VerificationRow* rows, int count, int cache_index,
    const std::array<LongVerifySequence, gqa_decode::long_requests>& sequences,
    int sequence_count) {
    return count == wide_mtp_rows && sequence_count == wide_mtp_lanes &&
        std::all_of(sequences.begin(), sequences.begin() + sequence_count,
                    [&](const LongVerifySequence& sequence) {
        const auto& first = rows[sequence.indices[0]];
        return sequence.count == gqa_decode::long_rows &&
               first.position >= rank_batch_min_position &&
               gqa_decode::supports_eager_wide_split(
                   rows[sequence.indices[gqa_decode::long_rows - 1]].position) &&
               first.request->caches[cache_index].kv_heads == gqa_rank_geometry.kv_heads;
    });
}

}

int plan_gqa_verify_long(const VerificationRow* rows, int count, int cache_index,
                         gqa_decode::LongBatch& read, GqaVerifySelection& selected) {
    using namespace gqa_decode;
    static_assert(max_wide_mtp_rows == long_requests * long_rows);
    if (count < 1 || count > max_wide_mtp_rows)
        return 0;
    std::array<LongVerifySequence, long_requests> sequences;
    const int sequence_count = collect_long_verify_sequences(rows, count, sequences);
    const bool rank_batch = is_rank_long_verify_batch(rows, count, cache_index, sequences,
                                                     sequence_count);
    const int minimum_position = rank_batch ? rank_batch_min_position : long_min_position;

    int requests = 0;
    int selected_count = 0;
    for (int lane = 0; lane < sequence_count; ++lane) {
        const auto& sequence = sequences[lane];
        int begin = 0;
        while (begin < sequence.count &&
               rows[sequence.indices[begin]].position < minimum_position) ++begin;
        if (begin == sequence.count) continue;
        const auto& seed = rows[sequence.indices[begin]];
        auto& request = read.requests[requests];
        request.cache = seed.request->caches[cache_index];
        request.pages = seed.request->pages;
        request.first = seed.position;
        request.count = sequence.count - begin;
        bool eligible = true;
        for (int depth = 0; depth < request.count; ++depth) {
            const auto& row = rows[sequence.indices[begin + depth]];
            request.rows[depth] = paged_gqa_buffers(*row.work);
            const auto input = request.lane(depth);
            eligible &= !row.constrained && readable(input) && tensor_aligned(input) &&
                        valid_buffers(input.buffers);
        }
        if (!eligible) continue;
        ++requests;
        for (int i = begin; i < sequence.count; ++i) selected[sequence.indices[i]] = true;
        selected_count += request.count;
    }
    return rank_batch && selected_count != count ? 0 : requests;
}

bool try_gqa_verify_long(const GqaLayer& layer, const VerificationRow* rows, int count,
                         int cache_index, bool vector_attention, int split_end, cudaStream_t stream,
                         GqaVerifyRemainder remainder) {
    using namespace gqa_decode;
    if (!vector_attention || split_end != LongBatch::end)
        return false;
    LongBatch read;
    GqaVerifySelection selected{};
    const int requests = plan_gqa_verify_long(rows, count, cache_index, read, selected);
    if (requests == 0)
        return false;
    const auto selected_count = std::count(selected.begin(), selected.begin() + count, true);

    const auto& geometry_cache = rows[0].request->caches[cache_index];
    const GqaGeometry geometry{gqa_query_heads(geometry_cache.kv_heads),
                               geometry_cache.kv_heads};

    check(cudaPeekAtLastError(), "Prepare long grouped target attention");
    if (selected_count != count) remainder.attend(remainder.context, selected);

    GqaVerifyRows<GqaVerifyPreprocessRow, max_wide_mtp_rows> preprocess{};
    GqaVerifyRows<GqaVerifyAppendRow, max_wide_mtp_rows> append{};
    GqaVerifyRows<GqaVerifyGateRow, max_wide_mtp_rows> gate{};
    int next = 0;
    for (int index = 0; index < count; ++index) {
        if (!selected[index]) continue;
        const auto& row = rows[index];
        const auto& work = *row.work;
        preprocess.rows[next] = {work.q_gate, work.keys, work.rope_queries, work.rope_keys,
                                 row.rotary_position < 0 ? row.position : row.rotary_position};
        append.rows[next] = {row.request->caches[cache_index], row.request->pages, work.rope_keys,
                             work.values, row.position};
        gate.rows[next++] = {work.q_gate, work.attention};
    }
    check(long_decode_status(gqa_verify_preprocess(preprocess, next, layer.query_norm,
                                                  layer.key_norm, stream, geometry)),
          "Normalize and rotate long grouped target Q/K");
    check(long_decode_status(gqa_verify_append(append, next, stream)),
          "Append long grouped target MXFP8 KV");
    if (const auto status = launch_long_decode(read, requests, stream)) {
        check(*status, "Read long grouped target MXFP8 attention");
    } else {
        std::array<PagedDecodeLane, verification_packet_rows> fallback;
        for (int first = 0; first < count; first += verification_packet_rows) {
            int size = 0;
            for (int i = first; i < std::min(first + verification_packet_rows, count); ++i) {
                if (!selected[i]) continue;
                const auto& row = rows[i];
                fallback[size++] = {row.request->caches[cache_index], row.request->pages,
                                    row.position, paged_gqa_buffers(*row.work)};
            }
            if (!size) continue;
            check(long_decode_status(target_split_gqa(fallback.data(), size, split_end, stream)),
                  "Read grouped target MXFP8 attention");
        }
    }
    check(long_decode_status(gqa_verify_postgate(gate, next, stream, geometry)),
          "Gate long grouped target attention");
    return true;
}
}
