#pragma once

#include "sglang/decoder_prefill.h"

#include <algorithm>

namespace sglang::decoder {

inline bool gqa_cached(const PrefillGrant& grant) {
    return grant.position + grant.rows <= grant.request->cached_gqa_tokens;
}

template <typename Buffers>
void prepare_prefill_gqa(const GqaLayer& layer, const Buffers& work, Mxfp8KvCache cache,
                         Mxfp8PageTable pages, int position, int rows, int cached,
                         const MropePosition* mrope, cudaStream_t stream) {
    const int reused = std::clamp(cached - position, 0, rows);
    if (reused == 0) {
        const auto rotation = mrope
            ? mrope_gqa_preprocess(mrope, rows, work.q_gate, work.keys, layer.query_norm,
                                    layer.key_norm, work.rope_queries, work.rope_keys, stream)
            : text_gqa_preprocess(position, rows, work.q_gate, work.keys, layer.query_norm,
                                   layer.key_norm, work.rope_queries, work.rope_keys, stream);
        check(rotation, "Normalize and rotate prefill Q/K");
    } else {
        check(prefill_gqa_queries(position, mrope, rows, work.q_gate, layer.query_norm,
                                  work.rope_queries, stream), "Normalize and rotate prefill Q");
        if (reused == rows) return;
        const auto offset = std::size_t(reused) * text_gqa_kv_width;
        const auto rotation = mrope
            ? mrope_gqa_preprocess_keys(mrope + reused, rows - reused, work.keys + offset,
                                       layer.key_norm, work.rope_keys + offset, stream)
            : text_gqa_preprocess_keys(position + reused, rows - reused, work.keys + offset,
                                      layer.key_norm, work.rope_keys + offset, stream);
        check(rotation, "Normalize and rotate suffix prefill K");
    }
    const auto offset = std::size_t(reused) * text_gqa_kv_width;
    check(mxfp8_kv_append(cache, pages, position + reused, rows - reused,
                          work.rope_keys + offset, work.values + offset, stream),
          "Append prefill MXFP8 KV");
}

}
