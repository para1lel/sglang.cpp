#pragma once

// Single-include implementation detail of src/decoder_verify.cpp: the
// verification workspace construction and rank-local relayout live here so
// decoder_verify.cpp stays under the canonical file-size limit. Include only
// from src/decoder_verify.cpp; the include site provides the full type
// context (sglang/decoder_verify.h and sglang/text/gdn_local.h above it).

#include <cstdio>

namespace sglang::decoder {

inline TextGdnReplayRows replay_rows(Bf16* storage, int count) {
    const int qkv = count * text_gdn_qkv_width;
    const int gates = count * text_gdn_value_heads;
    return {storage, storage + qkv, storage + 2 * qkv, storage + 2 * qkv + gates};
}

// Rank-local twin of replay_rows: same packing at GdnLocal widths.
inline TextGdnReplayRows replay_rows_local(Bf16* storage, int count) {
    const int qkv = count * GdnLocal::qkv_width;
    const int gates = count * GdnLocal::value_heads;
    return {storage, storage + qkv, storage + 2 * qkv, storage + 2 * qkv + gates};
}

// Re-lays out a full-width verification workspace with rank-local GDN
// strides (TP2): replay rows pack at GdnLocal qkv/gate widths and the z /
// attention rows stride at GdnLocal::value_width. Allocations stay
// full-width (safe over-provision); only the strides change, so the TP1
// layout and the k1 graph scratch stay byte-identical.
inline void relayout_verification_workspace_local(VerificationWorkspace& scratch, int rows) {
    const int full_layer_elements = 2 * rows * (text_gdn_qkv_width + text_gdn_value_heads);
    auto* const base = scratch.replay[0].inputs.qkv;
    for (int layer = 0; layer < gdn_layers; ++layer) {
        scratch.replay[layer].inputs =
            replay_rows_local(base + layer * full_layer_elements, rows);
    }
    auto* const z = scratch.work[0].gdn_z;
    auto* const attention = scratch.work[0].attention;
    for (int index = 0; index < rows; ++index) {
        scratch.work[index].gdn_z = z + index * GdnLocal::value_width;
        scratch.work[index].attention = attention + index * GdnLocal::value_width;
    }
}

inline void free_pinned(void* results) {
    const auto status = cudaFreeHost(results);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "sglang decoder: Free verifier pinned storage: %s\n",
                     cudaGetErrorString(status));
    }
}

}

