#pragma once

#include <cstdio>

namespace sglang::decoder {

inline TextGdnReplayRows replay_rows(Bf16* storage, int count) {
    const int qkv = count * text_gdn_qkv_width;
    const int gates = count * text_gdn_value_heads;
    return {storage, storage + qkv, storage + 2 * qkv, storage + 2 * qkv + gates};
}

inline TextGdnReplayRows replay_rows_local(Bf16* storage, int count) {
    const int qkv = count * GdnLocal::qkv_width;
    const int gates = count * GdnLocal::value_heads;
    return {storage, storage + qkv, storage + 2 * qkv, storage + 2 * qkv + gates};
}

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

