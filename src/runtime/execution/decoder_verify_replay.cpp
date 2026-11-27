#include "sglang/runtime/execution/decoder_verify_workspace.h"
#include "sglang/decoder_verify.h"
#include "sglang/text/gdn_local.h"
#include <cstdio>
#include <stdexcept>

namespace sglang::decoder {

TextGdnReplayRows replay_rows(Bf16* storage, int count) {
    const int qkv = count * text_gdn_qkv_width;
    const int gates = count * text_gdn_value_heads;
    return {storage, storage + qkv, storage + 2 * qkv, storage + 2 * qkv + gates};
}

TextGdnReplayRows replay_rows_local(Bf16* storage, int count) {
    const int qkv = count * GdnLocal::qkv_width;
    const int gates = count * GdnLocal::value_heads;
    return {storage, storage + qkv, storage + 2 * qkv, storage + 2 * qkv + gates};
}

void relayout_verification_workspace_local(VerificationWorkspace& scratch, int rows) {
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

void free_pinned(void* results) {
    const auto status = cudaFreeHost(results);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "sglang decoder: Free verifier pinned storage: %s\n",
                     cudaGetErrorString(status));
    }
}

namespace {

template <typename Source>
bool try_fill_gdn_replay_batch(TextGdnReplayBatch* batch, int count, int layers,
                               Source&& source) {
    if (batch == nullptr || batch->available() < layers) return false;
    for (int layer = 0; layer < layers; ++layer) {
        if (!text_gdn_replay_batchable(count, source(layer))) return false;
    }
    for (int layer = 0; layer < layers; ++layer) {
        if (!batch->add(source(layer))) {
            throw std::logic_error("Collect batched GDN prefix replay");
        }
    }
    return true;
}

}

void TargetVerification::commit(int count, TextGdnReplayBatch* gdn_batch) {
    if (!ready_ || count < 0 || count > pending_rows_) {
        throw std::logic_error("Invalid target verification commit");
    }
    if (count > 0 && count < pending_rows_) {
        const bool batched = try_fill_gdn_replay_batch(
            gdn_batch, count, gdn_layers, [&](int layer) {
                const auto& saved = workspace().replay[layer];
                return TextGdnReplayEntry{saved.inputs, saved.weights, request_.states[layer],
                                          tail_[layer], count};
            });
        if (!batched) {
            for (int layer = 0; layer < gdn_layers; ++layer) {
                const auto& saved = workspace().replay[layer];
                const auto status = rank_local_gdn_
                    ? text_gdn_replay_prefix_local(count, saved.inputs, saved.weights,
                                                   request_.states[layer], tail_[layer],
                                                   stream_)
                    : text_gdn_replay_prefix(count, saved.inputs, saved.weights,
                                             request_.states[layer], tail_[layer], stream_);
                check(status, "Replay committed target GDN prefix");
            }
        }
    }
    if (count > 0) {
        request_.states.swap(tail_);
        valid_prefix_ += count;
    }
    ready_ = false;
    pending_rows_ = 0;
    active_workspace_ = nullptr;
}

}
