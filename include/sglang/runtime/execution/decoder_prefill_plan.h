#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/decoder_prefill.h"
#include "sglang/prefill_fp8.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace sglang::decoder::prefill_detail {

struct PrefillGroup {
    int grants;
    int rows;
};

enum class PrefillAttentionMode { batched, full_group, serial_segments, causal_segments };

inline PrefillGroup partial_projection_group(const PrefillGrant* grants, int count) {
    PrefillGroup group{};
    while (group.grants < count) {
        const auto& grant = grants[group.grants];
        if (grant.rows < min_partial_prefill_rows || grant.rows > batch_prefill_tile_rows ||
            grant.rows != grant.tile_rows || grant.rows > grouped_prefill_max_rows - group.rows) {
            break;
        }
        ++group.grants;
        group.rows += grant.rows;
    }
    return group;
}

static_assert(grouped_prefill_max_rows % batch_prefill_tile_rows == 0);
inline constexpr PrefillGroup causal_full_group{
    grouped_prefill_max_rows / batch_prefill_tile_rows, grouped_prefill_max_rows};
inline constexpr PrefillGroup causal_three_tile_group{3, 3 * batch_prefill_tile_rows};

inline PrefillGroup causal_projection_group(int rows) {
    // Mixed tile sizes select different projection kernels and can change restored state.
    if (rows >= causal_full_group.rows) return causal_full_group;
    if (rows >= causal_three_tile_group.rows) return causal_three_tile_group;
    return {};
}

inline void validate_prefill_tile(const PrefillGrant& grant, const PrefillWorkspace& workspace,
                                  int capacity) {
    const auto rows = grant.rows;
    const auto position = grant.position;
    if (!workspace.stream() || !grant.request || !grant.tokens || rows < 1 || rows > capacity ||
        capacity > workspace.capacity() || position < 0 ||
        position > text_attention_max_tokens - rows ||
        (position + rows - 1) / mxfp8_page_size >= grant.request->pages.page_count ||
        (grant.draft && grant.prompt_remaining < rows)) {
        throw std::invalid_argument("Invalid decoder prefill tile");
    }
}

inline void record_projection_group(DecoderBatchStepStats& stats, int rows) {
    ++stats.layerwise_prefill_groups;
    stats.layerwise_prefill_rows += rows;
}

template <typename RunGroup, typename RunNative>
bool execute_partial_projection_groups(const PrefillGrant* grants, int count, cudaStream_t stream,
                                       DecoderBatchStepStats& stats, RunGroup run_group,
                                       RunNative run_native) {
    int serial_capacity = 1;
    int grouped_capacity = 0;
    for (int index = 0; index < count;) {
        const auto partial = partial_projection_group(grants + index, count - index);
        const int consumed = std::max(1, partial.grants);
        for (int lane = 0; lane < consumed; ++lane) {
            serial_capacity = std::max(serial_capacity, grants[index + lane].tile_rows);
        }
        if (partial.grants >= 2) grouped_capacity = std::max(grouped_capacity, partial.rows);
        if (grants[index].tile_rows == batch_prefill_tile_rows) {
            grouped_capacity = std::max(grouped_capacity,
                                        causal_projection_group(grants[index].rows).rows);
        }
        index += consumed;
    }
    if (!grouped_capacity) return false;

    PrefillWorkspace workspace(stream, std::max(serial_capacity, grouped_capacity));
    if (!workspace.data()) return false;
    stats.prefill_workspace_bytes = std::max(stats.prefill_workspace_bytes, workspace.bytes());
    for (int index = 0; index < count;) {
        const auto partial = partial_projection_group(grants + index, count - index);
        if (partial.grants >= 2) {
            run_group(grants + index, partial, workspace, PrefillAttentionMode::serial_segments,
                      serial_capacity);
            record_projection_group(stats, partial.rows);
            index += partial.grants;
            continue;
        }
        const auto& grant = grants[index++];
        int offset = 0;
        if (grant.tile_rows == batch_prefill_tile_rows) {
            std::array<PrefillGrant, causal_full_group.grants> segments;
            for (auto group = causal_projection_group(grant.rows); group.grants != 0;
                 group = causal_projection_group(grant.rows - offset)) {
                for (int lane = 0; lane < group.grants; ++lane) {
                    auto& segment = segments[lane];
                    segment = prefill_tile(grant, offset + lane * batch_prefill_tile_rows);
                    validate_prefill_tile(segment, workspace, serial_capacity);
                }
                // Each layer visits the request's cache and recurrent state in token order.
                run_group(segments.data(), group, workspace, PrefillAttentionMode::causal_segments,
                          serial_capacity);
                record_projection_group(stats, group.rows);
                offset += group.rows;
            }
        }
        if (offset < grant.rows) run_native(&grant, workspace, serial_capacity, offset);
    }
    workspace.close();
    return true;
}

}
