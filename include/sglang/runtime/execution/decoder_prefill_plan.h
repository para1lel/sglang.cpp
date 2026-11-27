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

PrefillGroup partial_projection_group(const PrefillGrant* grants, int count,
                                      int min_rows = min_partial_prefill_rows);

static_assert(grouped_prefill_max_rows % batch_prefill_tile_rows == 0);
inline constexpr PrefillGroup causal_full_group{
    grouped_prefill_max_rows / batch_prefill_tile_rows, grouped_prefill_max_rows};
inline constexpr PrefillGroup causal_three_tile_group{3, 3 * batch_prefill_tile_rows};

PrefillGroup causal_projection_group(int rows);

int long_serial_visible(const PrefillGrant* grants, int count, bool grouped);

void validate_prefill_tile(const PrefillGrant& grant, const PrefillWorkspace& workspace,
                           int capacity);

inline constexpr int paired_tile_grant_rows = 2 * batch_prefill_tile_rows;
inline constexpr PrefillGroup paired_tile_group{4, 2 * paired_tile_grant_rows};
static_assert(paired_tile_group.rows == grouped_prefill_max_rows);

bool can_pack_paired_tiles(const PrefillGrant* grants, int count);

std::array<PrefillGrant, 4> paired_tile_segments(const PrefillGrant* grants,
                                                 const PrefillWorkspace& workspace);

void record_projection_group(DecoderBatchStepStats& stats, int rows);
void peer_causal_segments(const PrefillGrant* peer_grants, int source, int offset,
                          PrefillGrant* segments, int lanes);

struct PartialProjectionSchedule {
    const void* state;
    void (*group)(const void*, int, int, const PrefillGrant*, PrefillGroup, PrefillWorkspace&,
                  PrefillAttentionMode, int);
    void (*native)(const void*, int, const PrefillGrant*, PrefillWorkspace&, int, int);
};

bool execute_partial_projection_groups(const PrefillGrant* grants, int count, cudaStream_t stream,
                                       DecoderBatchStepStats& stats,
                                       const PartialProjectionSchedule& schedule,
                                       int min_rows = min_partial_prefill_rows,
                                       int staged_visible = 0);

}
