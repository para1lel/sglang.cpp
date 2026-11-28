#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/decoder_prefill.h"
#include "sglang/prefill_fp8.h"

#include <array>
#include <stdexcept>

namespace sglang::decoder::prefill_detail {

struct PrefillGroup {
    int grants;
    int rows;
};

enum class PrefillAttentionMode { native, batched, full_group, serial_segments, causal_segments };

struct PrefillSegment {
    int source;
    int offset;
};

int long_serial_visible(const PrefillGrant* grants, int count, bool grouped);
bool wants_resident_gqa_staging(int cached_tokens, int position, int rows);
int resident_gqa_staging_visible(const PrefillGrant* grants, int count);

void validate_prefill_tile(const PrefillGrant& grant, const PrefillWorkspace& workspace,
                           int capacity);
void validate_ranked_prefill_grants(std::array<const PrefillGrant*, 2> ranks, int count);

struct PartialProjectionSchedule {
    const void* state;
    void (*group)(const void*, const PrefillSegment*, const PrefillGrant*, PrefillGroup,
                  PrefillWorkspace&, PrefillAttentionMode, int);
    void (*native)(const void*, int, const PrefillGrant*, PrefillWorkspace&, int, int);
    bool (*prepare)(const void*, int) = nullptr;
};

bool execute_partial_projection_groups(const PrefillGrant* grants, int count, cudaStream_t stream,
                                       DecoderBatchStepStats& stats,
                                       const PartialProjectionSchedule& schedule,
                                       int min_rows = min_partial_prefill_rows,
                                       GqaGeometry staged_geometry = gqa_rank_geometry,
                                       int staged_visible = 0, bool resident_staging = false,
                                       bool whole_tiles = true, bool ranked = false);
}
