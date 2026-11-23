#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/runtime/backend/cuda_backend_policy.h"

#include <algorithm>

namespace sglang::decoder {

using PrefillRows = std::array<int, decoder_batch_max_lanes>;

struct StepPlan {
    PrefillRows prefill{};
    PrefillRows decode{};
};

struct BudgetLane {
    int prefill = 0;
    int proposals = 0;
    bool decode = false;
};

using BudgetLanes = std::array<BudgetLane, decoder_batch_max_lanes>;

inline std::optional<StepPlan> plan_tiled_step(const BudgetLanes& work, int lanes, int unused,
                                             int& cursor, int tile) {
    StepPlan plan;
    int full_tiles = 0;
    for (int slot = 0; slot < lanes; ++slot) {
        plan.decode[slot] = work[slot].decode;
        unused -= plan.decode[slot];
        full_tiles += work[slot].prefill >= tile;
    }
    if (unused < tile || full_tiles <= unused / tile) {
        return std::nullopt;
    }
    const int start = cursor;
    int first_deferred = -1;
    for (int offset = 0; offset < lanes; ++offset) {
        const int slot = (start + offset) % lanes;
        const int rows = std::min(tile, work[slot].prefill);
        if (rows <= unused) {
            plan.prefill[slot] = rows;
            unused -= rows;
        } else if (first_deferred < 0) {
            first_deferred = slot;
        }
    }
    cursor = first_deferred;
    return plan;
}

inline StepPlan plan_budget_step(const BudgetLanes& work, int lanes, int unused, int& cursor,
                                 int tile) {
    if (tile) {
        if (const auto tiled = plan_tiled_step(work, lanes, unused, cursor, tile)) {
            return *tiled;
        }
    }
    StepPlan plan;
    const int start = cursor;
    for (int offset = 0; offset < lanes && unused > 0; ++offset) {
        const int slot = (start + offset) % lanes;
        if (work[slot].decode) {
            plan.decode[slot] = 1;
        } else if (work[slot].prefill) {
            plan.prefill[slot] = 1;
        } else {
            continue;
        }
        --unused;
        cursor = (slot + 1) % lanes;
    }
    if (unused > 0) {
        cursor = (start + 1) % lanes;
    }
    for (int offset = 0; offset < lanes && unused > 0; ++offset) {
        const int slot = (start + offset) % lanes;
        if (plan.decode[slot]) {
            const int extra = std::min(unused, work[slot].proposals);
            plan.decode[slot] += extra;
            unused -= extra;
        }
    }
    while (unused > 0) {
        int unfinished = 0;
        for (int slot = 0; slot < lanes; ++slot) {
            unfinished += plan.prefill[slot] < work[slot].prefill;
        }
        if (unfinished == 0) {
            break;
        }
        const int share = std::max(1, unused / unfinished);
        for (int offset = 0; offset < lanes && unused > 0; ++offset) {
            const int slot = (start + offset) % lanes;
            const int extra = std::min({unused, share, work[slot].prefill - plan.prefill[slot]});
            plan.prefill[slot] += extra;
            unused -= extra;
        }
    }
    return plan;
}

struct WideMtpCandidates {
    std::uint32_t speculative = 0;
    std::uint32_t eligible = 0;
    int eligible_count = 0;
};

struct WideMtpGroups {
    std::array<std::uint8_t, decoder_batch_max_lanes> width{};
    int largest = 0;
};

inline WideMtpGroups wide_mtp_groups(const PrefillRows& decode,
                                     const WideMtpCandidates& candidates, int limit) {
    WideMtpGroups groups;
    if (limit < wide_mtp_lanes || candidates.eligible_count < wide_mtp_lanes) return groups;
    std::array<int, max_wide_mtp_lanes / wide_mtp_lanes> starts;
    int pending = 0;
    const auto flush = [&] {
        for (int index = 0; index < pending;) {
            int width = limit;
            while (width > (pending - index) * wide_mtp_lanes) width /= 2;
            groups.width[starts[index]] = width;
            groups.largest = std::max(groups.largest, width);
            index += width / wide_mtp_lanes;
        }
        pending = 0;
    };
    int first = 0;
    int count = 0;
    bool eligible = true;
    // Keep four-lane packet boundaries and combine only consecutive eligible M40 groups.
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        const auto bit = std::uint32_t{1} << slot;
        if (!(candidates.speculative & bit) || !decode[slot]) continue;
        if (!count) first = slot;
        eligible &= decode[slot] == decoder_batch_max_mtp_steps + 1 && (candidates.eligible & bit);
        ++count;
        if (!eligible && count % decoder_batch_max_mtp_lanes == 0) {
            flush();
        } else if (count == wide_mtp_lanes) {
            starts[pending++] = first;
            if (pending * wide_mtp_lanes == limit) flush();
        } else {
            continue;
        }
        count = 0;
        eligible = true;
    }
    flush();
    return groups;
}

}
