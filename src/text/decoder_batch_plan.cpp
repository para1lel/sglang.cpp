#include "sglang/text/decoder_batch_plan.h"

#include "sglang/runtime/backend/cuda_backend_policy.h"

#include <algorithm>

namespace sglang::decoder {

StepPlan plan_budget_step(const BudgetLanes& work, const BatchAdmissionOrder& order,
                                 int unused, bool metered_decode) {
    StepPlan plan;
    for (const int slot : order) {
        if (!work[slot].decode) continue;
        const int reserve = metered_decode ? 1 + work[slot].proposals : 1;
        plan.decode[slot] = std::min(unused, reserve);
        unused -= plan.decode[slot];
        if (!unused) return plan;
    }
    for (const int slot : order) {
        plan.prefill[slot] = std::min(unused, work[slot].prefill);
        unused -= plan.prefill[slot];
        if (!unused) break;
    }
    return plan;
}

namespace {

void group_replay_packets(const PrefillRows& decode, const WideMtpCandidates& candidates,
                          WideMtpGroups& groups) {
    if (candidates.replay == 0)
        return;
    std::array<int, decoder_batch_max_lanes> slots{};
    int count = 0;
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (decode[slot] != 0 &&
            ((candidates.speculative >> static_cast<unsigned>(slot)) & 1U) != 0) {
            slots[count++] = slot;
        }
    }
    int pending = 0;
    int start = 0;
    const auto flush = [&] {
        if (pending > decoder_batch_max_mtp_lanes) {
            groups.width[start] = pending;
            groups.replay |= std::uint32_t{1} << static_cast<unsigned>(start);
        }
        pending = 0;
    };
    for (int first = 0; first < count;) {
        const int slot = slots[first];
        const int packed = groups.width[slot];
        const int width =
            packed != 0 ? packed : std::min(count - first, decoder_batch_max_mtp_lanes);
        const bool replay =
            packed == 0 &&
            std::all_of(slots.begin() + first, slots.begin() + first + width, [&](int lane) {
                return ((candidates.replay >> static_cast<unsigned>(lane)) & 1U) != 0;
            });
        if (replay) {
            if (pending == 0)
                start = slot;
            pending += width;
        } else {
            flush();
        }
        first += width;
    }
    flush();
}

WideMtpGroups packed_mtp_groups(const PrefillRows& decode, const WideMtpCandidates& candidates,
                                int limit) {
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

WideMtpGroups wide_mtp_groups(const PrefillRows& decode, const WideMtpCandidates& candidates,
                              int limit) {
    auto groups = packed_mtp_groups(decode, candidates, limit);
    group_replay_packets(decode, candidates, groups);
    return groups;
}

}
