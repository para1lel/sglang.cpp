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

WideMtpGroups wide_mtp_groups(const PrefillRows& decode,
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
