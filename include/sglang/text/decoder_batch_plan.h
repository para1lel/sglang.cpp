#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/runtime/execution/batch_admission_order.h"

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

StepPlan plan_budget_step(const BudgetLanes& work, const BatchAdmissionOrder& order,
                          int unused, bool metered_decode = true);

struct WideMtpCandidates {
    std::uint32_t speculative = 0;
    std::uint32_t eligible = 0;
    int eligible_count = 0;
    std::uint32_t replay = 0;
};

struct WideMtpGroups {
    std::array<std::uint8_t, decoder_batch_max_lanes> width{};
    int largest = 0;
    std::uint32_t replay = 0;
};

WideMtpGroups wide_mtp_groups(const PrefillRows& decode,
                              const WideMtpCandidates& candidates, int limit);

}
