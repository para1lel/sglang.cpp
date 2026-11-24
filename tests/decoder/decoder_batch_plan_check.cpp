#include "sglang/text/decoder_batch_plan.h"

#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

using sglang::BatchAdmissionOrder;
using sglang::decoder::BudgetLanes;
using sglang::decoder::PrefillRows;
using sglang::decoder::plan_budget_step;

void check(bool condition, const char* detail) {
    if (!condition) {
        throw std::runtime_error(detail);
    }
}

int total(const PrefillRows& rows) {
    return std::accumulate(rows.begin(), rows.end(), 0);
}

void check_decode_first() {
    BatchAdmissionOrder order(8);
    for (const int slot : {7, 0, 6, 1}) order.admitted(slot);
    BudgetLanes work{};
    work[7].prefill = 73;
    work[0].prefill = 8192;
    work[6] = {0, 4, true};
    work[1] = {0, 2, true};

    const auto plan = plan_budget_step(work, order, 128);
    check(plan.decode[6] == 5 && plan.decode[1] == 3, "Decode must reserve legal proposal rows");
    check(plan.prefill[7] == 73 && plan.prefill[0] == 47, "Give the head its full legal tail");
    check(total(plan.prefill) + total(plan.decode) == 128, "Use the entire step budget");

    const auto tight = plan_budget_step(work, order, 6);
    check(tight.decode[6] == 5 && tight.decode[1] == 1 && total(tight.prefill) == 0,
          "Clip the newer decode grant before allowing prefill");
    work[1] = {0, 0, true};
    const auto exhausted = plan_budget_step(work, order, 4);
    check(exhausted.decode[6] == 4 && exhausted.decode[1] == 0 &&
              total(exhausted.prefill) == 0,
          "The oldest running lane can use the only affordable rows");
    work[1] = {0, 2, true};
    const auto unmetered = plan_budget_step(work, order, 128, false);
    check(unmetered.decode[6] == 1 && unmetered.decode[1] == 1 &&
              unmetered.prefill[7] == 73 && unmetered.prefill[0] == 53,
          "Preserve the unmetered decode reservation");
}

void check_admission_order() {
    BatchAdmissionOrder order(8);
    order.admitted(0);
    order.admitted(7);
    order.admitted(0);
    BudgetLanes work{};
    work[0].prefill = 8192;
    work[7].prefill = 8192;

    const auto head = plan_budget_step(work, order, 8192);
    check(head.prefill[7] == 8192 && head.prefill[0] == 0, "A reused slot joins the queue tail");
    work[7] = {};
    const auto after_cancel = plan_budget_step(work, order, 8192);
    check(after_cancel.prefill[0] == 8192 && after_cancel.prefill[7] == 0,
          "An inactive slot cannot reserve budget");
    order.admitted(7);
    work[7].prefill = 8192;
    const auto after_reuse = plan_budget_step(work, order, 8192);
    check(after_reuse.prefill[0] == 8192 && after_reuse.prefill[7] == 0,
          "The surviving request keeps its original place in line");
}

void check_dense() {
    constexpr int width = sglang::decoder_batch_max_lanes;
    BatchAdmissionOrder order(width);
    BudgetLanes work{};
    for (int slot = width - 1; slot >= 0; --slot) {
        order.admitted(slot);
        work[slot].prefill = 32768;
    }
    for (int head = width - 1; head >= 0; --head) {
        for (int chunk = 0; chunk < 4; ++chunk) {
            const auto plan = plan_budget_step(work, order, 8192);
            check(total(plan.prefill) == 8192 && plan.prefill[head] == 8192,
                  "Occupancy does not change the head grant");
            work[head].prefill -= plan.prefill[head];
        }
    }
    check(total(plan_budget_step(work, order, 8192).prefill) == 0, "No work remains");
}

}

int main() {
    try {
        check_decode_first();
        check_admission_order();
        check_dense();
        std::cout << "Decoder batch planner checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
