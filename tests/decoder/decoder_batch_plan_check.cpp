#include "sglang/text/decoder_batch_plan.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

using sglang::decoder::BudgetLanes;
using sglang::decoder::PrefillRows;
using sglang::decoder::plan_budget_step;

constexpr int tile = 1024;

void check(bool condition, const char* detail) {
    if (!condition) {
        throw std::runtime_error(detail);
    }
}

int total(const PrefillRows& rows) {
    return std::accumulate(rows.begin(), rows.end(), 0);
}

void check_rotation() {
    constexpr int lanes = 16;
    constexpr int prefill_lanes = 14;
    constexpr int budget = 2611;
    BudgetLanes work{};
    PrefillRows waited{};
    for (int slot = 0; slot < prefill_lanes; ++slot) {
        work[slot].prefill = 3101 + 47 * slot;
    }
    work[14].decode = true;
    work[15].decode = true;
    int cursor = 13;
    int tiled_steps = 0;
    for (int iteration = 0; iteration < 800; ++iteration) {
        const auto full_tiles = std::count_if(work.begin(), work.end(), [](const auto& lane) {
            return lane.prefill >= tile;
        });
        const bool tiled = full_tiles > (budget - 2) / tile;
        tiled_steps += tiled;
        const auto plan = plan_budget_step(work, lanes, budget, cursor, tile);
        check(plan.decode[14] == 1 && plan.decode[15] == 1, "reserve ready decode rows");
        check(total(plan.prefill) + total(plan.decode) <= budget, "fit the step budget");
        for (int slot = 0; slot < prefill_lanes; ++slot) {
            const int grant = plan.prefill[slot];
            if (grant == 0) {
                check(++waited[slot] < lanes, "rotate past lanes already granted a tile");
                continue;
            }
            check(grant > 0 && grant <= work[slot].prefill, "fit the remaining prompt");
            if (tiled) {
                check(grant == std::min(tile, work[slot].prefill), "grant a tile or final tail");
            }
            waited[slot] = 0;
            work[slot].prefill -= grant;
            if (work[slot].prefill == 0) {
                work[slot].prefill = 2737 + (37 * iteration + slot) % 900;
            }
        }
    }
    check(tiled_steps > 0 && tiled_steps < 800, "exercise saturated and low-occupancy steps");
}

void check_mixed_work() {
    BudgetLanes work{};
    work[0].prefill = 3023;
    work[1].prefill = 6027;
    work[3].prefill = 1777;
    work[4].prefill = 73;
    work[5].prefill = 3100;
    work[6].decode = true;
    work[7].decode = true;
    int cursor = 0;
    const auto plan = plan_budget_step(work, 8, 2200, cursor, tile);
    check(total(plan.decode) == 2 && plan.decode[6] == 1 && plan.decode[7] == 1,
          "reserve ready decode rows before prefill");
    check(plan.prefill[0] == tile && plan.prefill[1] == tile && plan.prefill[4] == 73,
          "fit a finishing tail beside whole tiles");
    check(total(plan.prefill) == 2 * tile + 73 && cursor == 3,
          "leave residual space and resume at the first deferred lane");

    int stock_cursor = cursor;
    const auto small = plan_budget_step(work, 8, tile + 1, cursor, tile);
    const auto stock = plan_budget_step(work, 8, tile + 1, stock_cursor, 0);
    check(small.prefill == stock.prefill && small.decode == stock.decode && cursor == stock_cursor,
          "use fair share when decode reservations leave less than a tile");
    check(total(small.prefill) == tile - 1 && total(small.decode) == 2,
          "keep making progress with a sub-tile budget");
}

void check_occupancy() {
    BudgetLanes work{};
    int cursor = 0;
    constexpr int slots = sglang::decoder_batch_max_lanes;
    constexpr int budget = 8 * tile;
    for (int slot = 0; slot < 4; ++slot) {
        work[slot].prefill = 131072;
    }
    auto plan = plan_budget_step(work, slots, budget, cursor, tile);
    check(total(plan.prefill) == budget && cursor == 1, "use fair share at low occupancy");
    for (int slot = 0; slot < 4; ++slot) {
        check(plan.prefill[slot] == 2 * tile, "allow multiple tiles on each low-occupancy lane");
    }
    for (auto& lane : work) {
        lane.prefill = 32769;
    }
    plan = plan_budget_step(work, slots, budget, cursor, tile);
    check(total(plan.prefill) == budget && cursor == 9, "rotate saturated grants");
    for (int slot = 0; slot < slots; ++slot) {
        check(plan.prefill[slot] == (slot >= 1 && slot <= 8 ? tile : 0),
              "grant at most one tile per saturated lane");
    }
}

void check_stock_speculation() {
    BudgetLanes work{};
    work[0].prefill = 50;
    work[1].prefill = 90;
    work[2] = {0, 4, true};
    work[3] = {0, 2, true};
    int cursor = 2;
    const auto plan = plan_budget_step(work, 4, 14, cursor, 0);
    check(plan.decode[2] == 5 && plan.decode[3] == 3, "preserve speculative proposal allowance");
    check(plan.prefill[0] == 3 && plan.prefill[1] == 3 && cursor == 3,
          "preserve stock prefill share beside speculative decode");
}

}

int main() {
    try {
        check_rotation();
        check_mixed_work();
        check_occupancy();
        check_stock_speculation();
        std::cout << "Decoder batch planner checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
