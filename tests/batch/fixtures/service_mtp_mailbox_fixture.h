#pragma once

#include "sglang/decoder_batch.h"

#include <array>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <vector>

namespace sglang::server::mailbox_fixture {

using Selection = std::array<bool, decoder_batch_max_lanes>;

void require(bool condition, const char* message);
void check_progress();

struct Barrier {
    bool entered = false;
    bool released = false;
};

struct Fixture {
    std::mutex mutex;
    std::condition_variable changed;
    Barrier seed;
    Barrier writer;
    Barrier peer;
    Barrier fault;
    Barrier before_claim;
    Barrier after_claim;
    Barrier prefill_admit;
    Barrier prefill_advance;
    Barrier capacity_rejected;
    std::array<Barrier, decoder_batch_max_lanes> consumers{};
    std::array<int, decoder_batch_max_lanes> admissions{};
    std::vector<int> admission_order;
    std::vector<Selection> prefill_grants;
    std::vector<int> capacity_admissions;
    int capacity_rejections = 0;
    bool limit_capacity = false;
    bool exhaust_capacity = false;
    bool inspect_prefill = false;
    int max_lanes = 4;
    int hold_prefill_slot = 0;
    int fail_admission_slot = -1;
    int primary_steps = 0;
    int fail_at = 0;
    int cancels = 0;
    int terminal_stores = 0;
    bool claim_terminal = false;
    bool fail_after_claim = false;
    bool permanent = false;
    bool unavailable = false;

    Fixture();
    ~Fixture();
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
    Fixture(Fixture&&) = delete;
    Fixture& operator=(Fixture&&) = delete;
    static Fixture* current();

    void await(std::unique_lock<std::mutex>& lock, const std::function<bool()>& ready,
               const char* message = "Timed out at a decoder or writer barrier");
    void wait_for(const Barrier& barrier);
    void hold(std::unique_lock<std::mutex>& lock, Barrier& barrier);
    void release(Barrier& barrier);
};

}
