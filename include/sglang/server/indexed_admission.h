#pragma once

#include "sglang/decoder_batch.h"

#include <array>
#include <chrono>
#include <cstdint>

namespace sglang::server {

class IndexedAdmission {
public:
    using Clock = std::chrono::steady_clock;

    void wait(int owner, std::uint32_t slots);
    bool can_claim(int owner, int slot, Clock::time_point now) const;
    void claimed(int owner, int slot);
    void poll(int owner, Clock::time_point deadline);
    void pause(int owner);
    void finish(int owner);

private:
    struct Entry {
        std::uint32_t slots = 0;
        std::uint64_t ticket = 0;
        Clock::time_point eligible_until = Clock::time_point::max();
    };

    std::array<Entry, decoder_batch_max_lanes> waiting_{};
    std::uint64_t last_ticket_ = 0;
};

}
