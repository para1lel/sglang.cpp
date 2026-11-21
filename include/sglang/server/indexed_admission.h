#pragma once

#include "sglang/decoder_batch.h"

#include <array>
#include <chrono>
#include <cstdint>

namespace sglang::server {

class IndexedAdmission {
public:
    using Clock = std::chrono::steady_clock;

    void wait(int owner, std::uint32_t slots) {
        auto& entry = waiting_[owner];
        if (entry.ticket == 0) {
            entry.ticket = ++last_ticket_;
        }
        entry.slots = slots;
        entry.eligible_until = Clock::time_point::max();
    }

    bool can_claim(int owner, int slot, Clock::time_point now) const {
        const auto ticket = waiting_[owner].ticket;
        const auto bit = std::uint32_t{1} << slot;
        for (const auto& entry : waiting_) {
            if ((entry.slots & bit) != 0 && entry.ticket < ticket && now < entry.eligible_until) {
                return false;
            }
        }
        return true;
    }

    void claimed(int owner, int slot) {
        auto& entry = waiting_[owner];
        entry.slots &= ~(std::uint32_t{1} << slot);
        entry.ticket = ++last_ticket_;
    }

    void poll(int owner, Clock::time_point deadline) { waiting_[owner].eligible_until = deadline; }
    void pause(int owner) { waiting_[owner].slots = 0; }
    void finish(int owner) { waiting_[owner] = Entry{}; }

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
