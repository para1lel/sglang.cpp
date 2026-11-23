#pragma once

#include "sglang/decoder_batch.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace sglang::decoder {

class PrefixIndex {
public:
    // Consume a match before any recency change. A slot remains stable across touches.
    struct Match {
        int slot = -1;
        int tokens = 0;
        int draft_rows = 0;
        bool draft_miss = false;

    private:
        friend class PrefixIndex;
        int position = -1;
    };

    struct Write {
        int slot;
        bool evicted;

    private:
        friend class PrefixIndex;
        Write(int slot, bool evicted, int position)
            : slot(slot), evicted(evicted), position(position) {}

        int position;
    };

    struct Occupancy {
        int entries = 0;
        int tokens = 0;
    };

    explicit PrefixIndex(int capacity) : capacity_(capacity) {
        if (capacity < 1 || capacity > decoder_batch_prefix_max_entries) {
            throw std::invalid_argument("Invalid prefix cache capacity");
        }
        entries_.reset(new Entry[capacity]);
        std::iota(recency_.begin(), recency_.begin() + capacity_, 0);
    }

    // max_tokens bounds the readable part of prompt.
    Match find(const std::vector<int>& prompt, std::string_view salt, int min_tokens,
               int max_tokens, bool require_draft = false) const {
        Match match;
        int longest = std::max(min_tokens, 1) - 1;
        for (int position = 0; position < capacity_; ++position) {
            const int slot = recency_[position];
            const auto& entry = entries_[slot];
            if (entry.tokens <= longest || entry.tokens > max_tokens) continue;

            const bool eligible = !require_draft || entry.draft_rows == entry.tokens - 1;
            if ((!eligible && (match.slot != -1 || match.draft_miss)) ||
                salt != std::string_view(entry.salt.data(), entry.salt_size) ||
                !std::equal(entry.ids.begin(), entry.ids.begin() + entry.tokens, prompt.begin())) {
                continue;
            }
            if (!eligible) {
                match.draft_miss = true;
                continue;
            }
            match.slot = slot;
            match.tokens = entry.tokens;
            match.draft_rows = entry.draft_rows;
            match.draft_miss = false;
            match.position = position;
            longest = entry.tokens;
            if (longest == max_tokens) break;
        }
        return match;
    }

    void touch(const Match& match) { touch_at(match.position); }

    bool touch_if_covered(const Match& exact, int draft_rows) {
        if (exact.slot == -1 || exact.draft_rows < draft_rows) return false;
        touch(exact);
        return true;
    }

    // The caller preflights the transfer before invalidating a slot and publishes after completion.
    Write begin_write(const Match& exact) {
        int position = exact.position;
        if (position == -1) {
            const auto end = recency_.begin() + capacity_;
            const auto free_slot = std::find_if(
                recency_.begin(), end, [&](int slot) { return entries_[slot].tokens == 0; });
            position = free_slot == end ? capacity_ - 1 : free_slot - recency_.begin();
        }
        const int slot = recency_[position];
        auto& entry = entries_[slot];
        const bool evicted = exact.slot == -1 && entry.tokens != 0;
        entry.tokens = entry.draft_rows = 0;
        return Write(slot, evicted, position);
    }

    void publish(const Write& write, const std::vector<int>& prompt, std::string_view salt,
                 int tokens, int draft_rows) {
        auto& entry = entries_[write.slot];
        std::copy_n(prompt.begin(), tokens, entry.ids.begin());
        std::copy(salt.begin(), salt.end(), entry.salt.begin());
        entry.salt_size = salt.size();
        entry.tokens = tokens;
        entry.draft_rows = draft_rows;
        touch_at(write.position);
    }

    void reset() {
        for (int slot = 0; slot < capacity_; ++slot) {
            entries_[slot].tokens = entries_[slot].draft_rows = 0;
        }
    }

    Occupancy occupancy() const {
        Occupancy occupancy;
        for (int slot = 0; slot < capacity_; ++slot) {
            occupancy.entries += entries_[slot].tokens != 0;
            occupancy.tokens += entries_[slot].tokens;
        }
        return occupancy;
    }

private:
    struct Entry {
        std::array<int, decoder_batch_prefix_max_tokens> ids;
        std::array<char, decoder_prefix_max_salt_bytes> salt;
        std::size_t salt_size = 0;
        int tokens = 0;
        int draft_rows = 0;
    };

    void touch_at(int position) {
        std::rotate(recency_.begin(), recency_.begin() + position, recency_.begin() + position + 1);
    }

    std::unique_ptr<Entry[]> entries_;
    const int capacity_;
    // Reordering slots never moves their metadata or payloads.
    std::array<int, decoder_batch_prefix_max_entries> recency_;
};

}
