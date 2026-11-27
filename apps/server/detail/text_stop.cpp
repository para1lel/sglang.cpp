#include "sglang/server/text_stop.h"

#include <algorithm>
#include <array>
#include <utility>

namespace sglang::server {

TextStops::TextStops(const std::vector<std::string>& stops) {
    for (const auto& text : stops) {
        Pattern pattern{text, std::vector<std::uint32_t>(text.size())};
        for (std::size_t pos = 1; pos < text.size(); ++pos) {
            auto prefix = pattern.border[pos - 1];
            while (prefix != 0 && text[pos] != text[prefix]) {
                prefix = pattern.border[prefix - 1];
            }
            pattern.border[pos] = prefix + (text[pos] == text[prefix]);
        }
        patterns.push_back(std::move(pattern));
    }
}

TextStopMatcher::TextStopMatcher(const TextStops& stops)
    : stops_(stops), states_(stops.patterns.size()) {}

void TextStopMatcher::push(std::string_view text) {
    for (std::size_t index = 0; index < states_.size(); ++index) {
        const auto& pattern = stops_.patterns[index];
        auto& state = states_[index];
        if (pattern.text.empty()) {
            state.start = 0;
            state.end = 0;
            continue;
        }
        for (std::size_t pos = 0; pos < text.size(); ++pos) {
            while (state.prefix != 0 && text[pos] != pattern.text[state.prefix]) {
                state.prefix = pattern.border[state.prefix - 1];
            }
            state.prefix += text[pos] == pattern.text[state.prefix];
            if (state.prefix == pattern.text.size()) {
                if (!state.start) {
                    state.end = seen_ + pos + 1;
                    state.start = state.end - state.prefix;
                }
                state.prefix = pattern.border[state.prefix - 1];
            }
        }
    }
    seen_ += text.size();
}

std::optional<std::size_t> TextStopMatcher::match(bool configured_order) const {
    std::optional<std::size_t> best;
    for (std::size_t index = 0; index < states_.size(); ++index) {
        if (!states_[index].start) {
            continue;
        }
        if (configured_order) {
            return index;
        }
        if (!best || states_[index].start < states_[*best].start) {
            best = index;
        }
    }
    return best;
}

std::size_t TextStopMatcher::bytes() const { return seen_; }

bool TextStopMatcher::has_match(std::size_t index) const {
    return states_[index].start.has_value();
}

std::size_t TextStopMatcher::matched_start(std::size_t index) const {
    return states_[index].start.value();
}

std::size_t TextStopMatcher::matched_end(std::size_t index) const { return states_[index].end; }

std::size_t TextStopMatcher::safe_end() const {
    auto end = seen_;
    for (const auto& state : states_) {
        end = std::min(end, seen_ - state.prefix);
        if (state.start) {
            end = std::min(end, *state.start);
        }
    }
    return end;
}

TextStopBuffer::TextStopBuffer(const TextStops& stops) : matcher_(stops) {}

void TextStopBuffer::push(std::string_view text) {
    matcher_.push(text);
    pending_ += text;
}

std::optional<std::size_t> TextStopBuffer::match() const { return matcher_.match(false); }

bool TextStopBuffer::pending() const { return offset_ < pending_.size(); }

std::string TextStopBuffer::take(bool final, std::optional<std::size_t> match, bool include_match) {
    auto end = final ? matcher_.bytes() : matcher_.safe_end();
    if (match && matcher_.has_match(*match)) {
        end = include_match ? matcher_.matched_end(*match) : matcher_.matched_start(*match);
    }
    auto output = pending_.substr(offset_, end - consumed_);
    offset_ += output.size();
    consumed_ = end;
    if (final || match) {
        pending_.clear();
    } else if (offset_ >= pending_.size() - offset_) {
        pending_.erase(0, offset_);
        offset_ = 0;
    }
    return output;
}

TokenTextStop::TokenTextStop(std::shared_ptr<const TextStops> stops)
    : stops_(std::move(stops)), matcher_(*stops_) {}

void TokenTextStop::push(std::string_view piece, bool terminal) {
    matcher_.push(utf8_.push(piece));
    if (terminal) matcher_.push(utf8_.finish());
}

int TokenTextStop::match_native(const Tokenizer& tokenizer, const int* ids, int count,
                                bool terminal, std::optional<std::size_t>& matched) {
    std::array<std::size_t, decoder_max_mtp_steps + 1> ends;
    for (int index = 0; index < count; ++index) {
        push(tokenizer.piece(ids[index], false), terminal && index + 1 == count);
        ends[index] = matcher_.bytes();
    }
    matched = matcher_.match(true);
    if (!matched) return 0;
    const auto end = std::lower_bound(ends.begin(), ends.begin() + count,
                                      matcher_.matched_end(*matched));
    return 1 + static_cast<int>(end - ends.begin());
}

std::optional<std::size_t> TokenTextStop::match(bool configured_order) const {
    return matcher_.match(configured_order);
}

}
