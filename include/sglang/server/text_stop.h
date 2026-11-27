#pragma once

#include "sglang/decoder.h"
#include "sglang/tokenizer/tokenizer.h"
#include "sglang/utf8.h"

#include <memory>
#include <string_view>

namespace sglang::server {

class TextStops {
public:
    explicit TextStops(const std::vector<std::string>& stops);

    struct Pattern {
        std::string text;
        std::vector<std::uint32_t> border;
    };
    std::vector<Pattern> patterns;
};

class TextStopMatcher {
public:
    explicit TextStopMatcher(const TextStops& stops);

    void push(std::string_view text);
    std::optional<std::size_t> match(bool configured_order) const;
    std::size_t bytes() const;
    bool has_match(std::size_t index) const;
    std::size_t matched_start(std::size_t index) const;
    std::size_t matched_end(std::size_t index) const;
    std::size_t safe_end() const;

private:
    struct State {
        std::size_t prefix = 0;
        std::size_t end = 0;
        std::optional<std::size_t> start;
    };

    const TextStops& stops_;
    std::vector<State> states_;
    std::size_t seen_ = 0;
};

class TextStopBuffer {
public:
    explicit TextStopBuffer(const TextStops& stops);

    void push(std::string_view text);
    std::optional<std::size_t> match() const;
    bool pending() const;
    std::string take(bool final, std::optional<std::size_t> match, bool include_match = false);

private:
    TextStopMatcher matcher_;
    std::string pending_;
    std::size_t offset_ = 0;
    std::size_t consumed_ = 0;
};

struct NativeTextStops {
    const Tokenizer& tokenizer;
    TextStops stops;
};

class TokenTextStop {
public:
    explicit TokenTextStop(std::shared_ptr<const TextStops> stops);

    void push(std::string_view piece, bool terminal);
    int match_native(const Tokenizer& tokenizer, const int* ids, int count, bool terminal,
                     std::optional<std::size_t>& matched);
    std::optional<std::size_t> match(bool configured_order) const;

private:
    std::shared_ptr<const TextStops> stops_;
    TextStopMatcher matcher_;
    Utf8Stream utf8_;
};

}
