#pragma once

#include <array>
#include <string_view>
#include <vector>

namespace sglang::tokenizer::regex {

inline constexpr int max_dfa_states = 256;
inline constexpr int dead_state = -1;

struct Range {
    char32_t first;
    char32_t last;
};

struct Expression {
    enum class Kind { character, sequence, alternate, repeat };

    Kind kind;
    std::vector<Range> ranges;
    std::vector<Expression> children;
    int minimum = 0;
    int maximum = 0;
};

struct State {
    std::array<int, 256> next;
    bool accepting;
};

Expression parse(std::string_view pattern);
std::vector<State> determinize(const Expression& expression);

}
