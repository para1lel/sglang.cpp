#include "sglang/tokenizer/regex.h"

#include "sglang/utf8.h"

#include <algorithm>
#include <bitset>
#include <map>
#include <stdexcept>

namespace sglang::tokenizer::regex {
namespace {

constexpr int max_nfa_states = 4096;
constexpr int max_schema_subsets = 1024;
constexpr int max_tool_subsets = 4096;
using Kind = Expression::Kind;
using Utf8Bytes = std::array<unsigned, 4>;

struct Edge {
    unsigned first;
    unsigned last;
    int next;
};

struct Node {
    std::vector<int> empty;
    std::vector<Edge> bytes;
};

Utf8Bytes utf8_bytes(char32_t scalar) {
    std::string text;
    append_utf8(text, scalar);
    Utf8Bytes bytes{};
    std::copy(text.begin(), text.end(), bytes.begin());
    for (auto& byte : bytes) {
        byte &= 0xff;
    }
    return bytes;
}

class Nfa {
public:
    Nfa() { add_node(); add_node(); }
    void emit(const Expression& expression, int start, int finish);
    std::vector<State> determinize(int limit) const;

private:
    int add_node();
    void character(const std::vector<Range>& ranges, int start, int finish);
    void byte_range(const Utf8Bytes& low, const Utf8Bytes& high, unsigned offset,
                    unsigned size, int start, int finish);
    void repeat(const Expression& expression, int start, int finish);
    void xml_text(int start, int finish);
    std::vector<int> closure(std::vector<int> seeds) const;
    std::vector<int> advance(const std::vector<int>& states, unsigned byte) const;

    std::vector<Node> nodes_;
};

int Nfa::add_node() {
    if (nodes_.size() == max_nfa_states) {
        throw std::invalid_argument("regex exceeds the 4096-state NFA limit");
    }
    const int state = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    return state;
}

void Nfa::byte_range(const Utf8Bytes& low, const Utf8Bytes& high, unsigned offset,
                     unsigned size, int start, int finish) {
    if (offset + 1 == size) {
        nodes_[start].bytes.push_back({low[offset], high[offset], finish});
        return;
    }
    const auto emit_prefix = [&](unsigned first, unsigned last, const Utf8Bytes& suffix_low,
                                 const Utf8Bytes& suffix_high) {
        const int next = add_node();
        nodes_[start].bytes.push_back({first, last, next});
        byte_range(suffix_low, suffix_high, offset + 1, size, next, finish);
    };
    if (low[offset] == high[offset]) {
        emit_prefix(low[offset], high[offset], low, high);
        return;
    }
    Utf8Bytes floor{};
    Utf8Bytes ceiling{};
    floor.fill(0x80);
    ceiling.fill(0xbf);
    const auto lower_suffix = low.begin() + offset + 1;
    const auto upper_suffix = high.begin() + offset + 1;
    const auto lower_partial =
        !std::all_of(lower_suffix, low.begin() + size, [](unsigned byte) { return byte == 0x80; });
    const auto upper_partial =
        !std::all_of(upper_suffix, high.begin() + size, [](unsigned byte) { return byte == 0xbf; });
    if (lower_partial) {
        emit_prefix(low[offset], low[offset], low, ceiling);
    }
    const unsigned first = low[offset] + lower_partial;
    const unsigned last = high[offset] - upper_partial;
    if (first <= last) {
        emit_prefix(first, last, floor, ceiling);
    }
    if (upper_partial) {
        emit_prefix(high[offset], high[offset], floor, high);
    }
}

void Nfa::character(const std::vector<Range>& ranges, int start, int finish) {
    constexpr std::array<Range, 5> scalar_groups{
        Range{0, 0x7f}, Range{0x80, 0x7ff}, Range{0x800, 0xd7ff},
        Range{0xe000, 0xffff}, Range{0x10000, 0x10ffff}};
    for (const auto range : ranges) {
        unsigned size = 0;
        for (const auto group : scalar_groups) {
            size += group.first != 0xe000;
            const auto low = std::max(range.first, group.first);
            const auto high = std::min(range.last, group.last);
            if (low <= high) {
                byte_range(utf8_bytes(low), utf8_bytes(high), 0, size, start, finish);
            }
        }
    }
}

void Nfa::repeat(const Expression& expression, int start, int finish) {
    const auto& child = expression.children.front();
    for (int index = 0; index < expression.minimum; ++index) {
        const int next = add_node();
        emit(child, start, next);
        start = next;
    }
    if (expression.maximum < 0) {
        const int loop_end = add_node();
        emit(child, start, loop_end);
        nodes_[loop_end].empty.push_back(start);
    } else {
        for (int index = expression.minimum; index < expression.maximum; ++index) {
            nodes_[start].empty.push_back(finish);
            const int next = add_node();
            emit(child, start, next);
            start = next;
        }
    }
    nodes_[start].empty.push_back(finish);
}

using MarkerTransitions = std::array<int, 128>;

std::vector<MarkerTransitions> xml_marker_trie() {
    constexpr std::string_view markers[] = {"</parameter>", "<parameter=", "</function>",
                                            "</tool_call>"};
    std::vector<MarkerTransitions> trie(1);
    for (const auto marker : markers) {
        std::size_t parent = 0;
        for (std::size_t index = 0; index < marker.size(); ++index) {
            const auto byte = static_cast<unsigned char>(marker[index]);
            int next = trie[parent][byte];
            if (index + 1 == marker.size()) {
                trie[parent][byte] = dead_state;
                break;
            }
            if (next == 0) {
                next = static_cast<int>(trie.size());
                trie[parent][byte] = next;
                trie.emplace_back();
            }
            parent = next;
        }
    }
    return trie;
}

void Nfa::xml_text(int start, int finish) {
    static const auto trie = xml_marker_trie();
    const int root = add_node();
    nodes_[start].empty.push_back(root);
    for (std::size_t index = 1; index < trie.size(); ++index) {
        add_node();
    }
    const int unicode = add_node();
    character({{0x80, 0x10ffff}}, unicode, root);
    for (std::size_t index = 0; index < trie.size(); ++index) {
        auto& state = nodes_[root + index];
        state.empty = {finish, unicode};
        auto transitions = trie[index];
        transitions['<'] = trie.front()['<'];
        for (unsigned first = 0; first < transitions.size();) {
            const int next = transitions[first];
            unsigned end = first + 1;
            while (end < transitions.size() && transitions[end] == next) {
                ++end;
            }
            if (next != dead_state) {
                state.bytes.push_back({first, end - 1, root + next});
            }
            first = end;
        }
    }
}

void Nfa::emit(const Expression& expression, int start, int finish) {
    const auto& children = expression.children;
    switch (expression.kind) {
        case Kind::character:
            character(expression.ranges, start, finish);
            break;
        case Kind::sequence:
            for (const auto& child : children) {
                const int next = add_node();
                emit(child, start, next);
                start = next;
            }
            nodes_[start].empty.push_back(finish);
            break;
        case Kind::alternate:
            for (const auto& child : children) {
                const int branch = add_node();
                nodes_[start].empty.push_back(branch);
                emit(child, branch, finish);
            }
            break;
        case Kind::repeat: repeat(expression, start, finish); break;
        case Kind::xml_text: xml_text(start, finish); break;
    }
}

std::vector<int> Nfa::closure(std::vector<int> seeds) const {
    std::bitset<max_nfa_states> seen;
    std::vector<int> states;
    const auto enqueue = [&](int state) {
        if (!seen.test(state)) {
            seen.set(state);
            states.push_back(state);
        }
    };
    for (const int state : seeds) {
        enqueue(state);
    }
    for (std::size_t index = 0; index < states.size(); ++index) {
        for (const int next : nodes_[states[index]].empty) {
            enqueue(next);
        }
    }
    std::sort(states.begin(), states.end());
    return states;
}

std::vector<int> Nfa::advance(const std::vector<int>& states, unsigned byte) const {
    std::vector<int> next;
    for (const int state : states) {
        for (const auto edge : nodes_[state].bytes) {
            if (edge.first <= byte && byte <= edge.last) {
                next.push_back(edge.next);
            }
        }
    }
    return closure(std::move(next));
}

std::vector<State> Nfa::determinize(int limit) const {
    std::array<bool, 257> split{};
    split.front() = split.back() = true;
    for (const auto& node : nodes_) {
        for (const auto edge : node.bytes) {
            split[edge.first] = split[edge.last + 1] = true;
        }
    }
    std::vector<unsigned> alphabet;
    for (unsigned byte = 0; byte < split.size(); ++byte) {
        if (split[byte]) {
            alphabet.push_back(byte);
        }
    }
    std::vector<std::vector<int>> subsets{closure({0})};
    std::map<std::vector<int>, int> indices{{subsets.front(), 0}};
    std::vector<State> states;
    for (std::size_t index = 0; index < subsets.size(); ++index) {
        State state{{}, std::binary_search(subsets[index].begin(), subsets[index].end(), 1)};
        for (std::size_t letter = 1; letter < alphabet.size(); ++letter) {
            const unsigned start = alphabet[letter - 1];
            auto next = advance(subsets[index], start);
            int target = dead_state;
            if (!next.empty()) {
                const auto [entry, inserted] =
                    indices.try_emplace(next, static_cast<int>(subsets.size()));
                target = entry->second;
                if (inserted) {
                    if (subsets.size() == static_cast<std::size_t>(limit)) {
                        throw std::invalid_argument(
                            "regex exceeds the " + std::to_string(limit) + "-state DFA limit");
                    }
                    subsets.push_back(std::move(next));
                }
            }
            std::fill(state.next.begin() + start, state.next.begin() + alphabet[letter], target);
        }
        states.push_back(std::move(state));
    }
    return states;
}

std::array<int, 257> signature(const State& state, const std::vector<int>& classes) {
    std::array<int, 257> result;
    result.front() = state.accepting;
    std::transform(state.next.begin(), state.next.end(), result.begin() + 1,
                   [&](int target) { return target == dead_state ? dead_state : classes[target]; });
    return result;
}

std::vector<State> minimize(const std::vector<State>& states, std::size_t limit) {
    std::vector<int> classes(states.size(), 0);
    while (true) {
        std::map<std::array<int, 257>, int> groups;
        std::vector<int> next;
        next.reserve(states.size());
        for (const auto& state : states) {
            const int number = static_cast<int>(groups.size());
            const auto [entry, inserted] = groups.try_emplace(signature(state, classes), number);
            next.push_back(entry->second);
        }
        if (groups.size() > limit) {
            throw std::invalid_argument(
                "schema exceeds the " + std::to_string(limit) + "-state minimized DFA limit");
        }
        if (classes == next) {
            std::vector<State> result(groups.size());
            for (std::size_t index = 0; index < states.size(); ++index) {
                const auto key = signature(states[index], classes);
                auto& state = result[classes[index]];
                state.accepting = key.front() != 0;
                std::copy(key.begin() + 1, key.end(), state.next.begin());
            }
            return result;
        }
        classes = std::move(next);
    }
}

}

std::vector<State> determinize(const Expression& expression) {
    Nfa automaton;
    automaton.emit(expression, 0, 1);
    return automaton.determinize(max_dfa_states);
}

std::vector<State> determinize_schema(const Expression& expression) {
    Nfa automaton;
    automaton.emit(expression, 0, 1);
    return minimize(automaton.determinize(max_schema_subsets), max_dfa_states);
}

std::vector<State> determinize_tool(const Expression& expression) {
    Nfa automaton;
    automaton.emit(expression, 0, 1);
    return minimize(automaton.determinize(max_tool_subsets), max_tool_dfa_states);
}

}
