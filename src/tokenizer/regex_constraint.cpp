#include "sglang/decoder_constraint.h"

#include "sglang/tokenizer/regex.h"

#include <bitset>

namespace sglang {
namespace {

namespace regex = tokenizer::regex;
constexpr std::size_t max_token_edges = 4 * 1024 * 1024;

struct Piece {
    TokenId token;
    std::string_view tail;
};

using PiecesByFirstByte = std::array<std::vector<Piece>, 256>;

PiecesByFirstByte output_pieces(const Tokenizer& tokenizer) {
    PiecesByFirstByte pieces;
    for (TokenId token = 0; token < model_vocab_size; ++token) {
        const auto bytes = tokenizer.piece(token, true);
        if (!bytes.empty() && !constraint_eos(token)) {
            pieces[static_cast<unsigned char>(bytes.front())].push_back({token, bytes.substr(1)});
        }
    }
    return pieces;
}

int matching_state(const std::vector<regex::State>& states, const Piece& piece, int state) {
    for (const unsigned char byte : piece.tail) {
        state = states[state].next[byte];
        if (state == regex::dead_state) {
            break;
        }
    }
    return state;
}

std::vector<int> token_targets(const std::vector<regex::State>& states,
                               const PiecesByFirstByte& pieces, int state) {
    std::vector<int> targets(model_vocab_size, regex::dead_state);
    for (std::size_t byte = 0; byte < pieces.size(); ++byte) {
        const int next = states[state].next[byte];
        if (next != regex::dead_state) {
            for (const auto& piece : pieces[byte]) {
                targets[piece.token] = matching_state(states, piece, next);
            }
        }
    }
    return targets;
}

int common_target(const std::vector<int>& targets, std::size_t state_count) {
    std::vector<int> counts(state_count, 0);
    for (const int target : targets) {
        if (target != regex::dead_state) {
            ++counts[target];
        }
    }
    const auto best = std::max_element(counts.begin(), counts.end());
    return *best > constraint_max_candidates ? static_cast<int>(best - counts.begin())
                                             : regex::dead_state;
}

using StateSet = std::bitset<regex::max_tool_dfa_states>;

struct Trigger {
    std::string_view marker;
    int destination;
};

int plain_states(std::vector<regex::State>& states, const Trigger* triggers, int count) {
    const int start = static_cast<int>(states.size());
    std::vector<std::string_view> prefixes(1);
    for (int index = 0; index < count; ++index) {
        const auto marker = triggers[index].marker;
        for (std::size_t length = 1; length < marker.size(); ++length) {
            const auto prefix = marker.substr(0, length);
            if (std::find(prefixes.begin(), prefixes.end(), prefix) == prefixes.end()) {
                prefixes.push_back(prefix);
            }
        }
    }
    for (const auto prefix : prefixes) {
        regex::State state{{}, true};
        for (int byte = 0; byte < 256; ++byte) {
            const auto next = std::string(prefix) + static_cast<char>(byte);
            int destination = start;
            std::size_t longest = 0;
            for (std::size_t index = 1; index < prefixes.size(); ++index) {
                if (prefixes[index].size() > longest &&
                    next.size() >= prefixes[index].size() &&
                    next.compare(next.size() - prefixes[index].size(), prefixes[index].size(),
                                 prefixes[index]) == 0) {
                    longest = prefixes[index].size();
                    destination = start + static_cast<int>(index);
                }
            }
            for (int index = 0; index < count; ++index) {
                const auto& trigger = triggers[index];
                if (next.size() >= trigger.marker.size() &&
                    next.compare(next.size() - trigger.marker.size(), trigger.marker.size(),
                                 trigger.marker) == 0) {
                    destination = trigger.destination;
                    break;
                }
            }
            state.next[byte] = destination;
        }
        states.push_back(state);
    }
    return start;
}

int literal_states(std::vector<regex::State>& states, std::string_view text, int next) {
    for (const unsigned char byte : std::string(text.rbegin(), text.rend())) {
        regex::State state{{}, false};
        state.next.fill(regex::dead_state);
        state.next[byte] = next;
        next = static_cast<int>(states.size());
        states.push_back(state);
    }
    return next;
}

std::vector<int> shortest(const std::vector<regex::State>& states,
                          const std::vector<StateSet>& previous) {
    const auto count = static_cast<int>(states.size());
    std::vector<int> minimum(count, std::numeric_limits<int>::max());
    std::vector<int> frontier;
    for (int state = 0; state < count; ++state) {
        if (states[state].accepting) {
            minimum[state] = 0;
            frontier.push_back(state);
        }
    }
    for (std::size_t index = 0; index < frontier.size(); ++index) {
        const int next = frontier[index];
        for (int state = 0; state < count; ++state) {
            if (previous[next][state] && minimum[state] == std::numeric_limits<int>::max()) {
                minimum[state] = minimum[next] + 1;
                frontier.push_back(state);
            }
        }
    }
    return minimum;
}

}

std::shared_ptr<const TokenConstraint> TokenConstraint::compile_regex(
    const Tokenizer& tokenizer, std::string_view pattern) {
    return compile_regular(tokenizer, pattern, Kind::regex);
}

std::shared_ptr<const TokenConstraint> TokenConstraint::compile_regular(
    const Tokenizer& tokenizer, std::string_view pattern, Kind kind) {
    const auto states = kind == Kind::regex
        ? regex::determinize(regex::parse(pattern))
        : regex::determinize_schema(regex::parse_generated(pattern));
    return compile_states(tokenizer, states, kind);
}

std::shared_ptr<const TokenConstraint> TokenConstraint::compile_triggered(
    const Tokenizer& tokenizer, std::string_view continuation, std::string_view trigger,
    TriggeredConstraintMode mode, bool parallel, std::string_view prelude_end) {
    if (trigger.empty() || trigger.size() + prelude_end.size() > 64) {
        throw std::invalid_argument("Invalid constraint trigger length");
    }
    auto states = regex::determinize_tool(regex::parse_tool_pattern(continuation));
    const int frame_count = static_cast<int>(states.size());
    const Trigger next_frame{trigger, 0};
    const bool follow = mode != TriggeredConstraintMode::single && parallel;
    const int plain = follow || mode == TriggeredConstraintMode::optional
        ? plain_states(states, &next_frame, 1) : regex::dead_state;
    const int after = follow ? plain : regex::dead_state;
    if (after != regex::dead_state) {
        for (int index = 0; index < frame_count; ++index) {
            if (states[index].accepting) {
                states[index].next = states[after].next;
            }
        }
    }
    const int forced = mode == TriggeredConstraintMode::optional ? plain :
        literal_states(states, trigger, 0);
    int entry = forced;
    if (!prelude_end.empty()) {
        const Trigger triggers[] = {next_frame, {prelude_end, forced}};
        entry = plain_states(states, triggers, 2);
    }
    std::swap(states[0], states[entry]);
    for (auto& state : states) {
        for (auto& next : state.next) {
            if (next == 0) next = entry;
            else if (next == entry) next = 0;
        }
    }
    return compile_states(tokenizer, states, Kind::triggered);
}

std::shared_ptr<const TokenConstraint> TokenConstraint::compile_states(
    const Tokenizer& tokenizer, const std::vector<regex::State>& states, Kind kind) {
    const auto state_limit = kind == Kind::triggered ? regex::max_tool_dfa_states :
                                                      regex::max_dfa_states;
    if (states.size() > static_cast<std::size_t>(state_limit)) {
        throw std::invalid_argument("Output constraint exceeds the DFA state limit");
    }
    const auto pieces = output_pieces(tokenizer);
    const auto edge_limit = kind == Kind::triggered ? max_token_edges * 2 : max_token_edges;
    auto result = std::shared_ptr<TokenConstraint>(new TokenConstraint);
    result->kind_ = kind;
    result->nodes_.resize(states.size());
    std::vector<StateSet> previous(states.size());
    StateSet visited;
    visited.set(0);
    std::vector<int> frontier{0};
    std::size_t edge_count = 0;
    const auto reach = [&](int state, int next) {
        previous[next].set(state);
        if (!visited[next]) {
            visited.set(next);
            frontier.push_back(next);
        }
    };
    for (std::size_t index = 0; index < frontier.size(); ++index) {
        const int state = frontier[index];
        auto& node = result->nodes_[state];
        node.accepting = states[state].accepting;
        const auto targets = token_targets(states, pieces, state);
        node.fallback = common_target(targets, states.size());
        for (TokenId token = 0; token < model_vocab_size; ++token) {
            const int next = targets[token];
            if (next == node.fallback) {
                continue;
            }
            if (++edge_count > edge_limit) {
                throw std::invalid_argument("Output constraint exceeds the token-transition limit");
            }
            node.edges.push_back({token, next});
            if (next != regex::dead_state) {
                reach(state, next);
            }
        }
        if (node.fallback != regex::dead_state) {
            reach(state, node.fallback);
        }
    }
    const auto minimum = shortest(states, previous);
    for (std::size_t index = 0; index < states.size(); ++index) {
        auto& node = result->nodes_[index];
        node.minimum = minimum[index];
        const auto live = [&](int next) {
            return next != regex::dead_state && minimum[next] != std::numeric_limits<int>::max();
        };
        if (!live(node.fallback)) {
            node.fallback = regex::dead_state;
            node.edges.erase(std::remove_if(node.edges.begin(), node.edges.end(),
                                            [&](const Edge& edge) { return !live(edge.next); }),
                             node.edges.end());
        }
        for (auto& edge : node.edges) {
            edge.next = live(edge.next) ? edge.next : regex::dead_state;
        }
        const std::size_t candidates = node.fallback != regex::dead_state ? model_vocab_size :
            node.edges.size() + (node.accepting ? 2 : 0);
        result->uses_wide_mask_ |= candidates > constraint_max_candidates;
    }
    if (result->minimum_tokens() == std::numeric_limits<int>::max()) {
        throw std::invalid_argument("regex cannot be represented by the tokenizer");
    }
    return result;
}

}
