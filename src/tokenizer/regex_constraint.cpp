#include "sglang/decoder_constraint.h"

#include "sglang/tokenizer/regex.h"

#include <bitset>

namespace sglang {
namespace {

namespace regex = tokenizer::regex;
constexpr std::size_t max_token_edges = 4 * 1024 * 1024;

struct Piece {
    TokenId token;
    std::string_view bytes;
};

std::vector<Piece> output_pieces(const Tokenizer& tokenizer) {
    std::vector<Piece> pieces;
    pieces.reserve(model_vocab_size);
    for (TokenId token = 0; token < model_vocab_size; ++token) {
        const auto bytes = tokenizer.piece(token, true);
        if (!bytes.empty() && !constraint_eos(token)) {
            pieces.push_back({token, bytes});
        }
    }
    return pieces;
}

int matching_state(const std::vector<regex::State>& states, const Piece& piece, int state) {
    for (const unsigned char byte : piece.bytes) {
        state = states[state].next[byte];
        if (state == regex::dead_state) {
            break;
        }
    }
    return state;
}

using StateSet = std::bitset<regex::max_dfa_states>;

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
    const auto pieces = output_pieces(tokenizer);
    auto result = std::shared_ptr<TokenConstraint>(new TokenConstraint);
    result->kind_ = kind;
    result->nodes_.resize(states.size());
    std::vector<StateSet> previous(states.size());
    StateSet visited;
    visited.set(0);
    std::vector<int> frontier{0};
    std::size_t edge_count = 0;
    for (std::size_t index = 0; index < frontier.size(); ++index) {
        const int state = frontier[index];
        auto& node = result->nodes_[state];
        node.accepting = states[state].accepting;
        for (const auto& piece : pieces) {
            const int next = matching_state(states, piece, state);
            if (next == regex::dead_state) {
                continue;
            }
            if (++edge_count > max_token_edges) {
                throw std::invalid_argument("regex exceeds the 4194304-token-transition limit");
            }
            node.edges.push_back({piece.token, next});
            previous[next].set(state);
            if (!visited[next]) {
                visited.set(next);
                frontier.push_back(next);
            }
        }
    }
    const auto minimum = shortest(states, previous);
    for (std::size_t index = 0; index < states.size(); ++index) {
        auto& node = result->nodes_[index];
        node.minimum = minimum[index];
        const auto dead = [&](const Edge& edge) {
            return minimum[edge.next] == std::numeric_limits<int>::max();
        };
        node.edges.erase(std::remove_if(node.edges.begin(), node.edges.end(), dead),
                         node.edges.end());
        const std::size_t candidates = node.edges.size() + (node.accepting ? 2 : 0);
        result->uses_wide_mask_ |= candidates > constraint_max_candidates;
    }
    if (result->minimum_tokens() == std::numeric_limits<int>::max()) {
        throw std::invalid_argument("regex cannot be represented by the tokenizer");
    }
    return result;
}

}
