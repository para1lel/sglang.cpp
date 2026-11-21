#pragma once

#include "sglang/tokenizer/tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace sglang {

namespace json { class Value; }

inline constexpr int constraint_max_candidates = 4096;
inline constexpr int constraint_mask_words = (model_vocab_size + 31) / 32;

class ConstraintCursor;
namespace tokenizer::regex { struct State; }

enum class TriggeredConstraintMode { optional, required, single };

class TokenConstraint {
public:
    static std::shared_ptr<const TokenConstraint> compile(const Tokenizer& tokenizer,
                                                          std::string_view ebnf);
    static std::shared_ptr<const TokenConstraint> compile_regex(const Tokenizer& tokenizer,
                                                                std::string_view pattern);
    static std::shared_ptr<const TokenConstraint> compile_json_schema(
        const Tokenizer& tokenizer, const json::Value& schema);
    static std::shared_ptr<const TokenConstraint> compile_triggered(
        const Tokenizer& tokenizer, std::string_view continuation, std::string_view trigger,
        TriggeredConstraintMode mode, bool parallel, std::string_view prelude_end);
    int minimum_tokens() const {
        const int minimum = nodes_.front().minimum;
        return is_triggered() && minimum != std::numeric_limits<int>::max() ? 0 : minimum;
    }
    bool is_triggered() const { return kind_ == Kind::triggered; }
    bool uses_wide_mask() const { return uses_wide_mask_; }
    bool is_regex() const { return kind_ == Kind::regex; }
    const char* budget_error() const {
        switch (kind_) {
            case Kind::regex: return "max_tokens is too small to complete the regex";
            case Kind::json_schema: return "max_tokens is too small to complete the JSON Schema";
            case Kind::ebnf: return "max_tokens is too small to complete an ebnf choice";
            case Kind::triggered: return "max_tokens is too small to complete a tool call";
        }
        throw std::logic_error("Unknown output constraint");
    }

private:
    friend class ConstraintCursor;

    enum class Kind { ebnf, regex, json_schema, triggered };

    struct Edge {
        TokenId token;
        int next;
    };
    struct Node {
        std::vector<Edge> edges;
        bool accepting = false;
        int minimum = std::numeric_limits<int>::max();
        int fallback = -1;
    };

    TokenConstraint() = default;
    static std::shared_ptr<const TokenConstraint> compile_regular(
        const Tokenizer& tokenizer, std::string_view pattern, Kind kind);
    static std::shared_ptr<const TokenConstraint> compile_states(
        const Tokenizer& tokenizer, const std::vector<tokenizer::regex::State>& states, Kind kind);
    int target(const Node& node, int token) const {
        const auto edge = std::lower_bound(node.edges.begin(), node.edges.end(), token,
            [](const auto& entry, int id) { return static_cast<int>(entry.token) < id; });
        return edge != node.edges.end() && static_cast<int>(edge->token) == token
            ? edge->next : node.fallback;
    }
    std::vector<Node> nodes_{1};
    bool uses_wide_mask_ = false;
    Kind kind_ = Kind::ebnf;
};

inline bool constraint_eos(int token) {
    return token == end_of_text || token == end_of_message;
}

class ConstraintCursor {
public:
    void candidates(const TokenConstraint& constraint, int remaining, bool force_mask = false) {
        candidates_.clear();
        const auto& node = constraint.nodes_[node_];
        const int limit = constraint.is_triggered() ? std::numeric_limits<int>::max() : remaining;
        const auto allowed = [&](int next) {
            return next >= 0 && constraint.nodes_[next].minimum < limit;
        };
        if (node.fallback < 0) {
            for (const auto& edge : node.edges) {
                if (allowed(edge.next)) {
                    candidates_.push_back(edge.token);
                }
            }
        } else {
            auto edge = node.edges.begin();
            for (int token = 0; token < static_cast<int>(model_vocab_size); ++token) {
                int next = node.fallback;
                if (edge != node.edges.end() && static_cast<int>(edge->token) == token) {
                    next = (edge++)->next;
                }
                if (allowed(next)) {
                    candidates_.push_back(token);
                }
            }
        }
        if (node.accepting) {
            candidates_.insert(std::lower_bound(candidates_.begin(), candidates_.end(),
                                                end_of_text), end_of_text);
            candidates_.insert(std::lower_bound(candidates_.begin(), candidates_.end(),
                                                end_of_message), end_of_message);
        }
        mask_.clear();
        if (force_mask || candidates_.size() > constraint_max_candidates) {
            mask_.assign(constraint_mask_words, 0);
            for (const int id : candidates_) {
                mask_[id / 32] |= std::uint32_t{1} << (id % 32);
            }
        }
    }

    const std::vector<int>& ids() const { return candidates_; }
    const std::vector<std::uint32_t>& mask() const { return mask_; }

    void reset_to(const ConstraintCursor& other) { node_ = other.node_; }

    bool commit(const TokenConstraint& constraint, int token, int remaining) {
        const auto& node = constraint.nodes_[node_];
        if (constraint_eos(token) && node.accepting) {
            return true;
        }
        const int target = constraint.target(node, token);
        if (target < 0 || constraint.nodes_[target].minimum >
                (constraint.is_triggered() ? std::numeric_limits<int>::max() - 1 : remaining)) {
            throw std::logic_error("Selected token violates the output constraint");
        }
        node_ = target;
        const auto& next = constraint.nodes_[node_];
        return !constraint.is_triggered() && next.accepting &&
               (remaining == 0 || (next.edges.empty() && next.fallback < 0));
    }

private:
    std::vector<int> candidates_;
    std::vector<std::uint32_t> mask_;
    int node_ = 0;
};

}
