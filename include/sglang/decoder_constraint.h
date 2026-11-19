#pragma once

#include "sglang/tokenizer/tokenizer.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace sglang {

inline constexpr int constraint_max_candidates = 4096;

class ConstraintCursor;

class TokenConstraint {
public:
    static std::shared_ptr<const TokenConstraint> compile(const Tokenizer& tokenizer,
                                                          std::string_view ebnf);
    int minimum_tokens() const { return nodes_.front().minimum; }

private:
    friend class ConstraintCursor;

    struct Edge {
        TokenId token;
        int next;
    };
    struct Node {
        std::vector<Edge> edges;
        bool accepting = false;
        int minimum = std::numeric_limits<int>::max();
    };

    TokenConstraint() = default;
    std::vector<Node> nodes_{1};
};

inline bool constraint_eos(int token) {
    return token == end_of_text || token == end_of_message;
}

class ConstraintCursor {
public:
    void candidates(const TokenConstraint& constraint, int remaining) {
        candidates_.clear();
        const auto& node = constraint.nodes_[node_];
        for (const auto& edge : node.edges) {
            if (constraint.nodes_[edge.next].minimum < remaining) {
                candidates_.push_back(edge.token);
            }
        }
        if (node.accepting) {
            candidates_.insert(std::lower_bound(candidates_.begin(), candidates_.end(),
                                                end_of_text), end_of_text);
            candidates_.insert(std::lower_bound(candidates_.begin(), candidates_.end(),
                                                end_of_message), end_of_message);
        }
    }

    const std::vector<int>& ids() const { return candidates_; }

    bool commit(const TokenConstraint& constraint, int token, int remaining) {
        const auto& node = constraint.nodes_[node_];
        if (constraint_eos(token) && node.accepting) {
            return true;
        }
        const auto edge = std::lower_bound(node.edges.begin(), node.edges.end(), token,
            [](const auto& entry, int id) { return static_cast<int>(entry.token) < id; });
        if (edge == node.edges.end() || static_cast<int>(edge->token) != token ||
            constraint.nodes_[edge->next].minimum > remaining) {
            throw std::logic_error("Selected token violates the output constraint");
        }
        node_ = edge->next;
        const auto& next = constraint.nodes_[node_];
        return next.accepting && (remaining == 0 || next.edges.empty());
    }

private:
    std::vector<int> candidates_;
    int node_ = 0;
};

}
