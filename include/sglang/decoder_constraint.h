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

class TokenConstraint {
public:
    static std::shared_ptr<const TokenConstraint> compile(const Tokenizer& tokenizer,
                                                          std::string_view ebnf);
    static std::shared_ptr<const TokenConstraint> compile_regex(const Tokenizer& tokenizer,
                                                                std::string_view pattern);
    static std::shared_ptr<const TokenConstraint> compile_json_schema(
        const Tokenizer& tokenizer, const json::Value& schema);
    int minimum_tokens() const { return nodes_.front().minimum; }
    bool uses_wide_mask() const { return uses_wide_mask_; }
    bool is_regex() const { return kind_ == Kind::regex; }
    const char* budget_error() const {
        switch (kind_) {
            case Kind::regex: return "max_tokens is too small to complete the regex";
            case Kind::json_schema: return "max_tokens is too small to complete the JSON Schema";
            case Kind::ebnf: return "max_tokens is too small to complete an ebnf choice";
        }
        throw std::logic_error("Unknown output constraint");
    }

private:
    friend class ConstraintCursor;

    enum class Kind { ebnf, regex, json_schema };

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
    static std::shared_ptr<const TokenConstraint> compile_regular(
        const Tokenizer& tokenizer, std::string_view pattern, Kind kind);
    std::vector<Node> nodes_{1};
    bool uses_wide_mask_ = false;
    Kind kind_ = Kind::ebnf;
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
        mask_.clear();
        if (candidates_.size() > constraint_max_candidates) {
            mask_.assign(constraint_mask_words, 0);
            for (const int id : candidates_) {
                mask_[id / 32] |= std::uint32_t{1} << (id % 32);
            }
        }
    }

    const std::vector<int>& ids() const { return candidates_; }
    const std::vector<std::uint32_t>& mask() const { return mask_; }

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
    std::vector<std::uint32_t> mask_;
    int node_ = 0;
};

}
