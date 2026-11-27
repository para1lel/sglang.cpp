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
    int minimum_tokens() const;
    bool is_triggered() const;
    bool uses_wide_mask() const;
    bool is_regex() const;
    const char* budget_error() const;

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

    TokenConstraint();
    static std::shared_ptr<const TokenConstraint> compile_regular(
        const Tokenizer& tokenizer, std::string_view pattern, Kind kind);
    static std::shared_ptr<const TokenConstraint> compile_states(
        const Tokenizer& tokenizer, const std::vector<tokenizer::regex::State>& states, Kind kind);
    int target(const Node& node, int token) const;
    std::vector<Node> nodes_{1};
    bool uses_wide_mask_ = false;
    Kind kind_ = Kind::ebnf;
};

bool constraint_eos(int token);

class ConstraintCursor {
public:
    void candidates(const TokenConstraint& constraint, int remaining, bool force_mask = false);

    const std::vector<int>& ids() const;
    const std::vector<std::uint32_t>& mask() const;

    void reset_to(const ConstraintCursor& other);

    bool commit(const TokenConstraint& constraint, int token, int remaining);

private:
    std::vector<int> candidates_;
    std::vector<std::uint32_t> mask_;
    int node_ = 0;
};

}
