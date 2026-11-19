#include "sglang/decoder_constraint.h"

#include "sglang/json.h"
#include "sglang/utf8.h"

#include <map>
#include <string>

namespace sglang {
namespace {

constexpr int max_choices = 32;
constexpr int max_literal_bytes = 256;
constexpr int max_choice_bytes = 2048;
constexpr int max_grammar_bytes = 16384;

constexpr const char* grammar_error =
    "Expected ebnf with one root rule and nonempty double-quoted string alternatives";

void skip_space(std::string_view& text) {
    const auto start = text.find_first_not_of(" \t\r\n");
    text.remove_prefix(start == std::string_view::npos ? text.size() : start);
}

bool consume(std::string_view& text, std::string_view prefix) {
    skip_space(text);
    if (text.substr(0, prefix.size()) != prefix) {
        return false;
    }
    text.remove_prefix(prefix.size());
    return true;
}

std::string literal(std::string_view& text) {
    skip_space(text);
    if (text.empty() || text.front() != '"') {
        throw std::invalid_argument(grammar_error);
    }
    for (std::size_t end = 1; end < text.size(); ++end) {
        if (text[end] == '\\') {
            ++end;
        } else if (text[end] == '"') {
            try {
                const auto parsed = json::parse(text.substr(0, end + 1));
                text.remove_prefix(end + 1);
                return std::string(*parsed.string());
            } catch (const json::ParseError&) {
                break;
            }
        }
    }
    throw std::invalid_argument(grammar_error);
}

std::vector<std::string> choices(std::string_view text) {
    if (text.size() > max_grammar_bytes || !consume(text, "root") || !consume(text, "::=")) {
        throw std::invalid_argument(grammar_error);
    }
    std::vector<std::string> values;
    std::size_t total = 0;
    do {
        auto value = literal(text);
        if (!valid_utf8(value)) {
            throw std::invalid_argument("ebnf literals must contain valid UTF-8");
        }
        total += value.size();
        if (value.empty() || value.size() > max_literal_bytes || total > max_choice_bytes ||
            values.size() == max_choices) {
            throw std::invalid_argument("ebnf accepts up to 32 nonempty literals, "
                                        "256 bytes each and 2048 bytes total");
        }
        values.push_back(std::move(value));
    } while (consume(text, "|"));
    if (!text.empty()) {
        throw std::invalid_argument(grammar_error);
    }
    return values;
}

}

std::shared_ptr<const TokenConstraint> TokenConstraint::compile(const Tokenizer& tokenizer,
                                                               std::string_view ebnf) {
    const auto values = choices(ebnf);
    auto result = std::shared_ptr<TokenConstraint>(new TokenConstraint);
    std::vector<std::map<char, int>> trie(1);
    std::vector<TokenId> tokens;
    for (const auto& value : values) {
        std::vector<int> nodes{0};
        for (const char byte : value) {
            const auto [entry, inserted] =
                trie[nodes.back()].try_emplace(byte, static_cast<int>(trie.size()));
            nodes.push_back(entry->second);
            if (inserted) {
                trie.emplace_back();
                result->nodes_.emplace_back();
            }
        }
        result->nodes_[nodes.back()].accepting = true;
        for (std::size_t offset = 0; offset < value.size(); ++offset) {
            auto& edges = result->nodes_[nodes[offset]].edges;
            for (std::size_t length = 1; length <= value.size() - offset; ++length) {
                tokens.clear();
                tokenizer.matching_pieces(std::string_view(value).substr(offset, length), tokens);
                for (const TokenId token : tokens) {
                    edges.push_back({token, nodes[offset + length]});
                }
            }
        }
    }
    for (auto node = result->nodes_.rbegin(); node != result->nodes_.rend(); ++node) {
        auto& edges = node->edges;
        std::sort(edges.begin(), edges.end(),
                  [](const Edge& left, const Edge& right) { return left.token < right.token; });
        edges.erase(std::unique(edges.begin(), edges.end(),
                               [](const Edge& left, const Edge& right) {
                                   return left.token == right.token;
                               }), edges.end());
        if (edges.size() > constraint_max_candidates - 2) {
            throw std::invalid_argument("ebnf has too many next-token choices");
        }
        if (node->accepting) {
            node->minimum = 0;
        } else {
            for (const auto& edge : edges) {
                const auto next = result->nodes_[edge.next].minimum;
                if (next != std::numeric_limits<int>::max()) {
                    node->minimum = std::min(node->minimum, 1 + next);
                }
            }
        }
    }
    if (result->minimum_tokens() == std::numeric_limits<int>::max()) {
        throw std::invalid_argument("ebnf choices cannot be represented by the tokenizer");
    }
    return result;
}

}
