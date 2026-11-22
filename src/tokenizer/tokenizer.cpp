#include "sglang/tokenizer/tokenizer.h"

#include "sglang/json.h"
#include "sglang/tokenizer/digest.h"
#include "sglang/tokenizer/unicode.h"
#include "sglang/utf8.h"

#include <array>
#include <fstream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

namespace sglang {
namespace {

constexpr std::size_t file_size = 12809320;
constexpr std::string_view file_sha256 =
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3";
constexpr TokenId mapped_vocab_size = 248077;

json::Value read_checkpoint(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::string bytes(file_size, '\0');
    if (!input.read(bytes.data(), bytes.size()) ||
        input.peek() != std::char_traits<char>::eof() || input.bad() ||
        tokenizer::sha256(bytes) != file_sha256) {
        throw std::runtime_error("Cannot read pinned tokenizer.json or SHA-256 differs");
    }
    return json::parse(bytes);
}

std::uint64_t pair_key(TokenId left, TokenId right) {
    return (static_cast<std::uint64_t>(left) << 32) | right;
}

std::array<unsigned char, 324> inverse_alphabet() {
    std::array<unsigned char, 324> inverse{};
    unsigned extra = 256;
    for (unsigned byte = 0; byte < 256; ++byte) {
        const bool direct = (byte >= 33 && byte <= 126) ||
            (byte >= 161 && byte <= 172) || byte >= 174;
        inverse[direct ? byte : extra++] = static_cast<unsigned char>(byte);
    }
    return inverse;
}
}

struct Tokenizer::Model {
    struct Merge {
        std::uint32_t rank;
        TokenId result;
    };

    explicit Model(const json::Object& root);
    void bpe(std::string_view bytes, std::vector<TokenId>& output, std::size_t limit) const;

    std::vector<std::string> pieces{mapped_vocab_size};
    std::array<bool, mapped_vocab_size - checkpoint_base_vocab_size> special{};
    std::array<TokenId, 256> byte_ids{};
    std::vector<TokenId> added;
    std::unordered_map<std::uint64_t, Merge> merges;
    mutable std::once_flag piece_index_ready;
    mutable std::unordered_multimap<std::string_view, TokenId> piece_index;
};

Tokenizer::Model::Model(const json::Object& root) {
    const auto inverse = inverse_alphabet();
    const auto& model = *root.at("model").object();
    const auto& vocab = *model.at("vocab").object();
    std::unordered_map<std::string_view, TokenId> ids;
    ids.reserve(vocab.size());
    for (const auto& [text, value] : vocab) {
        const auto id = static_cast<TokenId>(*value.unsigned_integer());
        ids.emplace(text, id);
        auto& bytes = pieces[id];
        for (const auto scalar : read_utf8(text)) {
            bytes.push_back(static_cast<char>(inverse.at(scalar)));
        }
        if (bytes.size() == 1) {
            byte_ids[static_cast<unsigned char>(bytes.front())] = id;
        }
    }
    for (const auto& value : *root.at("added_tokens").array()) {
        const auto& entry = *value.object();
        const auto id = static_cast<TokenId>(*entry.at("id").unsigned_integer());
        special[id - checkpoint_base_vocab_size] = *entry.at("special").boolean();
        pieces[id] = *entry.at("content").string();
        added.push_back(id);
    }
    const auto& rules = *model.at("merges").array();
    merges.reserve(rules.size());
    for (std::uint32_t rank = 0; rank < rules.size(); ++rank) {
        const auto rule = *rules[rank].string();
        const auto divider = rule.find(' ');
        const auto left = rule.substr(0, divider);
        const auto right = rule.substr(divider + 1);
        const auto key = pair_key(ids.at(left), ids.at(right));
        auto combined = std::string(left);
        combined += right;
        const auto result = ids.at(combined);
        merges.emplace(key, Merge{rank, result});
    }
}

void Tokenizer::Model::bpe(std::string_view bytes, std::vector<TokenId>& output,
                           std::size_t limit) const {
    const auto append = [&](TokenId id) {
        if (output.size() == limit) {
            throw std::length_error("Prompt exceeds the available token budget");
        }
        output.push_back(id);
    };
    if (bytes.size() == 1) {
        append(byte_ids[static_cast<unsigned char>(bytes.front())]);
        return;
    }
    struct Node {
        TokenId id;
        std::size_t previous;
        std::size_t next;
    };
    const auto end = bytes.size();
    std::vector<Node> nodes;
    nodes.reserve(end);
    for (std::size_t index = 0; index < end; ++index) {
        nodes.push_back({byte_ids[static_cast<unsigned char>(bytes[index])],
                         index == 0 ? end : index - 1, index + 1});
    }
    using Candidate =
        std::tuple<std::uint32_t, std::size_t, std::size_t, TokenId, TokenId, TokenId>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> pending;
    const auto push_pair = [&](std::size_t index) {
        if (index == end || nodes[index].next == end) {
            return;
        }
        const auto next = nodes[index].next;
        const auto left = nodes[index].id;
        const auto right = nodes[next].id;
        const auto merge = merges.find(pair_key(left, right));
        if (merge != merges.end()) {
            pending.emplace(merge->second.rank, index, next, left, right, merge->second.result);
        }
    };
    for (std::size_t index = 0; index < end; ++index) {
        push_pair(index);
    }
    while (!pending.empty()) {
        const auto [rank, index, next, left, right, result] = pending.top();
        pending.pop();
        auto& node = nodes[index];
        if (node.next != next || node.id != left || nodes[next].id != right) {
            continue;
        }
        node.id = result;
        node.next = nodes[next].next;
        nodes[next].next = end;
        if (node.next != end) {
            nodes[node.next].previous = index;
        }
        push_pair(node.previous);
        push_pair(index);
    }
    for (std::size_t index = 0; index != end; index = nodes[index].next) {
        append(nodes[index].id);
    }
}

Tokenizer::Tokenizer(const std::filesystem::path& path)
    : model_(std::make_unique<Model>(*read_checkpoint(path).object())) {}

Tokenizer::~Tokenizer() = default;

std::vector<TokenId> Tokenizer::encode(std::string_view input, std::size_t limit,
                                       bool split_special) const {
    std::vector<TokenId> output;
    const auto ordinary = [&](std::string_view span) {
        tokenizer::split_pieces(span, [&](std::string_view bytes) {
            model_->bpe(bytes, output, limit);
        });
    };
    std::size_t start = 0;
    for (auto offset = input.find('<'); offset != std::string_view::npos;) {
        TokenId match = 0;
        std::size_t length = 0;
        for (const auto id : model_->added) {
            const auto spelling = piece(id, split_special);
            if (spelling.size() > length && input.compare(offset, spelling.size(), spelling) == 0) {
                match = id;
                length = spelling.size();
            }
        }
        if (length == 0) {
            offset = input.find('<', offset + 1);
            continue;
        }
        ordinary(input.substr(start, offset - start));
        if (output.size() == limit) {
            throw std::length_error("Prompt exceeds the available token budget");
        }
        output.push_back(match);
        start = offset + length;
        offset = input.find('<', start);
    }
    ordinary(input.substr(start));
    return output;
}

std::string_view Tokenizer::piece(TokenId id, bool skip_special) const {
    if (id >= mapped_vocab_size ||
        (skip_special && id >= checkpoint_base_vocab_size &&
         model_->special[id - checkpoint_base_vocab_size])) {
        return {};
    }
    return model_->pieces[id];
}

void Tokenizer::matching_pieces(std::string_view bytes, std::vector<TokenId>& ids) const {
    std::call_once(model_->piece_index_ready, [this] {
        decltype(model_->piece_index) index;
        index.reserve(mapped_vocab_size);
        for (TokenId token = 0; token < mapped_vocab_size; ++token) {
            const auto bytes = piece(token, true);
            if (!bytes.empty() && token != end_of_text && token != end_of_message) {
                index.emplace(bytes, token);
            }
        }
        model_->piece_index = std::move(index);
    });
    const auto [first, last] = model_->piece_index.equal_range(bytes);
    for (auto entry = first; entry != last; ++entry) {
        ids.push_back(entry->second);
    }
}

}
