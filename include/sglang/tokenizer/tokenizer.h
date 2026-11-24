#pragma once

#include "sglang/runtime/model/qwen_text_model.h"
#include "sglang/runtime/model/qwen_tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

namespace sglang {

using TokenId = std::uint32_t;
inline constexpr TokenId model_vocab_size = decoder::qwen_text_model.vocab_size;
inline constexpr TokenId end_of_text = model::qwen_end_of_text;
inline constexpr TokenId end_of_message = model::qwen_end_of_message;
inline constexpr std::size_t checkpoint_context = decoder::QwenTextModel::max_position_embeddings;
inline constexpr TokenId checkpoint_base_vocab_size = model::qwen_tokenizer_base_vocab;
inline constexpr std::string_view model_name = model::qwen_served_name;

class Tokenizer {
public:
    explicit Tokenizer(const std::filesystem::path& tokenizer_json);
    ~Tokenizer();

    std::vector<TokenId> encode(std::string_view input,
        std::size_t limit = std::numeric_limits<std::size_t>::max(),
        bool split_special = false) const;
    std::string_view piece(TokenId id, bool skip_special = false) const;
    void matching_pieces(std::string_view bytes, std::vector<TokenId>& ids) const;

private:
    struct Model;
    std::unique_ptr<const Model> model_;
};

}
