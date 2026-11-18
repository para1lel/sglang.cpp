#pragma once

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

namespace sglang {

using TokenId = std::uint32_t;
inline constexpr TokenId model_vocab_size = 248320;
inline constexpr TokenId end_of_text = 248044;
inline constexpr TokenId end_of_message = 248046;
inline constexpr std::size_t checkpoint_context = 262144;
inline constexpr std::string_view model_name = "Qwen3.8-27B-FP8";

class Tokenizer {
public:
    explicit Tokenizer(const std::filesystem::path& tokenizer_json);
    ~Tokenizer();

    std::vector<TokenId> encode(std::string_view input,
        std::size_t limit = std::numeric_limits<std::size_t>::max(),
        bool split_special = false) const;
    std::string_view piece(TokenId id, bool skip_special = false) const;

private:
    struct Model;
    std::unique_ptr<const Model> model_;
};

}
