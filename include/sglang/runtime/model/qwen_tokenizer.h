#pragma once

#include <cstddef>
#include <string_view>

namespace sglang::model {

inline constexpr std::string_view qwen_served_name = "Qwen3.8-27B-FP8";
inline constexpr std::string_view qwen_tokenizer_file = "tokenizer.json";
inline constexpr std::size_t qwen_tokenizer_size = 12809320;
inline constexpr std::string_view qwen_tokenizer_sha256 =
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3";
inline constexpr int qwen_tokenizer_base_vocab = 248044;
inline constexpr int qwen_tokenizer_mapped_vocab = 248077;
inline constexpr int qwen_end_of_text = 248044;
inline constexpr int qwen_end_of_message = 248046;

}
