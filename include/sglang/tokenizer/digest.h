#pragma once

#include <array>
#include <string>
#include <string_view>

namespace sglang::tokenizer {

using Sha256Digest = std::array<char, 32>;
Sha256Digest sha256_digest(std::string_view bytes);
std::string format_sha256(const Sha256Digest& digest);
std::string sha256(std::string_view bytes);

}
