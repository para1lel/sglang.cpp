#pragma once

#include <string>
#include <string_view>

namespace sglang::tokenizer {

std::string sha256(std::string_view bytes);

}
