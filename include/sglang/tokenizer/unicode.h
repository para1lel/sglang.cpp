#pragma once

#include <functional>
#include <string>
#include <string_view>

namespace sglang::tokenizer {

std::string normalize_nfc(std::string_view bytes);
void split_pieces(std::string_view bytes,
                  const std::function<void(std::string_view)>& consume);

}
