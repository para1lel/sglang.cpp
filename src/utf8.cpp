#include "sglang/utf8.h"

#include <stdexcept>

namespace sglang {
namespace {
constexpr std::string_view replacement = "\xef\xbf\xbd";
}

Utf8Lead utf8_lead(unsigned byte) {
    if (byte < 0x80) {
        return {1, 0, 0};
    }
    if (byte < 0xc2 || byte > 0xf4) {
        return {0, 0, 0};
    }
    return {
        byte < 0xe0 ? 2U : byte < 0xf0 ? 3U : 4U,
        byte == 0xe0 ? 0xa0U : byte == 0xf0 ? 0x90U : 0x80U,
        byte == 0xed ? 0x9fU : byte == 0xf4 ? 0x8fU : 0xbfU,
    };
}

namespace {

template <bool Decode> unsigned read_scalar(std::string_view bytes, char32_t& scalar) {
    const auto first = static_cast<unsigned char>(bytes.front());
    const auto [size, low, high] = utf8_lead(first);
    if (size == 0 || size > bytes.size()) {
        return 0;
    }
    if constexpr (Decode) {
        scalar = first & (0xffU >> (size == 1 ? 1 : size + 1));
    }
    for (unsigned index = 1; index < size; ++index) {
        const auto byte = static_cast<unsigned char>(bytes[index]);
        if (byte < (index == 1 ? low : 0x80) || byte > (index == 1 ? high : 0xbf)) {
            return 0;
        }
        if constexpr (Decode) {
            scalar = (scalar << 6) | (byte & 0x3fU);
        }
    }
    return size;
}

}

bool valid_utf8(std::string_view bytes) {
    while (!bytes.empty()) {
        char32_t unused = 0;
        const auto size = read_scalar<false>(bytes, unused);
        if (size == 0) {
            return false;
        }
        bytes.remove_prefix(size);
    }
    return true;
}

std::u32string read_utf8(std::string_view bytes) {
    std::u32string output;
    output.reserve(bytes.size());
    while (!bytes.empty()) {
        char32_t scalar = 0;
        const auto size = read_scalar<true>(bytes, scalar);
        if (size == 0) {
            throw std::invalid_argument("Invalid UTF-8 input");
        }
        output.push_back(scalar);
        bytes.remove_prefix(size);
    }
    return output;
}

void append_utf8(std::string& output, char32_t scalar) {
    if (scalar < 0x80) {
        output.push_back(static_cast<char>(scalar));
    } else {
        const unsigned size = scalar < 0x800 ? 2 : scalar < 0x10000 ? 3 : 4;
        const unsigned lead = 0xffU << (8 - size);
        output.push_back(static_cast<char>(lead | (scalar >> (6 * (size - 1)))));
        for (unsigned index = size - 1; index > 0; --index) {
            output.push_back(static_cast<char>(0x80 | ((scalar >> (6 * (index - 1))) & 0x3f)));
        }
    }
}

std::string Utf8Stream::push(std::string_view bytes) {
    std::string output;
    for (unsigned char byte : bytes) {
        if (size_ > 0) {
            if (byte >= low_ && byte <= high_) {
                pending_[size_++] = static_cast<char>(byte);
                low_ = 0x80;
                high_ = 0xbf;
                if (size_ == expected_) {
                    output.append(pending_.data(), size_);
                    size_ = 0;
                }
                continue;
            }
            output += replacement;
            size_ = 0;
        }
        const auto [size, low, high] = utf8_lead(byte);
        if (size == 1) {
            output.push_back(static_cast<char>(byte));
        } else if (size > 1) {
            pending_[0] = static_cast<char>(byte);
            size_ = 1;
            expected_ = size;
            low_ = low;
            high_ = high;
        } else {
            output += replacement;
        }
    }
    return output;
}

std::string Utf8Stream::finish() {
    if (size_ == 0) {
        return {};
    }
    size_ = 0;
    return std::string(replacement);
}

std::string repair_utf8(std::string_view bytes) {
    Utf8Stream stream;
    auto text = stream.push(bytes);
    text += stream.finish();
    return text;
}
}
