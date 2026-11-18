#include "sglang/utf8.h"

#include <stdexcept>

namespace sglang {
namespace {
constexpr std::string_view replacement = "\xef\xbf\xbd";
}

bool valid_utf8(std::string_view bytes) {
    while (!bytes.empty()) {
        const auto [size, low, high] = utf8_lead(static_cast<unsigned char>(bytes.front()));
        if (!size || size > bytes.size()) {
            return false;
        }
        for (unsigned index = 1; index < size; ++index) {
            const auto byte = static_cast<unsigned char>(bytes[index]);
            if (byte < (index == 1 ? low : 0x80) || byte > (index == 1 ? high : 0xbf)) {
                return false;
            }
        }
        bytes.remove_prefix(size);
    }
    return true;
}

std::u32string read_utf8(std::string_view bytes) {
    std::u32string output;
    output.reserve(bytes.size());
    for (std::size_t offset = 0; offset < bytes.size();) {
        const unsigned first = static_cast<unsigned char>(bytes[offset]);
        const auto [size, low, high] = utf8_lead(first);
        if (!size || size > bytes.size() - offset) {
            throw std::invalid_argument("Invalid UTF-8 input");
        }
        char32_t scalar = first & (0xffU >> (size == 1 ? 1 : (size + 1)));
        for (unsigned index = 1; index < size; ++index) {
            const unsigned byte = static_cast<unsigned char>(bytes[offset + index]);
            if (byte < (index == 1 ? low : 0x80) || byte > (index == 1 ? high : 0xbf)) {
                throw std::invalid_argument("Invalid UTF-8 input");
            }
            scalar = (scalar << 6) | (byte & 0x3fU);
        }
        output.push_back(scalar);
        offset += size;
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

}
