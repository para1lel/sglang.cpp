#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace sglang {

struct Utf8Lead {
    unsigned size;
    unsigned low;
    unsigned high;
};

constexpr Utf8Lead utf8_lead(unsigned byte) {
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

std::u32string read_utf8(std::string_view bytes);
bool valid_utf8(std::string_view bytes);
void append_utf8(std::string& output, char32_t scalar);

class Utf8Stream {
public:
    std::string push(std::string_view bytes);
    std::string finish();

private:
    std::array<char, 4> pending_{};
    unsigned size_ = 0;
    unsigned expected_ = 0;
    unsigned low_ = 0;
    unsigned high_ = 0;
};

}
