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

Utf8Lead utf8_lead(unsigned byte);

std::u32string read_utf8(std::string_view bytes);
bool valid_utf8(std::string_view bytes);
std::string repair_utf8(std::string_view bytes);
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
