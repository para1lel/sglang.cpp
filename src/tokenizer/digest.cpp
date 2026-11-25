#include "sglang/tokenizer/digest.h"

#include <array>
#include <cstdint>

namespace sglang::tokenizer {
namespace {

using Word = std::uint32_t;
using State = std::array<Word, 8>;

constexpr Word rotate_right(Word word, unsigned shift) {
    return (word >> shift) | (word << (32 - shift));
}

constexpr std::array<Word, 64> round_constants{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
};

void compress(State& state, const char* block) {
    std::array<Word, 64> schedule{};
    for (std::size_t index = 0; index < 64; ++index) {
        if (index < 16) {
            for (std::size_t byte = 0; byte < 4; ++byte) {
                schedule[index] = (schedule[index] << 8) |
                    static_cast<unsigned char>(block[index * 4 + byte]);
            }
        } else {
            const auto earlier = schedule[index - 15];
            const auto recent = schedule[index - 2];
            const auto sigma_0 =
                rotate_right(earlier, 7) ^ rotate_right(earlier, 18) ^ (earlier >> 3);
            const auto sigma_1 =
                rotate_right(recent, 17) ^ rotate_right(recent, 19) ^ (recent >> 10);
            schedule[index] = schedule[index - 16] + sigma_0 + schedule[index - 7] + sigma_1;
        }
    }
    auto work = state;
    for (std::size_t index = 0; index < 64; ++index) {
        const auto [first, second, third, fourth, fifth, sixth, seventh, eighth] = work;
        const Word sigma_0 =
            rotate_right(first, 2) ^ rotate_right(first, 13) ^ rotate_right(first, 22);
        const Word sigma_1 =
            rotate_right(fifth, 6) ^ rotate_right(fifth, 11) ^ rotate_right(fifth, 25);
        const Word choose = (fifth & sixth) ^ (~fifth & seventh);
        const Word majority = (first & second) ^ (first & third) ^ (second & third);
        const Word mix = eighth + sigma_1 + choose + round_constants[index] + schedule[index];
        work = {mix + sigma_0 + majority, first, second, third,
                fourth + mix, fifth, sixth, seventh};
    }
    for (std::size_t index = 0; index < state.size(); ++index) {
        state[index] += work[index];
    }
}

}

Sha256Digest sha256_digest(std::string_view bytes) {
    State state{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
        0x1f83d9ab, 0x5be0cd19,
    };
    const std::uint64_t bits = static_cast<std::uint64_t>(bytes.size()) * 8;
    while (bytes.size() >= 64) {
        compress(state, bytes.data());
        bytes.remove_prefix(64);
    }
    std::array<char, 128> tail{};
    bytes.copy(tail.data(), bytes.size());
    tail[bytes.size()] = static_cast<char>(0x80);
    const std::size_t padded = bytes.size() < 56 ? 64 : 128;
    for (std::size_t byte = 0; byte < 8; ++byte) {
        tail[padded - 1 - byte] = static_cast<char>(bits >> (8 * byte));
    }
    compress(state, tail.data());
    if (padded > 64) {
        compress(state, tail.data() + 64);
    }
    Sha256Digest digest;
    auto* out = digest.data();
    for (const auto word : state) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            *out++ = static_cast<char>(word >> shift);
        }
    }
    return digest;
}

std::string format_sha256(const Sha256Digest& digest) {
    constexpr std::string_view hex = "0123456789abcdef";
    std::string output(2 * digest.size(), '\0');
    auto* out = output.data();
    for (const unsigned char byte : digest) {
        *out++ = hex[byte >> 4];
        *out++ = hex[byte & 0xf];
    }
    return output;
}

std::string sha256(std::string_view bytes) {
    return format_sha256(sha256_digest(bytes));
}

}
