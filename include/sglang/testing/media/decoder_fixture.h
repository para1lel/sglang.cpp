#pragma once

#include "sglang/decoder_media.h"

#include <array>
#include <utility>

#include <cuda_runtime_api.h>

namespace media_test {

using sglang::DecoderMediaKind;
using sglang::DecoderMediaLayout;
using sglang::MropePosition;

inline constexpr int begin = 248053;
inline constexpr int end = 248054;
inline constexpr int image = 248056;
inline constexpr int video = 248057;
inline constexpr int physical = 262;
inline constexpr int rotary = 164;
inline constexpr int width = sglang::decoder_media_feature_width;

struct Span {
    int start;
    int height;
    int width;
    int item;
    int first;
    int position;
};

inline constexpr std::array spans{
    Span{126, 3, 4, 0, 0, 126},
    Span{144, 2, 3, 1, 0, 136},
    Span{153, 2, 3, 1, 6, 142},
    Span{163, 8, 12, 2, 0, 149},
};

void require(bool condition, const char* failure);
std::vector<int> prompt();
std::vector<DecoderMediaLayout> layouts();
std::uint16_t sample(int media, int row, int channel);
std::pair<int, int> feature(int row);
MropePosition axes(int row);
void cuda(cudaError_t status);
bool equal(MropePosition left, MropePosition right);

}
