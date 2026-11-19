#pragma once

#include "sglang/decoder_media.h"

#include <algorithm>
#include <array>
#include <stdexcept>
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

inline void require(bool condition, const char* failure) {
    if (!condition) {
        throw std::runtime_error(failure);
    }
}

inline std::vector<int> prompt() {
    std::vector<int> tokens(125, 8149);
    tokens.insert(tokens.end(), {begin, image, end, 17, begin, video, end, 18,
                                 begin, image, end, 19, 20});
    return tokens;
}

inline std::vector<DecoderMediaLayout> layouts() {
    return {{DecoderMediaKind::image, 1, 6, 8, {}},
            {DecoderMediaKind::video, 2, 4, 6, {{111, 112}, {113}}},
            {DecoderMediaKind::image, 1, 16, 24, {}}};
}

inline std::pair<int, int> feature(int row) {
    for (const auto& span : spans) {
        if (row >= span.start && row < span.start + span.height * span.width) {
            return {span.item, span.first + row - span.start};
        }
    }
    return {-1, -1};
}

inline MropePosition axes(int row) {
    int scalar = row;
    for (const auto& span : spans) {
        if (row < span.start) {
            break;
        }
        const int local = row - span.start;
        if (local < span.height * span.width) {
            return {span.position, span.position + local / span.width,
                    span.position + local % span.width};
        }
        scalar -= span.height * span.width - std::max(span.height, span.width);
    }
    return {scalar, scalar, scalar};
}

inline void cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

inline bool equal(MropePosition left, MropePosition right) {
    return left.temporal == right.temporal && left.height == right.height &&
        left.width == right.width;
}

}
