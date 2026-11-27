#include "sglang/testing/media/decoder_fixture.h"

#include <algorithm>
#include <stdexcept>

namespace media_test {

void require(bool condition, const char* failure) {
    if (!condition) {
        throw std::runtime_error(failure);
    }
}

std::vector<int> prompt() {
    std::vector<int> tokens(125, 8149);
    tokens.insert(tokens.end(), {begin, image, end, 17, begin, video, end, 18,
                                 begin, image, end, 19, 20});
    return tokens;
}

std::vector<DecoderMediaLayout> layouts() {
    return {{DecoderMediaKind::image, 1, 6, 8, {}},
            {DecoderMediaKind::video, 2, 4, 6, {{111, 112}, {113}}},
            {DecoderMediaKind::image, 1, 16, 24, {}}};
}

std::uint16_t sample(int media, int row, int channel) {
    constexpr std::uint16_t positive = 0x3e80;
    return positive + ((851 * media + 47 * row + channel) % 1024);
}

std::pair<int, int> feature(int row) {
    for (const auto& span : spans) {
        if (row >= span.start && row < span.start + span.height * span.width) {
            return {span.item, span.first + row - span.start};
        }
    }
    return {-1, -1};
}

MropePosition axes(int row) {
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

void cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

bool equal(MropePosition left, MropePosition right) {
    return left.temporal == right.temporal && left.height == right.height &&
        left.width == right.width;
}

}
