#pragma once

#include "sglang/media_preprocess.h"

#include <cstdint>
#include <vector>

namespace sglang::server {

inline constexpr int min_media_dimension = 16;
inline constexpr int max_media_dimension = 1024;
inline constexpr int max_selected_frames = 32;

struct DecodedMedia {
    int height = 0;
    int width = 0;
    std::vector<std::uint8_t> rgb;
    VideoMetadata video;
};

DecodedMedia decode_png(const std::vector<std::uint8_t>& bytes);
DecodedMedia decode_mp4(const std::vector<std::uint8_t>& bytes, int selected_frames);

}
