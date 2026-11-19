#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sglang {

struct DecodedRgb {
    const std::uint8_t* data = nullptr;
    std::size_t size_bytes = 0;
    int height = 0;
    int width = 0;
    int frame_count = 0;
};

enum class VideoPreprocess { SglangDefault, Checkpoint };

struct VideoMetadata {
    int source_frame_count = 0;
    double source_fps = 0;
    std::vector<int> frame_indices;
};

struct MediaPatches {
    static constexpr int patch_width = 1536;

    std::array<int, 3> grid_thw{};
    std::vector<float> pixels;
    std::vector<int> frame_indices;
    std::vector<double> timestamps_seconds;
};

std::vector<int> sample_video_frames(int source_frame_count, double source_fps);
MediaPatches preprocess_image(const DecodedRgb& image);
MediaPatches preprocess_video(const DecodedRgb& video, const VideoMetadata& metadata,
                              VideoPreprocess profile);

}
