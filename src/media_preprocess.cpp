#include "sglang/media_preprocess.h"
#include "sglang/media_resize.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace sglang {
namespace {

using media_detail::Size;
using media_detail::Resize;
using media_detail::ResizeFilter;
using media_detail::rgb_bytes;

constexpr int patch_size = 16;
constexpr int merge_size = 2;
constexpr int temporal_size = 2;
constexpr int spatial_factor = patch_size * merge_size;
constexpr int max_dimension = 16384;
constexpr int max_source_frames = 1000000000;
constexpr int max_selected_frames = 768;
constexpr std::size_t byte_limit = 256 * 1024 * 1024;

void validate_rgb(const DecodedRgb& rgb) {
    if (rgb.height <= 0 || rgb.width <= 0 || rgb.height > max_dimension ||
        rgb.width > max_dimension || rgb.frame_count <= 0 || !rgb.data) {
        throw std::invalid_argument("Invalid decoded RGB dimensions, frame count or data");
    }
    const auto frame_bytes = rgb_bytes({rgb.height, rgb.width});
    if (static_cast<std::size_t>(rgb.frame_count) > byte_limit / frame_bytes) {
        throw std::length_error("Decoded RGB exceeds the byte limit");
    }
    if (rgb.size_bytes != frame_bytes * rgb.frame_count) {
        throw std::invalid_argument("Decoded RGB byte length does not match its shape");
    }
}

void validate_source(int count, double fps) {
    if (count < temporal_size || count > max_source_frames ||
        !std::isfinite(fps) || fps < 0.001 || fps > 1000000) {
        throw std::invalid_argument("Invalid source video frame count or FPS");
    }
}

int rounded_multiple(int value, int unit) {
    const int quotient = value / unit;
    const int remainder = value % unit;
    return unit * (quotient + (2 * remainder > unit ||
                               (2 * remainder == unit && quotient % 2)));
}

Size smart_resize(Size input, int factor, int frames, int rounded_frames,
                  double min_pixels, double max_pixels, bool min_factor) {
    if (std::max(input.height, input.width) > 200 * std::min(input.height, input.width)) {
        throw std::invalid_argument("Media aspect ratio exceeds 200");
    }
    const int min_rounded = min_factor ? factor : 0;
    Size output{std::max(min_rounded, rounded_multiple(input.height, factor)),
                std::max(min_rounded, rounded_multiple(input.width, factor))};
    const double rounded_area = static_cast<double>(rounded_frames) * output.height * output.width;
    const double source_area = static_cast<double>(frames) * input.height * input.width;
    if (rounded_area > max_pixels) {
        const double scale = std::sqrt(source_area / max_pixels);
        output = {static_cast<int>(std::floor(input.height / scale / factor)) * factor,
                  static_cast<int>(std::floor(input.width / scale / factor)) * factor};
        if (!min_factor) {
            output = {std::max(output.height, factor), std::max(output.width, factor)};
        }
    } else if (rounded_area < min_pixels) {
        const double scale = std::sqrt(min_pixels / source_area);
        output = {static_cast<int>(std::ceil(input.height * scale / factor)) * factor,
                  static_cast<int>(std::ceil(input.width * scale / factor)) * factor};
    }
    if (output.height <= 0 || output.width <= 0) {
        throw std::invalid_argument("Media resize produced an empty dimension");
    }
    return output;
}

Size checkpoint_video_size(Size input, int count) {
    if (std::min(input.height, input.width) < spatial_factor) {
        const double scale = double(spatial_factor) / std::min(input.height, input.width);
        input = {static_cast<int>(input.height * scale), static_cast<int>(input.width * scale)};
    }
    return smart_resize(input, spatial_factor, count, rounded_multiple(count, temporal_size),
                        4096, 25165824, false);
}

Size sglang_video_size(Size input, int count) {
    const double max_pixels = std::max(
        std::min(768.0 * 28 * 28, 90316800.0 / count * temporal_size), 105369.0);
    return smart_resize(input, 28, 1, 1, 128 * 28 * 28, max_pixels, true);
}

std::size_t checked_patch_values(Size output, int groups) {
    const auto frame_bytes = rgb_bytes(output);
    if (frame_bytes > byte_limit / sizeof(float) / temporal_size / groups) {
        throw std::length_error("Normalized patches exceed the byte limit");
    }
    return frame_bytes * temporal_size * groups;
}

void pack_plane(const std::uint8_t* rgb, Size size, int plane, float* output) {
    constexpr int patch_area = patch_size * patch_size;
    for (int block_y = 0; block_y < size.height; block_y += spatial_factor) {
        for (int block_x = 0; block_x < size.width; block_x += spatial_factor) {
            for (int local_y = 0; local_y < spatial_factor; local_y += patch_size) {
                for (int local_x = 0; local_x < spatial_factor; local_x += patch_size) {
                    const auto start = (static_cast<std::size_t>(block_y + local_y) *
                                        size.width + block_x + local_x) * 3;
                    for (int channel = 0; channel < 3; ++channel) {
                        auto* patch = output + (temporal_size * channel + plane) * patch_area;
                        for (int row = 0; row < patch_size; ++row) {
                            auto* row_rgb = rgb + start + row * size.width * 3 + channel;
                            for (int column = 0; column < patch_size; ++column) {
                                *patch++ = (row_rgb[column * 3] - 127.5f) / 127.5f;
                            }
                        }
                    }
                    output += MediaPatches::patch_width;
                }
            }
        }
    }
}

MediaPatches prepare(const DecodedRgb& rgb, const std::vector<int>& decoded_indices,
                     Size intermediate, Size output) {
    const Size input{rgb.height, rgb.width};
    const int count = static_cast<int>(decoded_indices.size());
    const int groups = (count + temporal_size - 1) / temporal_size;
    const auto values = checked_patch_values(output, groups);
    Resize first(input, intermediate, ResizeFilter::Bilinear, byte_limit);
    Resize last(intermediate, output, ResizeFilter::Bicubic, byte_limit);
    MediaPatches result;
    result.grid_thw = {groups, output.height / patch_size, output.width / patch_size};
    result.pixels.resize(values);
    const auto frame_bytes = rgb_bytes(input);
    const auto group_values = values / groups;
    for (int index = 0; index < count; ++index) {
        const auto* data = last.apply(first.apply(rgb.data + decoded_indices[index] * frame_bytes));
        auto* patch = result.pixels.data() + (index / temporal_size) * group_values;
        const int plane = index % temporal_size;
        pack_plane(data, output, plane, patch);
        if (plane == 0 && index == count - 1) {
            pack_plane(data, output, 1, patch);
        }
    }
    return result;
}

void validate_indices(const DecodedRgb& video, const VideoMetadata& metadata) {
    const auto& indices = metadata.frame_indices;
    if (video.frame_count < temporal_size || video.frame_count > max_selected_frames ||
        indices.size() != static_cast<std::size_t>(video.frame_count) ||
        indices.front() < 0 || indices.back() >= metadata.source_frame_count ||
        !std::is_sorted(indices.begin(), indices.end())) {
        throw std::invalid_argument("Invalid preselected video frame indices");
    }
}

std::vector<double> timestamp_centers(const std::vector<int>& indices, double fps) {
    std::vector<double> centers;
    centers.reserve((indices.size() + 1) / temporal_size);
    for (std::size_t first = 0; first < indices.size(); first += temporal_size) {
        const auto last = std::min(first + temporal_size - 1, indices.size() - 1);
        centers.push_back((indices[first] / fps + indices[last] / fps) / 2);
    }
    return centers;
}

}

std::vector<int> sample_video_frames(int source_frame_count, double source_fps) {
    validate_source(source_frame_count, source_fps);
    const auto max_frames = std::min(max_selected_frames, source_frame_count);
    const double desired = std::min(std::max(source_frame_count / source_fps * 2, 4.0),
                                    double(max_frames));
    const auto count = static_cast<int>(desired / temporal_size) * temporal_size;
    const auto step = double(source_frame_count - 1) / (count - 1);
    std::vector<int> indices(count);
    for (int index = 0; index < count - 1; ++index) {
        indices[index] = static_cast<int>(index * step);
    }
    indices.back() = source_frame_count - 1;
    return indices;
}

MediaPatches preprocess_image(const DecodedRgb& image) {
    validate_rgb(image);
    if (image.frame_count != 1) {
        throw std::invalid_argument("Image preprocessing requires exactly one RGB frame");
    }
    const Size input{image.height, image.width};
    return prepare(image, {0}, input,
                   smart_resize(input, spatial_factor, 1, 1, 65536, 16777216, false));
}

MediaPatches preprocess_video(const DecodedRgb& video, const VideoMetadata& metadata,
                              VideoPreprocess profile) {
    validate_rgb(video);
    validate_source(metadata.source_frame_count, metadata.source_fps);
    const bool legacy = profile == VideoPreprocess::SglangDefault;
    if (!legacy && profile != VideoPreprocess::Checkpoint) {
        throw std::invalid_argument("Invalid video preprocessing profile");
    }
    const bool all_frames = legacy && metadata.frame_indices.empty();
    if (all_frames) {
        if (video.frame_count != metadata.source_frame_count) {
            throw std::invalid_argument("Decoded video frame count differs from the source count");
        }
    } else {
        validate_indices(video, metadata);
    }
    auto selected = legacy ? sample_video_frames(metadata.source_frame_count, metadata.source_fps)
                           : metadata.frame_indices;
    if (legacy && !all_frames && selected != metadata.frame_indices) {
        throw std::invalid_argument("Decoded video indices differ from the default sample");
    }
    auto decoded_indices = selected;
    if (!all_frames) {
        std::iota(decoded_indices.begin(), decoded_indices.end(), 0);
    }
    const Size input{video.height, video.width};
    const int count = static_cast<int>(selected.size());
    const Size intermediate = legacy ? sglang_video_size(input, count) : input;
    auto result = prepare(video, decoded_indices, intermediate,
                          checkpoint_video_size(intermediate, count));
    result.timestamps_seconds = timestamp_centers(selected, metadata.source_fps);
    result.frame_indices = std::move(selected);
    return result;
}

}
