#include "sglang/media_preprocess.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* detail) {
    if (!condition) {
        throw std::runtime_error(detail);
    }
}

template <typename Error, typename Operation>
void require_rejection(Operation operation, const char* detail) {
    try {
        operation();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(detail);
}

void check_contract() {
    constexpr int limit_height = 3328;
    constexpr int safe_width = 3360;
    constexpr int large_width = 3392;
    std::vector<std::uint8_t> source(std::size_t{limit_height} * large_width * 3, 255);
    {
        const auto bytes = std::size_t{limit_height} * safe_width * 3;
        const auto result = sglang::preprocess_image(
            {source.data(), bytes, limit_height, safe_width, 1});
        require(result.grid_thw == std::array{1, 208, 210}, "Near-limit image grid differs");
        require(result.pixels.size() * sizeof(float) == 268369920,
                "Near-limit normalized byte size differs");
        require(result.pixels.front() == 1.0f && result.pixels.back() == 1.0f,
                "Near-limit image content differs");
    }
    require_rejection<std::length_error>([&] {
        sglang::preprocess_image({source.data(), source.size(), limit_height, large_width, 1});
    }, "Oversized normalized patches were accepted");

    constexpr int selected = 768;
    constexpr int source_count = 1000000000;
    constexpr double min_fps = 0.001;
    const auto indices = sglang::sample_video_frames(source_count, min_fps);
    require(indices.size() == selected && indices.front() == 0 &&
                indices.back() == source_count - 1, "Sample limits differ");
    constexpr std::size_t frame_bytes = 32 * 32 * 3;
    std::vector<std::uint8_t> frames(selected * frame_bytes, 0);
    std::fill(frames.end() - frame_bytes, frames.end(), 255);
    const sglang::DecodedRgb rgb{frames.data(), frames.size(), 32, 32, selected};
    const sglang::VideoMetadata metadata{source_count, min_fps, indices};
    const auto video = sglang::preprocess_video(rgb, metadata, sglang::VideoPreprocess::Checkpoint);
    require(video.grid_thw == std::array{384, 2, 2} && video.frame_indices == indices &&
                video.timestamps_seconds.size() == 384, "Exact selected-frame cap differs");
    require(video.timestamps_seconds.back() ==
                (indices[selected - 2] / min_fps + indices.back() / min_fps) / 2,
            "Large source-index timestamp differs");
    std::fill(frames.begin(), frames.end(), 127);
    require(video.pixels.front() == -1.0f && video.pixels.back() == 1.0f,
            "Owned patch values changed with the borrowed buffer");

    require_rejection<std::invalid_argument>([&] {
        sglang::preprocess_video(rgb, metadata, static_cast<sglang::VideoPreprocess>(2));
    }, "Unknown profile was accepted");
    require_rejection<std::invalid_argument>([] {
        sglang::preprocess_image({nullptr, frame_bytes, 32, 32, 1});
    }, "Null data was accepted");
    require_rejection<std::invalid_argument>([&] {
        sglang::preprocess_image({frames.data(), frame_bytes + 1, 32, 32, 1});
    }, "Trailing RGB data was accepted");
    require_rejection<std::invalid_argument>([] {
        sglang::sample_video_frames(source_count + 1, 1);
    }, "Oversized source frame count was accepted");
    require_rejection<std::invalid_argument>([] {
        sglang::sample_video_frames(2, std::nextafter(min_fps, 0.0));
    }, "Out-of-range FPS was accepted");
}

template <typename Values>
void print(const Values& values) {
    for (const auto& value : values) {
        std::cout << value << ' ';
    }
    std::cout << '\n';
}

void check(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "contract" && argc == 2) {
        check_contract();
        return;
    }
    if (mode == "sample" && argc == 4) {
        print(sglang::sample_video_frames(std::stoi(argv[2]), std::stod(argv[3])));
        return;
    }
    if (argc < 7) {
        throw std::runtime_error("Expected a mode, input, output, height, width, and frame count");
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open RGB fixture");
    }
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(input), {}};
    const sglang::DecodedRgb rgb{
        bytes.data(), bytes.size(), std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6])};
    sglang::MediaPatches result;
    if (mode == "image" && argc == 7) {
        result = sglang::preprocess_image(rgb);
    } else if ((mode == "sglang" || mode == "checkpoint") && argc >= 9) {
        sglang::VideoMetadata metadata{std::stoi(argv[7]), std::stod(argv[8]), {}};
        for (int index = 9; index < argc; ++index) {
            metadata.frame_indices.push_back(std::stoi(argv[index]));
        }
        const auto profile = mode == "sglang" ? sglang::VideoPreprocess::SglangDefault
                                             : sglang::VideoPreprocess::Checkpoint;
        result = sglang::preprocess_video(rgb, metadata, profile);
    } else {
        throw std::runtime_error("Invalid fixture mode or missing mode arguments");
    }
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    std::ofstream output(argv[3], std::ios::binary);
    output.write(reinterpret_cast<const char*>(result.pixels.data()),
                 result.pixels.size() * sizeof(float));
    if (!output) {
        throw std::runtime_error("Could not write patch values");
    }
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    print(result.grid_thw);
    print(result.frame_indices);
    print(result.timestamps_seconds);
}

}

int main(int argc, char** argv) {
    try {
        check(argc, argv);
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::length_error& error) {
        std::cerr << error.what() << '\n';
        return 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
