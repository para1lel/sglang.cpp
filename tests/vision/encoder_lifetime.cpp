#include "sglang/checkpoint.h"
#include "sglang/vision.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using sglang::VisionEncoder;
using sglang::VisionGrid;
using sglang::vision_feature_rows;
using sglang::vision_feature_width;
using sglang::vision_patch_width;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed with " +
                                 cudaGetErrorString(result));
    }
}

struct Stream {
    Stream() { check(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking), "Create stream"); }
    ~Stream() { cudaStreamDestroy(handle); }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t handle = nullptr;
};

template <typename Element>
struct Allocation {
    explicit Allocation(std::size_t count) {
        void* pointer = nullptr;
        check(cudaMalloc(&pointer, count * sizeof(Element)), "Allocate test input or output");
        data = static_cast<Element*>(pointer);
    }
    ~Allocation() { cudaFree(data); }

    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;

    Element* data = nullptr;
};

struct Request {
    static constexpr std::size_t padding = 16;
    static constexpr std::uint16_t guard = 0x5a5a;

    VisionGrid grid;
    std::vector<float> pixels;
    std::size_t count;
    Allocation<float> input;
    Allocation<__nv_bfloat16> output;

    Request(VisionGrid shape, std::uint32_t seed)
        : grid(shape), pixels(vision_feature_rows(grid) * 4 * vision_patch_width),
          count(vision_feature_rows(grid) * vision_feature_width), input(pixels.size()),
          output(count + padding * 2) {
        for (auto& pixel : pixels) {
            seed = seed * 1664525u + 1013904223u;
            pixel = (static_cast<int>(seed >> 24) - 128) / 64.0f;
        }
    }

    std::vector<std::uint16_t> run(const VisionEncoder& encoder, cudaStream_t stream) const {
        std::vector<std::uint16_t> features(count + padding * 2);
        check(cudaMemsetAsync(output.data, 0x5a, features.size() * sizeof(features[0]), stream),
              "Set output canaries");
        check(cudaMemcpyAsync(input.data, pixels.data(), pixels.size() * sizeof(pixels[0]),
                              cudaMemcpyHostToDevice, stream), "Upload input");
        encoder.encode(grid, input.data, pixels.size(), output.data + padding, count, stream);
        check(cudaStreamQuery(stream), "Verify stream completion on return");
        check(cudaMemcpy(features.data(), output.data, features.size() * sizeof(features[0]),
                         cudaMemcpyDeviceToHost), "Read output and canaries");
        const auto first = features.begin() + padding;
        const auto last = first + count;
        const auto intact = [](std::uint16_t value) { return value == guard; };
        require(std::all_of(features.begin(), first, intact), "Output underflow");
        require(std::all_of(last, features.end(), intact), "Output overflow");
        require(std::none_of(first, last, [](std::uint16_t value) {
            return (value & 0x7f80) == 0x7f80;
        }), "Nonfinite BF16 feature");
        std::vector<float> pixels_after(pixels.size());
        check(cudaMemcpy(pixels_after.data(), input.data, pixels_after.size() * sizeof(float),
                         cudaMemcpyDeviceToHost), "Read input after encode");
        require(pixels_after == pixels, "Encoder modified the caller's input");
        return {first, last};
    }
};

std::unique_ptr<VisionEncoder> upload(const char* directory) {
    const sglang::Checkpoint checkpoint(directory);
    checkpoint.validate_vision();
    return std::make_unique<VisionEncoder>(checkpoint);
}

std::size_t free_memory() {
    std::size_t free = 0;
    std::size_t total = 0;
    check(cudaMemGetInfo(&free, &total), "Read device memory");
    return free;
}

bool same(const std::vector<std::uint16_t>& left, const std::vector<std::uint16_t>& right,
          std::size_t offset, std::size_t count) {
    return std::equal(left.begin() + offset, left.begin() + offset + count, right.begin() + offset);
}

void alternate(const VisionEncoder& encoder) {
    Stream stream_a;
    Stream stream_b;
    Request image({1, 6, 10}, 9966);
    Request video({3, 4, 6}, 314159);
    const auto image_first = image.run(encoder, stream_a.handle);
    const auto video_first = video.run(encoder, stream_b.handle);
    const std::size_t per_pair = video.count / 3;
    constexpr int changed_pair = 1;
    const auto first = video.pixels.begin() + changed_pair * video.pixels.size() / 3;
    std::fill(first, first + video.pixels.size() / 3, 0.625f);
    const auto changed = video.run(encoder, stream_a.handle);
    require(same(changed, video_first, 0, per_pair), "Earlier temporal pair changed");
    require(!same(changed, video_first, per_pair, per_pair), "Edited temporal pair had no effect");
    require(same(changed, video_first, 2 * per_pair, per_pair), "Later temporal pair changed");
    require(image.run(encoder, stream_b.handle) == image_first, "Image changed after video");

    const auto start = [&](const Request& request, cudaStream_t stream,
                           const std::shared_future<void>& launch) {
        check(cudaSetDevice(0), "Select visible device on worker");
        launch.wait();
        return request.run(encoder, stream);
    };
    const auto before = free_memory();
    for (int repeat = 0; repeat < 4; ++repeat) {
        std::promise<void> ready;
        const auto launch = ready.get_future().share();
        auto one = std::async(std::launch::async, start, std::cref(image), stream_a.handle, launch);
        std::future<std::vector<std::uint16_t>> two;
        try {
            two = std::async(std::launch::async, start, std::cref(video), stream_b.handle, launch);
        } catch (...) {
            ready.set_value();
            throw;
        }
        ready.set_value();
        require(one.get() == image_first, "Concurrent image changed");
        require(two.get() == changed, "Concurrent video changed");
    }
    require(free_memory() + 2 * 1024 * 1024 >= before, "Vision workspaces retained device memory");
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected a checkpoint directory");
        check(cudaSetDevice(0), "Select visible device");
        check(cudaFree(nullptr), "Initialize CUDA");
        const auto before = free_memory();
        {
            const auto encoder = upload(argv[1]);
            alternate(*encoder);
        }
        require(free_memory() + 16 * 1024 * 1024 >= before, "Encoder retained weights");
        std::cout << "Checkpoint lifetime, stream completion, alternating and concurrent requests, "
                     "temporal isolation, input immutability, canaries, and CUDA release passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
