#pragma once

#include "sglang/vision.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sglang::vision_detail {

using Bf16 = __nv_bfloat16;
inline constexpr int hidden = 1152;
inline constexpr int heads = 16;
inline constexpr int head_width = hidden / heads;
inline constexpr int feedforward = 4304;
inline constexpr int merge_width = 4 * hidden;
inline constexpr int depth = 27;
inline constexpr int table_side = 48;
inline constexpr int threads = 128;
inline constexpr std::size_t weight_bytes = 921460192;

inline void check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

inline bool cleanup(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

template <typename Element>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check(cudaGetDevice(&device_), "Get vision allocation device");
        void* storage = nullptr;
        check(cudaMalloc(&storage, count * sizeof(Element)), "Allocate vision buffer");
        data_ = static_cast<Element*>(storage);
    }
    ~DeviceBuffer() {
        int previous = 0;
        if (!cleanup(cudaGetDevice(&previous), "Get current device for vision cleanup")) {
            return;
        }
        const bool changed = previous != device_;
        if (changed && !cleanup(cudaSetDevice(device_), "Select vision allocation device")) {
            return;
        }
        cleanup(cudaFree(data_), "Free vision buffer");
        if (changed) {
            cleanup(cudaSetDevice(previous), "Restore device after vision cleanup");
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    Element* get() const { return data_; }

private:
    int device_;
    Element* data_ = nullptr;
};

struct Grid {
    int rows;
    int height;
    int width;

    __host__ __device__ int frame() const { return height * width; }
};

struct Affine {
    const Bf16* weight;
    const Bf16* bias;
};

struct Block {
    Affine norm1;
    Affine qkv;
    Affine attention;
    Affine norm2;
    Affine fc1;
    Affine fc2;
};

struct Weights {
    Affine patch;
    const Bf16* position;
    std::array<Block, depth> blocks;
    Affine merge_norm;
    Affine merge_fc1;
    Affine merge_fc2;
};

Weights load_weights(const Checkpoint& checkpoint, Bf16* destination);

void cast_patches(const float* input, Bf16* output, int count, cudaStream_t stream);
void linear(const Bf16* input, Affine weights, Bf16* output, int rows,
            int input_width, int output_width, cudaStream_t stream);
void positions(const Bf16* table, Bf16* output, Grid grid, cudaStream_t stream);
void add(Bf16* destination, const Bf16* value, int count, cudaStream_t stream);
void normalize(const Bf16* input, Affine weights, Bf16* output, int rows, cudaStream_t stream);
void gelu(Bf16* data, int count, bool approximate, cudaStream_t stream);
void make_rotary(float2* table, Grid grid, cudaStream_t stream);
void rotate(const Bf16* qkv, const float2* table, Bf16* qk, Grid grid, cudaStream_t stream);
void attend(const Bf16* qk, const Bf16* qkv, Bf16* context, Grid grid, cudaStream_t stream);

}
