#include "sglang/decoder_cuda.cuh"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using BFloat = __nv_bfloat16;
using sglang::TextFp8Activation;
using sglang::decoder::decode_bf16_gdn_gates;
using sglang::decoder::decode_bf16_lm_head;
using sglang::decoder::decode_quantize_fp8;

constexpr int hidden = 5120;
constexpr int gates = sglang::text_gdn_value_heads;

struct Launch {
    dim3 grid;
    dim3 block;
};

struct ExpectedLaunches {
    unsigned count;
    unsigned blocks;
    unsigned threads;
};

std::array<Launch, 2> launches;
unsigned successful_launches = 0;
unsigned attempted_launches = 0;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void require_launches(ExpectedLaunches expected, const char* message) {
    if (successful_launches != expected.count) {
        throw std::runtime_error(message);
    }
    for (unsigned index = 0; index < expected.count; ++index) {
        const auto [grid, block] = launches[index];
        if (grid.x != expected.blocks || grid.y != 1 || grid.z != 1 ||
            block.x != expected.threads || block.y != 1 || block.z != 1) {
            throw std::runtime_error(message);
        }
    }
}

template <typename Value>
class Device {
public:
    explicit Device(std::size_t count) : count_(count) {
        void* storage = nullptr;
        check(cudaMalloc(&storage, count * sizeof(Value)));
        data = static_cast<Value*>(storage);
    }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    void reset() { check(cudaMemset(data, 0x55, count_ * sizeof(Value))); }

    void upload(const std::vector<Value>& source) {
        check(cudaMemcpy(data, source.data(), count_ * sizeof(Value), cudaMemcpyHostToDevice));
    }

    void match(const Device& reference, const char* message) const {
        std::vector<Value> actual(count_);
        std::vector<Value> expected(count_);
        const auto bytes = count_ * sizeof(Value);
        check(cudaMemcpy(actual.data(), data, bytes, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(expected.data(), reference.data, bytes, cudaMemcpyDeviceToHost));
        if (std::memcmp(actual.data(), expected.data(), bytes) != 0) {
            throw std::runtime_error(message);
        }
    }

    Value* data = nullptr;

private:
    std::size_t count_;
};

void fill(Device<BFloat>& device, int count, int seed) {
    std::vector<BFloat> host(count);
    for (int index = 0; index < count; ++index) {
        const int sample = ((index + seed) * (31 + seed) ^ (index >> 4)) % 127 - 63;
        host[index] = __float2bfloat16(sample * 0.03125f);
    }
    device.upload(host);
}

void projection(int width, int rows, int offset, bool gdn, ExpectedLaunches expected_launches) {
    Device<BFloat> input(width + 1);
    Device<BFloat> first(rows * width + 1);
    Device<BFloat> second(rows * width + 1);
    Device<BFloat> expected(2 * rows + 2);
    Device<BFloat> actual(2 * rows + 2);
    fill(input, width + 1, 1);
    fill(first, rows * width + 1, 3);
    fill(second, rows * width + 1, 5);
    expected.reset();
    actual.reset();

    const auto* activation = input.data + offset;
    const auto* weights = first.data + offset;
    check(sglang::text_bf16_gemv(weights, activation, rows, width, expected.data + 1, nullptr));
    if (gdn) {
        check(sglang::text_bf16_gemv(second.data, activation, rows, width,
                                   expected.data + rows + 1, nullptr));
    }
    successful_launches = 0;
    if (gdn) {
        check(decode_bf16_gdn_gates(weights, second.data, activation, rows, width,
                                    actual.data + 1, actual.data + rows + 1, nullptr));
    } else {
        check(decode_bf16_lm_head(weights, activation, rows, width, actual.data + 1, nullptr));
    }
    check(cudaDeviceSynchronize());
    actual.match(expected,
                 gdn ? "GDN gate bits or guards differ" : "LM head bits or guards differ");
    require_launches(expected_launches,
                     gdn ? "GDN gate launch geometry differs" : "LM head launch geometry differs");
}

void quantization(int width, int input_offset, int output_offset,
                  ExpectedLaunches expected_launches) {
    const auto groups = sglang::text_fp8_group_count(width);
    Device<BFloat> input(width + 1);
    Device<std::uint8_t> actual_values(width + 8);
    Device<std::uint8_t> expected_values(width + 8);
    Device<float> actual_scales(groups + 2);
    Device<float> expected_scales(groups + 2);
    constexpr std::array<float, 10> samples{
        -448.0f, 416.0f, 1.0625f, -1.1875f, 0.0009765625f,
        -0.0029296875f, 0.001953125f, -0.005859375f, 0.0f, -0.0f,
    };
    std::vector<BFloat> host(width + 1);
    for (int index = 0; index <= width; ++index) {
        const float scale = index < sglang::text_fp8_group_size ? 1.0f : 0x1p-48f;
        host[index] = __float2bfloat16(samples[index % samples.size()] * scale);
    }
    input.upload(host);
    actual_values.reset();
    expected_values.reset();
    actual_scales.reset();
    expected_scales.reset();

    const TextFp8Activation actual{actual_values.data + output_offset, actual_scales.data + 1};
    const TextFp8Activation expected{
        expected_values.data + output_offset, expected_scales.data + 1};
    check(sglang::text_quantize_fp8(input.data + input_offset, width, expected, nullptr));
    successful_launches = 0;
    check(decode_quantize_fp8(input.data + input_offset, width, actual, nullptr));
    check(cudaDeviceSynchronize());
    actual_values.match(expected_values, "E4M3 bytes or guards differ");
    actual_scales.match(expected_scales, "FP8 scale bits or guards differ");
    require_launches(expected_launches, "FP8 quantization launch geometry differs");
}

void invalid_inputs() {
    Device<BFloat> input(hidden);
    Device<BFloat> weight(hidden * gates);
    Device<BFloat> output(gates * 2);
    Device<std::uint8_t> values(hidden);
    Device<float> scales(sglang::text_fp8_group_count(hidden));
    const TextFp8Activation fp8{values.data, scales.data};
    const auto before = attempted_launches;
    const std::array statuses{
        decode_bf16_lm_head(weight.data, input.data, 2, hidden, nullptr, nullptr),
        decode_bf16_lm_head(weight.data, input.data, 0, hidden, output.data, nullptr),
        decode_bf16_gdn_gates(weight.data, weight.data, input.data, gates, -1,
                              output.data, output.data + gates, nullptr),
        decode_bf16_gdn_gates(weight.data, nullptr, input.data, gates, hidden,
                              output.data, output.data + gates, nullptr),
        decode_quantize_fp8(input.data, 0, fp8, nullptr),
        decode_quantize_fp8(input.data, hidden, {values.data, nullptr}, nullptr),
    };
    for (const auto status : statuses) {
        if (status != cudaErrorInvalidValue) {
            throw std::runtime_error(
                "Invalid bandwidth input did not return cudaErrorInvalidValue");
        }
    }
    if (attempted_launches != before) {
        throw std::runtime_error("Invalid bandwidth input submitted a CUDA kernel");
    }
}

}

extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                               cudaStream_t);

extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream) {
    ++attempted_launches;
    const auto status = __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
    if (status == cudaSuccess) {
        if (successful_launches < launches.size()) {
            launches[successful_launches] = {grid, block};
        }
        ++successful_launches;
    }
    return status;
}

int main() try {
    check(cudaSetDevice(0));
    projection(hidden, 18, 0, false, {1, 2, 256});
    projection(hidden, 17, 0, false, {1, 3, 256});
    projection(hidden, 18, 1, false, {1, 3, 256});
    projection(hidden - 1, 18, 0, false, {1, 3, 256});
    projection(hidden, gates, 0, true, {1, 2 * gates, 32});
    projection(hidden, gates - 1, 0, true, {2, 6, 256});
    projection(hidden, gates, 1, true, {2, 6, 256});
    projection(hidden - 1, gates, 0, true, {2, 6, 256});
    quantization(256, 0, 4, {1, 2, 32});
    quantization(257, 0, 4, {1, 3, 128});
    quantization(256, 1, 4, {1, 2, 128});
    quantization(256, 0, 1, {1, 2, 128});
    invalid_inputs();
    std::cout << "Decode bandwidth operator checks passed\n";
} catch (const std::exception& error) {
    std::cerr << "Decode bandwidth operator check failed: " << error.what() << '\n';
    return 1;
}
