#include "sglang/prefill_fp8.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int max_rows = 129;
constexpr int columns = 197;
constexpr int width = 1024;
constexpr int group_width = sglang::text_fp8_group_size;
constexpr int groups = width / group_width;
constexpr int guard = 16;
constexpr std::uint16_t poison = 0x7fc1;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void expect(std::optional<cudaError_t> actual, std::optional<cudaError_t> expected,
            const char* name) {
    if (actual != expected) {
        throw std::runtime_error(std::string(name) + ": " +
                                 (actual ? cudaGetErrorString(*actual) : "declined"));
    }
}

template <typename Value>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        void* allocation = nullptr;
        check(cudaMalloc(&allocation, count * sizeof(Value)));
        data = static_cast<Value*>(allocation);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { cudaFree(data); }

    void upload(const std::vector<Value>& values) const {
        check(cudaMemcpy(data, values.data(), values.size() * sizeof(Value),
                         cudaMemcpyHostToDevice));
    }

    void download(std::vector<Value>& values) const {
        check(cudaMemcpy(values.data(), data, values.size() * sizeof(Value),
                         cudaMemcpyDeviceToHost));
    }

    Value* data = nullptr;
};

class Stream {
public:
    Stream() { check(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking)); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() { cudaStreamDestroy(handle); }

    cudaStream_t handle = nullptr;
};

double decode_fp8(std::uint8_t bits) {
    const int exponent = (bits >> 3) & 15;
    const int fraction = bits & 7;
    const double magnitude = exponent == 0 ? std::ldexp(fraction, -9)
                                           : std::ldexp(8 + fraction, exponent - 10);
    return (bits & 128) != 0 ? -magnitude : magnitude;
}

double decode_bf16(std::uint16_t bits) {
    const std::uint32_t expanded = std::uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

std::uint16_t round_bf16(double value) {
    if (value == 0.0) {
        return 0;
    }
    int exponent = 0;
    const double fraction = std::frexp(std::abs(value), &exponent);
    const double scaled = fraction * 256;
    auto significand = static_cast<int>(scaled);
    const double remainder = scaled - significand;
    if (remainder > 0.5 || (remainder == 0.5 && significand % 2 != 0)) {
        ++significand;
    }
    return static_cast<std::uint16_t>((value < 0 ? 0x8000 : 0) +
                                      ((exponent + 126) << 7) + significand - 128);
}

std::uint8_t sample_fp8(int row, int index, int salt) {
    const int pattern = (row * 13 + index * 17 + salt * (index / group_width + 1)) % 31;
    const auto magnitude = static_cast<std::uint8_t>(pattern == 0 ? 0 : 32 + pattern % 17);
    return pattern % 3 == 0 ? magnitude | 128 : magnitude;
}

struct Reference {
    std::uint16_t bf16;
    double lower;
    double upper;
};

std::vector<Reference> reference(const std::vector<std::uint8_t>& activation,
                                 const std::vector<float>& activation_scales,
                                 const std::vector<std::uint8_t>& weight,
                                 const std::vector<std::uint16_t>& weight_scales) {
    std::vector<double> decoded_activation(activation.size());
    std::vector<double> decoded_weight(weight.size());
    std::transform(activation.begin(), activation.end(), decoded_activation.begin(), decode_fp8);
    std::transform(weight.begin(), weight.end(), decoded_weight.begin(), decode_fp8);
    std::vector<Reference> output(max_rows * columns);
    for (int row = 0; row < max_rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            double sum = 0;
            double absolute_sum = 0;
            for (int group = 0; group < groups; ++group) {
                double dot = 0;
                double absolute_dot = 0;
                for (int inner = group * group_width; inner < (group + 1) * group_width; ++inner) {
                    const double product = decoded_activation[row * width + inner] *
                                           decoded_weight[column * width + inner];
                    dot += product;
                    absolute_dot += std::abs(product);
                }
                const double scale = activation_scales[row * groups + group] *
                                     decode_bf16(weight_scales[(column / group_width) *
                                                              groups + group]);
                sum += dot * scale;
                absolute_sum += absolute_dot * std::abs(scale);
            }
            const double bound = std::ldexp(absolute_sum * (group_width + groups), -23);
            output[row * columns + column] = {
                round_bf16(sum),
                decode_bf16(round_bf16(sum - bound)),
                decode_bf16(round_bf16(sum + bound)),
            };
        }
    }
    return output;
}

void run() {
    std::vector<std::uint8_t> activation(max_rows * width);
    std::vector<float> activation_scales(max_rows * groups);
    std::vector<std::uint8_t> weight(columns * width);
    std::vector<std::uint16_t> weight_scales(sglang::text_fp8_group_count(columns) * groups);
    for (int row = 0; row < max_rows; ++row) {
        for (int index = 0; index < width; ++index) {
            activation[row * width + index] = row == 0 ? 0 : sample_fp8(row, index, 11);
        }
        for (int group = 0; group < groups; ++group) {
            activation_scales[row * groups + group] = std::ldexp(1 + (row + group) % 3, -5);
        }
    }
    for (int column = 0; column < columns; ++column) {
        for (int index = 0; index < width; ++index) {
            weight[column * width + index] = sample_fp8(column, index, 19);
        }
    }
    for (std::size_t index = 0; index < weight_scales.size(); ++index) {
        weight_scales[index] = static_cast<std::uint16_t>(0x3e80 + 3 * index);
    }

    const auto expected = reference(activation, activation_scales, weight, weight_scales);
    std::vector<std::uint16_t> observed(2 * guard + expected.size(), poison);
    DeviceBuffer<std::uint8_t> device_activation(activation.size());
    DeviceBuffer<float> device_activation_scales(activation_scales.size());
    DeviceBuffer<std::uint8_t> device_weight(weight.size());
    DeviceBuffer<std::uint16_t> device_weight_scales(weight_scales.size());
    DeviceBuffer<std::uint16_t> device_output(observed.size());
    Stream stream;
    device_activation.upload(activation);
    device_activation_scales.upload(activation_scales);
    device_weight.upload(weight);
    device_weight_scales.upload(weight_scales);

    const sglang::TextFp8ActivationView input{device_activation.data,
                                              device_activation_scales.data};
    const auto* scales = reinterpret_cast<const __nv_bfloat16*>(device_weight_scales.data);
    auto* output = reinterpret_cast<__nv_bfloat16*>(device_output.data + guard);
    const auto launch = [&](const std::uint8_t* weights, sglang::TextFp8ActivationView activations,
                            int rows, int input_width) {
        return sglang::try_text_fp8_gemm_sm100(weights, scales, activations, rows, columns,
                                              input_width, output, stream.handle);
    };
    for (int rows : {2, 3, 127, 128, 129}) {
        std::fill(observed.begin(), observed.end(), poison);
        device_output.upload(observed);
        const auto status = launch(device_weight.data, input, rows, width);
        const bool use_tensor_core = rows >= 3 && rows <= sglang::text_embedding_max_rows;
        expect(status, use_tensor_core ? std::optional{cudaSuccess} : std::nullopt,
               "row dispatch");
        if (!use_tensor_core) {
            check(sglang::text_fp8_gemv_rows(device_weight.data, scales, input, columns, width,
                                             rows, output, stream.handle));
        }
        check(cudaStreamSynchronize(stream.handle));
        device_output.download(observed);
        for (std::size_t index = 0; index < observed.size(); ++index) {
            const bool interior = index >= guard && index < std::size_t(guard + rows * columns);
            bool matches = observed[index] == poison;
            if (interior) {
                const auto& value = expected[index - guard];
                const double actual = decode_bf16(observed[index]);
                matches = use_tensor_core ? observed[index] == value.bf16
                                           : actual >= value.lower && actual <= value.upper;
            }
            if (!matches) {
                throw std::runtime_error("Rows " + std::to_string(rows) +
                                         ", output offset " + std::to_string(index) +
                                         ": unexpected BF16 bits " +
                                         std::to_string(observed[index]));
            }
        }
        std::cout << "rows=" << rows << " path=" << (use_tensor_core ? "tensor-core" : "exact")
                  << " checked_outputs=" << rows * columns << '\n';
    }
    expect(launch(nullptr, input, 3, width), cudaErrorInvalidValue, "null weight");
    expect(launch(device_weight.data, input, 0, width), cudaErrorInvalidValue, "zero rows");
    expect(launch(device_weight.data, input, 3, width - 1), std::nullopt, "odd K");
    expect(launch(device_weight.data + 1, input, 3, width), std::nullopt, "offset weight");
    expect(launch(device_weight.data, {input.values + 1, input.scales}, 3, width),
           std::nullopt, "offset input");
    expect(sglang::try_text_fp8_gemm_sm100(
               device_weight.data, scales, input, 3, columns, width,
               reinterpret_cast<__nv_bfloat16*>(device_activation.data), stream.handle),
           cudaErrorInvalidValue, "aliased output");
    check(cudaStreamSynchronize(stream.handle));
    const auto saved = observed;
    device_output.download(observed);
    if (observed != saved) {
        throw std::runtime_error("Rejected launches changed output");
    }
}

}

int main() {
    try {
        run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
