#include "sglang/decoder_cuda.cuh"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using sglang::TextFp8ActivationView;
using sglang::decoder::DecodeFp8Projection;
using BFloat = __nv_bfloat16;

constexpr int width = 513;
constexpr int rows = 19;
constexpr int groups = 5;
constexpr std::uint16_t canary = 0x5555;
unsigned kernel_launches = 0;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Operation>
void reject(Operation operation, const char* message) {
    const auto launches_before = kernel_launches;
    const auto status = operation();
    check(cudaDeviceSynchronize());
    require(status == cudaErrorInvalidValue, message);
    require(kernel_launches == launches_before, "Rejected alias submitted a CUDA kernel");
}

template <typename Value>
class Device {
public:
    explicit Device(std::size_t count) : count_(count) {
        void* storage = nullptr;
        check(cudaMalloc(&storage, count_ * sizeof(Value)));
        data = static_cast<Value*>(storage);
    }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    void upload(const std::vector<Value>& values) {
        check(cudaMemcpy(data, values.data(), count_ * sizeof(Value), cudaMemcpyHostToDevice));
    }

    Value* data = nullptr;

private:
    std::size_t count_;
};

class Outputs {
public:
    BFloat* at(int offset) { return bits_.data + offset; }

    void reset() { check(cudaMemset(bits_.data, 0x55, size * sizeof(BFloat))); }

    std::vector<std::uint16_t> read() const {
        std::vector<std::uint16_t> result(size);
        check(cudaMemcpy(result.data(), bits_.data, size * sizeof(BFloat), cudaMemcpyDeviceToHost));
        return result;
    }

    static constexpr int size = 2 * rows + 2;

private:
    static_assert(sizeof(BFloat) == sizeof(std::uint16_t));
    Device<BFloat> bits_{size};
};

void ordinary(TextFp8ActivationView input, DecodeFp8Projection projection) {
    check(sglang::text_fp8_gemv(projection.values, projection.scales, input,
                               projection.rows, width, projection.output, nullptr));
}

void gate_up(TextFp8ActivationView input, DecodeFp8Projection gate, DecodeFp8Projection up,
             Outputs& actual, Outputs& expected) {
    const auto launch = [&] {
        return sglang::decoder::decode_fp8_gate_up(input, gate, up, width, nullptr);
    };
    expected.reset();
    ordinary(input, {gate.values, gate.scales, expected.at(1), rows});
    ordinary(input, {up.values, up.scales, expected.at(1 + rows), rows});
    check(sglang::text_swiglu(expected.at(1), expected.at(1 + rows), rows,
                             expected.at(1), nullptr));
    actual.reset();
    check(launch());
    check(cudaDeviceSynchronize());
    require(actual.read() == expected.read(), "Disjoint adjacent gate/up output values or guards");

    for (int offset : {0, 1, -1}) {
        actual.reset();
        gate.output = actual.at(2);
        up.output = actual.at(2 + offset);
        reject(launch, "Overlapping fallback gate/up outputs accepted");
        require(actual.read() == std::vector<std::uint16_t>(Outputs::size, canary),
                "Rejected fallback gate/up modified output");
    }
}

void qkv(TextFp8ActivationView input, DecodeFp8Projection query, Outputs& actual,
         Outputs& expected) {
    DecodeFp8Projection key{query.values, query.scales, actual.at(1 + rows), 4};
    DecodeFp8Projection value{query.values + width, query.scales,
                              actual.at(1 + rows + key.rows), 3};
    const auto launch = [&] {
        return sglang::decoder::decode_fp8_qkv(input, query, key, value, width, nullptr);
    };
    expected.reset();
    ordinary(input, {query.values, query.scales, expected.at(1), rows});
    ordinary(input, {key.values, key.scales, expected.at(1 + rows), key.rows});
    ordinary(input, {value.values, value.scales, expected.at(1 + rows + key.rows), value.rows});
    actual.reset();
    check(launch());
    check(cudaDeviceSynchronize());
    require(actual.read() == expected.read(), "Shared leading and interior QKV weight subsets");

    value.output = key.output + 1;
    reject(launch, "Overlapping QKV outputs accepted");
    value.output = const_cast<BFloat*>(key.scales);
    reject(launch, "QKV output overlapping another projection's scales accepted");
}

}

extern "C" cudaError_t __real_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream);

extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream) {
    ++kernel_launches;
    return __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
}

int main() try {
    check(cudaSetDevice(0));
    Device<std::uint8_t> activation(width);
    Device<float> activation_scales(groups);
    Device<std::uint8_t> weights((rows + 1) * width);
    Device<BFloat> scales(groups);
    std::vector<std::uint8_t> weights_host((rows + 1) * width);
    for (std::size_t index = 0; index < weights_host.size(); ++index) {
        weights_host[index] = static_cast<std::uint8_t>(0x30 + ((index / width) & 7));
    }
    activation.upload(std::vector<std::uint8_t>(width, 0x38));
    activation_scales.upload(std::vector<float>(groups, 1.0f / 512));
    weights.upload(weights_host);
    scales.upload(std::vector<BFloat>(groups, __float2bfloat16(1.0f)));
    const TextFp8ActivationView input{activation.data, activation_scales.data};
    Outputs actual;
    Outputs expected;
    const DecodeFp8Projection gate{weights.data, scales.data, actual.at(1), rows};
    const DecodeFp8Projection up{weights.data + width, scales.data, actual.at(1 + rows), rows};
    gate_up(input, gate, up, actual, expected);
    qkv(input, gate, actual, expected);
    auto projection = gate;
    projection.output = reinterpret_cast<BFloat*>(activation.data + 2);
    reject([&] { return sglang::decoder::decode_fp8_project(input, projection, width, nullptr); },
           "Projection output overlapping activation values accepted");
    std::cout << "FP8 alias checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
