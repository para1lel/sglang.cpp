#include "sglang/checkpoint.h"
#include "sglang/prefill_fp8.h"
#include "sglang/tokenizer/digest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sglang;
using Bf16 = __nv_bfloat16;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check(cudaError_t status) {
    require(status == cudaSuccess, cudaGetErrorString(status));
}

template<class Value>
class Device {
public:
    explicit Device(std::size_t size) {
        check(cudaMalloc(&data, size * sizeof(Value)));
    }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    void upload(std::span<const Value> values) {
        check(cudaMemcpy(data, values.data(), values.size_bytes(), cudaMemcpyHostToDevice));
    }

    Value* data = nullptr;
};

struct Launch {
    dim3 grid;
    std::string name;
};

Launch observed;
int launch_count = 0;

struct Case {
    int outputs;
    int inputs;
    const char* stem;
    std::vector<int> rows;
};

constexpr int group = text_fp8_group_size;
constexpr int guard = 256;
constexpr std::uint16_t poison = 0x7fce;

double e4m3(std::uint8_t byte) {
    const int exponent = (byte >> 3) & 15;
    const int mantissa = byte & 7;
    require(exponent != 15 || mantissa != 7, "Nonfinite FP8 input");
    const double magnitude = exponent ? std::ldexp(1 + mantissa / 8., exponent - 7)
                                      : std::ldexp(mantissa / 8., -6);
    return byte & 128 ? -magnitude : magnitude;
}

double bf16(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16);
}

int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

std::array<int, 2> expected_tile(bool tiling, int rows, int outputs, int inputs) {
    if ((rows <= 64 && outputs >= 4096) || (inputs >= 8192 && (!tiling || rows < 1024))) {
        return {64, 128};
    }
    if (outputs >= 8192 || inputs >= 6144) {
        return {128, 128};
    }
    return {128, outputs < 1024 ? 32 : 64};
}

double oracle(std::span<const std::uint8_t> activations, std::span<const float> activation_scales,
              std::span<const std::uint8_t> weights, std::span<const Bf16> weight_scales,
              int inputs, int row, int column) {
    const int groups = inputs / group;
    double total = 0;
    for (int index = 0; index < groups; ++index) {
        double partial = 0;
        for (int lane = 0; lane < group; ++lane) {
            const int inner = index * group + lane;
            partial += e4m3(activations[row * inputs + inner]) *
                       e4m3(weights[column * inputs + inner]);
        }
        total += partial * activation_scales[row * groups + index] *
                 __bfloat162float(weight_scales[column / group * groups + index]);
    }
    return total;
}

void load_weights(const Checkpoint& checkpoint, const Case& shape,
                  std::span<std::uint8_t> weights, std::span<Bf16> scales) {
    if (shape.stem) {
        const std::string stem = std::string("model.language_model.layers.0.") + shape.stem;
        const auto& payload = checkpoint.tensor(stem + ".weight", TensorDtype::F8_E4M3,
                                                {std::uint64_t(shape.outputs),
                                                 std::uint64_t(shape.inputs)});
        const auto& scale = checkpoint.tensor(stem + ".weight_scale_inv", TensorDtype::BF16,
                                              {std::uint64_t(ceil_div(shape.outputs, group)),
                                               std::uint64_t(shape.inputs / group)});
        std::memcpy(weights.data(), payload.data, payload.size_bytes);
        std::memcpy(scales.data(), scale.data, scale.size_bytes);
        return;
    }
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = (index * 37 + index / 101 * 11) % 127 |
                         ((index * 13 + index / 229) & 128);
    }
    for (std::size_t index = 0; index < scales.size(); ++index) {
        scales[index] = __float2bfloat16(std::ldexp(1.f, int(index % 5) - 12));
    }
}

void check_or_save_bf16(const std::filesystem::path& directory, const std::string& key,
                        std::span<const std::uint16_t> result, bool tiling) {
    const auto binary = directory / (key + ".bf16");
    if (tiling) {
        std::ifstream control(binary, std::ios::binary | std::ios::ate);
        require(control.tellg() == static_cast<std::streamoff>(result.size_bytes()),
                key + " wrong baseline byte count");
        control.seekg(0);
        std::vector<std::uint16_t> expected(result.size());
        control.read(reinterpret_cast<char*>(expected.data()), result.size_bytes());
        require(bool(control) && std::ranges::equal(expected, result), key + " BF16 differs");
        return;
    }
    std::ofstream control(binary, std::ios::binary);
    control.write(reinterpret_cast<const char*>(result.data()), result.size_bytes());
    require(bool(control), key + " failed to preserve baseline");
}

void projection(const Checkpoint& checkpoint, const Case& shape, bool tiling,
                const std::filesystem::path& baseline, cudaStream_t stream) {
    const int inputs = shape.inputs;
    const int outputs = shape.outputs;
    const int groups = inputs / group;
    const int max_rows = *std::max_element(shape.rows.begin(), shape.rows.end());
    const int allocated_rows = max_rows >= 1024 ? 2049 : max_rows;
    std::vector<std::uint8_t> host_weights(static_cast<std::size_t>(outputs) * inputs);
    std::vector<Bf16> host_weight_scales(ceil_div(outputs, group) * groups);
    load_weights(checkpoint, shape, host_weights, host_weight_scales);
    Device<std::uint8_t> weights(host_weights.size());
    Device<Bf16> weight_scales(host_weight_scales.size());
    weights.upload(host_weights);
    weight_scales.upload(host_weight_scales);

    std::vector<Bf16> input(static_cast<std::size_t>(max_rows) * inputs);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const int coordinate = static_cast<int>((index * 23 + index / 619 * 17) % 127) - 63;
        input[index] = __float2bfloat16(std::ldexp(float(coordinate), int(index / group % 5) - 7));
    }
    Device<Bf16> bf16_input(input.size());
    Device<std::uint8_t> activations(static_cast<std::size_t>(allocated_rows) * inputs);
    Device<float> activation_scales(allocated_rows * groups);
    bf16_input.upload(input);
    check(text_quantize_fp8_rows(bf16_input.data, inputs, max_rows,
                                 {activations.data, activation_scales.data}, stream));
    check(cudaStreamSynchronize(stream));
    std::vector<std::uint8_t> host_activations(input.size());
    std::vector<float> host_activation_scales(max_rows * groups);
    check(cudaMemcpy(host_activations.data(), activations.data, host_activations.size(),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(host_activation_scales.data(), activation_scales.data,
                     host_activation_scales.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const TextFp8ActivationView activation_view{activations.data, activation_scales.data};

    std::vector<std::uint16_t> host_output(guard * 2 + max_rows * outputs, poison);
    Device<Bf16> output(guard * 2 + allocated_rows * outputs);
    for (const int rows : shape.rows) {
        const std::string key = std::to_string(rows) + "-" + std::to_string(outputs) +
                                "-" + std::to_string(inputs);
        const auto words = static_cast<std::size_t>(rows) * outputs;
        const auto active = std::span(host_output).first(words + 2 * guard);
        const auto result = active.subspan(guard, words);
        std::ranges::fill(active, poison);
        check(cudaMemcpy(output.data, active.data(), active.size_bytes(), cudaMemcpyHostToDevice));
        launch_count = 0;
        const auto status = try_text_fp8_gemm_sm100(weights.data, weight_scales.data,
            activation_view, rows, outputs, inputs, output.data + guard, stream, true);
        require(status == cudaSuccess && launch_count == 1, key + " failed SM100 dispatch");
        const auto [tile_rows, tile_columns] = expected_tile(tiling, rows, outputs, inputs);
        const std::string kernel = "TileILi" + std::to_string(tile_rows) + "ELi" +
                                   std::to_string(tile_columns) + "E";
        require(observed.grid.x == ceil_div(outputs, tile_columns) &&
                observed.grid.y == ceil_div(rows, tile_rows) &&
                observed.name.find(kernel) != std::string::npos, key + " wrong launch geometry");
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(active.data(), output.data, active.size_bytes(), cudaMemcpyDeviceToHost));
        const auto is_poison = [](auto word) { return word == poison; };
        require(std::ranges::all_of(active.first(guard), is_poison) &&
                std::ranges::all_of(active.last(guard), is_poison),
                key + " overwrote guard words");
        require(std::ranges::none_of(result, [](auto word) { return (word & 0x7f80) == 0x7f80; }),
                key + " left poison or produced nonfinite outputs");
        double max_error = 0;
        for (const int row : {0, rows / 2, rows - 2, rows - 1}) {
            for (const int column : {0, 127, 128, outputs / 2, outputs - 2, outputs - 1}) {
                const double expected =
                    oracle(host_activations, host_activation_scales, host_weights,
                           host_weight_scales, inputs, row, column);
                const double actual = bf16(result[row * outputs + column]);
                const double error = std::abs(actual - expected);
                max_error = std::max(max_error, error);
                require(error <= 0.004 * std::max(.001, std::abs(expected)),
                        key + " FP64 comparison failed at " + std::to_string(row) + "," +
                        std::to_string(column));
            }
        }
        check_or_save_bf16(baseline, key, result, tiling);
        const auto* bytes = reinterpret_cast<const char*>(result.data());
        std::cout << key << '\t' << tile_rows << 'x' << tile_columns << '\t' << result.size()
                  << '\t' << std::setprecision(8) << max_error << '\t'
                  << tokenizer::sha256({bytes, result.size_bytes()}) << '\n';
    }
    if (max_rows >= 1024) {
        for (const auto [rows, allow_large] : {std::pair{2, true}, {129, false}, {2049, true}}) {
            launch_count = 0;
            const auto status = try_text_fp8_gemm_sm100(weights.data, weight_scales.data,
                activation_view, rows, outputs, inputs, output.data + guard, stream, allow_large);
            require(!status && launch_count == 0, "Expected no launch for rows=" +
                std::to_string(rows) + ", status=" +
                (status ? cudaGetErrorString(*status) : "declined"));
        }
    }
}

}

extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                              cudaStream_t);

extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                              void** args, std::size_t shared,
                                              cudaStream_t stream) {
    const auto status = __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
    const char* name = nullptr;
    if (status == cudaSuccess && cudaFuncGetName(&name, function) == cudaSuccess && name &&
        std::strstr(name, "fp8_tma")) {
        observed = {grid, name};
        ++launch_count;
    }
    return status;
}

int main(int argc, char** argv) try {
    require(argc == 4, "Pass checkpoint, base or tiling, and baseline output directory");
    const bool tiling = std::string_view(argv[2]) == "tiling";
    require(tiling || std::string_view(argv[2]) == "base", "Unexpected arm");
    const Checkpoint checkpoint(argv[1]);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const std::vector<Case> shapes{
        {5120, 17408, "mlp.down_proj",
         {3, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 1087, 1088, 1089, 2047, 2048}},
        {5120, 6144, "linear_attn.out_proj", {1023, 1024, 1025}},
        {17408, 5120, "mlp.up_proj", {1023, 1024, 1025}},
        {6144, 5120, "linear_attn.in_proj_z", {1023, 1024, 1025}},
        {129, 8064, nullptr, {1023, 1024, 1025}},
        {129, 8192, nullptr, {1023, 1024, 1025}},
        {129, 8320, nullptr, {1023, 1024, 1025}},
        {4095, 5120, nullptr, {64, 65}},
        {4096, 5120, nullptr, {64, 65}},
    };
    for (const auto& shape : shapes) {
        projection(checkpoint, shape, tiling, argv[3], stream);
    }
    check(cudaStreamDestroy(stream));
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
