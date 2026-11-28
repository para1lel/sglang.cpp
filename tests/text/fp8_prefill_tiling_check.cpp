#include "sglang/checkpoint.h"
#include "sglang/prefill_fp8.h"
#include "sglang/tokenizer/digest.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
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
    ~Device() {
        cudaFree(data);
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    void upload(const std::vector<Value>& values) {
        check(
            cudaMemcpy(data, values.data(), values.size() * sizeof(Value), cudaMemcpyHostToDevice));
    }

    Value* data = nullptr;
};

struct Launch {
    dim3 grid;
    std::string name;
};

struct LaunchState {
    Launch observed;
    int count{};
    int paired_attempts{};
    bool decline_paired{};
    bool fail_paired_launch{};
};

LaunchState& launches() {
    static LaunchState state{};
    return state;
}

constexpr char paired_kernel[] = "TileILi128ELi128ELi6ELi2ELi1ELi2EE";

struct Case {
    int outputs;
    int inputs;
    const char* stem;
    std::vector<int> rows;
};

constexpr int group = text_fp8_group_size;
constexpr std::size_t guard = 256;
constexpr std::uint16_t poison = 0x7fce;

double e4m3(std::uint8_t byte) {
    const unsigned bits = byte;
    const auto exponent = static_cast<int>((bits >> 3U) & 15U);
    const unsigned mantissa = bits & 7U;
    require(exponent != 15 || mantissa != 7, "Nonfinite FP8 input");
    const double magnitude = exponent != 0 ? std::ldexp(1 + (mantissa / 8.), exponent - 7)
                                           : std::ldexp(mantissa / 8., -6);
    return (bits & 128U) != 0 ? -magnitude : magnitude;
}

double bf16(std::uint16_t word) {
    const auto bits = static_cast<std::uint32_t>(word) << 16U;
    float value{};
    std::memcpy(&value, &bits, sizeof(value));
    return value;
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

struct Coordinate {
    int row;
    int column;
};

double oracle(const std::vector<std::uint8_t>& activations,
              const std::vector<float>& activation_scales, const std::vector<std::uint8_t>& weights,
              const std::vector<Bf16>& weight_scales, int inputs, Coordinate coordinate) {
    const auto [row, column] = coordinate;
    const int groups = inputs / group;
    double total = 0;
    for (int index = 0; index < groups; ++index) {
        double partial = 0;
        for (int lane = 0; lane < group; ++lane) {
            const int inner = (index * group) + lane;
            partial += e4m3(activations[(row * inputs) + inner]) *
                       e4m3(weights[(column * inputs) + inner]);
        }
        total += partial * activation_scales[(row * groups) + index] *
                 __bfloat162float(weight_scales[((column / group) * groups) + index]);
    }
    return total;
}

void load_weights(const Checkpoint& checkpoint, const Case& shape,
                  std::vector<std::uint8_t>& weights, std::vector<Bf16>& scales) {
    if (shape.stem != nullptr) {
        const std::string stem = std::string("model.language_model.layers.0.") + shape.stem;
        const auto& payload = checkpoint.tensor(stem + ".weight", TensorDtype::F8_E4M3,
                                                {
                                                    static_cast<std::uint64_t>(shape.outputs),
                                                    static_cast<std::uint64_t>(shape.inputs),
                                                });
        const auto& scale =
            checkpoint.tensor(stem + ".weight_scale_inv", TensorDtype::BF16,
                              {
                                  static_cast<std::uint64_t>(ceil_div(shape.outputs, group)),
                                  static_cast<std::uint64_t>(shape.inputs / group),
                              });
        std::memcpy(weights.data(), payload.data, payload.size_bytes);
        std::memcpy(scales.data(), scale.data, scale.size_bytes);
        return;
    }
    for (std::size_t index = 0; index < weights.size(); ++index) {
        const auto magnitude = ((index * 37) + ((index / 101) * 11)) % 127;
        const auto sign = ((index * 13) + (index / 229)) & 128U;
        weights[index] = static_cast<std::uint8_t>(magnitude | sign);
    }
    for (std::size_t index = 0; index < scales.size(); ++index) {
        scales[index] = __float2bfloat16(std::ldexp(1.F, static_cast<int>(index % 5) - 12));
    }
}

void check_or_save_bf16(const std::filesystem::path& directory, const std::string& key,
                        const std::uint16_t* result, std::size_t words, bool compare) {
    const auto binary = directory / (key + ".bf16");
    const auto bytes = static_cast<std::streamsize>(words * sizeof(*result));
    if (compare) {
        std::ifstream control(binary, std::ios::binary | std::ios::ate);
        require(control.tellg() == bytes, key + " wrong baseline byte count");
        control.seekg(0);
        std::vector<std::uint16_t> expected(words);
        control.read(reinterpret_cast<char*>(expected.data()), bytes);
        require(static_cast<bool>(control) && std::equal(expected.begin(), expected.end(), result),
                key + " BF16 differs");
        return;
    }
    std::ofstream control(binary, std::ios::binary);
    control.write(reinterpret_cast<const char*>(result), bytes);
    require(static_cast<bool>(control), key + " failed to preserve baseline");
}

void verify_paired_fallback(const std::uint8_t* weights, const Bf16* scales,
                            TextFp8ActivationView input, Bf16* output, cudaStream_t stream) {
    auto& state = launches();
    const auto run = [&](bool allow_large) {
        state.count = 0;
        state.paired_attempts = 0;
        return try_text_fp8_gemm_sm100(weights, scales, input, 1024, 5120, 17408,
                                       output, stream, allow_large);
    };
    require(!run(false) && state.count == 0, "Paired shape bypassed the large-row guard");

    state.decline_paired = true;
    const auto fallback = run(true);
    state.decline_paired = false;
    require(fallback == cudaSuccess && state.paired_attempts == 1 && state.count == 1 &&
                state.observed.grid.x == 40 && state.observed.grid.y == 8 &&
                state.observed.name.find(paired_kernel) == std::string::npos,
            "Unavailable paired kernel did not use the ordinary TMA kernel");
    check(cudaStreamSynchronize(stream));

    state.fail_paired_launch = true;
    const auto failure = run(true);
    state.fail_paired_launch = false;
    require(failure == cudaErrorInvalidConfiguration && state.paired_attempts == 1 &&
                state.count == 0,
            "Paired launch failure was retried or lost");
}

void verify_row_limits(const Case& shape, const std::uint8_t* weights, const Bf16* scales,
                       TextFp8ActivationView input, Bf16* output, cudaStream_t stream) {
    auto& state = launches();
    for (const auto [rows, allow_large] : {std::pair{2, true}, {129, false}, {2049, true}}) {
        state.count = 0;
        const auto status = try_text_fp8_gemm_sm100(weights, scales, input, rows, shape.outputs,
                                                   shape.inputs, output, stream, allow_large);
        require(!status && state.count == 0,
                "Expected no launch for rows=" + std::to_string(rows) +
                    ", status=" + (status ? cudaGetErrorString(*status) : "declined"));
    }
}

void verify_output(const std::uint16_t* active, std::size_t words, const std::string& key) {
    const auto* const result = active + guard;
    const auto is_poison = [](auto word) { return word == poison; };
    require(std::all_of(active, active + guard, is_poison) &&
                std::all_of(result + words, result + words + guard, is_poison),
            key + " overwrote guard words");
    require(
        std::none_of(result, result + words,
                     [](auto word) { return (static_cast<unsigned>(word) & 0x7F80U) == 0x7F80U; }),
        key + " left poison or produced nonfinite outputs");
}

void projection(const Checkpoint& checkpoint, const Case& shape, bool tiling, bool paired,
                bool compare,
                const std::filesystem::path& baseline, cudaStream_t stream) {
    auto& state = launches();
    const int inputs = shape.inputs;
    const int outputs = shape.outputs;
    const int groups = inputs / group;
    const int max_rows = *std::max_element(shape.rows.begin(), shape.rows.end());
    const int allocated_rows = max_rows >= 1024 ? 2049 : max_rows;
    std::vector<std::uint8_t> host_weights(static_cast<std::size_t>(outputs) * inputs);
    std::vector<Bf16> host_weight_scales(static_cast<std::size_t>(ceil_div(outputs, group)) *
                                         groups);
    load_weights(checkpoint, shape, host_weights, host_weight_scales);
    Device<std::uint8_t> weights(host_weights.size());
    Device<Bf16> weight_scales(host_weight_scales.size());
    weights.upload(host_weights);
    weight_scales.upload(host_weight_scales);

    std::vector<Bf16> input(static_cast<std::size_t>(max_rows) * inputs);
    for (std::size_t index = 0; index < input.size(); ++index) {
        const int coordinate = static_cast<int>(((index * 23) + ((index / 619) * 17)) % 127) - 63;
        input[index] = __float2bfloat16(
            std::ldexp(static_cast<float>(coordinate), static_cast<int>((index / group) % 5) - 7));
    }
    Device<Bf16> bf16_input(input.size());
    const Device<std::uint8_t> activations(static_cast<std::size_t>(allocated_rows) * inputs);
    const Device<float> activation_scales(static_cast<std::size_t>(allocated_rows) * groups);
    bf16_input.upload(input);
    check(text_quantize_fp8_rows(bf16_input.data, inputs, max_rows,
                                 {activations.data, activation_scales.data}, stream));
    check(cudaStreamSynchronize(stream));
    std::vector<std::uint8_t> host_activations(input.size());
    std::vector<float> host_activation_scales(static_cast<std::size_t>(max_rows) * groups);
    check(cudaMemcpy(host_activations.data(), activations.data, host_activations.size(),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(host_activation_scales.data(), activation_scales.data,
                     host_activation_scales.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const TextFp8ActivationView activation_view{activations.data, activation_scales.data};

    std::vector<std::uint16_t> host_output(
        (guard * 2) + (static_cast<std::size_t>(max_rows) * outputs), poison);
    const Device<Bf16> output((guard * 2) + (static_cast<std::size_t>(allocated_rows) * outputs));
    for (const int rows : shape.rows) {
        const std::string key = std::to_string(rows) + "-" + std::to_string(outputs) +
                                "-" + std::to_string(inputs);
        const auto words = static_cast<std::size_t>(rows) * outputs;
        const auto active_words = words + (2 * guard);
        const auto active_bytes = active_words * sizeof(std::uint16_t);
        auto* const active = host_output.data();
        const auto* const result = active + guard;
        std::fill_n(active, active_words, poison);
        check(cudaMemcpy(output.data, active, active_bytes, cudaMemcpyHostToDevice));
        state.count = 0;
        const bool paired_shape = paired && rows == 1024 && outputs == 5120 && inputs == 17408;
        const auto status = try_text_fp8_gemm_sm100(weights.data, weight_scales.data,
            activation_view, rows, outputs, inputs, output.data + guard, stream, true);
        require(status == cudaSuccess && state.count == 1, key + " failed SM100 dispatch");
        const auto [tile_rows, tile_columns] = expected_tile(tiling, rows, outputs, inputs);
        const std::string kernel = "TileILi" + std::to_string(tile_rows) + "ELi" +
                                   std::to_string(tile_columns) + "E";
        const auto row_tiles = static_cast<unsigned>(ceil_div(rows, tile_rows));
        const auto column_tiles = static_cast<unsigned>(ceil_div(outputs, tile_columns));
        const auto& observed = state.observed;
        require(observed.grid.x == (paired_shape ? column_tiles * row_tiles : column_tiles) &&
                    observed.grid.y == (paired_shape ? 1U : row_tiles) &&
                    (observed.name.find(paired_kernel) != std::string::npos) == paired_shape &&
                    observed.name.find(kernel) != std::string::npos,
                key + " wrong launch geometry");
        check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(active, output.data, active_bytes, cudaMemcpyDeviceToHost));
        verify_output(active, words, key);
        double max_error = 0;
        for (const int row : {0, rows / 2, rows - 2, rows - 1}) {
            for (const int column : {0, 127, 128, outputs / 2, outputs - 2, outputs - 1}) {
                const double expected =
                    oracle(host_activations, host_activation_scales, host_weights,
                           host_weight_scales, inputs, {row, column});
                const double actual = bf16(result[(row * outputs) + column]);
                const double error = std::abs(actual - expected);
                max_error = std::max(max_error, error);
                require(error <= 0.004 * std::max(.001, std::abs(expected)),
                        key + " FP64 comparison failed at " + std::to_string(row) + "," +
                        std::to_string(column));
            }
        }
        check_or_save_bf16(baseline, key, result, words, compare);
        const auto* bytes = reinterpret_cast<const char*>(result);
        std::cout << key << '\t' << tile_rows << 'x' << tile_columns << '\t' << words << '\t'
                  << std::setprecision(8) << max_error << '\t'
                  << tokenizer::sha256({bytes, words * sizeof(*result)}) << '\n';
        if (paired_shape) {
            verify_paired_fallback(weights.data, weight_scales.data, activation_view,
                                   output.data + guard, stream);
        }
    }
    if (max_rows >= 1024) {
        verify_row_limits(shape, weights.data, weight_scales.data, activation_view,
                          output.data + guard, stream);
    }
}

}

extern "C" cudaError_t real_launch_kernel(const void* function, dim3 grid, dim3 block, void** args,
                                          std::size_t shared,
                                          cudaStream_t stream) asm("__real_cudaLaunchKernel");
extern "C" cudaError_t
real_func_get_attributes(cudaFuncAttributes* attributes,
                         const void* function) asm("__real_cudaFuncGetAttributes");
extern "C" cudaError_t
wrap_func_get_attributes(cudaFuncAttributes* attributes,
                         const void* function) asm("__wrap_cudaFuncGetAttributes");
extern "C" cudaError_t wrap_launch_kernel(const void* function, dim3 grid, dim3 block, void** args,
                                          std::size_t shared,
                                          cudaStream_t stream) asm("__wrap_cudaLaunchKernel");

extern "C" cudaError_t wrap_func_get_attributes(cudaFuncAttributes* attributes,
                                                const void* function) {
    auto& state = launches();
    if (state.decline_paired) {
        const char* name = nullptr;
        if (cudaFuncGetName(&name, function) == cudaSuccess && name != nullptr &&
            std::strstr(name, paired_kernel) != nullptr) {
            ++state.paired_attempts;
            return cudaErrorInvalidDeviceFunction;
        }
    }
    return real_func_get_attributes(attributes, function);
}

extern "C" cudaError_t wrap_launch_kernel(const void* function, dim3 grid, dim3 block, void** args,
                                          std::size_t shared, cudaStream_t stream) {
    auto& state = launches();
    const char* name = nullptr;
    const bool tma = cudaFuncGetName(&name, function) == cudaSuccess && name != nullptr &&
                     std::strstr(name, "fp8_tma") != nullptr;
    if (state.fail_paired_launch && tma && std::strstr(name, paired_kernel) != nullptr) {
        ++state.paired_attempts;
        return cudaErrorInvalidConfiguration;
    }
    const auto status = real_launch_kernel(function, grid, block, args, shared, stream);
    if (status == cudaSuccess && tma) {
        state.observed = {grid, name};
        ++state.count;
    }
    return status;
}

int main(int argc, const char* const* argv) try {
    require(argc == 4, "Pass checkpoint, base/tiling/wide-base/paired, and baseline directory");
    const std::string_view arm = argv[2];
    require(arm == "base" || arm == "tiling" || arm == "wide-base" || arm == "paired",
            "Unexpected arm");
    const bool paired = arm == "paired";
    const bool tiling = arm != "base";
    const bool compare = paired || arm == "tiling";
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
        projection(checkpoint, shape, tiling, paired, compare, argv[3], stream);
    }
    check(cudaStreamDestroy(stream));
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
