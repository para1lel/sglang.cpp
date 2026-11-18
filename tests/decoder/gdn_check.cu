#include "sglang/text_attention.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using BFloat = __nv_bfloat16;
constexpr int head_dim = 128;
constexpr int qk_heads = 16;
constexpr int value_heads = 48;
constexpr int qk_width = qk_heads * head_dim;
constexpr int value_width = value_heads * head_dim;
constexpr int qkv_width = 2 * qk_width + value_width;
constexpr int history_width = 3;
constexpr std::size_t state_size = value_width * head_dim;
constexpr double epsilon = 1e-6;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

template <typename Value>
class Device {
public:
    explicit Device(std::size_t count) : count_(count) {
        check(cudaMalloc(&data_, count_ * sizeof(Value)));
    }
    ~Device() { cudaFree(data_); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Value* data() const { return data_; }

    void upload(const std::vector<Value>& values) const {
        if (values.size() > count_) {
            throw std::length_error("Fixture exceeds device allocation");
        }
        check(cudaMemcpy(data_, values.data(), values.size() * sizeof(Value),
                         cudaMemcpyHostToDevice));
    }

    std::vector<Value> download(std::size_t count) const {
        if (count > count_) {
            throw std::length_error("Readback exceeds device allocation");
        }
        std::vector<Value> result(count);
        check(cudaMemcpy(result.data(), data_, count * sizeof(Value), cudaMemcpyDeviceToHost));
        return result;
    }

private:
    Value* data_ = nullptr;
    std::size_t count_;
};

class Stream {
public:
    Stream() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t stream = nullptr;
};

double sigmoid(double value) {
    return 1 / (1 + std::exp(-value));
}

double silu(double value) {
    return value * sigmoid(value);
}

BFloat as_bf16(double value) {
    return __double2bfloat16(value);
}

double number(BFloat value) {
    return __bfloat162float(value);
}

std::vector<BFloat> input(int count, int seed, double scale) {
    std::vector<BFloat> result(count);
    for (int index = 0; index < count; ++index) {
        const int centered = (index * 31 + (index / 13) * 43 + seed * 47) % 113 - 56;
        result[index] = as_bf16(centered * scale);
    }
    return result;
}

struct Weights {
    std::vector<BFloat> convolution = input(qkv_width * 4, 3, 1.0 / 128);
    std::vector<BFloat> a_log = input(value_heads, 5, 1.0 / 64);
    std::vector<BFloat> dt_bias = input(value_heads, 7, 1.0 / 128);
    std::vector<BFloat> norm = input(head_dim, 11, 1.0 / 128);
};

struct Rows {
    int count;
    std::vector<BFloat> qkv;
    std::vector<BFloat> z;
    std::vector<BFloat> beta;
    std::vector<BFloat> decay;
};

Rows request_rows(int count, int seed) {
    return {count, input(count * qkv_width, seed, 1.0 / 64),
            input(count * value_width, seed + 1, 1.0 / 64),
            input(count * value_heads, seed + 2, 1.0 / 64),
            input(count * value_heads, seed + 3, 1.0 / 64)};
}

struct Reference {
    std::vector<BFloat> history = std::vector<BFloat>(qkv_width * history_width, as_bf16(0));
    std::vector<double> recurrent = std::vector<double>(state_size);
};

std::vector<double> convolve(const Rows& rows, int row, const Weights& weights, Reference& ref) {
    std::vector<double> mixed(qkv_width);
    for (int channel = 0; channel < qkv_width; ++channel) {
        auto* history = ref.history.data() + channel * history_width;
        const auto current = rows.qkv[row * qkv_width + channel];
        const std::array<double, 4> samples{
            number(history[0]), number(history[1]), number(history[2]), number(current)};
        const auto* taps = weights.convolution.data() + channel * 4;
        const double value = std::inner_product(samples.begin(), samples.end(), taps, 0.0,
                                                std::plus<>{}, [](double sample, BFloat weight) {
                                                    return sample * number(weight);
                                                });
        const float accumulator = static_cast<float>(value);
        const float activation = accumulator * (1.0f / (1.0f + std::exp(-accumulator)));
        mixed[channel] = number(__float2bfloat16(activation));
        std::move(history + 1, history + history_width, history);
        history[history_width - 1] = current;
    }
    return mixed;
}

std::array<double, head_dim> normalize(const double* channels, bool prefill, double scale) {
    std::array<double, head_dim> output;
    const double energy = std::inner_product(channels, channels + head_dim, channels, 0.0);
    const double inverse = 1 / std::sqrt(energy + epsilon);
    for (int channel = 0; channel < head_dim; ++channel) {
        const double normalized = channels[channel] * inverse;
        output[channel] = scale * (prefill ? number(as_bf16(normalized)) : normalized);
    }
    return output;
}

void update_head(const double* mixed, const Rows& rows, int row, int head, bool prefill,
                 const Weights& weights, Reference& ref, BFloat* output) {
    const int qk_head = head / (value_heads / qk_heads);
    const auto query = normalize(mixed + qk_head * head_dim, prefill, 1 / std::sqrt(head_dim));
    const auto key = normalize(mixed + qk_width + qk_head * head_dim, prefill, 1);
    const auto* value = mixed + 2 * qk_width + head * head_dim;
    const int gate = row * value_heads + head;
    const double alpha = number(rows.decay[gate]) + number(weights.dt_bias[head]);
    const double decay = std::exp(-std::exp(number(weights.a_log[head])) *
                                  std::log1p(std::exp(alpha)));
    const double beta = sigmoid(number(rows.beta[gate]));
    std::array<double, head_dim> response;
    for (int feature = 0; feature < head_dim; ++feature) {
        auto* state = ref.recurrent.data() + (head * head_dim + feature) * head_dim;
        for (int channel = 0; channel < head_dim; ++channel) {
            state[channel] *= decay;
        }
        const double estimate = std::inner_product(key.begin(), key.end(), state, 0.0);
        for (int channel = 0; channel < head_dim; ++channel) {
            state[channel] += beta * (value[feature] - estimate) * key[channel];
        }
        response[feature] = number(as_bf16(
            std::inner_product(query.begin(), query.end(), state, 0.0)));
    }
    const double energy = std::inner_product(response.begin(), response.end(), response.begin(),
                                             0.0);
    const double rms = std::sqrt(energy / head_dim + epsilon);
    for (int feature = 0; feature < head_dim; ++feature) {
        const double gate_value = number(rows.z[row * value_width + head * head_dim + feature]);
        output[feature] = as_bf16(response[feature] / rms * number(weights.norm[feature]) *
                                  silu(gate_value));
    }
}

std::vector<BFloat> reference_forward(const Rows& rows, bool prefill, const Weights& weights,
                                     Reference& ref) {
    std::vector<BFloat> outputs(rows.count * value_width);
    for (int row = 0; row < rows.count; ++row) {
        const auto mixed = convolve(rows, row, weights, ref);
        for (int head = 0; head < value_heads; ++head) {
            update_head(mixed.data(), rows, row, head, prefill, weights, ref,
                        outputs.data() + row * value_width + head * head_dim);
        }
    }
    return outputs;
}

double bf16_ulp(double value) {
    return value == 0 ? std::ldexp(1.0, -133)
                      : std::ldexp(1.0, std::max(std::ilogb(std::abs(value)) - 7, -133));
}

void require_close(const char* boundary, std::size_t index, double actual, double expected,
                   double budget) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > budget) {
        throw std::runtime_error(std::string(boundary) + "[" + std::to_string(index) +
                                 "] got " + std::to_string(actual) + " expected " +
                                 std::to_string(expected));
    }
}

struct GpuCase {
    Device<BFloat> qkv{2 * qkv_width};
    Device<BFloat> z{2 * value_width};
    Device<BFloat> beta{2 * value_heads};
    Device<BFloat> decay{2 * value_heads};
    Device<BFloat> history{qkv_width * history_width};
    Device<float> recurrent{state_size};
    Device<BFloat> scratch{2 * (qkv_width + value_width)};
    Device<BFloat> output{2 * value_width};
    Device<BFloat> convolution{qkv_width * 4};
    Device<BFloat> a_log{value_heads};
    Device<BFloat> dt_bias{value_heads};
    Device<BFloat> norm{head_dim};
    Stream stream;
};

void verify_transition(GpuCase& gpu, Reference& ref, const Weights& weights,
                       const Rows& rows, bool prefill) {
    const auto expected = reference_forward(rows, prefill, weights, ref);
    gpu.qkv.upload(rows.qkv);
    gpu.z.upload(rows.z);
    gpu.beta.upload(rows.beta);
    gpu.decay.upload(rows.decay);
    gpu.output.upload(std::vector<BFloat>(rows.count * value_width,
                                         as_bf16(std::numeric_limits<double>::quiet_NaN())));
    const auto mode = prefill ? sglang::TextGdnMode::prefill : sglang::TextGdnMode::decode;
    const sglang::TextGdnWeights device_weights{
        gpu.convolution.data(), gpu.a_log.data(), gpu.dt_bias.data(), gpu.norm.data()};
    const sglang::TextGdnState device_state{gpu.history.data(), gpu.recurrent.data()};
    const auto bytes = 2 * sizeof(BFloat) * (qkv_width + value_width);
    check(sglang::text_gdn_forward(rows.count, mode, gpu.qkv.data(), gpu.z.data(), gpu.beta.data(),
                                  gpu.decay.data(), device_weights, device_state,
                                  gpu.scratch.data(), bytes, gpu.output.data(), gpu.stream.stream));
    check(cudaStreamSynchronize(gpu.stream.stream));
    const auto actual = gpu.output.download(expected.size());
    const auto recurrent = gpu.recurrent.download(state_size);
    const auto history = gpu.history.download(ref.history.size());
    double max_output_delta = 0;
    double max_state_delta = 0;
    for (std::size_t index = 0; index < history.size(); ++index) {
        require_close("convolution history", index, number(history[index]),
                      number(ref.history[index]), 0);
    }
    for (std::size_t index = 0; index < recurrent.size(); ++index) {
        const double target = ref.recurrent[index];
        require_close("recurrent state", index, recurrent[index], target,
                      3e-6 + 5e-5 * std::abs(target));
        max_state_delta = std::max(max_state_delta, std::abs(recurrent[index] - target));
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double target = number(expected[index]);
        require_close("gated output", index, number(actual[index]), target,
                      2 * bf16_ulp(target) + 2e-4);
        max_output_delta = std::max(max_output_delta, std::abs(number(actual[index]) - target));
    }
    std::cout << (prefill ? "prefill" : "decode") << " rows=" << rows.count
              << " history=" << history.size() << " recurrent=" << recurrent.size()
              << " outputs=" << actual.size() << " max_state_delta=" << max_state_delta
              << " max_output_delta=" << max_output_delta << '\n';
}

}

int main() {
    try {
        GpuCase gpu;
        Reference reference;
        const Weights weights;
        gpu.convolution.upload(weights.convolution);
        gpu.a_log.upload(weights.a_log);
        gpu.dt_bias.upload(weights.dt_bias);
        gpu.norm.upload(weights.norm);
        gpu.history.upload(reference.history);
        gpu.recurrent.upload(std::vector<float>(state_size));
        verify_transition(gpu, reference, weights, request_rows(2, 13), true);
        verify_transition(gpu, reference, weights, request_rows(1, 29), false);
        verify_transition(gpu, reference, weights, request_rows(1, 37), false);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GDN check failed: " << error.what() << '\n';
        return 1;
    }
}
