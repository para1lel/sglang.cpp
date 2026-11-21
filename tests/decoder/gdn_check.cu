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

#ifdef SGLANG_TEST_GDN_TILED
#include "sglang/decoder_cuda.cuh"

#include <cstring>
#include <fstream>
#include <string_view>
#include <utility>

extern "C" cudaError_t __real_cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t,
                                              cudaStream_t);

namespace {

bool trace_launches = false;
std::vector<std::pair<bool, unsigned int>> recurrent_launches;
#ifdef SGLANG_TEST_GDN_CHUNK
std::vector<std::array<unsigned int, 4>> chunk_launches;
#endif
cudaEvent_t recurrence_start = nullptr;
cudaEvent_t recurrence_stop = nullptr;

struct PrefillCase {
    Device<BFloat> qkv;
    Device<BFloat> z;
    Device<BFloat> beta;
    Device<BFloat> decay;
    Device<BFloat> history{qkv_width * history_width};
    Device<float> recurrent{state_size};
    Device<BFloat> scratch;
    Device<float2> gates;
    Device<BFloat> output;
    Device<BFloat> convolution{qkv_width * 4};
    Device<BFloat> a_log{value_heads};
    Device<BFloat> dt_bias{value_heads};
    Device<BFloat> norm{head_dim};
    Stream stream;

    PrefillCase(int rows, const Weights& weights, const Reference& reference)
        : qkv(rows * qkv_width), z(rows * value_width), beta(rows * value_heads),
          decay(rows * value_heads), scratch(rows * (qkv_width + value_width) + 1),
          gates(rows * value_heads), output(rows * value_width) {
        convolution.upload(weights.convolution);
        a_log.upload(weights.a_log);
        dt_bias.upload(weights.dt_bias);
        norm.upload(weights.norm);
        history.upload(reference.history);
        recurrent.upload(std::vector<float>(reference.recurrent.begin(),
                                            reference.recurrent.end()));
    }

    void upload(const Rows& rows) const {
        qkv.upload(rows.qkv);
        z.upload(rows.z);
        beta.upload(rows.beta);
        decay.upload(rows.decay);
        check(cudaMemsetAsync(output.data(), 0xff, rows.count * value_width * sizeof(BFloat),
                              stream.stream));
    }

    template<typename... ChunkScratch>
    void launch(int rows, int offset, ChunkScratch... chunk) const {
        const sglang::TextGdnWeights weights{convolution.data(), a_log.data(), dt_bias.data(),
                                             norm.data()};
        const sglang::TextGdnState state{history.data(), recurrent.data()};
        check(sglang::decoder::tensor_core_prefill_gdn(
            rows, qkv.data(), z.data(), beta.data(), decay.data(), weights, state,
            scratch.data() + offset, gates.data(), output.data(), stream.stream, chunk...));
    }
};

template <typename Value>
void verify_exact(const std::vector<Value>& left, const std::vector<Value>& right,
                  const char* label, std::ofstream& capture) {
    const auto bytes = left.size() * sizeof(Value);
    if (left.size() != right.size() || std::memcmp(left.data(), right.data(), bytes) != 0) {
        throw std::runtime_error(std::string(label) + " differs for the offset workspace");
    }
    capture.write(reinterpret_cast<const char*>(left.data()), bytes);
    if (!capture) {
        throw std::runtime_error("Write capture");
    }
}

void check_launch(std::size_t index, bool paired) {
    if (recurrent_launches.size() <= index ||
        recurrent_launches[index] != std::pair{paired, paired ? 128U : 256U}) {
        throw std::runtime_error("Unexpected recurrence kernel or block width");
    }
}

void compare_prefill(PrefillCase& aligned, PrefillCase& offset, Reference& reference,
                     const Weights& weights, const Rows& rows, bool paired,
                     std::ofstream& capture) {
    const auto expected = reference_forward(rows, true, weights, reference);
    aligned.upload(rows);
    offset.upload(rows);
    recurrent_launches.clear();
    trace_launches = true;
    aligned.launch(rows.count, 0);
    offset.launch(rows.count, 1);
    trace_launches = false;
    check_launch(0, paired && rows.count >= 32);
    check_launch(1, false);
    if (recurrent_launches.size() != 2) {
        throw std::runtime_error("Unexpected recurrence launch count");
    }
    check(cudaStreamSynchronize(aligned.stream.stream));
    check(cudaStreamSynchronize(offset.stream.stream));
    const auto output = aligned.output.download(expected.size());
    const auto state = aligned.recurrent.download(state_size);
    const auto history = aligned.history.download(reference.history.size());
    verify_exact(output, offset.output.download(expected.size()), "Gated output", capture);
    verify_exact(state, offset.recurrent.download(state_size), "Recurrent carry", capture);
    verify_exact(history, offset.history.download(history.size()), "Convolution carry", capture);
    double output_delta = 0;
    double state_delta = 0;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const double target = number(expected[index]);
        require_close("gated output", index, number(output[index]), target,
                      2 * bf16_ulp(target) + 2e-4);
        output_delta = std::max(output_delta, std::abs(number(output[index]) - target));
    }
    for (std::size_t index = 0; index < state.size(); ++index) {
        const double target = reference.recurrent[index];
        require_close("recurrent carry", index, state[index], target,
                      3e-6 + 5e-5 * std::abs(target));
        state_delta = std::max(state_delta, std::abs(state[index] - target));
    }
    for (std::size_t index = 0; index < history.size(); ++index) {
        require_close("convolution carry", index, number(history[index]),
                      number(reference.history[index]), 0);
    }
    std::cout << "rows=" << rows.count << " offset=2"
              << " paired=" << recurrent_launches[0].first << " outputs=" << output.size()
              << " fp64_state_delta=" << state_delta << " fp64_output_delta=" << output_delta
              << std::endl;
}

void check_segmented(bool paired, bool short_run, const char* path) {
    Reference reference;
    reference.history = input(qkv_width * history_width, 17, 1.0 / 128);
    for (std::size_t index = 0; index < reference.recurrent.size(); ++index) {
        reference.recurrent[index] = (static_cast<int>(index % 127) - 63) * (1.0 / 8192);
    }
    const Weights weights;
    const int capacity = short_run ? 129 : 2048;
    PrefillCase aligned(capacity, weights, reference);
    PrefillCase offset(capacity, weights, reference);
    const auto sizes = short_run ? std::vector{31, 32, 33, 129} :
                                   std::vector{31, 32, 33, 128, 512, 1024, 2048, 1};
    std::ofstream capture(path, std::ios::binary);
    int seed = 17;
    for (const int rows : sizes) {
        compare_prefill(aligned, offset, reference, weights, request_rows(rows, seed++),
                        paired, capture);
    }
    capture.close();
    if (!capture) {
        throw std::runtime_error("Close capture");
    }
}

#ifdef SGLANG_TEST_GDN_CHUNK
constexpr int chunk_rows = 32;
constexpr int guard_bytes = 32;
constexpr unsigned char poison = 0xa5;

std::size_t chunk_bytes(int rows) {
    constexpr int bytes_per_head = 2 * chunk_rows * chunk_rows * sizeof(BFloat) +
                                   3 * chunk_rows * sizeof(float);
    return (rows + chunk_rows - 1) / chunk_rows * value_heads * bytes_per_head;
}

struct ChunkResult {
    std::vector<BFloat> output;
    std::vector<float> state;
    std::vector<BFloat> history;
};

ChunkResult launch_chunk(const PrefillCase& gpu, const Rows& rows, unsigned char* scratch,
                         std::size_t bytes, bool chunked) {
    gpu.upload(rows);
    recurrent_launches.clear();
    chunk_launches.clear();
    trace_launches = true;
    gpu.launch(rows.count, 0, scratch, bytes);
    trace_launches = false;
    if (chunked) {
        const auto chunks = static_cast<unsigned>((rows.count + chunk_rows - 1) / chunk_rows);
        if (!recurrent_launches.empty() ||
            chunk_launches != std::vector<std::array<unsigned int, 4>>{
                {0, value_heads, chunks, 128}, {1, value_heads, 2, 256}}) {
            throw std::runtime_error("Expected V9 matrix preparation and warp MMA recurrence");
        }
    } else {
        check_launch(0, true);
        if (recurrent_launches.size() != 1 || !chunk_launches.empty()) {
            throw std::runtime_error("Expected only the paired ordered recurrence");
        }
    }
    check(cudaStreamSynchronize(gpu.stream.stream));
    return {gpu.output.download(rows.count * value_width), gpu.recurrent.download(state_size),
            gpu.history.download(qkv_width * history_width)};
}

template<typename Value>
void same_bits(const std::vector<Value>& actual, const std::vector<Value>& expected,
               std::size_t count, const char* label) {
    if (std::memcmp(actual.data(), expected.data(), count * sizeof(Value))) {
        throw std::runtime_error(label);
    }
}

struct Error {
    double maximum = 0;
    double squared = 0;
    double energy = 0;

    void add(double actual, double expected) {
        const double delta = std::abs(actual - expected);
        if (!std::isfinite(delta)) {
            throw std::runtime_error("Nonfinite GDN result");
        }
        maximum = std::max(maximum, delta);
        squared += delta * delta;
        energy += expected * expected;
    }

    double relative() const { return std::sqrt(squared / energy); }
};

void verify_chunk(const ChunkResult& result, const std::vector<BFloat>& expected,
                  const Reference& reference, int rows, const char* label, bool chunked) {
    same_bits(result.history, reference.history, reference.history.size(),
              "Convolution history differs from FP64 sequence");
    Error output, state;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double target = number(expected[index]);
        output.add(number(result.output[index]), target);
    }
    for (std::size_t index = 0; index < result.state.size(); ++index) {
        state.add(result.state[index], reference.recurrent[index]);
    }
    std::cout << label << " rows=" << rows << " output_max=" << output.maximum
              << " output_nrmse=" << output.relative() << " state_max=" << state.maximum
              << " state_nrmse=" << state.relative() << std::endl;
    if (output.maximum > (chunked ? 0.012 : 0.005) ||
        output.relative() > (chunked ? 0.025 : 0.003) ||
        state.maximum > (chunked ? 0.003 : 0.0002) ||
        state.relative() > (chunked ? 0.02 : 0.002)) {
        throw std::runtime_error(std::string(label) + " GDN exceeded the FP64 error budget");
    }
}

void verify_chunk_workspace(const Device<unsigned char>& buffer, std::size_t size,
                            std::size_t bytes) {
    const auto after = buffer.download(size);
    const auto payload = after.begin() + guard_bytes;
    if (!std::all_of(after.begin(), payload, [](auto byte) { return byte == poison; }) ||
        !std::all_of(payload + bytes, after.end(), [](auto byte) { return byte == poison; }) ||
        std::all_of(payload, payload + bytes, [](auto byte) { return byte == poison; })) {
        throw std::runtime_error("Unexpected chunk workspace footprint");
    }
}

void check_chunked() {
    constexpr int capacity = 193;
    const Weights weights;
    Reference reference;
    reference.history = input(qkv_width * history_width, 43, 1.0 / 128);
    for (std::size_t index = 0; index < state_size; ++index) {
        reference.recurrent[index] = (static_cast<int>(index % 127) - 63) * (1.0 / 8192);
    }
    PrefillCase ordered(capacity, weights, reference);
    PrefillCase chunked(capacity, weights, reference);
    PrefillCase alternate(capacity, weights, reference);
    PrefillCase exact_boundary(capacity, weights, reference);
    PrefillCase perturbed_boundary(capacity, weights, reference);
    const auto size = chunk_bytes(capacity) + 2 * guard_bytes;
    Device<unsigned char> buffer(size);
    auto* scratch = buffer.data() + guard_bytes;
    const std::vector<unsigned char> initial(size, poison);

    for (const int count : {127, 128, 129, 193}) {
        auto rows = request_rows(count, 71 + count);
        for (std::size_t index = 0; index < rows.decay.size(); ++index) {
            rows.decay[index] = as_bf16(number(rows.decay[index]) - 3.0);
        }
        auto changed_reference = reference;
        const auto expected = reference_forward(rows, true, weights, reference);
        const auto baseline = launch_chunk(ordered, rows, nullptr, 0, false);
        verify_chunk(baseline, expected, reference, count, "ordered", false);
        const auto bytes = chunk_bytes(count);
        buffer.upload(initial);
        if (count <= 128) {
            unsigned char* fallback = count == 128 ? scratch + 1 : scratch;
            const auto legal = launch_chunk(chunked, rows, fallback, bytes, false);
            same_bits(legal.output, baseline.output, expected.size(), "Fallback output differs");
            same_bits(legal.state, baseline.state, state_size, "Fallback state differs");
            same_bits(legal.history, baseline.history, reference.history.size(),
                      "Fallback history differs");
            if (buffer.download(size) != initial) {
                throw std::runtime_error("Ordered fallback wrote chunk workspace");
            }
        } else {
            const auto actual = launch_chunk(chunked, rows, scratch, bytes, true);
            verify_chunk(actual, expected, reference, count, "chunked", true);
            verify_chunk_workspace(buffer, size, bytes);
        }
        if (count == 127) {
            launch_chunk(alternate, rows, nullptr, 0, false);
            launch_chunk(exact_boundary, rows, nullptr, 0, false);
            launch_chunk(perturbed_boundary, rows, nullptr, 0, false);
        }
        if (count == 128) {
            const auto shortfall = launch_chunk(alternate, rows, scratch, bytes - 1, false);
            same_bits(shortfall.output, baseline.output, expected.size(), "Wrong shortfall output");
            same_bits(shortfall.state, baseline.state, state_size, "Shortfall state differs");
            same_bits(shortfall.history, baseline.history, reference.history.size(),
                      "Shortfall history differs");
            if (buffer.download(size) != initial) {
                throw std::runtime_error("Shortfall wrote chunk workspace");
            }
            const auto boundary = launch_chunk(exact_boundary, rows, scratch, bytes, true);
            verify_chunk(boundary, expected, reference, count, "chunked boundary", true);
            verify_chunk_workspace(buffer, size, bytes);
        }
        if (count == 128 || count == 129) {
            constexpr int change = chunk_rows + 7;
            auto changed = rows;
            std::fill_n(changed.qkv.begin() + change * qkv_width, qkv_width, as_bf16(0));
            std::fill_n(changed.beta.begin() + change * value_heads, value_heads, as_bf16(3));
            const auto changed_expected = reference_forward(changed, true, weights,
                                                            changed_reference);
            const auto& perturbed = count == 128 ? perturbed_boundary : alternate;
            const auto distinct = launch_chunk(perturbed, changed, scratch, bytes, true);
            verify_chunk(distinct, changed_expected, changed_reference, count, "perturbed", true);
            verify_chunk_workspace(buffer, size, bytes);
            const auto& unperturbed = count == 128 ? exact_boundary : chunked;
            const auto original = unperturbed.output.download(expected.size());
            const auto prefix = change * value_width;
            same_bits(distinct.output, original, prefix, "A future input changed a causal output");
            same_bits(distinct.history, unperturbed.history.download(reference.history.size()),
                      reference.history.size(), "An earlier perturbation changed the raw tail");
            const auto boundary = 2 * chunk_rows * value_width;
            if (!std::memcmp(distinct.output.data() + prefix, original.data() + prefix,
                             (boundary - prefix) * sizeof(BFloat)) ||
                !std::memcmp(distinct.output.data() + boundary, original.data() + boundary,
                             (expected.size() - boundary) * sizeof(BFloat))) {
                throw std::runtime_error("Perturbation did not reach both windows");
            }
        }
    }
}
#endif

class Event {
public:
    Event() { check(cudaEventCreate(&event)); }
    ~Event() { cudaEventDestroy(event); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t event = nullptr;
};

float elapsed(const Event& start, const Event& stop) {
    check(cudaEventSynchronize(stop.event));
    float ms = 0;
    check(cudaEventElapsedTime(&ms, start.event, stop.event));
    return ms * 1000;
}

void time_prefill(int iterations) {
    PrefillCase gpu(2048, Weights{}, Reference{});
    const Event start, stop, kernel_start, kernel_stop;
    for (const int rows : {32, 128, 512, 1024, 2048}) {
        gpu.upload(request_rows(rows, 51));
        for (int warmup = 0; warmup < 15; ++warmup) {
            gpu.launch(rows, 0);
        }
        check(cudaStreamSynchronize(gpu.stream.stream));
        for (int index = 0; index < iterations; ++index) {
            check(cudaEventRecord(start.event, gpu.stream.stream));
            gpu.launch(rows, 0);
            check(cudaEventRecord(stop.event, gpu.stream.stream));
            const auto total_us = elapsed(start, stop);
            recurrence_start = kernel_start.event;
            recurrence_stop = kernel_stop.event;
            gpu.launch(rows, 0);
            recurrence_start = nullptr;
            recurrence_stop = nullptr;
            const auto recurrence_us = elapsed(kernel_start, kernel_stop);
            std::cout << rows << ',' << index << ',' << total_us << ',' << recurrence_us << '\n';
        }
        check(cudaStreamSynchronize(gpu.stream.stream));
    }
}

}

extern "C" cudaError_t __wrap_cudaLaunchKernel(const void* function, dim3 grid, dim3 block,
                                               void** args, std::size_t shared,
                                               cudaStream_t stream) {
    if (trace_launches || recurrence_start) {
        const char* name = nullptr;
        check(cudaFuncGetName(&name, function));
        const std::string_view kernel(name);
#ifdef SGLANG_TEST_GDN_CHUNK
        if (trace_launches && kernel.find("gdn_chunk") != std::string_view::npos &&
            (kernel.find("prepare") != std::string_view::npos ||
             kernel.find("recur_warp") != std::string_view::npos)) {
            const unsigned int stage = kernel.find("recur_warp") != std::string_view::npos;
            chunk_launches.push_back({stage, grid.x, grid.y, block.x});
        }
#endif
        if (kernel.find("gdn_prefill_recur") != std::string_view::npos) {
            if (trace_launches) {
                recurrent_launches.emplace_back(
                    kernel.find("gdn_prefill_recur_paired") != std::string_view::npos, block.x);
            }
            if (recurrence_start) {
                check(cudaEventRecord(recurrence_start, stream));
            }
            const auto status = __real_cudaLaunchKernel(function, grid, block, args,
                                                        shared, stream);
            if (recurrence_stop) {
                check(cudaEventRecord(recurrence_stop, stream));
            }
            return status;
        }
    }
    return __real_cudaLaunchKernel(function, grid, block, args, shared, stream);
}
#endif

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    try {
#ifdef SGLANG_TEST_GDN_TILED
#ifdef SGLANG_TEST_GDN_CHUNK
        if (argc == 2 && std::string_view(argv[1]) == "chunked") {
            check_chunked();
            return 0;
        }
#endif
        if (argc == 3 && std::string_view(argv[1]) == "timing") {
            time_prefill(std::stoi(argv[2]));
            return 0;
        }
        if (argc == 4) {
            const std::string_view mode = argv[1];
            const std::string_view expected = argv[2];
            if ((mode == "segmented" || mode == "short") &&
                (expected == "paired" || expected == "original")) {
                check_segmented(expected == "paired", mode == "short", argv[3]);
                return 0;
            }
        }
        if (argc != 1) {
            throw std::invalid_argument(
                "Expected segmented|short paired|original CAPTURE, timing N or chunked");
        }
#endif
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
