#include "sglang/decoder_cuda.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {

constexpr int warp_size = 32;
constexpr int values_per_warp = 4;
constexpr int recurrent_warps = 8;
constexpr int streaming_recurrent_warps = 4;
constexpr int streaming_min_rows = 8;
constexpr int value_rows_per_block = recurrent_warps * values_per_warp;
constexpr int recurrent_threads = recurrent_warps * warp_size;
constexpr int value_blocks_per_head = text_gdn_head_dim / value_rows_per_block;
constexpr int value_heads_per_qk_head = text_gdn_value_heads / text_gdn_qk_heads;
constexpr int keys_per_lane = text_gdn_head_dim / warp_size;
constexpr int conv_threads = 128;
constexpr int conv_taps = text_gdn_history_taps + 1;
constexpr int conv_blocks = text_gdn_qkv_width / conv_threads;
constexpr int history_blocks = text_gdn_qkv_width / text_gdn_head_dim;
constexpr int embedding_threads = 256;
constexpr int argmax_threads = 256;
constexpr int argmax_tile = 1024;
constexpr float norm_epsilon = 1.0e-6f;

static_assert(recurrent_threads == 2 * text_gdn_head_dim);
static_assert(conv_threads == text_gdn_head_dim);
static_assert(text_argmax_scratch_count(argmax_tile) == 1);
static_assert(text_argmax_scratch_count(argmax_tile + 1) == 2);

__device__ float warp_sum(float value) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ float2 warp_sum(float2 value) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        const float2 peer{__shfl_xor_sync(0xffffffffu, value.x, offset),
                          __shfl_xor_sync(0xffffffffu, value.y, offset)};
        value = __fadd2_rn(value, peer);
    }
    return value;
}

__device__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ float softplus(float value) {
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

__device__ std::size_t row_offset(int width) {
    return static_cast<std::size_t>(blockIdx.y) * width;
}

// Mirrors gdn_convolve for one decode token: taps read history, then the current row.
template <bool AdvanceHistory>
__global__ void packed_gdn_convolve(const int* active, decoder::PackedGdnRows rows,
                                    const TextGdnState* states, TextGdnWeights weights) {
    if (static_cast<int>(blockIdx.y) >= *active) {
        return;
    }
    auto* history = states[blockIdx.y].history;
    const auto* qkv = rows.qkv + row_offset(text_gdn_qkv_width);
    const int channel = blockIdx.x * conv_threads + threadIdx.x;
    float value = 0.0f;
    #pragma unroll
    for (int tap = 0; tap < conv_taps; ++tap) {
        const int sample_row = tap - text_gdn_history_taps;
        const auto sample = sample_row < 0
            ? history[channel * text_gdn_history_taps + sample_row + text_gdn_history_taps]
            : qkv[channel];
        value = fmaf(__bfloat162float(sample),
                     __bfloat162float(weights.convolution[channel * conv_taps + tap]), value);
        if constexpr (AdvanceHistory) {
            if (tap > 0) {
                history[channel * text_gdn_history_taps + tap - 1] = sample;
            }
        }
    }
    const auto mixed = __float2bfloat16(value * sigmoid(value));
    constexpr int qk_width = 2 * text_gdn_qk_width;
    if (channel >= qk_width) {
        const int feature = channel - qk_width;
        rows.convolved_values[row_offset(text_gdn_value_width) + feature] = mixed;
        if (threadIdx.x == 0) {
            const int head = feature / text_gdn_head_dim;
            const auto gate = row_offset(text_gdn_value_heads) + head;
            const float negative_a = -expf(__bfloat162float(weights.a_log[head]));
            const float dt_bias = __bfloat162float(weights.dt_bias[head]);
            rows.gates[gate] = {
                expf(negative_a * softplus(__bfloat162float(rows.a[gate]) + dt_bias)),
                sigmoid(__bfloat162float(rows.b[gate]))};
        }
        return;
    }
    __shared__ float warp_squares[conv_threads / warp_size];
    const float component = __bfloat162float(mixed);
    const float partial = warp_sum(component * component);
    if (threadIdx.x % warp_size == 0) {
        warp_squares[threadIdx.x / warp_size] = partial;
    }
    __syncthreads();
    float squares = 0.0f;
    #pragma unroll
    for (int part = 0; part < conv_threads / warp_size; ++part) {
        squares += warp_squares[part];
    }
    const float normalized = component * rsqrtf(squares + norm_epsilon);
    constexpr float query_scale = 0.08838834764831845f;
    rows.normalized_qk[row_offset(qk_width) + channel] =
        channel < text_gdn_qk_width ? normalized * query_scale : normalized;
}

// Mirrors gdn_recur<round_qk = false, observe = true> for one decode token.
__global__ void packed_gdn_recur(const int* active, decoder::PackedGdnRows rows,
                                 const TextGdnState* states) {
    if (static_cast<int>(blockIdx.y) >= *active) {
        return;
    }
    const int head = blockIdx.x / value_blocks_per_head;
    const int qk_head = head / value_heads_per_qk_head;
    const int warp = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int value = (blockIdx.x % value_blocks_per_head) * value_rows_per_block + warp;
    const int feature = head * text_gdn_head_dim + value;
    const auto* query = rows.normalized_qk + row_offset(2 * text_gdn_qk_width)
                        + qk_head * text_gdn_head_dim;
    const auto* key = query + text_gdn_qk_width;
    const auto gate = rows.gates[row_offset(text_gdn_value_heads) + head];

    auto* recurrent = states[blockIdx.y].recurrent + feature * text_gdn_head_dim + lane;
    constexpr int state_stride = recurrent_warps * text_gdn_head_dim;
    float state_keys[values_per_warp][keys_per_lane];
    float prediction[values_per_warp]{};
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        const int offset = part * warp_size;
        const float component = key[offset + lane];
        #pragma unroll
        for (int row = 0; row < values_per_warp; ++row) {
            state_keys[row][part] = recurrent[row * state_stride + offset] * gate.x;
            prediction[row] = fmaf(state_keys[row][part], component, prediction[row]);
        }
    }
    const auto index = row_offset(text_gdn_value_width) + feature;
    float correction[values_per_warp];
    #pragma unroll
    for (int row = 0; row < values_per_warp; ++row) {
        const auto entry = index + row * recurrent_warps;
        const float target = __bfloat162float(rows.convolved_values[entry]);
        correction[row] = gate.y * (target - warp_sum(prediction[row]));
    }
    float result[values_per_warp]{};
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        const int offset = part * warp_size;
        const float key_component = key[offset + lane];
        const float query_component = query[offset + lane];
        #pragma unroll
        for (int row = 0; row < values_per_warp; ++row) {
            const float updated = fmaf(correction[row], key_component, state_keys[row][part]);
            result[row] = fmaf(updated, query_component, result[row]);
            recurrent[row * state_stride + offset] = updated;
        }
    }
    #pragma unroll
    for (int row = 0; row < values_per_warp; ++row) {
        const float sum = warp_sum(result[row]);
        if (lane == 0) {
            rows.observable[index + row * recurrent_warps] = __float2bfloat16(sum);
        }
    }
}

__global__ void packed_gdn_recur_streaming(const int* active, decoder::PackedGdnRows rows,
                                           const TextGdnState* states) {
    if (static_cast<int>(blockIdx.y) >= *active) {
        return;
    }
    constexpr int warps = streaming_recurrent_warps;
    constexpr int rows_per_block = values_per_warp * warps;
    constexpr int blocks_per_head = text_gdn_head_dim / rows_per_block;
    const int head = blockIdx.x / blocks_per_head;
    const int qk_head = head / value_heads_per_qk_head;
    const int warp = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int value = (blockIdx.x % blocks_per_head) * rows_per_block + warp;
    const int feature = head * text_gdn_head_dim + value;
    const auto* query = rows.normalized_qk + row_offset(2 * text_gdn_qk_width)
                        + qk_head * text_gdn_head_dim;
    const auto* key = query + text_gdn_qk_width;
    const auto gate = rows.gates[row_offset(text_gdn_value_heads) + head];
    const float2 decay{gate.x, gate.x};
    const float2 beta{gate.y, gate.y};

    auto* recurrent = states[blockIdx.y].recurrent + feature * text_gdn_head_dim + lane;
    constexpr int state_stride = warps * text_gdn_head_dim;
    constexpr int pairs_per_warp = values_per_warp / 2;
    float2 state_keys[pairs_per_warp][keys_per_lane];
    float2 prediction[pairs_per_warp]{};
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        const int offset = part * warp_size;
        const float component = key[offset + lane];
        #pragma unroll
        for (int pair = 0; pair < pairs_per_warp; ++pair) {
            const auto* address = recurrent + 2 * pair * state_stride + offset;
            const float2 state{__ldcs(address), __ldcs(address + state_stride)};
            state_keys[pair][part] = __fmul2_rn(state, decay);
            prediction[pair] = __ffma2_rn(state_keys[pair][part], {component, component},
                                         prediction[pair]);
        }
    }
    const auto index = row_offset(text_gdn_value_width) + feature;
    float2 correction[pairs_per_warp];
    #pragma unroll
    for (int pair = 0; pair < pairs_per_warp; ++pair) {
        const auto entry = index + 2 * pair * warps;
        const auto predicted = warp_sum(prediction[pair]);
        const float2 delta{__bfloat162float(rows.convolved_values[entry]) - predicted.x,
                           __bfloat162float(rows.convolved_values[entry + warps]) - predicted.y};
        correction[pair] = __fmul2_rn(beta, delta);
    }
    float2 result[pairs_per_warp]{};
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        const int offset = part * warp_size;
        const float key_component = key[offset + lane];
        const float query_component = query[offset + lane];
        #pragma unroll
        for (int pair = 0; pair < pairs_per_warp; ++pair) {
            const auto updated = __ffma2_rn(correction[pair], {key_component, key_component},
                                            state_keys[pair][part]);
            result[pair] = __ffma2_rn(updated, {query_component, query_component}, result[pair]);
            auto* address = recurrent + 2 * pair * state_stride + offset;
            __stcs(address, updated.x);
            __stcs(address + state_stride, updated.y);
        }
    }
    #pragma unroll
    for (int pair = 0; pair < pairs_per_warp; ++pair) {
        const auto sum = warp_sum(result[pair]);
        if (lane == 0) {
            const auto entry = index + 2 * pair * warps;
            rows.observable[entry] = __float2bfloat16(sum.x);
            rows.observable[entry + warps] = __float2bfloat16(sum.y);
        }
    }
}

// Mirrors gdn_finish for one decode token: gated norm blocks, then history shift blocks.
template <bool AdvanceHistory>
__global__ void packed_gdn_finish(const int* active, decoder::PackedGdnRows rows,
                                  const __nv_bfloat16* norm, const TextGdnState* states) {
    if (static_cast<int>(blockIdx.y) >= *active) {
        return;
    }
    const int channel = threadIdx.x;
    if (!AdvanceHistory && static_cast<int>(blockIdx.x) >= text_gdn_value_heads) {
        auto* history = states[blockIdx.y].history;
        const auto* qkv = rows.qkv + row_offset(text_gdn_qkv_width);
        const int feature = (blockIdx.x - text_gdn_value_heads) * text_gdn_head_dim + channel;
        #pragma unroll
        for (int tap = 0; tap < text_gdn_history_taps; ++tap) {
            const int sample_row = 1 - text_gdn_history_taps + tap;
            history[feature * text_gdn_history_taps + tap] = sample_row < 0
                ? history[feature * text_gdn_history_taps + sample_row + text_gdn_history_taps]
                : qkv[feature];
        }
        return;
    }
    __shared__ float warp_squares[text_gdn_head_dim / warp_size];
    const std::size_t index = row_offset(text_gdn_value_width) +
                              static_cast<std::size_t>(blockIdx.x) * text_gdn_head_dim + channel;
    const float value = __bfloat162float(rows.observable[index]);
    const float partial = warp_sum(value * value);
    if (channel % warp_size == 0) {
        warp_squares[channel / warp_size] = partial;
    }
    __syncthreads();
    float squares = 0.0f;
    #pragma unroll
    for (int warp = 0; warp < text_gdn_head_dim / warp_size; ++warp) {
        squares += warp_squares[warp];
    }
    const float inverse_rms = rsqrtf(squares / text_gdn_head_dim + norm_epsilon);
    const float gate = __bfloat162float(rows.z[index]);
    rows.output[index] = __float2bfloat16(value * inverse_rms * __bfloat162float(norm[channel])
                                          * (gate * sigmoid(gate)));
}

__global__ void packed_embedding_kernel(const __nv_bfloat16* weight, int vocab_size, int width,
                                        const int* tokens, __nv_bfloat16* output) {
    const auto column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column < static_cast<std::size_t>(width)) {
        const int token_id = tokens[blockIdx.y];
        const int row = token_id >= 0 && token_id < vocab_size ? token_id : 0;
        output[row_offset(width) + column] = weight[static_cast<std::size_t>(row) * width + column];
    }
}

__device__ TextLogitCandidate better_candidate(TextLogitCandidate best, TextLogitCandidate other) {
    if (other.token_id >= 0 &&
        (best.token_id < 0 || other.score > best.score ||
         (other.score == best.score && other.token_id < best.token_id))) {
        return other;
    }
    return best;
}

__device__ TextLogitCandidate warp_argmax(TextLogitCandidate best) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        const TextLogitCandidate other{
            __shfl_down_sync(0xffffffffu, best.score, offset),
            __shfl_down_sync(0xffffffffu, best.token_id, offset)};
        best = better_candidate(best, other);
    }
    return best;
}

__device__ void store_argmax(TextLogitCandidate best, TextLogitCandidate* output) {
    constexpr int warps = argmax_threads / warp_size;
    __shared__ TextLogitCandidate candidates[warps];
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    best = warp_argmax(best);
    if (lane == 0) {
        candidates[warp] = best;
    }
    __syncthreads();
    if (warp == 0) {
        best = warp_argmax(lane < warps ? candidates[lane]
                                        : TextLogitCandidate{-CUDART_INF_F, -1});
        if (lane == 0) {
            *output = best;
        }
    }
}

__global__ void packed_argmax_tiles(const __nv_bfloat16* logits, int vocab_size, const int* active,
                                    int tiles, TextLogitCandidate* scratch) {
    if (static_cast<int>(blockIdx.y) >= *active) {
        return;
    }
    const auto* scores = logits + row_offset(vocab_size);
    const unsigned int start = blockIdx.x * argmax_tile;
    TextLogitCandidate best{-CUDART_INF_F, -1};
    for (unsigned int item = threadIdx.x; item < argmax_tile; item += argmax_threads) {
        const unsigned int token = start + item;
        if (token < static_cast<unsigned int>(vocab_size)) {
            const float score = __bfloat162float(scores[token]);
            if (!isnan(score)) {
                best = better_candidate(best, {score, static_cast<int>(token)});
            }
        }
    }
    store_argmax(best, scratch + row_offset(tiles) + blockIdx.x);
}

__global__ void packed_argmax_finish(const TextLogitCandidate* scratch, int tiles,
                                     const int* active, TextLogitCandidate* output) {
    if (static_cast<int>(blockIdx.y) >= *active) {
        return;
    }
    const auto* candidates = scratch + row_offset(tiles);
    TextLogitCandidate best{-CUDART_INF_F, -1};
    for (int item = threadIdx.x; item < tiles; item += argmax_threads) {
        best = better_candidate(best, candidates[item]);
    }
    store_argmax(best, output + blockIdx.y);
}

}

cudaError_t decoder::packed_decode_embedding(const __nv_bfloat16* embedding, int vocab_size,
                                             int width, const int* tokens, int rows,
                                             __nv_bfloat16* output, cudaStream_t stream) {
    if (!embedding || !tokens || !output || vocab_size <= 0 || width <= 0 || rows <= 0 ||
        rows > 65535) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid((width + embedding_threads - 1) / embedding_threads, rows);
    packed_embedding_kernel<<<grid, embedding_threads, 0, stream>>>(embedding, vocab_size, width,
                                                                     tokens, output);
    return cudaGetLastError();
}

cudaError_t decoder::packed_decode_gdn(int rows, const int* active, PackedGdnRows inputs,
                                       TextGdnWeights weights, const TextGdnState* states,
                                       cudaStream_t stream) {
    if (rows <= 0 || rows > 65535 || !active || !states || !inputs.qkv || !inputs.z ||
        !inputs.b || !inputs.a || !inputs.normalized_qk || !inputs.gates ||
        !inputs.convolved_values || !inputs.observable || !inputs.output ||
        !weights.convolution || !weights.a_log || !weights.dt_bias || !weights.norm) {
        return cudaErrorInvalidValue;
    }
    if (rows >= streaming_min_rows) {
        packed_gdn_convolve<true><<<dim3(conv_blocks, rows), conv_threads, 0, stream>>>(
            active, inputs, states, weights);
        auto status = cudaGetLastError();
        if (status != cudaSuccess) {
            return status;
        }
        constexpr int blocks = text_gdn_value_width / (streaming_recurrent_warps * values_per_warp);
        packed_gdn_recur_streaming<<<dim3(blocks, rows), streaming_recurrent_warps * warp_size,
                                     0, stream>>>(active, inputs, states);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            return status;
        }
        packed_gdn_finish<true><<<dim3(text_gdn_value_heads, rows), text_gdn_head_dim, 0, stream>>>(
            active, inputs, weights.norm, states);
        return cudaGetLastError();
    }
    packed_gdn_convolve<false><<<dim3(conv_blocks, rows), conv_threads, 0, stream>>>(
        active, inputs, states, weights);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    packed_gdn_recur<<<dim3(text_gdn_value_heads * value_blocks_per_head, rows),
                       recurrent_threads, 0, stream>>>(active, inputs, states);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    packed_gdn_finish<false><<<dim3(text_gdn_value_heads + history_blocks, rows), text_gdn_head_dim,
                               0, stream>>>(active, inputs, weights.norm, states);
    return cudaGetLastError();
}

cudaError_t decoder::packed_decode_argmax(const __nv_bfloat16* logits, int vocab_size, int rows,
                                          const int* active, TextLogitCandidate* scratch,
                                          TextLogitCandidate* output, cudaStream_t stream) {
    if (!logits || !active || !scratch || !output || vocab_size <= 0 || rows <= 0 || rows > 65535) {
        return cudaErrorInvalidValue;
    }
    const int tiles = text_argmax_scratch_count(vocab_size);
    packed_argmax_tiles<<<dim3(tiles, rows), argmax_threads, 0, stream>>>(
        logits, vocab_size, active, tiles, scratch);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    packed_argmax_finish<<<dim3(1, rows), argmax_threads, 0, stream>>>(
        scratch, tiles, active, output);
    return cudaGetLastError();
}

}
