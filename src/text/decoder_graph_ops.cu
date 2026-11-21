#include "sglang/decoder_cuda.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {

constexpr int warp_size = 32;
constexpr int value_rows_per_block = 8;
constexpr int recurrent_threads = value_rows_per_block * warp_size;
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

__device__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__device__ float softplus(float value) {
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

struct GdnRows {
    const __nv_bfloat16* qkv;
    const __nv_bfloat16* z;
    const __nv_bfloat16* b;
    const __nv_bfloat16* a;
    __nv_bfloat16* convolved;
    __nv_bfloat16* observable;
    __nv_bfloat16* output;
};

__device__ std::size_t row_offset(int width) {
    return static_cast<std::size_t>(blockIdx.y) * width;
}

// Mirrors gdn_convolve for one decode token: taps read history, then the current row.
__global__ void packed_gdn_convolve(const int* active, GdnRows rows, const TextGdnState* states,
                                    const __nv_bfloat16* weights) {
    if (static_cast<int>(blockIdx.y) >= *active) {
        return;
    }
    const auto* history = states[blockIdx.y].history;
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
                     __bfloat162float(weights[channel * conv_taps + tap]), value);
    }
    rows.convolved[row_offset(text_gdn_qkv_width) + channel] =
        __float2bfloat16(value * sigmoid(value));
}

// Mirrors gdn_recur<round_qk = false, observe = true> for one decode token.
__global__ void packed_gdn_recur(const int* active, GdnRows rows, TextGdnWeights weights,
                                 const TextGdnState* states) {
    if (static_cast<int>(blockIdx.y) >= *active) {
        return;
    }
    __shared__ float qk[2][text_gdn_head_dim];
    __shared__ float warp_squares[recurrent_threads / warp_size];
    __shared__ float decay;
    __shared__ float beta;

    const int head = blockIdx.x / value_blocks_per_head;
    const int qk_head = head / value_heads_per_qk_head;
    const int warp = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int value = (blockIdx.x % value_blocks_per_head) * value_rows_per_block + warp;
    const int feature = head * text_gdn_head_dim + value;
    const int q_or_k = threadIdx.x / text_gdn_head_dim;
    const int qk_channel = threadIdx.x % text_gdn_head_dim;
    const int qk_offset = q_or_k * text_gdn_qk_width + qk_head * text_gdn_head_dim;
    const int first_warp = q_or_k * (text_gdn_head_dim / warp_size);
    const float negative_a = -expf(__bfloat162float(weights.a_log[head]));
    const float dt_bias = __bfloat162float(weights.dt_bias[head]);

    float* recurrent = states[blockIdx.y].recurrent;
    float state_keys[keys_per_lane];
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state_keys[part] = recurrent[feature * text_gdn_head_dim + part * warp_size + lane];
    }
    const auto* mixed = rows.convolved + row_offset(text_gdn_qkv_width);
    const float component = __bfloat162float(mixed[qk_offset + qk_channel]);
    const float partial = warp_sum(component * component);
    if (lane == 0) {
        warp_squares[warp] = partial;
    }
    if (threadIdx.x == 0) {
        const auto gate = row_offset(text_gdn_value_heads) + head;
        decay = expf(negative_a * softplus(__bfloat162float(rows.a[gate]) + dt_bias));
        beta = sigmoid(__bfloat162float(rows.b[gate]));
    }
    __syncthreads();
    float squares = 0.0f;
    #pragma unroll
    for (int part = 0; part < text_gdn_head_dim / warp_size; ++part) {
        squares += warp_squares[first_warp + part];
    }
    const float normalized = component * rsqrtf(squares + norm_epsilon);
    constexpr float query_scale = 0.08838834764831845f;
    qk[q_or_k][qk_channel] = q_or_k == 0 ? normalized * query_scale : normalized;
    __syncthreads();

    float prediction = 0.0f;
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state_keys[part] *= decay;
        prediction = fmaf(state_keys[part], qk[1][part * warp_size + lane], prediction);
    }
    const float target = __bfloat162float(mixed[2 * text_gdn_qk_width + feature]);
    const float correction = beta * (target - warp_sum(prediction));
    float result = 0.0f;
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        const int key = part * warp_size + lane;
        state_keys[part] = fmaf(correction, qk[1][key], state_keys[part]);
        result = fmaf(state_keys[part], qk[0][key], result);
    }
    const float sum = warp_sum(result);
    if (lane == 0) {
        rows.observable[row_offset(text_gdn_value_width) + feature] = __float2bfloat16(sum);
    }
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        recurrent[feature * text_gdn_head_dim + part * warp_size + lane] = state_keys[part];
    }
}

// Mirrors gdn_finish for one decode token: gated norm blocks, then history shift blocks.
__global__ void packed_gdn_finish(const int* active, GdnRows rows, const __nv_bfloat16* norm,
                                  const TextGdnState* states) {
    if (static_cast<int>(blockIdx.y) >= *active) {
        return;
    }
    const int channel = threadIdx.x;
    if (static_cast<int>(blockIdx.x) >= text_gdn_value_heads) {
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
        !inputs.b || !inputs.a || !inputs.convolved || !inputs.observable || !inputs.output ||
        !weights.convolution || !weights.a_log || !weights.dt_bias || !weights.norm) {
        return cudaErrorInvalidValue;
    }
    const GdnRows packed{inputs.qkv, inputs.z, inputs.b, inputs.a,
                         inputs.convolved, inputs.observable, inputs.output};
    packed_gdn_convolve<<<dim3(conv_blocks, rows), conv_threads, 0, stream>>>(
        active, packed, states, weights.convolution);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    packed_gdn_recur<<<dim3(text_gdn_value_heads * value_blocks_per_head, rows),
                       recurrent_threads, 0, stream>>>(active, packed, weights, states);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    packed_gdn_finish<<<dim3(text_gdn_value_heads + history_blocks, rows), text_gdn_head_dim, 0,
                        stream>>>(active, packed, weights.norm, states);
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
