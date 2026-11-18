#include "sglang/text_attention.h"

#include <cuda_runtime.h>

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
constexpr int conv_blocks_per_token = text_gdn_qkv_width / conv_threads;
constexpr float norm_epsilon = 1.0e-6f;

static_assert(recurrent_threads == 2 * text_gdn_head_dim);
static_assert(text_gdn_head_dim % warp_size == 0);
static_assert(text_gdn_value_heads % text_gdn_qk_heads == 0);
static_assert(text_gdn_qkv_width % conv_threads == 0);
static_assert(conv_threads == text_gdn_head_dim);

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

__device__ __nv_bfloat16 raw_sample(const __nv_bfloat16* qkv,
                                    const __nv_bfloat16* history, int row, int channel) {
    if (row < 0) {
        return history[channel * text_gdn_history_taps + row + text_gdn_history_taps];
    }
    return qkv[static_cast<std::size_t>(row) * text_gdn_qkv_width + channel];
}

__global__ void gdn_convolve(const __nv_bfloat16* qkv, const __nv_bfloat16* history,
                             const __nv_bfloat16* weights, __nv_bfloat16* convolved) {
    const int row = blockIdx.x / conv_blocks_per_token;
    const int channel = (blockIdx.x % conv_blocks_per_token) * conv_threads + threadIdx.x;
    float value = 0.0f;
    #pragma unroll
    for (int tap = 0; tap < conv_taps; ++tap) {
        const int sample_row = row - text_gdn_history_taps + tap;
        const float raw = __bfloat162float(raw_sample(qkv, history, sample_row, channel));
        value = fmaf(raw, __bfloat162float(weights[channel * conv_taps + tap]), value);
    }
    convolved[static_cast<std::size_t>(row) * text_gdn_qkv_width + channel] =
        __float2bfloat16(value * sigmoid(value));
}

template <bool round_qk>
__global__ void gdn_recur(int token_count, const __nv_bfloat16* convolved,
                          const __nv_bfloat16* b, const __nv_bfloat16* a,
                          TextGdnWeights weights, float* recurrent, __nv_bfloat16* observable) {
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
    float* const state_row = recurrent + feature * text_gdn_head_dim;
    const int q_or_k = threadIdx.x / text_gdn_head_dim;
    const int qk_channel = threadIdx.x % text_gdn_head_dim;
    const int qk_offset = q_or_k * text_gdn_qk_width + qk_head * text_gdn_head_dim;
    const int first_warp = q_or_k * (text_gdn_head_dim / warp_size);
    const float negative_a = -expf(__bfloat162float(weights.a_log[head]));
    const float dt_bias = __bfloat162float(weights.dt_bias[head]);

    float state_keys[keys_per_lane];
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state_keys[part] = state_row[part * warp_size + lane];
    }

    for (int row = 0; row < token_count; ++row) {
        const __nv_bfloat16* const mixed =
            convolved + static_cast<std::size_t>(row) * text_gdn_qkv_width;
        const float component = __bfloat162float(mixed[qk_offset + qk_channel]);
        const float partial = warp_sum(component * component);
        if (lane == 0) {
            warp_squares[warp] = partial;
        }
        if (threadIdx.x == 0) {
            const int gate = row * text_gdn_value_heads + head;
            decay = expf(negative_a * softplus(__bfloat162float(a[gate]) + dt_bias));
            beta = sigmoid(__bfloat162float(b[gate]));
        }
        __syncthreads();

        float squares = 0.0f;
        #pragma unroll
        for (int part = 0; part < text_gdn_head_dim / warp_size; ++part) {
            squares += warp_squares[first_warp + part];
        }
        float normalized = component * rsqrtf(squares + norm_epsilon);
        if constexpr (round_qk) {
            normalized = __bfloat162float(__float2bfloat16(normalized));
        }
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
            observable[static_cast<std::size_t>(row) * text_gdn_value_width + feature] =
                __float2bfloat16(sum);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state_row[part * warp_size + lane] = state_keys[part];
    }
}

__global__ void gdn_finish(int token_count, const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                           const __nv_bfloat16* norm, const __nv_bfloat16* observable,
                           __nv_bfloat16* history, __nv_bfloat16* output) {
    const int norm_blocks = token_count * text_gdn_value_heads;
    const int channel = threadIdx.x;
    if (blockIdx.x >= norm_blocks) {
        const int feature = (blockIdx.x - norm_blocks) * text_gdn_head_dim + channel;
        #pragma unroll
        for (int tap = 0; tap < text_gdn_history_taps; ++tap) {
            history[feature * text_gdn_history_taps + tap] =
                raw_sample(qkv, history, token_count - text_gdn_history_taps + tap, feature);
        }
        return;
    }

    __shared__ float warp_squares[text_gdn_head_dim / warp_size];
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * text_gdn_head_dim + channel;
    const float value = __bfloat162float(observable[index]);
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
    const float gate = __bfloat162float(z[index]);
    output[index] = __float2bfloat16(value * inverse_rms * __bfloat162float(norm[channel])
                                    * (gate * sigmoid(gate)));
}

}

cudaError_t text_gdn_forward(int token_count, TextGdnMode mode, const __nv_bfloat16* qkv,
                             const __nv_bfloat16* z, const __nv_bfloat16* b,
                             const __nv_bfloat16* a, TextGdnWeights weights,
                             TextGdnState state, __nv_bfloat16* workspace,
                             std::size_t workspace_bytes, __nv_bfloat16* output,
                             cudaStream_t stream) {
    if (token_count == 0) {
        return cudaSuccess;
    }
    const std::size_t required_bytes = text_gdn_workspace_bytes(token_count);
    if (required_bytes == 0 || workspace_bytes < required_bytes
        || (mode != TextGdnMode::prefill && mode != TextGdnMode::decode)
        || !qkv || !z || !b || !a || !weights.convolution || !weights.a_log
        || !weights.dt_bias || !weights.norm || !state.history || !state.recurrent
        || !workspace || !output) {
        return cudaErrorInvalidValue;
    }

    auto* observable = workspace + static_cast<std::size_t>(token_count) * text_gdn_qkv_width;
    void* conv_arguments[] = {&qkv, &state.history, &weights.convolution, &workspace};
    cudaError_t status = cudaLaunchKernel(gdn_convolve, dim3(token_count * conv_blocks_per_token),
                                         dim3(conv_threads), conv_arguments, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    constexpr int recurrent_blocks = text_gdn_value_heads * value_blocks_per_head;
    const auto recur = mode == TextGdnMode::prefill ? gdn_recur<true> : gdn_recur<false>;
    void* recurrent_arguments[] = {
        &token_count, &workspace, &b, &a, &weights, &state.recurrent, &observable};
    status = cudaLaunchKernel(recur, dim3(recurrent_blocks), dim3(recurrent_threads),
                              recurrent_arguments, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    const int finish_blocks =
        token_count * text_gdn_value_heads + text_gdn_qkv_width / text_gdn_head_dim;
    void* finish_arguments[] = {
        &token_count, &qkv, &z, &weights.norm, &observable, &state.history, &output};
    return cudaLaunchKernel(gdn_finish, dim3(finish_blocks), dim3(text_gdn_head_dim),
                            finish_arguments, 0, stream);
}

}
