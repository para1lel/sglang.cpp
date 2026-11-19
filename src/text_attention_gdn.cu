#include "sglang/decoder_cuda.cuh"

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

__device__ __nv_bfloat16* history_value(TextGdnState state) { return state.history; }
__device__ __nv_bfloat16* history_value(const TextGdnState* state) { return state->history; }
__device__ float* recurrent_value(TextGdnState state) { return state.recurrent; }
__device__ float* recurrent_value(const TextGdnState* state) { return state->recurrent; }

__device__ __nv_bfloat16 convolve(const __nv_bfloat16* qkv, const __nv_bfloat16* history,
                                  const __nv_bfloat16* weights, int row, int channel) {
    float value = 0.0f;
    #pragma unroll
    for (int tap = 0; tap < conv_taps; ++tap) {
        const int sample_row = row - text_gdn_history_taps + tap;
        const float raw = __bfloat162float(raw_sample(qkv, history, sample_row, channel));
        value = fmaf(raw, __bfloat162float(weights[channel * conv_taps + tap]), value);
    }
    return __float2bfloat16(value * sigmoid(value));
}

template <typename State>
__global__ void gdn_convolve(const __nv_bfloat16* qkv, State binding,
                             const __nv_bfloat16* weights, __nv_bfloat16* convolved) {
    const auto* history = history_value(binding);
    const int row = blockIdx.x / conv_blocks_per_token;
    const int channel = (blockIdx.x % conv_blocks_per_token) * conv_threads + threadIdx.x;
    convolved[static_cast<std::size_t>(row) * text_gdn_qkv_width + channel] =
        convolve(qkv, history, weights, row, channel);
}

__global__ void gdn_prefill_prepare(const __nv_bfloat16* qkv, const __nv_bfloat16* history,
                                    TextGdnWeights weights, const __nv_bfloat16* a,
                                    const __nv_bfloat16* b, __nv_bfloat16* convolved,
                                    float2* gates) {
    constexpr int blocks_per_row = conv_blocks_per_token + 1;
    const int row = blockIdx.x / blocks_per_row;
    const int block = blockIdx.x % blocks_per_row;
    const int feature = threadIdx.x;
    if (block == conv_blocks_per_token) {
        if (feature < text_gdn_value_heads) {
            const int index = row * text_gdn_value_heads + feature;
            const float negative_a = -expf(__bfloat162float(weights.a_log[feature]));
            const float dt_bias = __bfloat162float(weights.dt_bias[feature]);
            gates[index] = {expf(negative_a * softplus(__bfloat162float(a[index]) + dt_bias)),
                            sigmoid(__bfloat162float(b[index]))};
        }
        return;
    }
    const int channel = block * conv_threads + feature;
    auto mixed = convolve(qkv, history, weights.convolution, row, channel);
    if (channel < 2 * text_gdn_qk_width) {
        constexpr int warps = conv_threads / warp_size;
        __shared__ float warp_squares[warps];
        const float component = __bfloat162float(mixed);
        const float partial = warp_sum(component * component);
        if (feature % warp_size == 0) {
            warp_squares[feature / warp_size] = partial;
        }
        __syncthreads();
        float squares = 0.0f;
        #pragma unroll
        for (int warp = 0; warp < warps; ++warp) {
            squares += warp_squares[warp];
        }
        mixed = __float2bfloat16(component * rsqrtf(squares + norm_epsilon));
    }
    convolved[static_cast<std::size_t>(row) * text_gdn_qkv_width + channel] = mixed;
}

__global__ void gdn_prefill_recur(int rows, const __nv_bfloat16* convolved, const float2* gates,
                                  float* recurrent, __nv_bfloat16* observable) {
    const int head = blockIdx.x / value_blocks_per_head;
    const int qk_head = head / value_heads_per_qk_head;
    const int lane = threadIdx.x % warp_size;
    const int value = (blockIdx.x % value_blocks_per_head) * value_rows_per_block +
                      threadIdx.x / warp_size;
    const int feature = head * text_gdn_head_dim + value;
    const int q_offset = qk_head * text_gdn_head_dim + lane;
    const int k_offset = q_offset + text_gdn_qk_width;
    auto* state = recurrent + feature * text_gdn_head_dim + lane;
    float state_keys[keys_per_lane];
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state_keys[part] = state[part * warp_size];
    }
    for (int row = 0; row < rows; ++row) {
        const auto* mixed = convolved + static_cast<std::size_t>(row) * text_gdn_qkv_width;
        const auto gate = gates[row * text_gdn_value_heads + head];
        const float target = __bfloat162float(mixed[2 * text_gdn_qk_width + feature]);
        float keys[keys_per_lane];
        float queries[keys_per_lane];
        float prediction = 0.0f;
        #pragma unroll
        for (int part = 0; part < keys_per_lane; ++part) {
            const int key = part * warp_size;
            constexpr float query_scale = 0.08838834764831845f;
            keys[part] = __bfloat162float(mixed[k_offset + key]);
            queries[part] = __bfloat162float(mixed[q_offset + key]) * query_scale;
            state_keys[part] *= gate.x;
            prediction = fmaf(state_keys[part], keys[part], prediction);
        }
        const float correction = gate.y * (target - warp_sum(prediction));
        float result = 0.0f;
        #pragma unroll
        for (int part = 0; part < keys_per_lane; ++part) {
            state_keys[part] = fmaf(correction, keys[part], state_keys[part]);
            result = fmaf(state_keys[part], queries[part], result);
        }
        const float sum = warp_sum(result);
        if (lane == 0) {
            observable[static_cast<std::size_t>(row) * text_gdn_value_width + feature] =
                __float2bfloat16(sum);
        }
    }
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state[part * warp_size] = state_keys[part];
    }
}

constexpr int paired_recurrent_threads = recurrent_threads / 2;
constexpr int paired_recurrent_min_rows = 32;
constexpr int recurrent_tile_rows = 8;
constexpr int value_pairs_per_block = value_rows_per_block / 2;

__global__ void gdn_prefill_recur_paired(int rows, const __nv_bfloat16* convolved,
                                        const float2* gates, float* recurrent,
                                        __nv_bfloat16* observable) {
    constexpr int components_per_vector = sizeof(uint4) / sizeof(__nv_bfloat16);
    constexpr int vectors_per_head = text_gdn_head_dim / components_per_vector;
    __shared__ uint4 packed_qk[recurrent_tile_rows][2][vectors_per_head];
    __shared__ float2 tile_gates[recurrent_tile_rows];
    __shared__ __nv_bfloat162 tile_targets[recurrent_tile_rows][value_pairs_per_block];
    const int head = blockIdx.x / value_blocks_per_head;
    const int qk_head = head / value_heads_per_qk_head;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int first_feature = head * text_gdn_head_dim +
                              (blockIdx.x % value_blocks_per_head) * value_rows_per_block;
    const int feature = first_feature + 2 * warp;
    auto* first_state = recurrent + feature * text_gdn_head_dim + lane;
    auto* second_state = first_state + text_gdn_head_dim;
    float first_keys[keys_per_lane], second_keys[keys_per_lane];
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        first_keys[part] = first_state[part * warp_size];
        second_keys[part] = second_state[part * warp_size];
    }
    for (int first = 0; first < rows; first += recurrent_tile_rows) {
        const int count = min(recurrent_tile_rows, rows - first);
        constexpr int vectors_per_row = 2 * vectors_per_head;
        #pragma unroll
        for (int index = threadIdx.x; index < recurrent_tile_rows * vectors_per_row;
             index += paired_recurrent_threads) {
            const int local_row = index / vectors_per_row;
            const int plane = index / vectors_per_head % 2;
            const int vector = index % vectors_per_head;
            if (local_row < count) {
                const auto* mixed =
                    convolved + static_cast<std::size_t>(first + local_row) * text_gdn_qkv_width;
                const auto* source = reinterpret_cast<const uint4*>(
                    mixed + plane * text_gdn_qk_width + qk_head * text_gdn_head_dim);
                packed_qk[local_row][plane][vector] = source[vector];
            }
        }
        if (threadIdx.x < count * value_pairs_per_block) {
            const int local_row = threadIdx.x / value_pairs_per_block;
            const int pair = threadIdx.x % value_pairs_per_block;
            const int row = first + local_row;
            const auto* mixed = convolved + static_cast<std::size_t>(row) * text_gdn_qkv_width;
            const auto* target = reinterpret_cast<const __nv_bfloat162*>(
                mixed + 2 * text_gdn_qk_width + first_feature);
            tile_targets[local_row][pair] = target[pair];
            if (pair == 0) {
                tile_gates[local_row] = gates[row * text_gdn_value_heads + head];
            }
        }
        __syncthreads();
        for (int local_row = 0; local_row < count; ++local_row) {
            const auto* queries_bf16 =
                reinterpret_cast<const __nv_bfloat16*>(packed_qk[local_row][0]);
            const auto* keys_bf16 =
                reinterpret_cast<const __nv_bfloat16*>(packed_qk[local_row][1]);
            const auto gate = tile_gates[local_row];
            const auto targets = __bfloat1622float2(tile_targets[local_row][warp]);
            float keys[keys_per_lane], queries[keys_per_lane];
            float first_prediction = 0.0f, second_prediction = 0.0f;
            #pragma unroll
            for (int part = 0; part < keys_per_lane; ++part) {
                const int key = part * warp_size + lane;
                constexpr float query_scale = 0.08838834764831845f;
                keys[part] = __bfloat162float(keys_bf16[key]);
                queries[part] = __bfloat162float(queries_bf16[key]) * query_scale;
                first_keys[part] *= gate.x;
                second_keys[part] *= gate.x;
                first_prediction = fmaf(first_keys[part], keys[part], first_prediction);
                second_prediction = fmaf(second_keys[part], keys[part], second_prediction);
            }
            const float first_correction = gate.y * (targets.x - warp_sum(first_prediction));
            const float second_correction = gate.y * (targets.y - warp_sum(second_prediction));
            float first_result = 0.0f, second_result = 0.0f;
            #pragma unroll
            for (int part = 0; part < keys_per_lane; ++part) {
                first_keys[part] = fmaf(first_correction, keys[part], first_keys[part]);
                second_keys[part] = fmaf(second_correction, keys[part], second_keys[part]);
                first_result = fmaf(first_keys[part], queries[part], first_result);
                second_result = fmaf(second_keys[part], queries[part], second_result);
            }
            const float first_sum = warp_sum(first_result);
            const float second_sum = warp_sum(second_result);
            if (lane == 0) {
                auto* row_output =
                    observable + static_cast<std::size_t>(first + local_row) * text_gdn_value_width;
                reinterpret_cast<__nv_bfloat162*>(row_output)[feature / 2] =
                    __floats2bfloat162_rn(first_sum, second_sum);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        first_state[part * warp_size] = first_keys[part];
        second_state[part * warp_size] = second_keys[part];
    }
}

template <bool round_qk, bool observe, typename State>
__global__ void gdn_recur(int token_count, const __nv_bfloat16* convolved,
                          const __nv_bfloat16* b, const __nv_bfloat16* a,
                          TextGdnWeights weights, State initial, State final,
                          __nv_bfloat16* observable) {
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

    const auto* recurrent = recurrent_value(initial);
    float state_keys[keys_per_lane];
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state_keys[part] = recurrent[feature * text_gdn_head_dim + part * warp_size + lane];
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
            if constexpr (observe) {
                result = fmaf(state_keys[part], qk[0][key], result);
            }
        }
        if constexpr (observe) {
            const float sum = warp_sum(result);
            if (lane == 0) {
                observable[static_cast<std::size_t>(row) * text_gdn_value_width + feature] =
                    __float2bfloat16(sum);
            }
        }
        __syncthreads();
    }

    auto* final_recurrent = recurrent_value(final);
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        final_recurrent[feature * text_gdn_head_dim + part * warp_size + lane] =
            state_keys[part];
    }
}

template <typename State>
__global__ void gdn_finish(int token_count, const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                           const __nv_bfloat16* norm, const __nv_bfloat16* observable,
                           State initial, State final, __nv_bfloat16* output) {
    const int norm_blocks = output ? token_count * text_gdn_value_heads : 0;
    const int channel = threadIdx.x;
    if (blockIdx.x >= norm_blocks) {
        const auto* history = history_value(initial);
        auto* final_history = history_value(final);
        const int feature = (blockIdx.x - norm_blocks) * text_gdn_head_dim + channel;
        #pragma unroll
        for (int tap = 0; tap < text_gdn_history_taps; ++tap) {
            final_history[feature * text_gdn_history_taps + tap] =
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

struct GdnInputs {
    const __nv_bfloat16* qkv;
    __nv_bfloat16* convolved;
    const __nv_bfloat16* b;
    const __nv_bfloat16* a;
};

template <bool observe, typename State>
cudaError_t recur_and_finish(int token_count, TextGdnMode mode, GdnInputs rows,
                             const __nv_bfloat16* z, TextGdnWeights weights,
                             State initial, State final,
                             __nv_bfloat16* observable, __nv_bfloat16* output,
                             cudaStream_t stream) {
    constexpr int recurrent_blocks = text_gdn_value_heads * value_blocks_per_head;
    const auto recur = mode == TextGdnMode::prefill ? gdn_recur<true, observe, State> :
                                                    gdn_recur<false, observe, State>;
    void* recurrent_arguments[] = {
        &token_count, &rows.convolved, &rows.b, &rows.a, &weights, &initial, &final, &observable};
    const auto status = cudaLaunchKernel(recur, dim3(recurrent_blocks), dim3(recurrent_threads),
                                         recurrent_arguments, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    const int finish_blocks =
        (observe ? token_count * text_gdn_value_heads : 0) +
        text_gdn_qkv_width / text_gdn_head_dim;
    void* finish_arguments[] = {
        &token_count, &rows.qkv, &z, &weights.norm, &observable, &initial, &final, &output};
    return cudaLaunchKernel(gdn_finish<State>, dim3(finish_blocks), dim3(text_gdn_head_dim),
                            finish_arguments, 0, stream);
}

cudaError_t prefill_gdn(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state,
    __nv_bfloat16* workspace, float2* gates, __nv_bfloat16* output, cudaStream_t stream) {
    void* prepare_arguments[] = {&qkv, &state.history, &weights, &a, &b, &workspace, &gates};
    auto status = cudaLaunchKernel(gdn_prefill_prepare, dim3(rows * (conv_blocks_per_token + 1)),
                                   dim3(conv_threads), prepare_arguments, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    auto* observable = workspace + static_cast<std::size_t>(rows) * text_gdn_qkv_width;
    void* recurrent_arguments[] = {&rows, &workspace, &gates, &state.recurrent, &observable};
    const bool paired = rows >= paired_recurrent_min_rows &&
                        reinterpret_cast<std::uintptr_t>(workspace) % alignof(uint4) == 0;
    const auto recur = paired ? gdn_prefill_recur_paired : gdn_prefill_recur;
    const int threads = paired ? paired_recurrent_threads : recurrent_threads;
    status = cudaLaunchKernel(recur, dim3(text_gdn_value_heads * value_blocks_per_head),
                              dim3(threads), recurrent_arguments, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    const int finish_blocks =
        rows * text_gdn_value_heads + text_gdn_qkv_width / text_gdn_head_dim;
    void* finish_arguments[] = {
        &rows, &qkv, &z, &weights.norm, &observable, &state, &state, &output};
    return cudaLaunchKernel(gdn_finish<TextGdnState>, dim3(finish_blocks), dim3(text_gdn_head_dim),
                            finish_arguments, 0, stream);
}

template <typename State>
cudaError_t convolve(int token_count, GdnInputs rows, TextGdnWeights weights, State initial,
                     cudaStream_t stream) {
    void* args[] = {&rows.qkv, &initial, &weights.convolution, &rows.convolved};
    return cudaLaunchKernel(gdn_convolve<State>, dim3(token_count * conv_blocks_per_token),
                            dim3(conv_threads), args, 0, stream);
}

cudaError_t convolve_and_advance(int token_count, TextGdnMode mode, GdnInputs rows,
                                  const __nv_bfloat16* z, TextGdnWeights weights,
                                  TextGdnState initial, TextGdnState final,
                                  __nv_bfloat16* observable, __nv_bfloat16* output,
                                  cudaStream_t stream) {
    if ((mode != TextGdnMode::prefill && mode != TextGdnMode::decode) ||
        !rows.qkv || !rows.convolved || !rows.b || !rows.a || !z ||
        !weights.convolution || !weights.a_log || !weights.dt_bias || !weights.norm ||
        !initial.history || !initial.recurrent || !observable || !output) {
        return cudaErrorInvalidValue;
    }
    const auto status = convolve(token_count, rows, weights, initial, stream);
    if (status != cudaSuccess) {
        return status;
    }
    return recur_and_finish<true>(token_count, mode, rows, z, weights, initial, final,
                                  observable, output, stream);
}

bool valid_replay(int count, TextGdnReplayRows rows, TextGdnWeights weights,
                  TextGdnState initial, TextGdnState result) {
    return count >= 1 && count <= text_gdn_max_verify_rows && rows.qkv && rows.convolved &&
           rows.b && rows.a && weights.a_log && weights.dt_bias && initial.history &&
           initial.recurrent && result.history && result.recurrent &&
           result.history != initial.history && result.recurrent != initial.recurrent;
}

}

cudaError_t decoder::tensor_core_prefill_gdn(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state,
    __nv_bfloat16* workspace, float2* gates, __nv_bfloat16* output, cudaStream_t stream) {
    return prefill_gdn(rows, qkv, z, b, a, weights, state, workspace, gates, output, stream);
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
    const auto required_bytes = text_gdn_workspace_bytes(token_count);
    if (!workspace || required_bytes == 0 || workspace_bytes < required_bytes) {
        return cudaErrorInvalidValue;
    }
    auto* observable = workspace + static_cast<std::size_t>(token_count) * text_gdn_qkv_width;
    const GdnInputs rows{qkv, workspace, b, a};
    return convolve_and_advance(token_count, mode, rows, z, weights, state, state, observable,
                                output, stream);
}

cudaError_t text_gdn_verify_tail(int token_count, TextGdnReplayRows rows,
                                const __nv_bfloat16* z, TextGdnWeights weights,
                                TextGdnState initial, TextGdnState tail,
                                __nv_bfloat16* observable, __nv_bfloat16* output,
                                cudaStream_t stream) {
    if (!valid_replay(token_count, rows, weights, initial, tail)) {
        return cudaErrorInvalidValue;
    }
    return convolve_and_advance(token_count, TextGdnMode::decode,
                                {rows.qkv, rows.convolved, rows.b, rows.a}, z, weights,
                                initial, tail, observable, output, stream);
}

cudaError_t text_gdn_replay_prefix(int token_count, TextGdnReplayRows rows, TextGdnWeights weights,
                                  TextGdnState initial, TextGdnState result, cudaStream_t stream) {
    if (!valid_replay(token_count, rows, weights, initial, result)) {
        return cudaErrorInvalidValue;
    }
    return recur_and_finish<false>(token_count, TextGdnMode::decode,
                                   {rows.qkv, rows.convolved, rows.b, rows.a}, nullptr,
                                   weights, initial, result, nullptr, nullptr, stream);
}

cudaError_t decoder::decode_gdn(const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                                 const __nv_bfloat16* b, const __nv_bfloat16* a,
                                 TextGdnWeights weights, const TextGdnState* state,
                                 __nv_bfloat16* workspace, __nv_bfloat16* output,
                                 cudaStream_t stream) {
    const GdnInputs rows{qkv, workspace, b, a};
    const auto status = convolve(1, rows, weights, state, stream);
    if (status != cudaSuccess) {
        return status;
    }
    return recur_and_finish<true>(1, TextGdnMode::decode, rows, z, weights, state, state,
                                  workspace + text_gdn_qkv_width, output, stream);
}

}
