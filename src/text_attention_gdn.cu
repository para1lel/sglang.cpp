#include "sglang/decoder_cuda.cuh"
#include "sglang/text/gdn/gdn_chunk.cuh"
#include "sglang/text/gdn_local.h"
#include "sglang/text/text_cuda_primitives.cuh"
#include "sglang/text/gdn/gdn_finish_history.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace sglang {
namespace {

template <typename S, typename State>
__global__ void gdn_convolve(const __nv_bfloat16* qkv, State binding,
                             const __nv_bfloat16* weights, __nv_bfloat16* convolved) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    const auto* history = history_value(binding);
    const int row = blockIdx.x / conv_blocks_per_token;
    const int channel = (blockIdx.x % conv_blocks_per_token) * conv_threads + threadIdx.x;
    convolved[static_cast<std::size_t>(row) * S::qkv_width + channel] =
        convolve<S>(qkv, history, weights, row, channel);
}

template <typename S>
__global__ void gdn_convolve_lanes(int token_count,
                                   const __grid_constant__ GdnVerifyTailLanes jobs,
                                   const __nv_bfloat16* weights) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    const auto& job = jobs.lanes[blockIdx.z];
    const auto* history = history_value(job.initial);
    const int row = blockIdx.x / conv_blocks_per_token;
    const int channel = (blockIdx.x % conv_blocks_per_token) * conv_threads + threadIdx.x;
    job.inputs.convolved[static_cast<std::size_t>(row) * S::qkv_width + channel] =
        convolve<S>(job.inputs.qkv, history, weights, row, channel);
}

template <typename S, bool round_qk, bool observe, typename State>
__device__ void gdn_recur_core(int token_count, const __nv_bfloat16* convolved,
                          const __nv_bfloat16* b, const __nv_bfloat16* a,
                          TextGdnWeights weights, State initial, State final,
                          __nv_bfloat16* observable) {
    __shared__ float qk[2][text_gdn_head_dim];
    __shared__ float warp_squares[recurrent_threads / warp_size];
    __shared__ float decay;
    __shared__ float beta;

    const int head = blockIdx.x / value_blocks_per_head;
    const int qk_head = head / S::heads_per_qk;
    const int warp = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int value = (blockIdx.x % value_blocks_per_head) * value_rows_per_block + warp;
    const int feature = head * text_gdn_head_dim + value;
    const int q_or_k = threadIdx.x / text_gdn_head_dim;
    const int qk_channel = threadIdx.x % text_gdn_head_dim;
    const int qk_offset = q_or_k * S::qk_width + qk_head * text_gdn_head_dim;
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
            convolved + static_cast<std::size_t>(row) * S::qkv_width;
        const float component = __bfloat162float(mixed[qk_offset + qk_channel]);
        const float partial = warp_sum(component * component);
        if (lane == 0) {
            warp_squares[warp] = partial;
        }
        if (threadIdx.x == 0) {
            const int gate = row * S::value_heads + head;
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
        const float target = __bfloat162float(mixed[2 * S::qk_width + feature]);
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
                observable[static_cast<std::size_t>(row) * S::value_width + feature] =
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

template <typename S, bool round_qk, bool observe, typename State>
__global__ void gdn_recur(int token_count, const __nv_bfloat16* convolved,
                          const __nv_bfloat16* b, const __nv_bfloat16* a,
                          TextGdnWeights weights, State initial, State final,
                          __nv_bfloat16* observable) {
    gdn_recur_core<S, round_qk, observe, State>(token_count, convolved, b, a, weights, initial,
                                                final, observable);
}

template <typename S, bool round_qk, bool observe>
__global__ void gdn_recur_lanes(int token_count, const __grid_constant__ GdnVerifyTailLanes jobs,
                                TextGdnWeights weights) {
    const auto& job = jobs.lanes[blockIdx.z];
    gdn_recur_core<S, round_qk, observe, TextGdnState>(
        token_count, job.inputs.convolved, job.inputs.b, job.inputs.a, weights, job.initial,
        job.tail, job.observable);
}

__device__ __nv_bfloat16 finish_value(float value, float inverse_rms, __nv_bfloat16 norm,
                                      __nv_bfloat16 z) {
    const float gate = __bfloat162float(z);
    return __float2bfloat16(value * inverse_rms * __bfloat162float(norm)
                            * (gate * sigmoid(gate)));
}

template <typename S, typename State>
__device__ void gdn_finish_core(int token_count, const __nv_bfloat16* qkv,
                                const __nv_bfloat16* z, const __nv_bfloat16* norm,
                                const __nv_bfloat16* observable, State initial, State final,
                                __nv_bfloat16* output) {
    const int norm_blocks = output ? token_count * S::value_heads : 0;
    const int channel = threadIdx.x;
    if (blockIdx.x >= norm_blocks) {
        const int feature = (blockIdx.x - norm_blocks) * text_gdn_head_dim + channel;
        finish_history<S>(token_count, qkv, history_value(initial), history_value(final),
                          feature);
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
    output[index] = finish_value(value, inverse_rms, norm[channel], z[index]);
}

template <typename S, typename State>
__global__ void gdn_finish(int token_count, const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                           const __nv_bfloat16* norm, const __nv_bfloat16* observable,
                           State initial, State final, __nv_bfloat16* output) {
    gdn_finish_core<S, State>(token_count, qkv, z, norm, observable, initial, final, output);
}

template <typename S>
__global__ void gdn_finish_lanes(int token_count, const __grid_constant__ GdnVerifyTailLanes jobs,
                                 const __nv_bfloat16* norm) {
    const auto& job = jobs.lanes[blockIdx.z];
    gdn_finish_core<S, TextGdnState>(token_count, job.inputs.qkv, job.z, norm, job.observable,
                                     job.initial, job.tail, job.output);
}

constexpr int finish_heads_per_block = 4;
constexpr int finish_threads = finish_heads_per_block * warp_size;

template <typename S>
__host__ __device__ constexpr int quantized_finish_blocks(int rows) {
    return rows * (S::value_heads / finish_heads_per_block) +
           S::qkv_width / finish_threads;
}

template <typename S>
__device__ __forceinline__ void gdn_write_quantized(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* norm,
    const __nv_bfloat16* observable, TextGdnState state, TextFp8Activation output) {
    static_assert(text_gdn_head_dim == text_fp8_group_size);
    static_assert(S::value_heads % finish_heads_per_block == 0);
    constexpr int parts = text_gdn_head_dim / warp_size;
    const int norm_blocks = rows * (S::value_heads / finish_heads_per_block);
    if (blockIdx.x >= norm_blocks) {
        const int feature = (blockIdx.x - norm_blocks) * finish_threads + threadIdx.x;
        finish_history<S>(rows, qkv, state.history, state.history, feature);
        return;
    }

    const int lane = threadIdx.x % warp_size;
    const int group = blockIdx.x * finish_heads_per_block + threadIdx.x / warp_size;
    const auto offset = static_cast<std::size_t>(group) * text_gdn_head_dim + lane;
    float values[parts];
    float squares = 0.0f;
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const float value = __bfloat162float(observable[offset + part * warp_size]);
        values[part] = value;
        squares += warp_sum(value * value);
    }
    const float inverse_rms = rsqrtf(squares / text_gdn_head_dim + norm_epsilon);
    float maximum = 0.0f;
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const int channel = lane + part * warp_size;
        const auto value = finish_value(values[part], inverse_rms, norm[channel],
                                        z[offset + part * warp_size]);
        values[part] = __bfloat162float(value);
        maximum = fmaxf(maximum, fabsf(values[part]));
    }
    const float amax = text_kernel::warp_max(maximum);
    float reciprocal = 0.0f;
    if (lane == 0) {
        const float scale = __fdiv_rn(fmaxf(text_kernel::quant_epsilon, amax),
                                      text_kernel::fp8_max);
        output.scales[group] = scale;
        reciprocal = __fdiv_rn(1.0f, scale);
    }
    reciprocal = __shfl_sync(text_kernel::warp_mask, reciprocal, 0);
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        output.values[offset + part * warp_size] = __nv_cvt_float_to_fp8(
            values[part] * reciprocal, __NV_SATFINITE, __NV_E4M3);
    }
}

template <typename S>
__global__ void gdn_finish_quantized(int rows, const __nv_bfloat16* qkv,
                                     const __nv_bfloat16* z, const __nv_bfloat16* norm,
                                     const __nv_bfloat16* observable, TextGdnState state,
                                     TextFp8Activation output) {
    gdn_write_quantized<S>(rows, qkv, z, norm, observable, state, output);
}

struct GdnPrefillSegments {
    decoder::GdnPrefillSegment values[decoder::max_gdn_prefill_segments];
};

template <typename S>
__global__ void gdn_prepare_segments(const __grid_constant__ GdnPrefillSegments segments,
                                     TextGdnWeights weights) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    const auto& segment = segments.values[blockIdx.z];
    constexpr int blocks_per_row_group = conv_blocks_per_token + 1;
    if (blockIdx.x < segment.rows / prefill_rows_per_block * blocks_per_row_group) {
        gdn_prepare_aligned_rows<S>(segment.qkv, segment.state.history, weights, segment.a,
                                    segment.b, segment.workspace, segment.gates);
    }
}

template <typename S>
__global__ void gdn_finish_segments(const __grid_constant__ GdnPrefillSegments segments,
                                    const __nv_bfloat16* norm) {
    const auto& segment = segments.values[blockIdx.z];
    if (blockIdx.x < quantized_finish_blocks<S>(segment.rows)) {
        const auto* observable = segment.workspace +
                                 static_cast<std::size_t>(segment.rows) * S::qkv_width;
        gdn_write_quantized<S>(segment.rows, segment.qkv, segment.z, norm, observable,
                               segment.state, segment.output);
    }
}

struct GdnInputs {
    const __nv_bfloat16* qkv;
    __nv_bfloat16* convolved;
    const __nv_bfloat16* b;
    const __nv_bfloat16* a;
};

template <typename S, bool observe, typename State>
cudaError_t recur_and_finish(int token_count, TextGdnMode mode, GdnInputs rows,
                             const __nv_bfloat16* z, TextGdnWeights weights,
                             State initial, State final,
                             __nv_bfloat16* observable, __nv_bfloat16* output,
                             cudaStream_t stream) {
    constexpr int recurrent_blocks = S::value_heads * value_blocks_per_head;
    const auto recur = mode == TextGdnMode::prefill ? gdn_recur<S, true, observe, State> :
                                                    gdn_recur<S, false, observe, State>;
    void* recurrent_arguments[] = {
        &token_count, &rows.convolved, &rows.b, &rows.a, &weights, &initial, &final, &observable};
    const auto status = cudaLaunchKernel(recur, dim3(recurrent_blocks), dim3(recurrent_threads),
                                         recurrent_arguments, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    const int finish_blocks =
        (observe ? token_count * S::value_heads : 0) +
        S::qkv_width / text_gdn_head_dim;
    void* finish_arguments[] = {
        &token_count, &rows.qkv, &z, &weights.norm, &observable, &initial, &final, &output};
    return cudaLaunchKernel(gdn_finish<S, State>, dim3(finish_blocks), dim3(text_gdn_head_dim),
                            finish_arguments, 0, stream);
}

template <typename S, typename Output>
cudaError_t prefill_gdn(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state,
    __nv_bfloat16* workspace, float2* gates, Output output, cudaStream_t stream,
    void* chunk_scratch, std::size_t chunk_bytes) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    bool chunked = gdn_chunk::eligible<S>(rows, chunk_scratch, chunk_bytes);
    if (chunked) {
        const auto status = gdn_chunk::configure<S>(chunked);
        if (status != cudaSuccess) {
            return status;
        }
    }
    void* prepare_arguments[] = {&qkv, &state.history, &weights, &a, &b, &workspace, &gates};
    const bool grouped = rows >= gdn_chunk::min_count && rows % prefill_rows_per_block == 0;
    const auto prepare = grouped ? gdn_prefill_prepare_grouped<S> : gdn_prefill_prepare<S>;
    const int prepare_groups = grouped ? rows / prefill_rows_per_block : rows;
    auto status = cudaLaunchKernel(prepare, dim3(prepare_groups * (conv_blocks_per_token + 1)),
                                   dim3(conv_threads), prepare_arguments, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    auto* observable = workspace + static_cast<std::size_t>(rows) * S::qkv_width;
    void* recurrent_arguments[] = {&rows, &workspace, &gates, &state.recurrent, &observable};
    const bool paired = rows >= paired_recurrent_min_rows &&
                        reinterpret_cast<std::uintptr_t>(workspace) % alignof(uint4) == 0;
    const auto recur = paired ? gdn_prefill_recur_paired<S> : gdn_prefill_recur<S>;
    const int threads = paired ? paired_recurrent_threads : recurrent_threads;
    if (chunked) {
        status = gdn_chunk::launch<S>(rows, workspace, gates, chunk_scratch,
                                      state.recurrent, observable, stream);
    } else {
        status = cudaLaunchKernel(recur, dim3(S::value_heads * value_blocks_per_head),
                                  dim3(threads), recurrent_arguments, 0, stream);
    }
    if (status != cudaSuccess) {
        return status;
    }
    if constexpr (std::is_same_v<Output, TextFp8Activation>) {
        const int blocks = quantized_finish_blocks<S>(rows);
        void* arguments[] = {&rows, &qkv, &z, &weights.norm, &observable, &state, &output};
        return cudaLaunchKernel(gdn_finish_quantized<S>, dim3(blocks), dim3(finish_threads),
                                arguments, 0, stream);
    } else {
        const int finish_blocks =
            rows * S::value_heads + S::qkv_width / text_gdn_head_dim;
        void* finish_arguments[] = {
            &rows, &qkv, &z, &weights.norm, &observable, &state, &state, &output};
        return cudaLaunchKernel(gdn_finish<S, TextGdnState>, dim3(finish_blocks),
                                dim3(text_gdn_head_dim), finish_arguments, 0, stream);
    }
}

template <typename S, typename State>
cudaError_t convolve(int token_count, GdnInputs rows, TextGdnWeights weights, State initial,
                     cudaStream_t stream) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    void* args[] = {&rows.qkv, &initial, &weights.convolution, &rows.convolved};
    return cudaLaunchKernel(gdn_convolve<S, State>, dim3(token_count * conv_blocks_per_token),
                            dim3(conv_threads), args, 0, stream);
}

template <typename S, typename State>
cudaError_t enqueue_gdn(int token_count, TextGdnMode mode, GdnInputs rows,
                         const __nv_bfloat16* z, TextGdnWeights weights, State initial,
                         State final, __nv_bfloat16* observable, __nv_bfloat16* output,
                         cudaStream_t stream) {
    const auto status = convolve<S>(token_count, rows, weights, initial, stream);
    if (status != cudaSuccess) {
        return status;
    }
    return recur_and_finish<S, true>(token_count, mode, rows, z, weights, initial, final,
                                  observable, output, stream);
}

template <typename S>
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
    return enqueue_gdn<S>(token_count, mode, rows, z, weights, initial, final, observable,
                          output, stream);
}

bool valid_replay(int count, TextGdnReplayRows rows, TextGdnWeights weights,
                  TextGdnState initial, TextGdnState result) {
    return count >= 1 && count <= text_gdn_max_verify_rows && rows.qkv && rows.convolved &&
           rows.b && rows.a && weights.a_log && weights.dt_bias && initial.history &&
           initial.recurrent && result.history && result.recurrent &&
           result.history != initial.history && result.recurrent != initial.recurrent;
}

template <typename S>
cudaError_t verify_tail_lanes(int token_count, GdnVerifyTailLanes jobs,
                              TextGdnWeights weights, cudaStream_t stream) {
    if (token_count < 1 || token_count > text_gdn_max_verify_rows || jobs.lane_count < 1 ||
        jobs.lane_count > gdn_verify_tail_max_lanes) {
        return cudaErrorInvalidValue;
    }
    if (!weights.convolution || !weights.norm) {
        return cudaErrorInvalidValue;
    }
    for (int lane = 0; lane < jobs.lane_count; ++lane) {
        const auto& job = jobs.lanes[lane];
        if (!valid_replay(token_count, job.inputs, weights, job.initial, job.tail) || !job.z ||
            !job.observable || !job.output) {
            return cudaErrorInvalidValue;
        }
    }
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    void* convolve_arguments[] = {&token_count, &jobs, &weights.convolution};
    auto status = cudaLaunchKernel(
        gdn_convolve_lanes<S>, dim3(token_count * conv_blocks_per_token, 1, jobs.lane_count),
        dim3(conv_threads), convolve_arguments, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    constexpr int recurrent_blocks = S::value_heads * value_blocks_per_head;
    void* recur_arguments[] = {&token_count, &jobs, &weights};
    status = cudaLaunchKernel(gdn_recur_lanes<S, false, true>,
                              dim3(recurrent_blocks, 1, jobs.lane_count),
                              dim3(recurrent_threads), recur_arguments, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    const int finish_blocks =
        token_count * S::value_heads + S::qkv_width / text_gdn_head_dim;
    void* finish_arguments[] = {&token_count, &jobs, &weights.norm};
    return cudaLaunchKernel(gdn_finish_lanes<S>, dim3(finish_blocks, 1, jobs.lane_count),
                            dim3(text_gdn_head_dim), finish_arguments, 0, stream);
}

template <typename S>
std::size_t workspace_bytes(int rows) {
    return rows > 0 && rows <= text_attention_max_tokens
               ? sizeof(__nv_bfloat16) * static_cast<std::size_t>(rows) *
                     (S::qkv_width + S::value_width)
               : 0;
}

template <typename S>
cudaError_t forward_gdn(int token_count, TextGdnMode mode, const __nv_bfloat16* qkv,
                        const __nv_bfloat16* z, const __nv_bfloat16* b, const __nv_bfloat16* a,
                        TextGdnWeights weights, TextGdnState state, __nv_bfloat16* workspace,
                        std::size_t bytes, __nv_bfloat16* output, cudaStream_t stream) {
    if (token_count == 0) {
        return cudaSuccess;
    }
    const auto required_bytes = workspace_bytes<S>(token_count);
    if (!workspace || required_bytes == 0 || bytes < required_bytes) {
        return cudaErrorInvalidValue;
    }
    auto* observable = workspace + static_cast<std::size_t>(token_count) * S::qkv_width;
    const GdnInputs rows{qkv, workspace, b, a};
    return convolve_and_advance<S>(token_count, mode, rows, z, weights, state, state, observable,
                                   output, stream);
}

template <typename S>
std::optional<cudaError_t> prefill_gdn_group(const decoder::GdnPrefillSegment* input, int count,
                                             TextGdnWeights weights, cudaStream_t stream) {
    if (!input || count < 2 || count > decoder::max_gdn_prefill_segments ||
        !weights.convolution || !weights.a_log || !weights.dt_bias || !weights.norm) {
        return cudaErrorInvalidValue;
    }
    GdnPrefillSegments segments{};
    gdn_chunk::Segments chunk{};
    int max_rows = 0;
    for (int index = 0; index < count; ++index) {
        const auto& segment = input[index];
        const auto rows = segment.rows;
        if (workspace_bytes<S>(rows) == 0 || rows % prefill_rows_per_block || !segment.qkv ||
            !segment.z || !segment.b || !segment.a || !segment.workspace ||
            reinterpret_cast<std::uintptr_t>(segment.workspace) % alignof(uint4) ||
            !segment.gates || !segment.state.history || !segment.state.recurrent ||
            !segment.output.values || !segment.output.scales ||
            !gdn_chunk::eligible<S>(rows, segment.chunk_scratch, segment.chunk_bytes)) {
            return cudaErrorInvalidValue;
        }
        if (rows > max_rows) {
            max_rows = rows;
        }
        segments.values[index] = segment;
        chunk.values[index] = {
            rows, segment.workspace, segment.gates,
            static_cast<gdn_chunk::Matrices*>(segment.chunk_scratch), segment.state.recurrent,
            segment.workspace + static_cast<std::size_t>(rows) * S::qkv_width,
        };
    }

    bool supported = true;
    auto status = gdn_chunk::configure<S>(supported, true);
    if (status != cudaSuccess) {
        return status;
    }
    if (!supported) {
        return std::nullopt;
    }
    void* prepare_args[] = {&segments, &weights};
    const int prepare_blocks =
        max_rows / prefill_rows_per_block * (S::qkv_width / conv_threads + 1);
    status = cudaLaunchKernel(gdn_prepare_segments<S>, dim3(prepare_blocks, 1, count),
                              dim3(conv_threads), prepare_args, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    status = gdn_chunk::launch_segments<S>(chunk, count, max_rows, stream);
    if (status != cudaSuccess) {
        return status;
    }
    void* finish_args[] = {&segments, &weights.norm};
    return cudaLaunchKernel(gdn_finish_segments<S>,
                            dim3(quantized_finish_blocks<S>(max_rows), 1, count),
                            dim3(finish_threads), finish_args, 0, stream);
}

}

cudaError_t decoder::tensor_core_prefill_gdn(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state,
    __nv_bfloat16* workspace, float2* gates, __nv_bfloat16* output, cudaStream_t stream,
    void* chunk_scratch, std::size_t chunk_bytes) {
    return prefill_gdn<GdnFull>(rows, qkv, z, b, a, weights, state, workspace, gates, output,
                                stream, chunk_scratch, chunk_bytes);
}

cudaError_t decoder::tensor_core_prefill_gdn_local(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state,
    __nv_bfloat16* workspace, float2* gates, __nv_bfloat16* output, cudaStream_t stream,
    void* chunk_scratch, std::size_t chunk_bytes) {
    return prefill_gdn<GdnLocal>(rows, qkv, z, b, a, weights, state, workspace, gates, output,
                                 stream, chunk_scratch, chunk_bytes);
}

cudaError_t decoder::tensor_core_prefill_gdn(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state,
    __nv_bfloat16* workspace, float2* gates, TextFp8Activation output, cudaStream_t stream,
    void* chunk_scratch, std::size_t chunk_bytes) {
    if (workspace_bytes<GdnFull>(rows) == 0 || !output.values || !output.scales) {
        return cudaErrorInvalidValue;
    }
    return prefill_gdn<GdnFull>(rows, qkv, z, b, a, weights, state, workspace, gates, output,
                                stream, chunk_scratch, chunk_bytes);
}

cudaError_t decoder::tensor_core_prefill_gdn_local(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state,
    __nv_bfloat16* workspace, float2* gates, TextFp8Activation output, cudaStream_t stream,
    void* chunk_scratch, std::size_t chunk_bytes) {
    if (workspace_bytes<GdnLocal>(rows) == 0 || !output.values || !output.scales) {
        return cudaErrorInvalidValue;
    }
    return prefill_gdn<GdnLocal>(rows, qkv, z, b, a, weights, state, workspace, gates, output,
                                 stream, chunk_scratch, chunk_bytes);
}

std::optional<cudaError_t> decoder::tensor_core_prefill_gdn_group(
    const GdnPrefillSegment* input, int count, TextGdnWeights weights, cudaStream_t stream) {
    return prefill_gdn_group<GdnFull>(input, count, weights, stream);
}

std::optional<cudaError_t> decoder::tensor_core_prefill_gdn_group_local(
    const GdnPrefillSegment* input, int count, TextGdnWeights weights, cudaStream_t stream) {
    return prefill_gdn_group<GdnLocal>(input, count, weights, stream);
}

cudaError_t text_gdn_forward(int token_count, TextGdnMode mode, const __nv_bfloat16* qkv,
                             const __nv_bfloat16* z, const __nv_bfloat16* b,
                             const __nv_bfloat16* a, TextGdnWeights weights,
                             TextGdnState state, __nv_bfloat16* workspace,
                             std::size_t bytes, __nv_bfloat16* output, cudaStream_t stream) {
    return forward_gdn<GdnFull>(token_count, mode, qkv, z, b, a, weights, state, workspace,
                                bytes, output, stream);
}

cudaError_t text_gdn_forward_local(int token_count, TextGdnMode mode, const __nv_bfloat16* qkv,
                                   const __nv_bfloat16* z, const __nv_bfloat16* b,
                                   const __nv_bfloat16* a, TextGdnWeights weights,
                                   TextGdnState state, __nv_bfloat16* workspace,
                                   std::size_t bytes, __nv_bfloat16* output,
                                   cudaStream_t stream) {
    return forward_gdn<GdnLocal>(token_count, mode, qkv, z, b, a, weights, state, workspace,
                                 bytes, output, stream);
}

cudaError_t text_gdn_verify_tail(int token_count, TextGdnReplayRows rows,
                                const __nv_bfloat16* z, TextGdnWeights weights,
                                TextGdnState initial, TextGdnState tail,
                                __nv_bfloat16* observable, __nv_bfloat16* output,
                                cudaStream_t stream) {
    if (!valid_replay(token_count, rows, weights, initial, tail)) {
        return cudaErrorInvalidValue;
    }
    return convolve_and_advance<GdnFull>(token_count, TextGdnMode::decode,
                                         {rows.qkv, rows.convolved, rows.b, rows.a}, z, weights,
                                         initial, tail, observable, output, stream);
}

cudaError_t text_gdn_verify_tail_local(int token_count, TextGdnReplayRows rows,
                                       const __nv_bfloat16* z, TextGdnWeights weights,
                                       TextGdnState initial, TextGdnState tail,
                                       __nv_bfloat16* observable, __nv_bfloat16* output,
                                       cudaStream_t stream) {
    if (!valid_replay(token_count, rows, weights, initial, tail)) {
        return cudaErrorInvalidValue;
    }
    return convolve_and_advance<GdnLocal>(token_count, TextGdnMode::decode,
                                          {rows.qkv, rows.convolved, rows.b, rows.a}, z,
                                          weights, initial, tail, observable, output, stream);
}

cudaError_t text_gdn_verify_tail_local_lanes(int token_count, const GdnVerifyTailLanes& jobs,
                                             TextGdnWeights weights, cudaStream_t stream) {
    return verify_tail_lanes<GdnLocal>(token_count, jobs, weights, stream);
}

cudaError_t text_gdn_replay_prefix(int token_count, TextGdnReplayRows rows, TextGdnWeights weights,
                                  TextGdnState initial, TextGdnState result, cudaStream_t stream) {
    if (!valid_replay(token_count, rows, weights, initial, result)) {
        return cudaErrorInvalidValue;
    }
    return recur_and_finish<GdnFull, false>(token_count, TextGdnMode::decode,
                                            {rows.qkv, rows.convolved, rows.b, rows.a}, nullptr,
                                            weights, initial, result, nullptr, nullptr, stream);
}

cudaError_t text_gdn_replay_prefix_local(int token_count, TextGdnReplayRows rows,
                                         TextGdnWeights weights, TextGdnState initial,
                                         TextGdnState result, cudaStream_t stream) {
    if (!valid_replay(token_count, rows, weights, initial, result)) {
        return cudaErrorInvalidValue;
    }
    return recur_and_finish<GdnLocal, false>(token_count, TextGdnMode::decode,
                                             {rows.qkv, rows.convolved, rows.b, rows.a},
                                             nullptr, weights, initial, result, nullptr,
                                             nullptr, stream);
}

cudaError_t decoder::decode_gdn(const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                                 const __nv_bfloat16* b, const __nv_bfloat16* a,
                                 TextGdnWeights weights, const TextGdnState* state,
                                 __nv_bfloat16* workspace, __nv_bfloat16* output,
                                 cudaStream_t stream) {
    return enqueue_gdn<GdnFull, const TextGdnState*>(
        1, TextGdnMode::decode, {qkv, workspace, b, a}, z, weights, state, state,
        workspace + GdnFull::qkv_width, output, stream);
}

cudaError_t decoder::decode_gdn_local(const __nv_bfloat16* qkv, const __nv_bfloat16* z,
                                      const __nv_bfloat16* b, const __nv_bfloat16* a,
                                      TextGdnWeights weights, const TextGdnState* state,
                                      __nv_bfloat16* workspace, __nv_bfloat16* output,
                                      cudaStream_t stream) {
    return enqueue_gdn<GdnLocal, const TextGdnState*>(
        1, TextGdnMode::decode, {qkv, workspace, b, a}, z, weights, state, state,
        workspace + GdnLocal::qkv_width, output, stream);
}

cudaError_t decoder::decode_verify_gdn(int rows, TextGdnReplayRows inputs,
                                      const __nv_bfloat16* z, TextGdnWeights weights,
                                      const TextGdnState* initial, const TextGdnState* tail,
                                      __nv_bfloat16* observable, __nv_bfloat16* output,
                                      cudaStream_t stream) {
    const GdnInputs replay{inputs.qkv, inputs.convolved, inputs.b, inputs.a};
    return enqueue_gdn<GdnFull, const TextGdnState*>(rows, TextGdnMode::decode, replay, z,
                                                      weights, initial, tail, observable, output,
                                                      stream);
}

cudaError_t decoder::decode_verify_gdn_local(int rows, TextGdnReplayRows inputs,
                                            const __nv_bfloat16* z, TextGdnWeights weights,
                                            const TextGdnState* initial,
                                            const TextGdnState* tail,
                                            __nv_bfloat16* observable, __nv_bfloat16* output,
                                            cudaStream_t stream) {
    const GdnInputs replay{inputs.qkv, inputs.convolved, inputs.b, inputs.a};
    return enqueue_gdn<GdnLocal, const TextGdnState*>(rows, TextGdnMode::decode, replay, z,
                                                       weights, initial, tail, observable,
                                                       output, stream);
}

std::size_t text_gdn_local_chunk_scratch_bytes(int rows) {
    return gdn_chunk::scratch_bytes<GdnLocal>(rows);
}

}
