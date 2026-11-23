#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_internal.h"
#include "sglang/text/quantize_warp.cuh"

#include <cuda_runtime.h>

namespace sglang::decoder {
namespace {

using Batch = GdnVerificationBatch;
constexpr int warp_width = 32;
constexpr int parts = text_gdn_head_dim / warp_width;
constexpr int conv_threads = 128;
constexpr int tap_count = text_gdn_history_taps + 1;
constexpr int value_heads_per_qk = text_gdn_value_heads / text_gdn_qk_heads;
constexpr int recurrent_threads = 2 * text_gdn_head_dim;
constexpr int values_per_warp = 2;
constexpr int tile_values = recurrent_threads / warp_width * values_per_warp;
constexpr int tiles_per_head = text_gdn_head_dim / tile_values;
constexpr int finish_heads_per_block = 4;
constexpr int finish_threads = finish_heads_per_block * warp_width;
constexpr float epsilon = 1.0e-6f;
static_assert(text_gdn_head_dim == parts * warp_width && tiles_per_head * tile_values ==
              text_gdn_head_dim);

__device__ float sum_warp(float value) {
    for (int offset = warp_width / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

template <typename Rows, bool StoreHistory = false>
__global__ void gdn_verify_convolve(const __grid_constant__ Rows batch,
                                    const __nv_bfloat16* weights) {
    const auto& job = batch.entries[blockIdx.z];
    const int channel = blockIdx.x * conv_threads + threadIdx.x;
    float samples[Batch::rows + text_gdn_history_taps];
    float taps[tap_count];
    #pragma unroll
    for (int tap = 0; tap < tap_count; ++tap) {
        taps[tap] = __bfloat162float(weights[channel * tap_count + tap]);
    }
    #pragma unroll
    for (int tap = 0; tap < text_gdn_history_taps; ++tap) {
        samples[tap] = __bfloat162float(
            job.initial.history[channel * text_gdn_history_taps + tap]);
    }
    #pragma unroll
    for (int row = 0; row < Batch::rows; ++row) {
        samples[row + text_gdn_history_taps] =
            __bfloat162float(job.inputs.qkv[row * text_gdn_qkv_width + channel]);
    }
    if constexpr (StoreHistory) {
        #pragma unroll
        for (int tap = 0; tap < text_gdn_history_taps; ++tap) {
            constexpr int first = Batch::rows - text_gdn_history_taps;
            job.tail.history[channel * text_gdn_history_taps + tap] =
                job.inputs.qkv[(first + tap) * text_gdn_qkv_width + channel];
        }
    }
    #pragma unroll
    for (int row = 0; row < Batch::rows; ++row) {
        float value = 0.0f;
        #pragma unroll
        for (int tap = 0; tap < tap_count; ++tap) {
            value = fmaf(samples[row + tap], taps[tap], value);
        }
        job.inputs.convolved[row * text_gdn_qkv_width + channel] =
            __float2bfloat16(value * sigmoid(value));
    }
}

struct RecurrentInputs {
    float qk[Batch::rows][2][text_gdn_head_dim];
    float squares[Batch::rows][recurrent_threads / warp_width];
    float decay[Batch::rows];
    float beta[Batch::rows];
};

__device__ void prepare_recurrent(const GdnVerificationLane& job, TextGdnWeights weights,
                                   int head, RecurrentInputs& shared) {
    const int warp = threadIdx.x / warp_width;
    const int lane = threadIdx.x % warp_width;
    const int q_or_k = threadIdx.x / text_gdn_head_dim;
    const int channel = threadIdx.x % text_gdn_head_dim;
    const int qk_offset = q_or_k * text_gdn_qk_width +
                          head / value_heads_per_qk * text_gdn_head_dim;
    float components[Batch::rows];
    #pragma unroll
    for (int row = 0; row < Batch::rows; ++row) {
        const float component =
            __bfloat162float(job.inputs.convolved[row * text_gdn_qkv_width + qk_offset + channel]);
        components[row] = component;
        const float partial = sum_warp(component * component);
        if (lane == 0) {
            shared.squares[row][warp] = partial;
        }
    }
    if (threadIdx.x < Batch::rows) {
        const int row = threadIdx.x;
        const int gate = row * text_gdn_value_heads + head;
        const float negative_a = -expf(__bfloat162float(weights.a_log[head]));
        const float value = __bfloat162float(job.inputs.a[gate]) +
                            __bfloat162float(weights.dt_bias[head]);
        const float dt = fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
        shared.decay[row] = expf(negative_a * dt);
        shared.beta[row] = sigmoid(__bfloat162float(job.inputs.b[gate]));
    }
    __syncthreads();

    #pragma unroll
    for (int row = 0; row < Batch::rows; ++row) {
        float squares = 0.0f;
        #pragma unroll
        for (int part = 0; part < parts; ++part) {
            squares += shared.squares[row][q_or_k * parts + part];
        }
        const float normalized = components[row] * rsqrtf(squares + epsilon);
        constexpr float query_scale = 0.08838834764831845f;
        shared.qk[row][q_or_k][channel] = q_or_k == 0 ? normalized * query_scale : normalized;
    }
    __syncthreads();
}

__device__ float2 multiply_channels(float2 values, float factor) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1000
    return __fmul2_rn(values, {factor, factor});
#else
    return {values.x * factor, values.y * factor};
#endif
}

__device__ float2 multiply_add_channels(float2 values, float factor, float2 addends) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1000
    return __ffma2_rn(values, {factor, factor}, addends);
#else
    return {fmaf(values.x, factor, addends.x), fmaf(values.y, factor, addends.y)};
#endif
}

template <typename Rows>
__global__ void gdn_verify_recur(const __grid_constant__ Rows batch, TextGdnWeights weights) {
    const auto& job = batch.entries[blockIdx.z];
    const int head = blockIdx.x / tiles_per_head;
    const int warp = threadIdx.x / warp_width;
    const int lane = threadIdx.x % warp_width;
    const int first_value = head * text_gdn_head_dim +
                            (blockIdx.x % tiles_per_head) * tile_values + warp;
    constexpr int warps = recurrent_threads / warp_width;
    float2 state[parts];
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const auto* initial = job.initial.recurrent + first_value * text_gdn_head_dim +
                              part * warp_width + lane;
        state[part] = {initial[0], initial[warps * text_gdn_head_dim]};
    }

    __shared__ RecurrentInputs shared;
    prepare_recurrent(job, weights, head, shared);
    #pragma unroll
    for (int row = 0; row < Batch::rows; ++row) {
        float2 prediction{};
        #pragma unroll
        for (int part = 0; part < parts; ++part) {
            const int key = part * warp_width + lane;
            state[part] = multiply_channels(state[part], shared.decay[row]);
            prediction = multiply_add_channels(state[part], shared.qk[row][1][key], prediction);
        }
        const auto* target = job.inputs.convolved + row * text_gdn_qkv_width +
                             2 * text_gdn_qk_width + first_value;
        const float2 correction{
            shared.beta[row] * (__bfloat162float(target[0]) - sum_warp(prediction.x)),
            shared.beta[row] * (__bfloat162float(target[warps]) - sum_warp(prediction.y)),
        };
        float2 result{};
        #pragma unroll
        for (int part = 0; part < parts; ++part) {
            const int key = part * warp_width + lane;
            state[part] = multiply_add_channels(correction, shared.qk[row][1][key], state[part]);
            result = multiply_add_channels(state[part], shared.qk[row][0][key], result);
        }
        const float sum0 = sum_warp(result.x);
        const float sum1 = sum_warp(result.y);
        if (lane == 0) {
            auto* out = job.observable + row * text_gdn_value_width + first_value;
            out[0] = __float2bfloat16(sum0);
            out[warps] = __float2bfloat16(sum1);
        }
    }

    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        auto* tail = job.tail.recurrent + first_value * text_gdn_head_dim +
                     part * warp_width + lane;
        tail[0] = state[part].x;
        tail[warps * text_gdn_head_dim] = state[part].y;
    }
}

constexpr int finish_norm_blocks = Batch::rows * text_gdn_value_heads / finish_heads_per_block;

template <typename Rows, bool Quantize = false>
__device__ __forceinline__ void gdn_verify_finish_impl(const Rows& batch,
    const __nv_bfloat16* norm, const CompactGdnFp8Rows* packed) {
    const auto& job = batch.entries[blockIdx.z];
    if (!Quantize && blockIdx.x >= finish_norm_blocks) {
        const int feature = (blockIdx.x - finish_norm_blocks) * finish_threads + threadIdx.x;
        #pragma unroll
        for (int tap = 0; tap < text_gdn_history_taps; ++tap) {
            const int row = Batch::rows - text_gdn_history_taps + tap;
            job.tail.history[feature * text_gdn_history_taps + tap] =
                job.inputs.qkv[row * text_gdn_qkv_width + feature];
        }
        return;
    }
    const int lane = threadIdx.x % warp_width;
    const int group = blockIdx.x * finish_heads_per_block + threadIdx.x / warp_width;
    float values[parts];
    float squares = 0.0f;
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const int channel = part * warp_width + lane;
        const float value = __bfloat162float(job.observable[group * text_gdn_head_dim + channel]);
        values[part] = value;
        squares += sum_warp(value * value);
    }
    const float inverse_rms = rsqrtf(squares / text_gdn_head_dim + epsilon);
    __nv_bfloat16 activated[parts];
    #pragma unroll
    for (int part = 0; part < parts; ++part) {
        const int channel = part * warp_width + lane;
        const int offset = group * text_gdn_head_dim + channel;
        const float gate = __bfloat162float(job.z[offset]);
        activated[part] = __float2bfloat16(
            values[part] * inverse_rms * __bfloat162float(norm[channel]) * (gate * sigmoid(gate)));
        job.output[offset] = activated[part];
    }
    if constexpr (Quantize) {
        static_assert(text_fp8_group_size == text_gdn_head_dim);
        const int row = compact_gdn_row(blockIdx.z, group / text_gdn_value_heads);
        const int head = group % text_gdn_value_heads;
        const int offset = row * text_gdn_value_width + head * text_fp8_group_size + lane;
        float elements[parts];
        float maximum = text_kernel::quant_epsilon;
        #pragma unroll
        for (int part = 0; part < parts; ++part) {
            elements[part] = __bfloat162float(activated[part]);
            maximum = fmaxf(maximum, fabsf(elements[part]));
        }
        float reciprocal = 0.0f;
        if (const float warp_maximum = text_kernel::warp_max(maximum); lane == 0) {
            const float scale = __fdiv_rn(warp_maximum, text_kernel::fp8_max);
            packed->scales[row][head] = scale;
            reciprocal = __fdiv_rn(1.0f, scale);
        }
        reciprocal = __shfl_sync(text_kernel::warp_mask, reciprocal, 0);
        #pragma unroll
        for (int part = 0; part < parts; ++part) {
            packed->values[offset + part * warp_width] = __nv_cvt_float_to_fp8(
                elements[part] * reciprocal, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

__global__ void gdn_verify_finish(const __grid_constant__ Batch batch,
                                  const __nv_bfloat16* norm) {
    gdn_verify_finish_impl<Batch>(batch, norm, nullptr);
}

__global__ void gdn_verify_finish_compact(const __grid_constant__ CompactGdnVerificationBatch batch,
                                         const __nv_bfloat16* norm,
                                         const __grid_constant__ CompactGdnFp8Rows packed) {
    gdn_verify_finish_impl<CompactGdnVerificationBatch, true>(batch, norm, &packed);
}

}

cudaError_t decode_verify_gdn_batch(Batch batch, TextGdnWeights weights, cudaStream_t stream) {
    void* conv_args[] = {&batch, &weights.convolution};
    auto status = cudaLaunchKernel(gdn_verify_convolve<Batch>,
                                   dim3(text_gdn_qkv_width / conv_threads, 1, Batch::lanes),
                                   conv_threads, conv_args, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    void* recur_args[] = {&batch, &weights};
    status = cudaLaunchKernel(gdn_verify_recur<Batch>,
                              dim3(text_gdn_value_heads * tiles_per_head, 1, Batch::lanes),
                              recurrent_threads, recur_args, 0, stream);
    if (status != cudaSuccess) {
        return status;
    }
    constexpr int finish_blocks = finish_norm_blocks + text_gdn_qkv_width / finish_threads;
    void* finish_args[] = {&batch, &weights.norm};
    return cudaLaunchKernel(gdn_verify_finish, dim3(finish_blocks, 1, Batch::lanes),
                            finish_threads, finish_args, 0, stream);
}

namespace {

cudaError_t compact_launch_status(cudaError_t direct) {
    const auto recorded = cudaGetLastError();
    if (direct == cudaSuccess || (!CudaError::terminal(direct) && CudaError::terminal(recorded))) {
        return recorded;
    }
    return direct;
}

}

cudaError_t decode_verify_gdn_compact(CompactGdnVerificationBatch batch, int lanes,
                                      TextGdnWeights weights, CompactGdnFp8Rows output,
                                      cudaStream_t stream) {
    using Compact = CompactGdnVerificationBatch;
    if ((lanes != Compact::lanes / 2 && lanes != Compact::lanes) || !output.values) {
        return cudaErrorInvalidValue;
    }
    void* conv_args[] = {&batch, &weights.convolution};
    auto status = compact_launch_status(cudaLaunchKernel(gdn_verify_convolve<Compact, true>,
        dim3(text_gdn_qkv_width / conv_threads, 1, lanes), conv_threads, conv_args, 0, stream));
    if (status != cudaSuccess) {
        return status;
    }
    void* recur_args[] = {&batch, &weights};
    status = compact_launch_status(cudaLaunchKernel(gdn_verify_recur<Compact>,
        dim3(text_gdn_value_heads * tiles_per_head, 1, lanes), recurrent_threads,
        recur_args, 0, stream));
    if (status != cudaSuccess) {
        return status;
    }
    void* finish_args[] = {&batch, &weights.norm, &output};
    return compact_launch_status(cudaLaunchKernel(gdn_verify_finish_compact,
        dim3(finish_norm_blocks, 1, lanes), finish_threads, finish_args, 0, stream));
}

}
