#include "sglang/decoder_cuda.cuh"
#include "sglang/text/gdn/gdn_chunk.cuh"
#include "sglang/text/gdn_local.h"
#include "sglang/text/text_cuda_primitives.cuh"
#include "sglang/text/gdn/gdn_prefill_kernels.cuh"

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace sglang::gdn_prefill {

__device__ __inline_hint__ float warp_sum(float value) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

}

namespace sglang {
namespace {

using namespace gdn_prefill;

struct GdnRecurInput {
    const __nv_bfloat16* convolved;
    const __nv_bfloat16* b;
    const __nv_bfloat16* a;
};

struct GdnFinish {
    const __nv_bfloat16* qkv;
    const __nv_bfloat16* z;
    const __nv_bfloat16* norm;
    const __nv_bfloat16* observable;
    __nv_bfloat16* history;
    __nv_bfloat16* output;
};

template <typename S>
__global__ void gdn_convolve(const __nv_bfloat16* qkv, TextGdnState state,
                             const __nv_bfloat16* weights, __nv_bfloat16* convolved) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    const auto* history = state.history;
    const int row = blockIdx.x / conv_blocks_per_token;
    const int channel = ((blockIdx.x % conv_blocks_per_token) * conv_threads) + threadIdx.x;
    convolved[static_cast<std::size_t>(row) * S::qkv_width + channel] =
        convolve<S>({qkv, history}, weights, {row, channel});
}

__device__ __forceinline__ void store_checkpoint(const Checkpoints& checkpoints, int& next,
                                                 RowChannel point,
                                                 const float (&state)[keys_per_lane]) {
    if (next < checkpoints.count && checkpoints.values[next].rows == point.row) {
        auto* target = checkpoints.values[next++].state.recurrent + point.channel;
#pragma unroll
        for (int part = 0; part < keys_per_lane; ++part) {
            target[static_cast<std::size_t>(part) * warp_size] = state[part];
        }
    }
}

template <typename S, bool round_qk, bool Snapshot = false>
__device__ __forceinline__ void
gdn_recur_core(int token_count, GdnRecurInput input, TextGdnWeights weights, float* recurrent,
               __nv_bfloat16* observable, const Checkpoints* checkpoints = nullptr) {
    __shared__ float qk[2][text_gdn_head_dim];
    __shared__ float warp_squares[recurrent_threads / warp_size];
    __shared__ float decay;
    __shared__ float beta;

    const int block = static_cast<int>(blockIdx.x);
    const int head = block / value_blocks_per_head;
    const int qk_head = head / S::heads_per_qk;
    const int warp = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int value = ((block % value_blocks_per_head) * value_rows_per_block) + warp;
    const int feature = (head * text_gdn_head_dim) + value;
    const int state_offset = (feature * text_gdn_head_dim) + lane;
    const int q_or_k = threadIdx.x / text_gdn_head_dim;
    const int qk_channel = threadIdx.x % text_gdn_head_dim;
    const int qk_offset = q_or_k * S::qk_width + qk_head * text_gdn_head_dim;
    const int first_warp = q_or_k * (text_gdn_head_dim / warp_size);
    const float negative_a = -expf(__bfloat162float(weights.a_log[head]));
    const float dt_bias = __bfloat162float(weights.dt_bias[head]);

    float state_keys[keys_per_lane];
    int next_checkpoint = 0;
#pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state_keys[part] = recurrent[state_offset + (part * warp_size)];
    }

    for (int row = 0; row < token_count; ++row) {
        const auto* mixed = input.convolved + (static_cast<std::size_t>(row) * S::qkv_width);
        const float component = __bfloat162float(mixed[qk_offset + qk_channel]);
        const float partial = warp_sum(component * component);
        if (lane == 0) {
            warp_squares[warp] = partial;
        }
        if (threadIdx.x == 0) {
            const int gate = row * S::value_heads + head;
            decay = expf(negative_a * softplus(__bfloat162float(input.a[gate]) + dt_bias));
            beta = sigmoid(__bfloat162float(input.b[gate]));
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
            result = fmaf(state_keys[part], qk[0][key], result);
        }
        const float sum = warp_sum(result);
        if (lane == 0) {
            observable[static_cast<std::size_t>(row) * S::value_width + feature] =
                __float2bfloat16(sum);
        }
        if constexpr (Snapshot) {
            store_checkpoint(*checkpoints, next_checkpoint, {row + 1, state_offset}, state_keys);
        }
        __syncthreads();
    }

#pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        recurrent[state_offset + (part * warp_size)] = state_keys[part];
    }
}

template <typename S, bool round_qk>
__global__ void gdn_recur(int token_count, GdnRecurInput input, TextGdnWeights weights,
                          float* recurrent, __nv_bfloat16* observable) {
    gdn_recur_core<S, round_qk>(token_count, input, weights, recurrent, observable);
}

template <typename S>
__global__ void gdn_recur_prefill_checkpoints(int token_count, GdnRecurInput input,
                                              TextGdnWeights weights, float* recurrent,
                                              __nv_bfloat16* observable,
                                              const __grid_constant__ Checkpoints checkpoints) {
    gdn_recur_core<S, true, true>(token_count, input, weights, recurrent, observable, &checkpoints);
}

__device__ __nv_bfloat16 finish_value(float value, float inverse_rms, __nv_bfloat16 norm,
                                      __nv_bfloat16 z) {
    const float gate = __bfloat162float(z);
    return __float2bfloat16(value * inverse_rms * __bfloat162float(norm)
                            * (gate * sigmoid(gate)));
}

template <typename S> __global__ void gdn_finish(int token_count, GdnFinish finish) {
    const int norm_blocks = token_count * S::value_heads;
    const int channel = threadIdx.x;
    if (blockIdx.x >= norm_blocks) {
        const int feature = (blockIdx.x - norm_blocks) * text_gdn_head_dim + channel;
        finish_history<S>(token_count, {finish.qkv, finish.history}, finish.history, feature);
        return;
    }

    __shared__ float warp_squares[text_gdn_head_dim / warp_size];
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * text_gdn_head_dim + channel;
    const float value = __bfloat162float(finish.observable[index]);
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
    finish.output[index] = finish_value(value, inverse_rms, finish.norm[channel], finish.z[index]);
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
        finish_history<S>(rows, {qkv, state.history}, state.history, feature);
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

template <typename S>
__global__ void gdn_finish_segments(const __grid_constant__ gdn_prefill::Segments segments,
                                    const __nv_bfloat16* norm) {
    const auto& segment = segments.values[blockIdx.z];
    if (blockIdx.x < quantized_finish_blocks<S>(segment.rows)) {
        const auto* observable =
            segment.workspace + static_cast<std::size_t>(segment.rows) * S::qkv_width;
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

template <typename S> cudaError_t finish_gdn(int rows, GdnFinish finish, cudaStream_t stream) {
    const int blocks = rows * S::value_heads + S::qkv_width / text_gdn_head_dim;
    void* args[] = {&rows, &finish};
    return cudaLaunchKernel(gdn_finish<S>, dim3(blocks), dim3(text_gdn_head_dim), args, 0, stream);
}

template <typename S, typename Output>
cudaError_t prefill_gdn(int rows, GdnInputs input, const __nv_bfloat16* z, TextGdnWeights weights,
                        TextGdnState state, float2* gates, Output output, cudaStream_t stream,
                        void* chunk_scratch, std::size_t chunk_bytes,
                        TextGdnPrefillCheckpoints checkpoints) {
    std::optional<Checkpoints> points;
    if (checkpoints.size != 0 && !pack_checkpoints(rows, state, checkpoints, points.emplace())) {
        return cudaErrorInvalidValue;
    }
    auto* capture = points ? &*points : nullptr;
    auto* workspace = input.convolved;
    bool chunked = gdn_chunk::eligible<S>(rows, chunk_scratch, chunk_bytes);
    auto recurrence = std::is_same_v<S, GdnLocal> && rows >= 1024 ? gdn_chunk::Recurrence::staged
                                                                  : gdn_chunk::Recurrence::unstaged;
    if (chunked) {
        const auto status =
            gdn_chunk::configure<S>(chunked, false, &recurrence, capture != nullptr);
        if (status != cudaSuccess) {
            return status;
        }
    }
    auto status = launch_prepare<S>(rows, input.qkv, state.history, weights, input.a, input.b,
                                    workspace, gates, stream);
    if (status != cudaSuccess) {
        return status;
    }
    auto* observable = workspace + static_cast<std::size_t>(rows) * S::qkv_width;
    if (chunked) {
        status = gdn_chunk::launch<S>(rows, workspace, gates, chunk_scratch, state.recurrent,
                                      observable, stream, recurrence, capture);
    } else {
        status = gdn_prefill::launch_recur<S>(rows, workspace, gates, state.recurrent, observable,
                                              stream, capture);
    }
    if (status != cudaSuccess) {
        return status;
    }
    if (capture) {
        status = launch_history<S>(input.qkv, state.history, *capture, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    if constexpr (std::is_same_v<Output, TextFp8Activation>) {
        const int blocks = quantized_finish_blocks<S>(rows);
        void* arguments[] = {
            &rows,
            static_cast<void*>(&input.qkv),
            static_cast<void*>(&z),
            static_cast<void*>(&weights.norm),
            static_cast<void*>(&observable),
            &state,
            &output,
        };
        return cudaLaunchKernel(gdn_finish_quantized<S>, dim3(blocks), dim3(finish_threads),
                                arguments, 0, stream);
    } else {
        return finish_gdn<S>(rows, {input.qkv, z, weights.norm, observable, state.history, output},
                             stream);
    }
}

template <typename S>
cudaError_t enqueue_gdn(int token_count, TextGdnMode mode, GdnInputs rows, const __nv_bfloat16* z,
                        TextGdnWeights weights, TextGdnState state, __nv_bfloat16* observable,
                        __nv_bfloat16* output, cudaStream_t stream, Checkpoints* checkpoints) {
    if ((mode != TextGdnMode::prefill && mode != TextGdnMode::decode) || !rows.qkv ||
        !rows.convolved || !rows.b || !rows.a || !z || !weights.convolution || !weights.a_log ||
        !weights.dt_bias || !weights.norm || !state.history || !state.recurrent || !observable ||
        !output) {
        return cudaErrorInvalidValue;
    }
    constexpr int conv_blocks = S::qkv_width / conv_threads;
    void* convolution_args[] = {
        static_cast<void*>(&rows.qkv),
        &state,
        static_cast<void*>(&weights.convolution),
        static_cast<void*>(&rows.convolved),
    };
    auto status = cudaLaunchKernel(gdn_convolve<S>, dim3(token_count * conv_blocks),
                                   dim3(conv_threads), convolution_args, 0, stream);
    if (status != cudaSuccess)
        return status;
    if (checkpoints) {
        status = launch_history<S>(rows.qkv, state.history, *checkpoints, stream);
        if (status != cudaSuccess)
            return status;
    }
    constexpr int recurrent_blocks = S::value_heads * value_blocks_per_head;
    GdnRecurInput input{rows.convolved, rows.b, rows.a};
    void* recur_args[] = {
        &token_count,
        &input,
        &weights,
        static_cast<void*>(&state.recurrent),
        static_cast<void*>(&observable),
        checkpoints,
    };
    const void* recur = reinterpret_cast<const void*>(
        mode == TextGdnMode::prefill ? gdn_recur<S, true> : gdn_recur<S, false>);
    if (checkpoints != nullptr) {
        recur = reinterpret_cast<const void*>(gdn_recur_prefill_checkpoints<S>);
    }
    status = cudaLaunchKernel(recur, dim3(recurrent_blocks), dim3(recurrent_threads), recur_args, 0,
                              stream);
    if (status != cudaSuccess)
        return status;
    return finish_gdn<S>(token_count,
                         {rows.qkv, z, weights.norm, observable, state.history, output}, stream);
}

template <typename S>
std::size_t workspace_bytes(int rows) {
    if constexpr (std::is_same_v<S, GdnFull>) {
        return text_gdn_workspace_bytes(rows);
    } else {
        return text_gdn_local_workspace_bytes(rows);
    }
}

template <typename S>
cudaError_t forward_gdn(int token_count, TextGdnMode mode, const __nv_bfloat16* qkv,
                        const __nv_bfloat16* z, const __nv_bfloat16* b, const __nv_bfloat16* a,
                        TextGdnWeights weights, TextGdnState state, __nv_bfloat16* workspace,
                        std::size_t bytes, __nv_bfloat16* output, cudaStream_t stream,
                        TextGdnPrefillCheckpoints checkpoints) {
    if (token_count == 0) {
        return checkpoints.size == 0 ? cudaSuccess : cudaErrorInvalidValue;
    }
    const auto required_bytes = workspace_bytes<S>(token_count);
    if (!workspace || required_bytes == 0 || bytes < required_bytes) {
        return cudaErrorInvalidValue;
    }
    std::optional<Checkpoints> points;
    if (checkpoints.size != 0 &&
        (mode != TextGdnMode::prefill ||
         !pack_checkpoints(token_count, state, checkpoints, points.emplace()))) {
        return cudaErrorInvalidValue;
    }
    auto* observable = workspace + static_cast<std::size_t>(token_count) * S::qkv_width;
    const GdnInputs rows{qkv, workspace, b, a};
    return enqueue_gdn<S>(token_count, mode, rows, z, weights, state, observable, output, stream,
                          points ? &*points : nullptr);
}

template <typename S>
std::optional<cudaError_t> prefill_gdn_group(const decoder::GdnPrefillSegment* input, int count,
                                             TextGdnWeights weights, cudaStream_t stream) {
    if (!input || count < 2 || count > decoder::max_gdn_prefill_segments || !weights.convolution ||
        !weights.a_log || !weights.dt_bias || !weights.norm) {
        return cudaErrorInvalidValue;
    }
    gdn_prefill::Segments segments{};
    std::optional<SegmentCheckpoints> points;
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
        if (segment.checkpoints.size != 0 && !points) {
            points.emplace();
        }
        if (points &&
            !pack_checkpoints(rows, segment.state, segment.checkpoints, points->values[index])) {
            return cudaErrorInvalidValue;
        }
        segments.values[index] = {
            rows,          segment.qkv,       segment.z,     segment.b,      segment.a,
            segment.state, segment.workspace, segment.gates, segment.output,
        };
        chunk.values[index] = {
            rows,
            segment.workspace,
            segment.gates,
            static_cast<gdn_chunk::Matrices*>(segment.chunk_scratch),
            segment.state.recurrent,
            segment.workspace + static_cast<std::size_t>(rows) * S::qkv_width,
        };
    }

    bool supported = true;
    auto status = gdn_chunk::configure<S>(supported, true, nullptr, points.has_value());
    if (status != cudaSuccess) {
        return status;
    }
    if (!supported) {
        return std::nullopt;
    }
    status = launch_prepare_segments<S>(segments, {count, max_rows}, weights, stream);
    if (status != cudaSuccess) {
        return status;
    }
    status =
        gdn_chunk::launch_segments<S>(chunk, count, max_rows, stream, points ? &*points : nullptr);
    if (status != cudaSuccess) {
        return status;
    }
    if (points) {
        status = launch_history_segments<S>(segments, count, *points, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    void* finish_args[] = {&segments, static_cast<void*>(&weights.norm)};
    return cudaLaunchKernel(gdn_finish_segments<S>,
                            dim3(quantized_finish_blocks<S>(max_rows), 1, count),
                            dim3(finish_threads), finish_args, 0, stream);
}
}

std::size_t text_gdn_local_history_bytes() {
    return sizeof(__nv_bfloat16) * static_cast<std::size_t>(GdnLocal::qkv_width) *
           GdnLocal::history_taps;
}

std::size_t text_gdn_local_recurrent_bytes() {
    return sizeof(float) * static_cast<std::size_t>(GdnLocal::value_heads) * GdnLocal::head_dim *
           GdnLocal::head_dim;
}

cudaError_t decoder::tensor_core_prefill_gdn(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state, __nv_bfloat16* workspace,
    float2* gates, __nv_bfloat16* output, cudaStream_t stream, void* chunk_scratch,
    std::size_t chunk_bytes, TextGdnPrefillCheckpoints checkpoints) {
    return prefill_gdn<GdnFull>(rows, {qkv, workspace, b, a}, z, weights, state, gates, output,
                                stream, chunk_scratch, chunk_bytes, checkpoints);
}

cudaError_t decoder::tensor_core_prefill_gdn_local(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state, __nv_bfloat16* workspace,
    float2* gates, __nv_bfloat16* output, cudaStream_t stream, void* chunk_scratch,
    std::size_t chunk_bytes, TextGdnPrefillCheckpoints checkpoints) {
    return prefill_gdn<GdnLocal>(rows, {qkv, workspace, b, a}, z, weights, state, gates, output,
                                 stream, chunk_scratch, chunk_bytes, checkpoints);
}

cudaError_t decoder::tensor_core_prefill_gdn(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state, __nv_bfloat16* workspace,
    float2* gates, TextFp8Activation output, cudaStream_t stream, void* chunk_scratch,
    std::size_t chunk_bytes, TextGdnPrefillCheckpoints checkpoints) {
    if (workspace_bytes<GdnFull>(rows) == 0 || output.values == nullptr ||
        output.scales == nullptr) {
        return cudaErrorInvalidValue;
    }
    return prefill_gdn<GdnFull>(rows, {qkv, workspace, b, a}, z, weights, state, gates, output,
                                stream, chunk_scratch, chunk_bytes, checkpoints);
}

cudaError_t decoder::tensor_core_prefill_gdn_local(
    int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* z, const __nv_bfloat16* b,
    const __nv_bfloat16* a, TextGdnWeights weights, TextGdnState state, __nv_bfloat16* workspace,
    float2* gates, TextFp8Activation output, cudaStream_t stream, void* chunk_scratch,
    std::size_t chunk_bytes, TextGdnPrefillCheckpoints checkpoints) {
    if (workspace_bytes<GdnLocal>(rows) == 0 || output.values == nullptr ||
        output.scales == nullptr) {
        return cudaErrorInvalidValue;
    }
    return prefill_gdn<GdnLocal>(rows, {qkv, workspace, b, a}, z, weights, state, gates, output,
                                 stream, chunk_scratch, chunk_bytes, checkpoints);
}

std::optional<cudaError_t> decoder::tensor_core_prefill_gdn_group(const GdnPrefillSegment* input,
                                                                  int count, TextGdnWeights weights,
                                                                  cudaStream_t stream) {
    return prefill_gdn_group<GdnFull>(input, count, weights, stream);
}

std::optional<cudaError_t>
decoder::tensor_core_prefill_gdn_group_local(const GdnPrefillSegment* input, int count,
                                             TextGdnWeights weights, cudaStream_t stream) {
    return prefill_gdn_group<GdnLocal>(input, count, weights, stream);
}

cudaError_t text_gdn_forward(int token_count, TextGdnMode mode, const __nv_bfloat16* qkv,
                             const __nv_bfloat16* z, const __nv_bfloat16* b, const __nv_bfloat16* a,
                             TextGdnWeights weights, TextGdnState state, __nv_bfloat16* workspace,
                             std::size_t bytes, __nv_bfloat16* output, cudaStream_t stream,
                             TextGdnPrefillCheckpoints checkpoints) {
    return forward_gdn<GdnFull>(token_count, mode, qkv, z, b, a, weights, state, workspace, bytes,
                                output, stream, checkpoints);
}

cudaError_t text_gdn_forward_local(int token_count, TextGdnMode mode, const __nv_bfloat16* qkv,
                                   const __nv_bfloat16* z, const __nv_bfloat16* b,
                                   const __nv_bfloat16* a, TextGdnWeights weights,
                                   TextGdnState state, __nv_bfloat16* workspace, std::size_t bytes,
                                   __nv_bfloat16* output, cudaStream_t stream,
                                   TextGdnPrefillCheckpoints checkpoints) {
    return forward_gdn<GdnLocal>(token_count, mode, qkv, z, b, a, weights, state, workspace, bytes,
                                 output, stream, checkpoints);
}
}
