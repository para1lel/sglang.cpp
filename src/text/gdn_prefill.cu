#include "sglang/text/gdn/gdn_prefill_kernels.cuh"
#include "sglang/text/gdn_local.h"

namespace sglang::gdn_prefill {
namespace {

template <typename S>
__global__ void gdn_prefill_recur(int rows, const __nv_bfloat16* convolved, const float2* gates,
                                  float* recurrent, __nv_bfloat16* observable) {
    const int head = blockIdx.x / value_blocks_per_head;
    const int qk_head = head / S::heads_per_qk;
    const int lane = threadIdx.x % warp_size;
    const int value = (blockIdx.x % value_blocks_per_head) * value_rows_per_block +
                      threadIdx.x / warp_size;
    const int feature = head * text_gdn_head_dim + value;
    const int q_offset = qk_head * text_gdn_head_dim + lane;
    const int k_offset = q_offset + S::qk_width;
    auto* state = recurrent + feature * text_gdn_head_dim + lane;
    float state_keys[keys_per_lane];
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state_keys[part] = state[part * warp_size];
    }
    for (int row = 0; row < rows; ++row) {
        const auto* mixed = convolved + static_cast<std::size_t>(row) * S::qkv_width;
        const auto gate = gates[row * S::value_heads + head];
        const float target = __bfloat162float(mixed[2 * S::qk_width + feature]);
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
            observable[static_cast<std::size_t>(row) * S::value_width + feature] =
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

__device__ float2 add_channels(float2 values, float2 addends) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1000
    return __fadd2_rn(values, addends);
#else
    return {values.x + addends.x, values.y + addends.y};
#endif
}

__device__ float2 warp_sum_channels(float2 value) {
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        const float2 peer = {__shfl_xor_sync(0xffffffffu, value.x, offset),
                             __shfl_xor_sync(0xffffffffu, value.y, offset)};
        value = add_channels(value, peer);
    }
    return value;
}

template <typename S>
__global__ void gdn_prefill_recur_paired(int rows, const __nv_bfloat16* convolved,
                                        const float2* gates, float* recurrent,
                                        __nv_bfloat16* observable) {
    constexpr int components_per_vector = sizeof(uint4) / sizeof(__nv_bfloat16);
    constexpr int vectors_per_head = text_gdn_head_dim / components_per_vector;
    __shared__ uint4 packed_qk[recurrent_tile_rows][2][vectors_per_head];
    __shared__ float2 tile_gates[recurrent_tile_rows];
    __shared__ __nv_bfloat162 tile_targets[recurrent_tile_rows][value_pairs_per_block];
    const int head = blockIdx.x / value_blocks_per_head;
    const int qk_head = head / S::heads_per_qk;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int first_feature = head * text_gdn_head_dim +
                              (blockIdx.x % value_blocks_per_head) * value_rows_per_block;
    const int feature = first_feature + 2 * warp;
    auto* first_state = recurrent + feature * text_gdn_head_dim + lane;
    auto* second_state = first_state + text_gdn_head_dim;
    float2 state_keys[keys_per_lane];
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state_keys[part] = {first_state[part * warp_size], second_state[part * warp_size]};
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
                    convolved + static_cast<std::size_t>(first + local_row) * S::qkv_width;
                const auto* source = reinterpret_cast<const uint4*>(
                    mixed + plane * S::qk_width + qk_head * text_gdn_head_dim);
                packed_qk[local_row][plane][vector] = source[vector];
            }
        }
        if (threadIdx.x < count * value_pairs_per_block) {
            const int local_row = threadIdx.x / value_pairs_per_block;
            const int pair = threadIdx.x % value_pairs_per_block;
            const int row = first + local_row;
            const auto* mixed = convolved + static_cast<std::size_t>(row) * S::qkv_width;
            const auto* target = reinterpret_cast<const __nv_bfloat162*>(
                mixed + 2 * S::qk_width + first_feature);
            tile_targets[local_row][pair] = target[pair];
            if (pair == 0) {
                tile_gates[local_row] = gates[row * S::value_heads + head];
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
            float2 prediction = {0.0f, 0.0f};
            #pragma unroll
            for (int part = 0; part < keys_per_lane; ++part) {
                const int key = part * warp_size + lane;
                constexpr float query_scale = 0.08838834764831845f;
                keys[part] = __bfloat162float(keys_bf16[key]);
                queries[part] = __bfloat162float(queries_bf16[key]) * query_scale;
                state_keys[part] = multiply_channels(state_keys[part], gate.x);
                prediction = multiply_add_channels(state_keys[part], keys[part], prediction);
            }
            const float2 predicted = warp_sum_channels(prediction);
            const auto correction = multiply_channels(
                add_channels(targets, {-predicted.x, -predicted.y}), gate.y);
            float2 result = {0.0f, 0.0f};
            #pragma unroll
            for (int part = 0; part < keys_per_lane; ++part) {
                state_keys[part] =
                    multiply_add_channels(correction, keys[part], state_keys[part]);
                result = multiply_add_channels(state_keys[part], queries[part], result);
            }
            const float2 sum = warp_sum_channels(result);
            if (lane == 0) {
                auto* row_output =
                    observable + static_cast<std::size_t>(first + local_row) * S::value_width;
                reinterpret_cast<__nv_bfloat162*>(row_output)[feature / 2] =
                    __float22bfloat162_rn(sum);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        first_state[part * warp_size] = state_keys[part].x;
        second_state[part * warp_size] = state_keys[part].y;
    }
}

}

template <typename S>
cudaError_t launch_recur(int rows, __nv_bfloat16* workspace, float2* gates, float* recurrent,
                         __nv_bfloat16* observable, cudaStream_t stream) {
    void* arguments[] = {&rows, &workspace, &gates, &recurrent, &observable};
    const bool paired = rows >= paired_recurrent_min_rows &&
                        reinterpret_cast<std::uintptr_t>(workspace) % alignof(uint4) == 0;
    const auto recur = paired ? gdn_prefill_recur_paired<S> : gdn_prefill_recur<S>;
    const int threads = paired ? paired_recurrent_threads : recurrent_threads;
    return cudaLaunchKernel(recur, dim3(S::value_heads * value_blocks_per_head),
                            dim3(threads), arguments, 0, stream);
}

template cudaError_t launch_recur<GdnFull>(int, __nv_bfloat16*, float2*, float*,
                                           __nv_bfloat16*, cudaStream_t);
template cudaError_t launch_recur<GdnLocal>(int, __nv_bfloat16*, float2*, float*,
                                            __nv_bfloat16*, cudaStream_t);

}
