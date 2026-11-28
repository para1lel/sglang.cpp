#include "sglang/text/gdn/gdn_prefill_kernels.cuh"
#include "sglang/text/gdn/gdn_chunk.cuh"
#include "sglang/text/gdn_local.h"

#include <algorithm>

namespace sglang::gdn_prefill {
__device__ __inline_hint__ float sigmoid(float value) {
    return 1.0F / (1.0F + expf(-value));
}

__device__ __inline_hint__ float softplus(float value) {
    return fmaxf(value, 0.0F) + log1pf(expf(-fabsf(value)));
}

template <typename S>
static __device__ __nv_bfloat16 raw_sample(RawInput input, int row, int channel) {
    if (row < 0) {
        return input.history[(channel * text_gdn_history_taps) + row + text_gdn_history_taps];
    }
    return input.qkv[(static_cast<std::size_t>(row) * S::qkv_width) + channel];
}
template <typename S>
__device__ __nv_bfloat16 convolve(RawInput input, const __nv_bfloat16* weights, RowChannel point) {
    float value = 0.0F;
#pragma unroll
    for (int tap = 0; tap < conv_taps; ++tap) {
        const int sample_row = point.row - text_gdn_history_taps + tap;
        const float raw = __bfloat162float(raw_sample<S>(input, sample_row, point.channel));
        value = fmaf(raw, __bfloat162float(weights[(point.channel * conv_taps) + tap]), value);
    }
    return __float2bfloat16(value * sigmoid(value));
}

template <typename S>
static __global__ void gdn_prefill_prepare(RawInput input, TextGdnWeights weights,
                                           const __nv_bfloat16* a, const __nv_bfloat16* b,
                                           __nv_bfloat16* convolved, float2* gates) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    constexpr int blocks_per_row = conv_blocks_per_token + 1;
    const int row = blockIdx.x / blocks_per_row;
    const int block = blockIdx.x % blocks_per_row;
    const int feature = static_cast<int>(threadIdx.x);
    if (block == conv_blocks_per_token) {
        if (feature < S::value_heads) {
            const int index = row * S::value_heads + feature;
            const float negative_a = -expf(__bfloat162float(weights.a_log[feature]));
            const float dt_bias = __bfloat162float(weights.dt_bias[feature]);
            gates[index] = {
                expf(negative_a * softplus(__bfloat162float(a[index]) + dt_bias)),
                sigmoid(__bfloat162float(b[index])),
            };
        }
        return;
    }
    const int channel = (block * conv_threads) + feature;
    auto mixed = convolve<S>(input, weights.convolution, {row, channel});
    if (channel < 2 * S::qk_width) {
        constexpr int warps = conv_threads / warp_size;
        __shared__ float warp_squares[warps];
        const float component = __bfloat162float(mixed);
        const float partial = warp_sum(component * component);
        if (feature % warp_size == 0) {
            warp_squares[feature / warp_size] = partial;
        }
        __syncthreads();
        float squares = 0.0F;
#pragma unroll
        for (const float warp_square : warp_squares) {
            squares += warp_square;
        }
        mixed = __float2bfloat16(component * rsqrtf(squares + norm_epsilon));
    }
    convolved[static_cast<std::size_t>(row) * S::qkv_width + channel] = mixed;
}

template <typename S>
static __device__ __forceinline__ void
gdn_prepare_aligned_rows(RawInput input, TextGdnWeights weights, const __nv_bfloat16* a,
                         const __nv_bfloat16* b, __nv_bfloat16* convolved, float2* gates) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    constexpr int blocks_per_group = conv_blocks_per_token + 1;
    constexpr int warps = conv_threads / warp_size;
    const int first = blockIdx.x / blocks_per_group * prefill_rows_per_block;
    const int block = blockIdx.x % blocks_per_group;
    const int feature = static_cast<int>(threadIdx.x);
    if (block == conv_blocks_per_token) {
        if (feature < S::value_heads) {
            const float negative_a = -expf(__bfloat162float(weights.a_log[feature]));
            const float dt_bias = __bfloat162float(weights.dt_bias[feature]);
#pragma unroll
            for (int row = 0; row < prefill_rows_per_block; ++row) {
                const int index = (first + row) * S::value_heads + feature;
                gates[index] = {
                    expf(negative_a * softplus(__bfloat162float(a[index]) + dt_bias)),
                    sigmoid(__bfloat162float(b[index])),
                };
            }
        }
        return;
    }

    const int channel = (block * conv_threads) + feature;
    const auto* taps = weights.convolution + (static_cast<std::size_t>(channel) * conv_taps);
    const float weight0 = __bfloat162float(taps[0]);
    const float weight1 = __bfloat162float(taps[1]);
    const float weight2 = __bfloat162float(taps[2]);
    const float weight3 = __bfloat162float(taps[3]);
    float oldest = __bfloat162float(raw_sample<S>(input, first - 3, channel));
    float middle = __bfloat162float(raw_sample<S>(input, first - 2, channel));
    float latest = __bfloat162float(raw_sample<S>(input, first - 1, channel));
    __nv_bfloat16 mixed[prefill_rows_per_block];
#pragma unroll
    for (int row = 0; row < prefill_rows_per_block; ++row) {
        const float current = __bfloat162float(raw_sample<S>(input, first + row, channel));
        float value = fmaf(oldest, weight0, 0.0F);
        value = fmaf(middle, weight1, value);
        value = fmaf(latest, weight2, value);
        value = fmaf(current, weight3, value);
        mixed[row] = __float2bfloat16(value * sigmoid(value));
        oldest = middle;
        middle = latest;
        latest = current;
    }
    if (channel < 2 * S::qk_width) {
        __shared__ float warp_squares[prefill_rows_per_block][warps];
#pragma unroll
        for (int row = 0; row < prefill_rows_per_block; ++row) {
            const float component = __bfloat162float(mixed[row]);
            const float partial = warp_sum(component * component);
            if (feature % warp_size == 0) {
                warp_squares[row][feature / warp_size] = partial;
            }
        }
        __syncthreads();
#pragma unroll
        for (int row = 0; row < prefill_rows_per_block; ++row) {
            float squares = 0.0F;
#pragma unroll
            for (int warp = 0; warp < warps; ++warp) {
                squares += warp_squares[row][warp];
            }
            const float component = __bfloat162float(mixed[row]);
            mixed[row] = __float2bfloat16(component * rsqrtf(squares + norm_epsilon));
        }
    }
#pragma unroll
    for (int row = 0; row < prefill_rows_per_block; ++row) {
        convolved[static_cast<std::size_t>(first + row) * S::qkv_width + channel] = mixed[row];
    }
}

template <typename S>
static __global__ void gdn_prefill_prepare_grouped(RawInput input, TextGdnWeights weights,
                                                   const __nv_bfloat16* a, const __nv_bfloat16* b,
                                                   __nv_bfloat16* convolved, float2* gates) {
    gdn_prepare_aligned_rows<S>(input, weights, a, b, convolved, gates);
}

template <typename S>
__device__ void finish_history(int token_count, RawInput input, __nv_bfloat16* final_history,
                               int feature) {
#pragma unroll
    for (int tap = 0; tap < text_gdn_history_taps; ++tap) {
        final_history[(feature * text_gdn_history_taps) + tap] =
            raw_sample<S>(input, token_count - text_gdn_history_taps + tap, feature);
    }
}

namespace {
template <typename S>
__global__ void gdn_prepare_segments(const __grid_constant__ Segments segments,
                                     TextGdnWeights weights) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    const auto& segment = segments.values[blockIdx.z];
    constexpr int blocks_per_row_group = conv_blocks_per_token + 1;
    if (static_cast<int>(blockIdx.x) <
        (segment.rows / prefill_rows_per_block) * blocks_per_row_group) {
        gdn_prepare_aligned_rows<S>({segment.qkv, segment.state.history}, weights, segment.a,
                                    segment.b, segment.workspace, segment.gates);
    }
}

template <typename S>
__global__ void gdn_checkpoint_history(RawInput input,
                                       const __grid_constant__ Checkpoints checkpoints) {
    const auto& point = checkpoints.values[blockIdx.y];
    finish_history<S>(point.rows, input, point.state.history,
                      (static_cast<int>(blockIdx.x) * conv_threads) +
                          static_cast<int>(threadIdx.x));
}

template <typename S>
__global__ void
gdn_checkpoint_segment_history(const __grid_constant__ Segments segments,
                               const __grid_constant__ SegmentCheckpoints checkpoints) {
    const auto& points = checkpoints.values[blockIdx.z];
    if (static_cast<int>(blockIdx.y) < points.count) {
        const auto& segment = segments.values[blockIdx.z];
        const auto& point = points.values[blockIdx.y];
        finish_history<S>(point.rows, {segment.qkv, segment.state.history}, point.state.history,
                          (static_cast<int>(blockIdx.x) * conv_threads) +
                              static_cast<int>(threadIdx.x));
    }
}

template <typename S, bool Snapshot>
__device__ __forceinline__ void
recur_ordered(int rows, const __nv_bfloat16* convolved, const float2* gates, float* recurrent,
              __nv_bfloat16* observable, const Checkpoints* checkpoints) {
    const int head = blockIdx.x / value_blocks_per_head;
    const int qk_head = head / S::heads_per_qk;
    const int lane = threadIdx.x % warp_size;
    const int value =
        ((static_cast<int>(blockIdx.x) % value_blocks_per_head) * value_rows_per_block) +
        (static_cast<int>(threadIdx.x) / warp_size);
    const int feature = head * text_gdn_head_dim + value;
    const int q_offset = qk_head * text_gdn_head_dim + lane;
    const int k_offset = q_offset + S::qk_width;
    auto* state = recurrent + feature * text_gdn_head_dim + lane;
    float state_keys[keys_per_lane];
    int next_checkpoint = 0;
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
        if constexpr (Snapshot) {
            if (next_checkpoint < checkpoints->count &&
                checkpoints->values[next_checkpoint].rows == row + 1) {
                auto* target = checkpoints->values[next_checkpoint++].state.recurrent +
                               (static_cast<std::size_t>(feature) * text_gdn_head_dim) + lane;
#pragma unroll
                for (int part = 0; part < keys_per_lane; ++part) {
                    target[static_cast<std::size_t>(part) * warp_size] = state_keys[part];
                }
            }
        }
    }
#pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        state[part * warp_size] = state_keys[part];
    }
}

template <typename S>
__global__ void gdn_prefill_recur(int rows, const __nv_bfloat16* convolved, const float2* gates,
                                  float* recurrent, __nv_bfloat16* observable) {
    recur_ordered<S, false>(rows, convolved, gates, recurrent, observable, nullptr);
}

template <typename S>
__global__ void gdn_prefill_recur_checkpoints(int rows, const __nv_bfloat16* convolved,
                                              const float2* gates, float* recurrent,
                                              __nv_bfloat16* observable,
                                              const __grid_constant__ Checkpoints checkpoints) {
    recur_ordered<S, true>(rows, convolved, gates, recurrent, observable, &checkpoints);
}

constexpr int paired_recurrent_threads = recurrent_threads / 2;
constexpr int paired_recurrent_min_rows = 32;
constexpr int recurrent_tile_rows = 8;
constexpr int value_pairs_per_block = value_rows_per_block / 2;

__device__ float2 multiply_channels(float2 values, float factor) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1000
    asm("{ .reg .b64 pair, factors;\n"
        "mov.b64 pair, {%0,%1};\n"
        "mov.b64 factors, {%2,%2};\n"
        "mul.rn.f32x2 pair, pair, factors;\n"
        "mov.b64 {%0,%1}, pair; }"
        : "+f"(values.x), "+f"(values.y)
        : "f"(factor));
    return values;
#else
    return {values.x * factor, values.y * factor};
#endif
}

__device__ float2 multiply_add_channels(float2 values, float factor, float2 addends) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1000
    asm("{ .reg .b64 pair, factors, sum;\n"
        "mov.b64 pair, {%2,%3};\n"
        "mov.b64 factors, {%4,%4};\n"
        "mov.b64 sum, {%0,%1};\n"
        "fma.rn.f32x2 sum, pair, factors, sum;\n"
        "mov.b64 {%0,%1}, sum; }"
        : "+f"(addends.x), "+f"(addends.y)
        : "f"(values.x), "f"(values.y), "f"(factor));
    return addends;
#else
    return {fmaf(values.x, factor, addends.x), fmaf(values.y, factor, addends.y)};
#endif
}

__device__ float2 add_channels(float2 values, float2 addends) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1000
    asm("{ .reg .b64 pair, sum;\n"
        "mov.b64 pair, {%2,%3};\n"
        "mov.b64 sum, {%0,%1};\n"
        "add.rn.f32x2 sum, pair, sum;\n"
        "mov.b64 {%0,%1}, sum; }"
        : "+f"(addends.x), "+f"(addends.y)
        : "f"(values.x), "f"(values.y));
    return addends;
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

__device__ __forceinline__ void store_paired_carry(float* first, float* second,
                                                   const float2 (&state)[keys_per_lane]) {
#pragma unroll
    for (int part = 0; part < keys_per_lane; ++part) {
        const auto key = static_cast<std::size_t>(part) * warp_size;
        first[key] = state[part].x;
        second[key] = state[part].y;
    }
}

__device__ __forceinline__ void store_paired_checkpoint(const Checkpoints& checkpoints, int rows,
                                                        int& next, int offset,
                                                        const float2 (&state)[keys_per_lane]) {
    if (next < checkpoints.count && checkpoints.values[next].rows == rows) {
        auto* target = checkpoints.values[next++].state.recurrent + offset;
        store_paired_carry(target, target + text_gdn_head_dim, state);
    }
}

template <typename S>
__device__ __forceinline__ void store_paired_observable(__nv_bfloat16* observable, RowChannel point,
                                                        int lane, float2 value) {
    if (lane == 0) {
        auto* row_output = observable + (static_cast<std::size_t>(point.row) * S::value_width);
        reinterpret_cast<__nv_bfloat162*>(row_output)[point.channel / 2] =
            __float22bfloat162_rn(value);
    }
}

template <typename S, bool Snapshot>
__device__ __forceinline__ void
recur_ordered_paired(int rows, const __nv_bfloat16* convolved, const float2* gates,
                     float* recurrent, __nv_bfloat16* observable, const Checkpoints* checkpoints) {
    constexpr int components_per_vector = sizeof(uint4) / sizeof(__nv_bfloat16);
    constexpr int vectors_per_head = text_gdn_head_dim / components_per_vector;
    __shared__ uint4 packed_qk[recurrent_tile_rows][2][vectors_per_head];
    __shared__ float2 tile_gates[recurrent_tile_rows];
    __shared__ __nv_bfloat162 tile_targets[recurrent_tile_rows][value_pairs_per_block];
    const int head = blockIdx.x / value_blocks_per_head;
    const int qk_head = head / S::heads_per_qk;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const int first_feature =
        (head * text_gdn_head_dim) +
        ((static_cast<int>(blockIdx.x) % value_blocks_per_head) * value_rows_per_block);
    const int feature = first_feature + 2 * warp;
    auto* first_state = recurrent + feature * text_gdn_head_dim + lane;
    auto* second_state = first_state + text_gdn_head_dim;
    float2 state_keys[keys_per_lane];
    int next_checkpoint = 0;
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
                const auto* source = reinterpret_cast<const uint4*>(mixed + plane * S::qk_width +
                                                                    qk_head * text_gdn_head_dim);
                packed_qk[local_row][plane][vector] = source[vector];
            }
        }
        if (threadIdx.x < count * value_pairs_per_block) {
            const int local_row = threadIdx.x / value_pairs_per_block;
            const int pair = threadIdx.x % value_pairs_per_block;
            const int row = first + local_row;
            const auto* mixed = convolved + static_cast<std::size_t>(row) * S::qkv_width;
            const auto* target =
                reinterpret_cast<const __nv_bfloat162*>(mixed + 2 * S::qk_width + first_feature);
            tile_targets[local_row][pair] = target[pair];
            if (pair == 0) {
                tile_gates[local_row] = gates[row * S::value_heads + head];
            }
        }
        __syncthreads();
        for (int local_row = 0; local_row < count; ++local_row) {
            const auto* queries_bf16 =
                reinterpret_cast<const __nv_bfloat16*>(packed_qk[local_row][0]);
            const auto* keys_bf16 = reinterpret_cast<const __nv_bfloat16*>(packed_qk[local_row][1]);
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
            const auto correction =
                multiply_channels(add_channels(targets, {-predicted.x, -predicted.y}), gate.y);
            float2 result = {0.0f, 0.0f};
#pragma unroll
            for (int part = 0; part < keys_per_lane; ++part) {
                state_keys[part] = multiply_add_channels(correction, keys[part], state_keys[part]);
                result = multiply_add_channels(state_keys[part], queries[part], result);
            }
            store_paired_observable<S>(observable, {first + local_row, feature}, lane,
                                       warp_sum_channels(result));
            if constexpr (Snapshot) {
                store_paired_checkpoint(*checkpoints, first + local_row + 1, next_checkpoint,
                                        (feature * text_gdn_head_dim) + lane, state_keys);
            }
        }
        __syncthreads();
    }
    store_paired_carry(first_state, second_state, state_keys);
}

template <typename S>
__global__ void gdn_prefill_recur_paired(int rows, const __nv_bfloat16* convolved,
                                         const float2* gates, float* recurrent,
                                         __nv_bfloat16* observable) {
    recur_ordered_paired<S, false>(rows, convolved, gates, recurrent, observable, nullptr);
}

template <typename S>
__global__ void
gdn_prefill_recur_paired_checkpoints(int rows, const __nv_bfloat16* convolved, const float2* gates,
                                     float* recurrent, __nv_bfloat16* observable,
                                     const __grid_constant__ Checkpoints checkpoints) {
    recur_ordered_paired<S, true>(rows, convolved, gates, recurrent, observable, &checkpoints);
}

}

bool pack_checkpoints(int rows, TextGdnState active, TextGdnPrefillCheckpoints input,
                      Checkpoints& result) {
    if (input.size < 0 || input.size > text_gdn_max_prefill_checkpoints ||
        (input.size != 0 && input.data == nullptr)) {
        return false;
    }
    int previous = 0;
    result.count = 0;
    for (int point_index = 0; point_index < input.size; ++point_index) {
        const auto& point = input.data[point_index];
        if (point.rows <= previous || point.rows > rows || point.state.history == nullptr ||
            point.state.recurrent == nullptr || point.state.history == active.history ||
            point.state.recurrent == active.recurrent) {
            return false;
        }
        for (int index = 0; index < result.count; ++index) {
            const auto& destination = result.values[index].state;
            if (point.state.history == destination.history ||
                point.state.recurrent == destination.recurrent) {
                return false;
            }
        }
        result.values[result.count++] = point;
        previous = point.rows;
    }
    return true;
}

template <typename S>
cudaError_t launch_prepare(int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* history,
                           TextGdnWeights weights, const __nv_bfloat16* a, const __nv_bfloat16* b,
                           __nv_bfloat16* workspace, float2* gates, cudaStream_t stream) {
    constexpr int conv_blocks_per_token = S::qkv_width / conv_threads;
    RawInput input{qkv, history};
    void* args[] = {
        &input,
        &weights,
        static_cast<void*>(&a),
        static_cast<void*>(&b),
        static_cast<void*>(&workspace),
        static_cast<void*>(&gates),
    };
    const bool grouped = rows >= gdn_chunk::min_count && rows % prefill_rows_per_block == 0;
    const auto prepare = grouped ? gdn_prefill_prepare_grouped<S> : gdn_prefill_prepare<S>;
    const int groups = grouped ? rows / prefill_rows_per_block : rows;
    return cudaLaunchKernel(prepare, dim3(groups * (conv_blocks_per_token + 1)), dim3(conv_threads),
                            args, 0, stream);
}

template <typename S>
cudaError_t launch_prepare_segments(Segments& segments, SegmentExtent extent,
                                    TextGdnWeights weights, cudaStream_t stream) {
    void* args[] = {&segments, &weights};
    const int blocks =
        (extent.max_rows / prefill_rows_per_block) * (S::qkv_width / conv_threads + 1);
    return cudaLaunchKernel(gdn_prepare_segments<S>, dim3(blocks, 1, extent.count),
                            dim3(conv_threads), args, 0, stream);
}

template <typename S>
cudaError_t launch_history(const __nv_bfloat16* qkv, const __nv_bfloat16* history,
                           Checkpoints& checkpoints, cudaStream_t stream) {
    RawInput input{qkv, history};
    void* args[] = {&input, &checkpoints};
    return cudaLaunchKernel(gdn_checkpoint_history<S>,
                            dim3(S::qkv_width / conv_threads, checkpoints.count),
                            dim3(conv_threads), args, 0, stream);
}

template <typename S>
cudaError_t launch_history_segments(Segments& segments, int count, SegmentCheckpoints& checkpoints,
                                    cudaStream_t stream) {
    int points = 0;
    for (int lane = 0; lane < count; ++lane) {
        points = std::max(points, checkpoints.values[lane].count);
    }
    void* args[] = {&segments, &checkpoints};
    return cudaLaunchKernel(gdn_checkpoint_segment_history<S>,
                            dim3(S::qkv_width / conv_threads, points, count), dim3(conv_threads),
                            args, 0, stream);
}

template <typename S>
cudaError_t launch_recur(int rows, __nv_bfloat16* workspace, float2* gates, float* recurrent,
                         __nv_bfloat16* observable, cudaStream_t stream, Checkpoints* checkpoints) {
    void* arguments[] = {
        &rows,
        static_cast<void*>(&workspace),
        static_cast<void*>(&gates),
        static_cast<void*>(&recurrent),
        static_cast<void*>(&observable),
        checkpoints,
    };
    const bool paired = rows >= paired_recurrent_min_rows &&
                        reinterpret_cast<std::uintptr_t>(workspace) % alignof(uint4) == 0;
    const void* recur = nullptr;
    if (checkpoints) {
        recur = reinterpret_cast<const void*>(paired ? gdn_prefill_recur_paired_checkpoints<S>
                                                     : gdn_prefill_recur_checkpoints<S>);
    } else {
        recur = reinterpret_cast<const void*>(paired ? gdn_prefill_recur_paired<S>
                                                     : gdn_prefill_recur<S>);
    }
    const int threads = paired ? paired_recurrent_threads : recurrent_threads;
    return cudaLaunchKernel(recur, dim3(S::value_heads * value_blocks_per_head), dim3(threads),
                            arguments, 0, stream);
}

template __device__ __nv_bfloat16 convolve<GdnFull>(RawInput, const __nv_bfloat16*, RowChannel);
template __device__ __nv_bfloat16 convolve<GdnLocal>(RawInput, const __nv_bfloat16*, RowChannel);
template __device__ void finish_history<GdnFull>(int, RawInput, __nv_bfloat16*, int);
template __device__ void finish_history<GdnLocal>(int, RawInput, __nv_bfloat16*, int);
template cudaError_t launch_prepare<GdnFull>(int, const __nv_bfloat16*, const __nv_bfloat16*,
                                             TextGdnWeights, const __nv_bfloat16*,
                                             const __nv_bfloat16*, __nv_bfloat16*, float2*,
                                             cudaStream_t);
template cudaError_t launch_prepare<GdnLocal>(int, const __nv_bfloat16*, const __nv_bfloat16*,
                                              TextGdnWeights, const __nv_bfloat16*,
                                              const __nv_bfloat16*, __nv_bfloat16*, float2*,
                                              cudaStream_t);
template cudaError_t launch_prepare_segments<GdnFull>(Segments&, SegmentExtent, TextGdnWeights,
                                                      cudaStream_t);
template cudaError_t launch_prepare_segments<GdnLocal>(Segments&, SegmentExtent, TextGdnWeights,
                                                       cudaStream_t);
template cudaError_t launch_history<GdnFull>(const __nv_bfloat16*, const __nv_bfloat16*,
                                             Checkpoints&, cudaStream_t);
template cudaError_t launch_history<GdnLocal>(const __nv_bfloat16*, const __nv_bfloat16*,
                                              Checkpoints&, cudaStream_t);
template cudaError_t launch_history_segments<GdnFull>(Segments&, int, SegmentCheckpoints&,
                                                      cudaStream_t);
template cudaError_t launch_history_segments<GdnLocal>(Segments&, int, SegmentCheckpoints&,
                                                       cudaStream_t);
template cudaError_t launch_recur<GdnFull>(int, __nv_bfloat16*, float2*, float*, __nv_bfloat16*,
                                           cudaStream_t, Checkpoints*);
template cudaError_t launch_recur<GdnLocal>(int, __nv_bfloat16*, float2*, float*, __nv_bfloat16*,
                                            cudaStream_t, Checkpoints*);
}
