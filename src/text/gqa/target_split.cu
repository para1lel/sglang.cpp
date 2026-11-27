#include "sglang/text/gqa/decode_kernels.cuh"
#include <algorithm>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang {
namespace {

using namespace gqa_device;

namespace combine {
using namespace gqa_decode;

template <int FixedSplits = 0>
__device__ inline bool combine_head(const decoder::PagedDecodeLane& lane, int head) {
    constexpr int stride = FixedSplits ? FixedSplits : max_splits;
    static_assert(stride <= 32);
    __shared__ float weights[stride];
    __shared__ float total;
    const int channel = static_cast<int>(threadIdx.x);
    const int splits = FixedSplits ? FixedSplits : split_count(lane.position);
    auto* part = partials<stride>(lane.buffers.scratch, head);
    const float maximum = channel < splits ? part[channel].maximum : -CUDART_INF_F;
    bool invalid = __syncthreads_or(channel < splits && !isfinite(maximum));
    if (!invalid) {
        if (channel < 32) {
            float global = maximum;
            for (int offset = 16; offset > 0; offset /= 2) {
                global = fmaxf(global, __shfl_xor_sync(0xffffffffu, global, offset));
            }
            const float delta = maximum - global;
            const float weight = channel >= splits ? 0.0f :
                delta < -126.0f ? CUDART_NAN_F : exp2f(delta);
            float contribution = channel < splits ? weight * part[channel].mass : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                contribution += __shfl_xor_sync(0xffffffffu, contribution, offset);
            }
            if (channel < stride) {
                weights[channel] = weight;
            }
            if (channel == 0) {
                total = contribution;
            }
        }
        __syncthreads();
        float numerator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            numerator = fmaf(weights[split], part[split].value[channel], numerator);
        }
        const auto result = __float2bfloat16_rn(numerator / total);
        invalid = __syncthreads_or(!(total > 0.0f) || !isfinite(total) || !isfinite(numerator) ||
                                   !isfinite(__bfloat162float(result)));
        if (!invalid) {
            lane.buffers.output[head * mxfp8_head_dim + channel] = result;
            return true;
        }
    }
    if (channel == 0) {
        part[0].maximum = CUDART_NAN_F;
    }
    return false;
}

}

template <typename Binding, int FixedSplits = 0>
__global__ void target_split_combine(const __grid_constant__ Binding binding) {
    const auto lane = gqa_decode::resolve(binding);
    if (!decoder::target_split_position(lane.position, binding.end) ||
        !gqa_decode::readable(lane)) {
        return;
    }
    const int head = blockIdx.x;
    if (head >= gqa_query_heads(lane.cache.kv_heads)) {
        return;
    }
    if (!gqa_decode::tensor_aligned(lane) || !combine::combine_head<FixedSplits>(lane, head)) {
        wide_decode_head(lane, head);
    }
}

int target_split_heads(const decoder::PagedDecodeLane& lane) {
    return gqa_query_heads(lane.cache.kv_heads);
}

int target_split_heads(const decoder::BoundPagedDecodeLane&) {
    return mxfp8_query_heads;
}

int target_split_heads(const gqa_decode::TargetLane& lane) {
    return lane.bound.position ? target_split_heads(lane.bound)
                               : target_split_heads(lane.eager);
}

cudaError_t launch_target_split(gqa_decode::TargetLane lane, cudaStream_t stream) {
    const auto status = gqa_decode::launch(lane, stream);
    if (status != cudaSuccess) {
        return status;
    }
    target_split_combine<<<target_split_heads(lane), mxfp8_head_dim, 0, stream>>>(lane);
    return cudaGetLastError();
}

template <typename Batch>
cudaError_t launch_target_batch_combine(const Batch& batch, int count, cudaStream_t stream) {
    target_split_combine<<<dim3(target_split_heads(batch.lanes[0]), 1, count),
                          mxfp8_head_dim, 0, stream>>>(batch);
    return cudaGetLastError();
}

template <typename Batch>
cudaError_t launch_target_batch(const Batch& batch, int count, cudaStream_t stream) {
    const auto status = gqa_decode::launch(batch, count, stream);
    return status == cudaSuccess ? launch_target_batch_combine(batch, count, stream) : status;
}

bool valid_target_lane(const decoder::PagedDecodeLane& lane, int end) {
    return decoder::target_split_position(lane.position, end) && gqa_decode::readable(lane) &&
           gqa_decode::valid_buffers(lane.buffers);
}

bool valid_target_binding(const decoder::BoundPagedDecodeLane& lane) {
    return lane.cache && lane.pages && lane.position && lane.scores &&
           gqa_decode::valid_buffers(lane.buffers);
}

int eager_target_batch_width(const decoder::PagedDecodeLane* lanes, int count, int end) {
    constexpr int reference_width = text_gdn_max_verify_rows;
    if (end != text_attention_max_tokens || count < 2 * reference_width) {
        return reference_width;
    }
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        if (!gqa_decode::tensor_aligned(lane) ||
            !gqa_decode::supports_eager_wide_split(lane.position)) {
            return reference_width;
        }
    }
    return count;
}

}

cudaError_t decoder::target_split_gqa(PagedDecodeLane lane, cudaStream_t stream, int end) {
    if (!valid_target_lane(lane, end)) {
        return cudaErrorInvalidValue;
    }
    return launch_target_split({lane, {}, end}, stream);
}

cudaError_t decoder::target_split_gqa(BoundPagedDecodeLane lane, cudaStream_t stream, int end) {
    if (!valid_target_binding(lane)) {
        return cudaErrorInvalidValue;
    }
    const auto status = launch_target_split({{}, lane, end}, stream);
    return status == cudaSuccess ? launch_target_fallback(lane, end, stream) : status;
}

cudaError_t decoder::target_split_gqa(const BoundPagedDecodeLane* lanes, int count, int end,
                                     cudaStream_t stream) {
    if (!lanes || count < 1 || count > text_gdn_max_verify_rows) {
        return cudaErrorInvalidValue;
    }
    gqa_decode::TargetBatch batch{{}, end};
    for (int index = 0; index < count; ++index) {
        if (!valid_target_binding(lanes[index])) {
            return cudaErrorInvalidValue;
        }
        batch.lanes[index] = lanes[index];
    }
    auto status = launch_target_batch(batch, count, stream);
    for (int index = 0; index < count && status == cudaSuccess; ++index) {
        status = launch_target_fallback(lanes[index], end, stream);
    }
    return status;
}

cudaError_t decoder::target_split_gqa(const PagedDecodeLane* lanes, int count, int end,
                                     cudaStream_t stream) {
    if (!lanes || count < 1 || count > target_split_eager_max_rows ||
        (count > text_gdn_max_verify_rows && end != text_attention_max_tokens)) {
        return cudaErrorInvalidValue;
    }
    for (int index = 0; index < count; ++index) {
        if (!valid_target_lane(lanes[index], end)) {
            return cudaErrorInvalidValue;
        }
    }
    if (count == target_split_eager_max_rows) {
        gqa_decode::EagerTargetBatch batch{{}, end};
        std::copy_n(lanes, count, batch.lanes);
        if (const auto status = gqa_decode::try_launch_reuse(batch, count, stream)) {
            if (*status != cudaSuccess) {
                return *status;
            }
            target_split_combine<<<
                dim3(target_split_heads(batch.lanes[0]), 1, count), mxfp8_head_dim, 0, stream>>>(
                batch);
            return cudaGetLastError();
        }
    }

    const int width = eager_target_batch_width(lanes, count, end);
    for (int offset = 0; offset < count; offset += width) {
        const int size = std::min(width, count - offset);
        gqa_decode::EagerTargetBatch batch{{}, end};
        std::copy_n(lanes + offset, size, batch.lanes);
        const auto status = launch_target_batch(batch, size, stream);
        if (status != cudaSuccess) {
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t gqa_decode::combine_long_decode(const LongBatch& batch, int requests, int splits,
                                            cudaStream_t stream) {
    static_assert(long_min_position >= (max_splits - 1) * min_split_chunks * split_chunk_tokens);
    void* arguments[] = {const_cast<LongBatch*>(&batch)};
    const auto kernel = splits == long_rank_wide_splits
        ? target_split_combine<LongBatch, long_rank_wide_splits>
        : target_split_combine<LongBatch, max_splits>;
    return cudaLaunchKernel(kernel, dim3(gqa_query_heads(batch.requests[0].cache.kv_heads),
                                        1, requests * long_rows),
                            mxfp8_head_dim, arguments, 0, stream);
}

namespace gqa_device {

cudaError_t combine_target_batch(const gqa_decode::EagerTargetBatch& batch, int count,
                                 cudaStream_t stream) {
    return launch_target_batch_combine(batch, count, stream);
}

}

}
