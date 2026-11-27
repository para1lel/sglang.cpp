#pragma once

template <int Count>
struct Bf16Projections {
    TextBf16Projection entries[Count];
};

template <int Count>
__global__ void bf16_batch_packed_kernel(const __nv_bfloat16* weight, int out_width,
                                         int in_width, Bf16Projections<Count> batch) {
    const int lane = threadIdx.x % warp_width;
    const int row = blockIdx.x * block_warps + threadIdx.x / warp_width;
    if (row >= out_width) {
        return;
    }
    const auto* row_weight = weight + static_cast<std::size_t>(row) * in_width;
    constexpr int pack = 8;
    float sums[Count]{};
    float second_sums[Count]{};
    for (std::size_t col = lane * pack; col < static_cast<std::size_t>(in_width);
         col += warp_width * pack) {
        const uint4 lhs = load_128(row_weight + col);
#pragma unroll
        for (int index = 0; index < Count; ++index) {
            const uint4 rhs = load_128(batch.entries[index].input + col);
            sums[index] = bf16x2_dot(lhs.x, rhs.x, sums[index]);
            second_sums[index] = bf16x2_dot(lhs.y, rhs.y, second_sums[index]);
            sums[index] = bf16x2_dot(lhs.z, rhs.z, sums[index]);
            second_sums[index] = bf16x2_dot(lhs.w, rhs.w, second_sums[index]);
        }
    }
#pragma unroll
    for (int index = 0; index < Count; ++index) {
        const float sum = warp_sum(sums[index] + second_sums[index]);
        if (lane == 0) {
            batch.entries[index].output[row] = __float2bfloat16_rn(sum);
        }
    }
}

template <int Count>
cudaError_t launch_bf16_batch(const __nv_bfloat16* weight, int out_width, int in_width,
                              const TextBf16Projection* projections, cudaStream_t stream) {
    Bf16Projections<Count> batch;
    std::copy_n(projections, Count, batch.entries);
    void* args[] = {&weight, &out_width, &in_width, &batch};
    return cudaLaunchKernel(bf16_batch_packed_kernel<Count>, divide_up(out_width, block_warps),
                            block_threads, args, 0, stream);
}
