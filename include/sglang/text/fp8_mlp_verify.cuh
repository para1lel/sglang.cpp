#pragma once

constexpr int mlp_verify_rows = 5;
constexpr int mlp_verify_channels = 4;
constexpr int mlp_verify_steps = 2;
constexpr int mlp_verify_threads = fp8_packed_warps * warp_width;
constexpr int mlp_verify_outputs = mlp_verify_channels * fp8_packed_warps;
constexpr int mlp_verify_groups = mlp_verify_steps * fp8_groups_per_warp;
static_assert(text_fp8_group_size % mlp_verify_channels == 0);
static_assert(decoder::hidden_size % mlp_verify_outputs == 0 &&
              decoder::intermediate_size % mlp_verify_outputs == 0);
static_assert(decoder::hidden_size % (mlp_verify_groups * text_fp8_group_size) == 0 &&
              decoder::intermediate_size % (mlp_verify_groups * text_fp8_group_size) == 0);
static_assert(text_gdn_qkv_width % mlp_verify_outputs == 0 &&
              text_gdn_value_width % mlp_verify_outputs == 0 &&
              text_gdn_value_width % (mlp_verify_groups * text_fp8_group_size) == 0);

bool is_fp8_mlp_or_gdn_shape(int out_width, int in_width) {
    if (in_width == decoder::hidden_size) {
        return out_width == decoder::intermediate_size || out_width == text_gdn_qkv_width ||
               out_width == text_gdn_value_width;
    }
    return out_width == decoder::hidden_size &&
           (in_width == decoder::intermediate_size || in_width == text_gdn_value_width);
}

struct Fp8MlpStage {
    uint4 weight[mlp_verify_channels][fp8_packed_warps][mlp_verify_steps][warp_width];
    uint4 activation[mlp_verify_rows * mlp_verify_steps * warp_width];
};

__device__ void copy_mlp_pack(uint4* target, const std::uint8_t* source) {
    const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(target));
    const auto global = __cvta_generic_to_global(source);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                 "r"(shared), "l"(global) : "memory");
}

__device__ void load_mlp_stage(Fp8MlpStage& stage,
                               const std::uint8_t* weight, int in_width, int begin,
                               const Fp8Projections<mlp_verify_rows>& projections) {
    const int warp = threadIdx.x / warp_width;
    const int lane = threadIdx.x % warp_width;
    const auto weight_offset = static_cast<std::size_t>(
        mlp_verify_channels * (blockIdx.x * fp8_packed_warps + warp)) * in_width;
    const auto col = begin + lane * fp8_pack_bytes;
#pragma unroll
    for (int channel = 0; channel < mlp_verify_channels; ++channel) {
#pragma unroll
        for (int step = 0; step < mlp_verify_steps; ++step) {
            copy_mlp_pack(&stage.weight[channel][warp][step][lane],
                          weight + weight_offset + static_cast<std::size_t>(channel) * in_width +
                              col + step * warp_width * fp8_pack_bytes);
        }
    }
    constexpr int packs_per_row = mlp_verify_steps * warp_width;
    for (int index = threadIdx.x; index < mlp_verify_rows * packs_per_row;
         index += mlp_verify_threads) {
        const int item = index / packs_per_row;
        const int pack = index % packs_per_row;
        copy_mlp_pack(&stage.activation[index],
                      projections.entries[item].input.values + begin + pack * fp8_pack_bytes);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__global__ void fp8_mlp_staged_kernel(const std::uint8_t* weight,
                                      const __nv_bfloat16* weight_scales, int in_width, int groups,
                                      const __grid_constant__
                                          Fp8Projections<mlp_verify_rows> projections) {
    constexpr int lanes_per_group = warp_width / fp8_groups_per_warp;
    const int warp = threadIdx.x / warp_width;
    const int lane = threadIdx.x % warp_width;
    const int row = mlp_verify_channels * (blockIdx.x * fp8_packed_warps + warp);
    const auto* row_scales =
        weight_scales + static_cast<std::size_t>(row / text_fp8_group_size) * groups;
    __shared__ Fp8MlpStage stages[2];
    int current = 0;
    float sums[mlp_verify_rows][mlp_verify_channels]{};
    load_mlp_stage(stages[current], weight, in_width, 0, projections);
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int group = 0; group < groups; group += mlp_verify_groups) {
        if (group + mlp_verify_groups < groups) {
            load_mlp_stage(stages[1 - current], weight, in_width,
                           (group + mlp_verify_groups) * text_fp8_group_size, projections);
        }
        const auto& stage = stages[current];
#pragma unroll
        for (int step = 0; step < mlp_verify_steps; ++step) {
            const int scale_index = group + step * fp8_groups_per_warp + lane / lanes_per_group;
            const float scale = __bfloat162float(row_scales[scale_index]);
            uint4 lhs[mlp_verify_channels];
#pragma unroll
            for (int channel = 0; channel < mlp_verify_channels; ++channel) {
                lhs[channel] = stage.weight[channel][warp][step][lane];
            }
#pragma unroll
            for (int item = 0; item < mlp_verify_rows; ++item) {
                const auto input = projections.entries[item].input;
                const auto rhs = stage.activation[(item * mlp_verify_steps + step) * warp_width +
                                                  lane];
                const float input_scale = input.scales[scale_index];
#pragma unroll
                for (int channel = 0; channel < mlp_verify_channels; ++channel) {
                    const auto left = lhs[channel];
                    const float low = fp8x4_dot(left.y, rhs.y, fp8x4_dot(left.x, rhs.x, 0.0f));
                    const float high = fp8x4_dot(left.w, rhs.w, fp8x4_dot(left.z, rhs.z, 0.0f));
                    const float scaled = (low + high) * input_scale;
                    sums[item][channel] = fmaf(scaled, scale, sums[item][channel]);
                }
            }
        }
        if (group + mlp_verify_groups < groups) {
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
        }
        current = 1 - current;
    }
#pragma unroll
    for (int item = 0; item < mlp_verify_rows; ++item) {
#pragma unroll
        for (int channel = 0; channel < mlp_verify_channels; ++channel) {
            const float sum = warp_sum(sums[item][channel]);
            if (lane == 0) {
                projections.entries[item].output[row + channel] = __float2bfloat16_rn(sum);
            }
        }
    }
}

cudaError_t launch_fp8_mlp_staged(const std::uint8_t* weight, const __nv_bfloat16* weight_scales,
                                 int out_width, int in_width,
                                 const TextFp8Projection* projections, cudaStream_t stream) {
    Fp8Projections<mlp_verify_rows> batch;
    for (int index = 0; index < mlp_verify_rows; ++index) {
        batch.entries[index] = projections[index];
    }
    int groups = text_fp8_group_count(in_width);
    void* args[] = {&weight, &weight_scales, &in_width, &groups, &batch};
    return cudaLaunchKernel(fp8_mlp_staged_kernel, out_width / mlp_verify_outputs,
                            mlp_verify_threads, args, 0, stream);
}
