#pragma once

#include <cstddef>
#include <optional>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "sglang/text_ops.h"

namespace sglang {

struct Fp8DecodeInputProducer {
    const void* context;
    cudaError_t (*launch)(const void* context, bool prefetch_weights);
};

std::optional<cudaError_t> try_text_fp8_decode_sm100(const TextFp8DecodeTarget* targets,
                                                     int count, TextFp8ActivationView input,
                                                     int in_width, int rows,
                                                     cudaStream_t stream,
                                                     const Fp8DecodeInputProducer* producer =
                                                         nullptr);

std::optional<cudaError_t> try_text_fp8_decode_sm100_rows(
    const TextFp8DecodeTarget* targets, int target_count, const std::uint8_t* packed_values,
    int in_width, const TextFp8Projection* rows, int count, cudaStream_t stream);

std::optional<cudaError_t> try_text_bf16_decode_sm100(
    const __nv_bfloat16* weight, const __nv_bfloat16* input, int out_width, int in_width,
    int rows, __nv_bfloat16* output, cudaStream_t stream);

std::optional<cudaError_t> try_text_bf16_decode_gates(
    const __nv_bfloat16* weight_b, const __nv_bfloat16* weight_a,
    const __nv_bfloat16* input, int rows, __nv_bfloat16* output_b,
    __nv_bfloat16* output_a, int channels, cudaStream_t stream);

namespace decode_proj {

struct LaunchTarget {
    int device;
    int sm_count;
    PFN_cuTensorMapEncodeTiled_v12000 encode;
};

std::optional<cudaError_t> prepare_launch(cudaStream_t stream, LaunchTarget& target);

bool aligned16(const void* pointer);

struct SplitPlan {
    int splits;
    int shared_bytes;
};

cudaError_t configure(const void* kernel, int slot, int device, int& limit);

std::optional<cudaError_t> prepare_gate_cluster(const void* kernel, const LaunchTarget& target,
                                                int clusters, int shared_bytes);

cudaError_t plan_splits(const void* kernel, int slot, const LaunchTarget& target,
                        int row_blocks, int k_blocks, int fixed_bytes, int bytes_per_block,
                        SplitPlan& plan);

std::optional<cudaError_t> prepare_bf16_single_split(const LaunchTarget& target, int row_blocks,
                                                     int chunks);

CUresult encode_map(const LaunchTarget& target, CUtensorMap& map, const void* data,
                    CUtensorMapDataType type, int element_bytes, int rows, int width,
                    int box_rows);

cudaError_t launch_split(const void* kernel, dim3 grid, int shared_bytes, void** args,
                         cudaStream_t stream, unsigned cluster);
cudaError_t launch_split(const void* kernel, dim3 grid, int shared_bytes, void** args,
                         cudaStream_t stream, unsigned cluster, int block_threads);

}

}
