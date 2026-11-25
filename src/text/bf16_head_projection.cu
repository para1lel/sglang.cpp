#include "sglang/text/bf16_head_projection.h"

#include "sglang/text/decode_proj.h"

#include <cstdint>

namespace sglang {
namespace {

constexpr int max_rows = 32;
constexpr int width_limit = 1 << 28;  // keeps 32-bit block index math safe
// Mirrors the tcgen05 operator's TMA chunk (chunk_elements = 64 bf16 values
// in try_text_bf16_decode_sm100). Checked here so unsupported layouts fail
// before the pack launch.
constexpr int width_quantum = 64;
constexpr int pack_bytes = 16;
constexpr int pack_width = pack_bytes / int(sizeof(__nv_bfloat16));

struct RowInputs {
    const __nv_bfloat16* input[max_rows];
};

struct RowOutputs {
    __nv_bfloat16* output[max_rows];
};

__global__ void pack_bf16_rows(RowInputs sources, __nv_bfloat16* packed, int in_width) {
    const int row = blockIdx.y;
    const int base = blockIdx.x * blockDim.x * pack_width;
    const auto* source = sources.input[row] + base;
    auto* target = packed + static_cast<std::size_t>(row) * in_width + base;
    for (int lane = threadIdx.x; lane < blockDim.x * pack_width; lane += blockDim.x) {
        if (base + lane < in_width) {
            target[lane] = source[lane];
        }
    }
}

__global__ void scatter_bf16_logits(const __nv_bfloat16* packed, int out_width,
                                    RowOutputs destinations) {
    const int row = blockIdx.y;
    const int base = blockIdx.x * blockDim.x * pack_width;
    const auto* source =
        packed + static_cast<std::size_t>(row) * out_width + base;
    auto* output = destinations.output[row];
    for (int lane = threadIdx.x; lane < blockDim.x * pack_width; lane += blockDim.x) {
        if (base + lane < out_width) {
            output[base + lane] = source[lane];
        }
    }
}

int block_count(int width, int threads) {
    return (width + threads * pack_width - 1) / (threads * pack_width);
}

}  // namespace

std::optional<cudaError_t> project_bf16_head(const __nv_bfloat16* weight, int out_width,
                                             int in_width, const Bf16HeadRow* rows, int count,
                                             __nv_bfloat16* packed, __nv_bfloat16* packed_logits,
                                             cudaStream_t stream) {
    // Unsupported-but-valid shapes: decline before touching any buffer.
    if (count <= 0 || count > max_rows) {
        return std::nullopt;
    }
    if (in_width <= 0 || out_width <= 0 || in_width > width_limit || out_width > width_limit) {
        return cudaErrorInvalidValue;
    }
    // Null buffers are an invalid argument; aligned16(nullptr) would otherwise
    // pass and fault inside a launch.
    if (!weight || !packed || !packed_logits) {
        return cudaErrorInvalidValue;
    }
    // Layouts the tcgen05 operator would decline: fall back before any launch.
    if (!decode_proj::aligned16(weight) || !decode_proj::aligned16(packed) ||
        !decode_proj::aligned16(packed_logits) || in_width % width_quantum != 0) {
        return std::nullopt;
    }
    // Null row array is an invalid argument, not a fallback shape.
    if (!rows) {
        return cudaErrorInvalidValue;
    }
    for (int r = 0; r < count; ++r) {
        if (!rows[r].input || !rows[r].output) {
            return cudaErrorInvalidValue;
        }
    }
    RowInputs inputs{};
    RowOutputs outputs{};
    for (int r = 0; r < count; ++r) {
        inputs.input[r] = rows[r].input;
        outputs.output[r] = rows[r].output;
    }
    constexpr int threads = 128;
    const dim3 pack_grid(block_count(in_width, threads), count);
    pack_bf16_rows<<<pack_grid, threads, 0, stream>>>(inputs, packed, in_width);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) {
        return status;
    }
    auto projected = try_text_bf16_decode_sm100(weight, packed, out_width, in_width, count,
                                                packed_logits, stream);
    if (!projected || *projected != cudaSuccess) {
        return projected;
    }
    const dim3 scatter_grid(block_count(out_width, threads), count);
    scatter_bf16_logits<<<scatter_grid, threads, 0, stream>>>(packed_logits, out_width, outputs);
    return cudaGetLastError();
}

}  // namespace sglang
