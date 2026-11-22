#include "sglang/text/target_fp8.h"

#include "sglang/text/decode_proj.h"

#include <cuda_runtime.h>

namespace sglang {
namespace {

constexpr int group_size = text_fp8_group_size;

struct Operands {
    const std::uint8_t* values[text_fp8_max_batch];
};

__global__ void pack_input(Operands operands, std::uint8_t* packed, int width) {
    const int row = blockIdx.y;
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element < width / 16) {
        reinterpret_cast<uint4*>(packed)[row * (width / 16) + element] =
            reinterpret_cast<const uint4*>(operands.values[row])[element];
    }
}

}

std::optional<cudaError_t> try_target_fp8_tc(const TextFp8DecodeTarget* targets,
                                           int target_count, int in_width,
                                           const TextFp8Projection* rows, int count,
                                           __nv_bfloat16* scratch, cudaStream_t stream) {
    if (!rows || !targets || target_count < 1 || target_count > text_decode_max_targets ||
        in_width < group_size || in_width % group_size || count < 1 ||
        count > text_fp8_max_batch) {
        return std::nullopt;
    }
    for (int index = 0; index < target_count; ++index) {
        const auto& target = targets[index];
        if (!target.weight || !target.weight_scales || target.out_width < group_size ||
            target.out_width % group_size || !decode_proj::aligned16(target.weight)) {
            return std::nullopt;
        }
        for (int row = 0; row < count; ++row) {
            const auto& projection = rows[index * count + row];
            if (!projection.output || projection.input.values != rows[row].input.values ||
                projection.input.scales != rows[row].input.scales) {
                return std::nullopt;
            }
        }
    }
    if (count == 1) {
        return try_text_fp8_decode_sm100(targets, target_count, rows[0].input, in_width, 1, stream);
    }
    if (!decode_proj::aligned16(scratch)) {
        return std::nullopt;
    }
    Operands operands{};
    auto* packed_input = reinterpret_cast<std::uint8_t*>(scratch);
    bool already_packed = true;
    for (int row = 0; row < count; ++row) {
        if (!rows[row].input.values || !decode_proj::aligned16(rows[row].input.values) ||
            !rows[row].input.scales) {
            return std::nullopt;
        }
        operands.values[row] = rows[row].input.values;
        already_packed &= operands.values[row] == packed_input + row * in_width;
    }
    if (!already_packed) {
        constexpr int threads = 128;
        pack_input<<<dim3((in_width / 16 + threads - 1) / threads, count), threads, 0, stream>>>(
            operands, packed_input, in_width);
        if (const auto status = cudaGetLastError(); status != cudaSuccess) {
            return status;
        }
    }
    return try_text_fp8_decode_sm100_rows(
        targets, target_count, packed_input, in_width, rows, count, stream);
}

}
