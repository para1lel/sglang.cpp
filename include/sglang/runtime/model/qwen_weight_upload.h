#pragma once

#include "sglang/checkpoint.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/model/qwen_layers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

namespace sglang::decoder {

class QwenWeightUpload {
public:
    QwenWeightUpload(const Checkpoint& checkpoint, Allocations& memory, cudaStream_t stream)
        : checkpoint_(checkpoint), memory_(memory), stream_(stream) {}

    const Bf16* bf16(const std::string& name, std::initializer_list<std::uint64_t> shape) {
        return reinterpret_cast<const Bf16*>(upload(name, TensorDtype::BF16, shape));
    }

    Bf16Weight bf16_matrix(const std::string& name, int rows, int columns) {
        return {bf16(name, {extent(rows), extent(columns)}), rows, columns};
    }

    Fp8Weight fp8_matrix(const std::string& stem, int rows, int columns) {
        const auto* values = upload(stem + ".weight", TensorDtype::F8_E4M3,
                                    {extent(rows), extent(columns)});
        const auto* scales = bf16(stem + ".weight_scale_inv",
                                  {extent(text_fp8_group_count(rows)),
                                   extent(text_fp8_group_count(columns))});
        return {values, scales, rows, columns};
    }

    template <int... Widths>
    std::array<Fp8Weight, sizeof...(Widths)> packed_fp8(
        const std::array<std::string, sizeof...(Widths)>& stems, int columns) {
        static_assert(((Widths > 0 && Widths % text_fp8_group_size == 0) && ...));
        constexpr int total_rows = (Widths + ...);
        constexpr std::array rows{Widths...};
        const int scale_columns = text_fp8_group_count(columns);
        auto* values = memory_.allocate<std::uint8_t>(std::size_t(total_rows) * columns);
        auto* scales = memory_.allocate<Bf16>(std::size_t(text_fp8_group_count(total_rows)) *
                                             scale_columns);
        std::array<Fp8Weight, sizeof...(Widths)> weights;
        for (std::size_t index = 0; index < rows.size(); ++index) {
            const int width = rows[index];
            const int scale_rows = text_fp8_group_count(width);
            const auto& stem = stems[index];
            upload(stem + ".weight", TensorDtype::F8_E4M3, {extent(width), extent(columns)},
                   values);
            upload(stem + ".weight_scale_inv", TensorDtype::BF16,
                   {extent(scale_rows), extent(scale_columns)},
                   reinterpret_cast<std::uint8_t*>(scales));
            weights[index] = {values, scales, width, columns};
            values += std::size_t(width) * columns;
            scales += std::size_t(scale_rows) * scale_columns;
        }
        return weights;
    }

private:
    static std::uint64_t extent(int size) {
        return static_cast<std::uint64_t>(size);
    }

    const std::uint8_t* upload(const std::string& name, TensorDtype dtype,
                               std::initializer_list<std::uint64_t> shape,
                               std::uint8_t* destination = nullptr) {
        const auto& tensor = checkpoint_.tensor(name, dtype, shape);
        if (!destination) {
            destination = memory_.allocate<std::uint8_t>(tensor.size_bytes);
        }
        copy_cuda_async(destination, tensor.data, tensor.size_bytes, cudaMemcpyHostToDevice,
                        stream_, name.c_str());
        return destination;
    }

    const Checkpoint& checkpoint_;
    Allocations& memory_;
    cudaStream_t stream_;
};

}
