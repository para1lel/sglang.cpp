#pragma once

#include "sglang/checkpoint.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/model/qwen_layers.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace sglang::decoder {

/// One row slice of an FP8 checkpoint matrix inside a packed bank.
/// `stem` names the checkpoint tensor pair stem+".weight"/stem+".weight_scale_inv";
/// `full_rows` is the checkpoint row count (validation keeps checkpoint
/// dimensions global); `first_row`/`row_count` select the rank-local rows.
/// A slice with `joined` set appends its rows and scale rows to the previous
/// returned weight instead of starting a new one, so several slices of one
/// checkpoint tensor can form a single device matrix.
struct Fp8PackSlice {
    std::string stem;
    int full_rows;
    int first_row;
    int row_count;
    bool joined = false;
};

/// One row slice of a BF16 checkpoint tensor (see QwenWeightUpload::bf16_packed_rows).
struct Bf16RowSlice {
    int first_row;
    int row_count;
};

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

    /// Row-slice variant of bf16_matrix: takes rows [first_row, first_row + row_count)
    /// of a [full_rows, columns] matrix into a [row_count, columns] device weight.
    Bf16Weight bf16_matrix_rows(const std::string& name, int full_rows, int columns,
                                int first_row, int row_count) {
        return {reinterpret_cast<const Bf16*>(upload_rows(name, TensorDtype::BF16,
                                                          {extent(full_rows), extent(columns)},
                                                          first_row, row_count)),
                row_count, columns};
    }

    /// Packs the given row slices of one row-major BF16 tensor end to end and
    /// returns the device allocation. The first shape dimension counts rows;
    /// every row is the product of the remaining dimensions wide (a [48]
    /// vector has 48 one-element rows, a [10240,1,4] conv has 10240
    /// four-element rows).
    const Bf16* bf16_packed_rows(const std::string& name,
                                 std::initializer_list<std::uint64_t> shape,
                                 std::initializer_list<Bf16RowSlice> slices) {
        std::uint64_t row_elements = 1;
        bool leading = true;
        for (auto dimension : shape) {
            if (leading) {
                leading = false;
            } else {
                row_elements *= dimension;
            }
        }
        std::size_t total_rows = 0;
        for (const auto& slice : slices) {
            if (slice.row_count < 0) {
                throw std::invalid_argument("negative row slice of " + name);
            }
            total_rows += static_cast<std::size_t>(slice.row_count);
        }
        auto* destination = memory_.allocate<Bf16>(total_rows * row_elements);
        auto* cursor = destination;
        for (const auto& slice : slices) {
            upload_rows(name, TensorDtype::BF16, shape, slice.first_row, slice.row_count,
                        reinterpret_cast<std::uint8_t*>(cursor));
            cursor += static_cast<std::size_t>(slice.row_count) * row_elements;
        }
        return destination;
    }

    /// Column-slice variant of fp8_matrix: takes columns
    /// [first_column, first_column + column_count) of a [rows, full_columns]
    /// matrix into a [rows, column_count] device weight. The slice must land
    /// on whole FP8 scale blocks.
    Fp8Weight fp8_matrix_columns(const std::string& stem, int rows, int full_columns,
                                 int first_column, int column_count) {
        if (first_column % text_fp8_group_size != 0 || column_count % text_fp8_group_size != 0) {
            throw std::invalid_argument("FP8 column slice of " + stem + " must align to " +
                                        std::to_string(text_fp8_group_size) + "-column blocks");
        }
        const auto* values = upload_columns(stem + ".weight", TensorDtype::F8_E4M3,
                                            {extent(rows), extent(full_columns)},
                                            first_column, column_count);
        const auto* scales = reinterpret_cast<const Bf16*>(upload_columns(
            stem + ".weight_scale_inv", TensorDtype::BF16,
            {extent(text_fp8_group_count(rows)), extent(text_fp8_group_count(full_columns))},
            first_column / text_fp8_group_size, text_fp8_group_count(column_count)));
        return {values, scales, rows, column_count};
    }

    /// Packs row slices of FP8 checkpoint matrices into one contiguous value
    /// buffer and one contiguous scale buffer, in listed order. Returns one
    /// Fp8Weight per run of slices: a slice with `joined` set extends the
    /// previous weight instead of starting a new one. Row slices and their
    /// scale blocks must align to whole 128-row groups.
    std::vector<Fp8Weight> packed_fp8(std::initializer_list<Fp8PackSlice> slices, int columns) {
        if (columns % text_fp8_group_size != 0) {
            throw std::invalid_argument("FP8 bank width must align to " +
                                        std::to_string(text_fp8_group_size) + " columns");
        }
        std::vector<Fp8Weight> weights;
        int total_rows = 0;
        for (const auto& slice : slices) {
            if (slice.first_row % text_fp8_group_size != 0 ||
                slice.row_count % text_fp8_group_size != 0) {
                throw std::invalid_argument("FP8 row slice of " + slice.stem +
                                            " must align to " +
                                            std::to_string(text_fp8_group_size) + "-row blocks");
            }
            if (slice.joined && weights.empty()) {
                throw std::invalid_argument("first FP8 slice of a bank cannot be joined");
            }
            if (slice.joined) {
                weights.back().rows += slice.row_count;
            } else {
                weights.push_back(Fp8Weight{nullptr, nullptr, slice.row_count, columns});
            }
            total_rows += slice.row_count;
        }
        const int scale_columns = text_fp8_group_count(columns);
        auto* values = memory_.allocate<std::uint8_t>(std::size_t(total_rows) * columns);
        auto* scales = memory_.allocate<Bf16>(
            std::size_t(text_fp8_group_count(total_rows)) * scale_columns);
        std::size_t index = 0;
        bool started = false;
        for (const auto& slice : slices) {
            if (!started || !slice.joined) {
                if (started) {
                    ++index;
                }
                weights[index].values = values;
                weights[index].scales = scales;
                started = true;
            }
            const int scale_rows = text_fp8_group_count(slice.row_count);
            upload_rows(slice.stem + ".weight", TensorDtype::F8_E4M3,
                        {extent(slice.full_rows), extent(columns)},
                        slice.first_row, slice.row_count, values);
            upload_rows(slice.stem + ".weight_scale_inv", TensorDtype::BF16,
                        {extent(text_fp8_group_count(slice.full_rows)), extent(scale_columns)},
                        text_fp8_group_count(slice.first_row), scale_rows,
                        reinterpret_cast<std::uint8_t*>(scales));
            values += std::size_t(slice.row_count) * columns;
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
        return upload_rows(name, dtype, shape, 0, static_cast<int>(*shape.begin()),
                           destination);
    }

    /// Copies rows [first_row, first_row + row_count) of a row-major tensor
    /// whose first shape dimension counts rows; one contiguous run of bytes.
    const std::uint8_t* upload_rows(const std::string& name, TensorDtype dtype,
                                    std::initializer_list<std::uint64_t> shape, int first_row,
                                    int row_count, std::uint8_t* destination = nullptr) {
        const auto& tensor = checkpoint_.tensor(name, dtype, shape);
        const auto rows = *shape.begin();
        if (first_row < 0 || row_count < 0 || rows < first_row + row_count) {
            throw std::invalid_argument("row slice outside " + name);
        }
        const std::uint64_t row_bytes = tensor.size_bytes / rows;
        if (!destination) {
            destination = memory_.allocate<std::uint8_t>(row_count * row_bytes);
        }
        const std::uint64_t offset = first_row * row_bytes;
        const std::uint64_t bytes = row_count * row_bytes;
        copy_cuda_async(destination, tensor.data + offset, bytes, cudaMemcpyHostToDevice,
                        stream_, name.c_str());
        return destination;
    }

    /// Copies columns [first_column, first_column + column_count) of a
    /// row-major 2-D tensor: `rows` strided runs of `column_count` elements.
    const std::uint8_t* upload_columns(const std::string& name, TensorDtype dtype,
                                       std::initializer_list<std::uint64_t> shape,
                                       int first_column, int column_count,
                                       std::uint8_t* destination = nullptr) {
        const auto& tensor = checkpoint_.tensor(name, dtype, shape);
        if (shape.size() != 2) {
            throw std::invalid_argument("column slicing needs a 2-D tensor: " + name);
        }
        const auto rows = *shape.begin();
        const auto full_columns = *(shape.begin() + 1);
        const std::uint64_t elements = rows * full_columns;
        const std::uint64_t element_bytes = elements ? tensor.size_bytes / elements : 0;
        if (first_column < 0 || column_count < 0 ||
            full_columns < first_column + column_count) {
            throw std::invalid_argument("column slice outside " + name);
        }
        const std::uint64_t chunk_bytes = column_count * element_bytes;
        const std::uint64_t pitch_bytes = full_columns * element_bytes;
        if (!destination) {
            destination = memory_.allocate<std::uint8_t>(chunk_bytes * rows);
        }
        const std::uint64_t offset = first_column * element_bytes;
        if (chunk_bytes == pitch_bytes) {
            copy_cuda_async(destination, tensor.data + offset, chunk_bytes * rows,
                            cudaMemcpyHostToDevice, stream_, name.c_str());
        } else {
            check(cudaMemcpy2DAsync(destination, chunk_bytes, tensor.data + offset,
                                    pitch_bytes, chunk_bytes, rows, cudaMemcpyHostToDevice,
                                    stream_),
                  name.c_str());
        }
        return destination;
    }

    const Checkpoint& checkpoint_;
    Allocations& memory_;
    cudaStream_t stream_;
};

}  // namespace sglang::decoder
