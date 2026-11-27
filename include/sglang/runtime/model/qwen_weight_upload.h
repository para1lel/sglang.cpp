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

struct Fp8PackSlice {
    std::string stem;
    int full_rows;
    int first_row;
    int row_count;
    bool joined = false;
};

struct Bf16RowSlice {
    int first_row;
    int row_count;
};

class QwenWeightUpload {
public:
    QwenWeightUpload(const Checkpoint& checkpoint, Allocations& memory, cudaStream_t stream);

    const Bf16* bf16(const std::string& name, std::initializer_list<std::uint64_t> shape);

    Bf16Weight bf16_matrix(const std::string& name, int rows, int columns);

    Fp8Weight fp8_matrix(const std::string& stem, int rows, int columns);

    Bf16Weight bf16_matrix_rows(const std::string& name, int full_rows, int columns,
                                int first_row, int row_count);

    const Bf16* bf16_packed_rows(const std::string& name,
                                 std::initializer_list<std::uint64_t> shape,
                                 std::initializer_list<Bf16RowSlice> slices);

    Fp8Weight fp8_matrix_columns(const std::string& stem, int rows, int full_columns,
                                 int first_column, int column_count);

    std::vector<Fp8Weight> packed_fp8(std::initializer_list<Fp8PackSlice> slices, int columns);

private:

    static std::uint64_t extent(int size);

    const std::uint8_t* upload(const std::string& name, TensorDtype dtype,
                               std::initializer_list<std::uint64_t> shape,
                               std::uint8_t* destination = nullptr);

    const std::uint8_t* upload_rows(const std::string& name, TensorDtype dtype,
                                    std::initializer_list<std::uint64_t> shape, int first_row,
                                    int row_count, std::uint8_t* destination = nullptr);

    const std::uint8_t* upload_columns(const std::string& name, TensorDtype dtype,
                                       std::initializer_list<std::uint64_t> shape,
                                       int first_column, int column_count,
                                       std::uint8_t* destination = nullptr);

    const Checkpoint& checkpoint_;
    Allocations& memory_;
    cudaStream_t stream_;
};

}
