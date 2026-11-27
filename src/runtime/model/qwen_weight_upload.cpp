#include "sglang/runtime/model/qwen_weight_upload.h"

namespace sglang::decoder {

QwenWeightUpload::QwenWeightUpload(const Checkpoint& checkpoint, Allocations& memory,
                                   cudaStream_t stream)
    : checkpoint_(checkpoint), memory_(memory), stream_(stream) {}

const Bf16* QwenWeightUpload::bf16(const std::string& name,
                                  std::initializer_list<std::uint64_t> shape) {
    return reinterpret_cast<const Bf16*>(upload(name, TensorDtype::BF16, shape));
}

Bf16Weight QwenWeightUpload::bf16_matrix(const std::string& name, int rows, int columns) {
    return {bf16(name, {extent(rows), extent(columns)}), rows, columns};
}

Fp8Weight QwenWeightUpload::fp8_matrix(const std::string& stem, int rows, int columns) {
    const auto* values = upload(stem + ".weight", TensorDtype::F8_E4M3,
                                {extent(rows), extent(columns)});
    const auto* scales = bf16(stem + ".weight_scale_inv",
                              {extent(text_fp8_group_count(rows)),
                               extent(text_fp8_group_count(columns))});
    return {values, scales, rows, columns};
}

Bf16Weight QwenWeightUpload::bf16_matrix_rows(const std::string& name, int full_rows, int columns,
                            int first_row, int row_count) {
    return {reinterpret_cast<const Bf16*>(upload_rows(name, TensorDtype::BF16,
                                                      {extent(full_rows), extent(columns)},
                                                      first_row, row_count)),
            row_count, columns};
}

const Bf16* QwenWeightUpload::bf16_packed_rows(const std::string& name,
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

Fp8Weight QwenWeightUpload::fp8_matrix_columns(const std::string& stem, int rows, int full_columns,
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

std::vector<Fp8Weight> QwenWeightUpload::packed_fp8(std::initializer_list<Fp8PackSlice> slices,
                                                  int columns) {
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

std::uint64_t QwenWeightUpload::extent(int size) {
    return static_cast<std::uint64_t>(size);
}

const std::uint8_t* QwenWeightUpload::upload(const std::string& name, TensorDtype dtype,
                           std::initializer_list<std::uint64_t> shape,
                           std::uint8_t* destination) {
    return upload_rows(name, dtype, shape, 0, static_cast<int>(*shape.begin()),
                       destination);
}

const std::uint8_t* QwenWeightUpload::upload_rows(const std::string& name, TensorDtype dtype,
                                std::initializer_list<std::uint64_t> shape, int first_row,
                                int row_count, std::uint8_t* destination) {
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

const std::uint8_t* QwenWeightUpload::upload_columns(const std::string& name, TensorDtype dtype,
                                   std::initializer_list<std::uint64_t> shape,
                                   int first_column, int column_count,
                                   std::uint8_t* destination) {
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

}
