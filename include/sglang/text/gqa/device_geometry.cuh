#pragma once

#include "sglang/text/gqa/geometry.cuh"

namespace sglang::gqa_device {

__device__ std::size_t row_offset(std::size_t token, int heads, int head);
__device__ std::size_t payload_offset(int kv_heads, int physical_page, int token, int head);
__device__ std::size_t scale_offset(int kv_heads, int physical_page, int token, int head,
                                    int group = 0);
__device__ int position_value(int position);
__device__ int position_value(const int* position);

}
