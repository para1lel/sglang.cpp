#pragma once

#include "sglang/mxfp8_gqa.h"

namespace sglang::decoder {

constexpr int prefix_kv_pages(int rows) {
    return (rows + mxfp8_page_size - 1) / mxfp8_page_size;
}

void copy_mxfp8_scale_tail(std::uint8_t* dst, int dst_pages, const std::uint8_t* src,
                           int src_pages, int layers, int rows, cudaStream_t stream,
                           const char* operation);

}
