#pragma once

#include "sglang/text/gdn_local.h"

namespace sglang::gdn_chunk {

using Bf16 = __nv_bfloat16;
constexpr int tile = 16;
constexpr int rows = 32;
constexpr int min_count = 128;
struct alignas(32) Matrices {
    Bf16 inverse[rows * rows];
    Bf16 scores[rows * rows];
    float initial[rows];
    float tail[rows];
    float beta[rows];
};

struct Segment {
    int count;
    const Bf16* convolved;
    const float2* gates;
    Matrices* matrices;
    float* recurrent;
    Bf16* observable;
};

struct Segments {
    Segment values[decoder::max_gdn_prefill_segments];
};

template <typename S = GdnFull>
bool eligible(int count, const void* scratch, std::size_t bytes);

template <typename S = GdnFull>
cudaError_t configure(bool& enabled, bool grouped = false, bool* staged_values = nullptr,
                      bool* pipelined = nullptr);

template <typename S = GdnFull>
cudaError_t launch(int count, const Bf16* convolved, const float2* gates,
                   void* scratch, float* recurrent, Bf16* observable,
                   cudaStream_t stream, bool staged_values, bool pipelined);

template <typename S = GdnFull>
cudaError_t launch_segments(Segments segments, int count, int max_rows, cudaStream_t stream);

}
