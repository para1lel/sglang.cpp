#pragma once

#include "sglang/text_attention.h"

namespace sglang::gdn_prefill {

constexpr int warp_size = 32;
constexpr int value_rows_per_block = 8;
constexpr int recurrent_threads = value_rows_per_block * warp_size;
constexpr int value_blocks_per_head = text_gdn_head_dim / value_rows_per_block;
constexpr int keys_per_lane = text_gdn_head_dim / warp_size;
constexpr int conv_threads = 128;
constexpr int conv_taps = text_gdn_history_taps + 1;
constexpr int prefill_rows_per_block = 4;
constexpr float norm_epsilon = 1.0e-6f;

static_assert(recurrent_threads == 2 * text_gdn_head_dim);
static_assert(text_gdn_head_dim % warp_size == 0);
static_assert(conv_threads == text_gdn_head_dim);
static_assert(conv_taps == 4);

__device__ __inline_hint__ float warp_sum(float value);

template <typename S>
cudaError_t launch_recur(int rows, __nv_bfloat16* workspace, float2* gates, float* recurrent,
                         __nv_bfloat16* observable, cudaStream_t stream);

}
