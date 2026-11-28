#pragma once

#include "sglang/decoder_cuda.cuh"

namespace sglang::gdn_prefill {

constexpr int warp_size = 32;
constexpr int value_rows_per_block = 8;
constexpr int recurrent_threads = value_rows_per_block * warp_size;
constexpr int value_blocks_per_head = text_gdn_head_dim / value_rows_per_block;
constexpr int keys_per_lane = text_gdn_head_dim / warp_size;
constexpr int conv_threads = 128;
constexpr int conv_taps = text_gdn_history_taps + 1;
constexpr int prefill_rows_per_block = 4;
constexpr float norm_epsilon = 1.0e-6F;

static_assert(recurrent_threads == 2 * text_gdn_head_dim);
static_assert(text_gdn_head_dim % warp_size == 0);
static_assert(conv_threads == text_gdn_head_dim);
static_assert(conv_taps == 4);

__device__ __inline_hint__ float warp_sum(float value);
__device__ __inline_hint__ float sigmoid(float value);
struct GateParameters {
    float negative_a;
    float dt_bias;
};

struct OrderedQK {
    const float* keys;
    const float* queries;
    int stride;
};

__device__ __inline_hint__ float2 gated_decay(GateParameters weights, const __nv_bfloat162& inputs);
__device__ __inline_hint__ float ordered_row(float (&state)[keys_per_lane], OrderedQK qk,
                                             float2 gates, float target);
__device__ __inline_hint__ __nv_bfloat16 finish_value(float2 value_rms,
                                                      const __nv_bfloat162& norm_z);
__device__ __inline_hint__ float finish_rms(const __nv_bfloat16* observable,
                                            float (&values)[keys_per_lane]);

struct RawInput {
    const __nv_bfloat16* qkv;
    const __nv_bfloat16* history;
};

struct RowChannel {
    int row;
    int channel;
};

template <typename S>
__device__ __nv_bfloat16 convolve(RawInput input, const __nv_bfloat16* weights, RowChannel point);

template <typename S>
__device__ void finish_history(int token_count, RawInput input, __nv_bfloat16* final_history,
                               int feature);

struct Checkpoints {
    int count;
    TextGdnPrefillCheckpoint values[text_gdn_max_prefill_checkpoints];
};

__device__ __inline_hint__ void store_checkpoint(const Checkpoints& checkpoints, int& next,
                                                 RowChannel point,
                                                 const float (&state)[keys_per_lane]);

struct Segment {
    int rows;
    const __nv_bfloat16* qkv;
    const __nv_bfloat16* z;
    const __nv_bfloat16* b;
    const __nv_bfloat16* a;
    TextGdnState state;
    __nv_bfloat16* workspace;
    float2* gates;
    TextFp8Activation output;
};

struct Segments {
    Segment values[decoder::max_gdn_prefill_segments];
};

struct SegmentExtent {
    int count;
    int max_rows;
};

struct SegmentCheckpoints {
    Checkpoints values[decoder::max_gdn_prefill_segments];
};

bool pack_checkpoints(int rows, TextGdnState active, TextGdnPrefillCheckpoints input,
                      Checkpoints& result);

template <typename S>
cudaError_t launch_prepare(int rows, const __nv_bfloat16* qkv, const __nv_bfloat16* history,
                           TextGdnWeights weights, const __nv_bfloat16* a, const __nv_bfloat16* b,
                           __nv_bfloat16* workspace, float2* gates, cudaStream_t stream);

template <typename S>
cudaError_t launch_prepare_segments(Segments& segments, SegmentExtent extent,
                                    TextGdnWeights weights, cudaStream_t stream);

template <typename S>
cudaError_t launch_history(const __nv_bfloat16* qkv, const __nv_bfloat16* history,
                           Checkpoints& checkpoints, cudaStream_t stream);

template <typename S>
cudaError_t launch_history_segments(Segments& segments, int count, SegmentCheckpoints& checkpoints,
                                    cudaStream_t stream);

template <typename S>
cudaError_t launch_recur(int rows, __nv_bfloat16* workspace, float2* gates, float* recurrent,
                         __nv_bfloat16* observable, cudaStream_t stream, Checkpoints* checkpoints);
}
