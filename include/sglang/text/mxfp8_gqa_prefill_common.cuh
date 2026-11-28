#pragma once

#include "sglang/mxfp8_gqa.h"
#include "sglang/text/gqa/device_geometry.cuh"
#include "sglang/text/prefill_attn_sm100.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <optional>

namespace sglang {

namespace decoder {

std::optional<cudaError_t> launch_batched_ragged_tmem(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream);

}

std::optional<cudaError_t> launch_tmem_prefill(Mxfp8KvCache cache, Mxfp8PageTable pages,
                                               int position, const __nv_bfloat16* queries,
                                               __nv_bfloat16* output, cudaStream_t stream,
                                               int rows);

std::optional<cudaError_t> launch_paired_tmem_prefill(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream);

std::optional<cudaError_t> launch_staged_tmem_prefill(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream,
    decoder::PairedPrefillScratch scratch);
namespace gqa_prefill {

using Bf16 = __nv_bfloat16;
using Word = unsigned int;
inline constexpr int channels = mxfp8_head_dim;
inline constexpr int ratio = mxfp8_query_heads / mxfp8_kv_heads;
inline constexpr int ordered_heads = 2;
inline constexpr int keys_per_tile = 32;
inline constexpr Word warp_mask = 0xffffffffu;
inline constexpr float softmax_scale = 1.0f / 16.0f;

static_assert(channels == 256 && ratio == 6 && mxfp8_page_size % keys_per_tile == 0);
static_assert(keys_per_tile == mxfp8_group_size && ratio % ordered_heads == 0);

template <typename Real, int Heads = ordered_heads>
struct OrderedTile {
    Real scores[Heads][keys_per_tile];
    Real history_weight[Heads][keys_per_tile];
    Real token_weight[Heads][keys_per_tile];
    Real denominator[Heads];
};

struct OrderedScratch {
    OrderedTile<float> ordered;
    OrderedTile<double> wide;
};

struct Attention {
    Mxfp8KvCache cache;
    const std::int32_t* pages;
    int position;
    const Bf16* queries;
    Bf16* output;
};

using gqa_device::payload_offset;
using gqa_device::scale_offset;

inline constexpr int flash_vector_elements = 16;
inline constexpr int flash_token_lanes = channels / flash_vector_elements;

template <int Tokens>
struct PrefillPayload {
    alignas(16) std::uint8_t key[Tokens][channels];
    alignas(16) std::uint8_t value[Tokens][channels];
    alignas(8) std::uint8_t key_scale[Tokens][mxfp8_groups];
    alignas(8) std::uint8_t value_scale[Tokens][mxfp8_groups];
};

inline constexpr int flash_rows = 64;
inline constexpr int flash_queries = flash_rows * ordered_heads;
inline constexpr int flash_keys = 64;
inline constexpr int flash_chunk = 1024;
inline constexpr int flash_wide_rows = 88;
inline constexpr int flash_short_rows = 32;
inline constexpr int flash_medium_rows = 48;
inline constexpr int paged_batch_sequences = 32;
inline constexpr int head_pairs = mxfp8_query_heads / ordered_heads;
inline constexpr int min_flash_rows = 144;
inline constexpr int default_ragged_min_position = 8192;
static_assert(mxfp8_page_size % flash_keys == 0);

struct PagedAttentionBatch {
    Attention requests[paged_batch_sequences];
    int rows[paged_batch_sequences];
    int tile_end[paged_batch_sequences];
    int count;

    __device__ int locate(int tile) const;

    __device__ int row(int tile, int index, int tile_rows) const;
};

struct FlashSoftmax {
    float maximum;
    float mass;
};

inline constexpr int max_multirow_visible_tokens = 32768;

__device__ std::size_t row_offset(int query_heads, int row, int head);

template <int Tokens, int Threads>
__device__ void prefetch_payload(PrefillPayload<Tokens>& raw, Attention request, int first,
                                 int visible, int kv_head);

__device__ float ordered_decode(std::uint8_t payload, std::uint8_t exponent);

template <typename Real>
__device__ Real ordered_prefix(Real prefix, int lane);

template <int Heads>
__device__ int score_recovery_mask(int warp, int lane, bool invalid_score);

template <typename Real, bool recover_scores = false>
__device__ int ordered_attention(OrderedTile<Real>& shared, Attention request, int row_index,
                                 int query_head, int invalid = 0);

#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || \
    (defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1000)
constexpr int recover_wide_max_registers = 128;
#else
constexpr int recover_wide_max_registers = 64;
#endif

__device__ __noinline__ __local_maxnreg__(recover_wide_max_registers) void recover_wide(
    OrderedTile<double>& shared, Attention request, int row, int head, int invalid);

__device__ void protected_attention(
    OrderedScratch& shared, Attention request, int row, int head);

template <bool CommonScale>
__device__ void flash_store_vector(std::uint16_t* destination, uint4 payload,
                                                  int scale, bool& bad);

bool raw_pipeline_before_flash(Mxfp8KvCache cache, int rows, int visible);

bool default_ragged_prefill(Mxfp8KvCache cache, int position, int rows);

int long_flash_tile_rows(int rows, int tiles_per_head);

std::optional<cudaError_t> flash_setup_error(cudaError_t status, bool unavailable);

bool paired_kernel_unavailable(cudaError_t status);

bool valid_paged_prefill_sequence(const decoder::PagedPrefillSequence& sequence,
                                        int min_rows, int max_visible);

bool default_ragged_prefill_tail(const decoder::PagedPrefillSequence& sequence,
                                       int max_visible);

std::optional<cudaError_t> launch_flash_prefill(Attention request, int rows,
                                               cudaStream_t stream);

}
}
