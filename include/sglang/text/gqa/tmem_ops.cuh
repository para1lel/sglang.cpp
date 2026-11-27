#pragma once

#include "sglang/text/mxfp8_gqa_prefill_common.cuh"

#include <cooperative_groups.h>
#include <cuda.h>

namespace sglang::gqa_prefill::tmem {

struct PairMaps {
    CUtensorMap keys;
    CUtensorMap values;
    CUtensorMap key_scales;
    CUtensorMap value_scales;
};

static_assert(alignof(PairMaps) >= 64 && sizeof(CUtensorMap) % 64 == 0);

std::optional<cudaError_t> encode_pair_maps(Mxfp8KvCache cache, PairMaps& maps);
std::optional<cudaError_t> encode_flash_maps(Mxfp8KvCache cache, CUtensorMap& key_map,
                                             CUtensorMap& value_map, bool automatic);

inline constexpr int tmem_swizzle = 64;
inline constexpr int tmem_producers = 256;
inline constexpr int pair_key_rows = flash_keys / 2;
inline constexpr int pair_value_channels = channels / 2;
inline constexpr int pair_value_swizzle = 128;
inline constexpr int pair_page_halves = mxfp8_page_size / flash_keys;
inline constexpr int pair_scale_bytes = mxfp8_page_size * mxfp8_groups;

struct alignas(128) PairPayload {
    uint4 key[pair_page_halves][pair_key_rows][channels / sizeof(uint4)];
    uint4 value[pair_page_halves][flash_keys][pair_value_channels / sizeof(uint4)];
    std::uint8_t key_scales[pair_scale_bytes];
    std::uint8_t value_scales[pair_scale_bytes];
};

struct alignas(1024) PairTile {
    std::uint16_t query[channels / tmem_swizzle][flash_queries][tmem_swizzle];
    std::uint16_t key[channels / tmem_swizzle][pair_key_rows][tmem_swizzle];
    std::uint16_t value[2][pair_value_channels / tmem_swizzle][flash_keys][tmem_swizzle];
    std::uint16_t probability[2][flash_queries][flash_keys];
    PairPayload raw[2];
    int bad_query[flash_queries];
    int bad_key;
    Word tmem;
    alignas(8) unsigned long long copy_ready;
    alignas(8) unsigned long long scores_ready[2];
    alignas(8) unsigned long long output_ready[2];
    alignas(8) unsigned long long operands_ready[2];
    alignas(8) unsigned long long key_ready;
};

static_assert(mxfp8_page_size == 2 * flash_keys && mxfp8_page_size == 2 * flash_rows);
static_assert(flash_chunk % mxfp8_page_size == 0);
static_assert(sizeof(PairPayload) ==
              pair_page_halves * (pair_key_rows * channels + flash_keys * pair_value_channels) +
                  2 * pair_scale_bytes);
static_assert(pair_value_channels == pair_value_swizzle && sizeof(PairPayload) % 1024 == 0 &&
              offsetof(PairPayload, value) % 1024 == 0 && offsetof(PairTile, raw) % 1024 == 0);
static_assert(offsetof(PairPayload, key) % 128 == 0 &&
              offsetof(PairPayload, value) % 128 == 0 &&
              offsetof(PairPayload, key_scales) % 128 == 0 &&
              offsetof(PairPayload, value_scales) % 128 == 0);
static_assert(offsetof(PairTile, raw) % alignof(PairPayload) == 0 &&
              offsetof(PairTile, query) % 1024 == 0 && offsetof(PairTile, key) % 1024 == 0 &&
              offsetof(PairTile, value) % 1024 == 0 &&
              offsetof(PairTile, probability) % 1024 == 0);

struct alignas(128) FlashPayload {
    alignas(16) uint4 key[flash_keys][channels / sizeof(uint4)];
    alignas(16) uint4 value[flash_keys][channels / sizeof(uint4)];
};

inline constexpr int tmem_cols = 512;
inline constexpr int tmem_scores = channels;
inline constexpr int tmem_threads = flash_queries + tmem_producers;
inline constexpr int pair_threads = tmem_threads + 32;

static_assert(tmem_scores + 2 * flash_keys <= tmem_cols);
inline constexpr int tmem_parts = 3;

struct alignas(1024) TmemFlashTile {
    std::uint16_t query[channels / tmem_swizzle][flash_queries][tmem_swizzle];
    std::uint16_t key[channels / tmem_swizzle][flash_keys][tmem_swizzle];
    std::uint16_t value[channels / tmem_swizzle][flash_keys][tmem_swizzle];
    std::uint16_t probability[flash_queries][flash_keys];
    FlashPayload raw[2];
    int bad_query[flash_queries];
    int bad_key;
    Word tmem;
    alignas(8) unsigned long long copy_ready;
    alignas(8) unsigned long long mma_ready;
};

inline constexpr int pipeline_members = 64;
inline constexpr int pipeline_rows = pipeline_members / ordered_heads;
inline constexpr int pipeline_threads = tmem_threads + 32;
inline constexpr int pipeline_query_rows = 256;

struct alignas(1024) TmemPipelineTile {
    std::uint16_t query[channels / tmem_swizzle][pipeline_members][tmem_swizzle];
    std::uint16_t key[channels / tmem_swizzle][flash_keys][tmem_swizzle];
    std::uint16_t value[2][channels / tmem_swizzle][flash_keys][tmem_swizzle];
    std::uint16_t probability[2][pipeline_members][flash_keys];
    FlashPayload raw[2];
    int bad_query[pipeline_members];
    int bad_key;
    Word tmem;
    alignas(8) unsigned long long copy_ready;
    alignas(8) unsigned long long key_ready;
    alignas(8) unsigned long long scores_ready[2];
    alignas(8) unsigned long long output_ready[2];
    alignas(8) unsigned long long operands_ready[2];
};

static_assert(offsetof(TmemPipelineTile, raw) % alignof(FlashPayload) == 0 &&
              offsetof(TmemPipelineTile, key) % 1024 == 0 &&
              offsetof(TmemPipelineTile, value) % 1024 == 0 &&
              offsetof(TmemPipelineTile, probability) % 1024 == 0);
static_assert(pipeline_query_rows % pipeline_rows == 0 &&
              sizeof(TmemPipelineTile) == sizeof(TmemFlashTile));

struct alignas(1024) StagedPairTile {
    std::uint16_t query[channels / tmem_swizzle][flash_queries][tmem_swizzle];
    std::uint16_t key[2][channels / tmem_swizzle][pair_key_rows][tmem_swizzle];
    std::uint16_t value[2][pair_value_channels / tmem_swizzle][flash_keys][tmem_swizzle];
    std::uint16_t probability[2][flash_queries][flash_keys];
    int bad_query[flash_queries];
    int bad_key;
    Word tmem;
    alignas(8) unsigned long long copy_ready;
    alignas(8) unsigned long long value_copy_ready;
    alignas(8) unsigned long long scores_ready[2];
    alignas(8) unsigned long long output_ready[2];
    alignas(8) unsigned long long operands_ready[2];
    alignas(8) unsigned long long key_ready[2];
};

static_assert(mxfp8_page_size == 2 * flash_keys && mxfp8_page_size == 2 * flash_rows);
static_assert(flash_chunk % mxfp8_page_size == 0);
static_assert(offsetof(StagedPairTile, query) % 1024 == 0 &&
              offsetof(StagedPairTile, key) % 1024 == 0 &&
              offsetof(StagedPairTile, value) % 1024 == 0 &&
              offsetof(StagedPairTile, probability) % 1024 == 0);

struct StagedPairMaps {
    CUtensorMap keys;
    CUtensorMap values;
};

static_assert(alignof(StagedPairMaps) >= 64 && sizeof(CUtensorMap) % 64 == 0);

inline constexpr int staged_planes = channels / tmem_swizzle;
inline constexpr int staged_head_elements = channels * mxfp8_page_size;
inline constexpr int staged_producers = tmem_producers / 2;
inline constexpr int staged_value_leader = flash_queries + staged_producers;
inline constexpr int staged_compact_threads = channels;
inline constexpr int staged_compact_value_leader = flash_queries + 32;
inline constexpr int staged_compact_issuer = staged_compact_value_leader + 32;
inline constexpr std::size_t staged_alignment = 128;
static_assert(staged_compact_issuer + 32 <= staged_compact_threads);
static_assert(tmem_swizzle * sizeof(std::uint16_t) == staged_alignment);
static_assert(staged_head_elements * sizeof(std::uint16_t) % staged_alignment == 0);

template <int Count>
struct StagedRequests {
    Attention items[Count];
    int rows[Count];
    int page_base[Count];

    __device__ int lane() const;
};

inline constexpr int staged_single_score_slots = 3;
inline constexpr int staged_single_probability =
    tmem_scores + staged_single_score_slots * flash_keys;
static_assert(staged_single_probability + flash_keys <= tmem_cols);

struct alignas(1024) StagedSingleTile {
    std::uint16_t query[staged_planes][flash_queries][tmem_swizzle];
    std::uint16_t key[2][staged_planes][flash_keys][tmem_swizzle];
    std::uint16_t value[2][staged_planes][flash_keys][tmem_swizzle];
    int bad_query[flash_queries];
    int bad_key;
    Word tmem;
    unsigned long long key_ready[2];
    unsigned long long value_ready[2];
    unsigned long long scores_ready[3];
    unsigned long long output_ready[2];
    unsigned long long probability_ready[2];
};
static_assert(offsetof(StagedSingleTile, key) % 1024 == 0 &&
              offsetof(StagedSingleTile, value) % 1024 == 0);

inline constexpr int staged_page_keys = mxfp8_page_size;
static_assert(staged_page_keys == 2 * flash_keys &&
              tmem_scores + 2 * staged_page_keys == tmem_cols);

struct alignas(1024) StagedPageTile {
    std::uint16_t query[staged_planes][flash_queries][tmem_swizzle];
    std::uint16_t key[staged_planes][staged_page_keys][tmem_swizzle];
    std::uint16_t value[2][staged_planes][flash_keys][tmem_swizzle];
    int bad_query[flash_queries];
    int bad_key;
    Word tmem;
    unsigned long long key_ready[2];
    unsigned long long value_ready[2];
    unsigned long long scores_ready[2];
    unsigned long long output_ready[2];
    unsigned long long probability_ready[2];
};
static_assert(offsetof(StagedPageTile, key) % 1024 == 0 &&
              offsetof(StagedPageTile, value) % 1024 == 0);
static_assert(sizeof(StagedPageTile) == sizeof(StagedSingleTile));

__device__ Word tmem_smem(const void* pointer);

template <int Members = flash_queries, int LoadWarps = tmem_threads / 32, typename Tile>
__device__ void tmem_stage_query(Tile& tile, Attention request, int row, int head,
                                 int rows = flash_rows);

template <bool MnMajor = false>
__device__ std::uint64_t tmem_descriptor(const void* pointer);

template <bool Paired = false>
__device__ void tmem_commit(Word barrier);

__device__ void tmem_mma_wait(Word barrier, Word phase);

__device__ void tmem_copy_wait(Word barrier, Word phase);

__device__ void pipeline_init(unsigned long long& barrier, Word arrivals);

__device__ void tmem_load(Word pointer, float (&values)[32]);

template <bool Historical, bool Ordered, bool TmemProbability = false, typename Probability>
__device__ float tmem_probabilities(Probability& probability_tile, Word row_address, int member,
                                    int visible, FlashSoftmax& state,
                                    Word score_column = tmem_scores, Word probability_column = 0);

template <bool Paired = false, bool Staged = false, int Members = flash_queries, typename Tile>
__device__ void tmem_query_key(Tile& tile, Word mma,
                                              Word score_column = tmem_scores, int key_slot = -1);

template <bool Paired = false, int Members = flash_queries, typename Probability, typename Value>
__device__ void tmem_probability_value(
    Word tmem, const Probability& probability, const Value& value, Word mma, bool history);

template <bool Paired = false, bool Staged = false, typename Tile>
__device__ void tmem_initialize(Tile& tile, int warp,
                                               Word copy, Word mma);

__device__ void tmem_correct_output(Word row_address, float history);

template <bool Ordered, typename Probability>
__device__ float tmem_softmax_chunk(Probability& probability, Word row_address,
                                                    int position, int first, FlashSoftmax& state,
                                                    Word score_column = tmem_scores,
                                                    int member = threadIdx.x);

template <bool Packed = false>
__device__ void tmem_store_output(
    Word row_address, FlashSoftmax state, Attention request, int row, int head,
    int rows = flash_rows, int member = threadIdx.x);

__device__ void pair_arrive(Word ready);

template <typename Tile>
__device__ void pair_repair(Tile& tile, Attention request,
                            cooperative_groups::cluster_group pair, int row, int head);

template <int Barrier, int Count, int Leader>
__device__ void pair_publish(Word ready);

template <bool Staged = false, typename Tile>
__device__ void pair_issue_mmas(Tile& tile, int end);

template <typename Tile>
__device__ void tmem_repair_ordered(const Tile& tile, Attention request, int row, int head,
                                    int rows);

__device__ int tmem_swizzled(int row, int channel);

__device__ void tmem_decode(std::uint16_t* destination, uint4 payload,
                                             int scale, bool& bad);

template <bool Key>
__device__ void pair_decode_tail(
    std::uint16_t* output, const Mxfp8KvCache& cache, int page, int token, int kv_head,
    int channel, bool& bad);

template <int Count, bool Partial = false>
__global__ void pair_predecode(const __grid_constant__ StagedRequests<Count> requests,
                               std::uint16_t* keys, std::uint16_t* values, int* invalid);

template <bool PartialPage, int Count>
__launch_bounds__(Count > 1 ? staged_compact_threads : pair_threads, 1)
__attribute__((cluster_dims(2, 1, 1)))
__global__ void tmem_staged_pair_attention(const __grid_constant__ StagedRequests<Count> requests,
                                           const __grid_constant__ StagedPairMaps maps,
                                           const int* invalid);

template <int Count>
__global__ __launch_bounds__(256, 1) void tmem_staged_single_attention(
    const __grid_constant__ StagedRequests<Count> requests,
    const __grid_constant__ StagedPairMaps maps, const int* invalid);

template <int Count, bool ScoreLookahead = false>
__global__ __launch_bounds__(256, 1) void tmem_staged_page_attention(
    const __grid_constant__ StagedRequests<Count> requests,
    const __grid_constant__ StagedPairMaps maps, const int* invalid);

std::optional<cudaError_t> prepare_flash_kernel(
    const void* kernel, std::size_t dynamic_shared, const cudaDeviceProp& properties,
    bool automatic);

std::optional<cudaError_t> validate_paired_request(const Attention& request,
                                                  Mxfp8PageTable pages, int rows = flash_chunk);

cudaLaunchConfig_t paired_config(std::size_t shared, int head_pairs, cudaStream_t stream);

std::optional<cudaError_t> prepare_paired_kernel(const void* kernel,
                                                 const cudaLaunchConfig_t& config);

cudaError_t consume_prefill_submission(cudaError_t status);

}
