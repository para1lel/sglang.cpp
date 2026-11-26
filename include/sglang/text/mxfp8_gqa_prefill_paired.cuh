#pragma once

namespace sglang {
namespace {

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

template <bool Key>
__device__ void pair_stage_tail(StagedPairTile& tile, int first, int rank, const Attention& request,
                               int kv_head, int end, int stage) {
    constexpr int vector_elements = 16;
    constexpr int width = Key ? channels : pair_value_channels;
    constexpr int row_count = Key ? pair_key_rows : flash_keys;
    constexpr int token_lanes = width / vector_elements;
    constexpr int producer_count = tmem_producers / 2;
    const int producer = (threadIdx.x - flash_queries) % producer_count;
    const int vector = producer % token_lanes;
    const int channel = vector * vector_elements;
    const int scale_channel = channel + (Key ? 0 : rank * pair_value_channels);
    const int page = request.pages[first / mxfp8_page_size];
    for (int index = producer / token_lanes; index < row_count;
         index += producer_count / token_lanes) {
        const int sub = channel / tmem_swizzle;
        const int column = channel % tmem_swizzle;
        const int low = tmem_swizzled(index, column);
        const int high = tmem_swizzled(index, column + 8);
        const int logical = first + index + (Key ? rank * pair_key_rows : 0);
        const int token = logical % mxfp8_page_size;
        bool bad = false;
        uint4 packed[2]{};
        if (logical < end) {
            pair_decode_tail<Key>(reinterpret_cast<std::uint16_t*>(packed), request.cache,
                                  page, token, kv_head, scale_channel, bad);
        }
        auto* destination = Key ? tile.key[stage][sub] : tile.value[stage][sub];
        *reinterpret_cast<uint4*>(destination[index] + low) = packed[0];
        *reinterpret_cast<uint4*>(destination[index] + high) = packed[1];
        if (bad) {
            atomicMin(&tile.bad_key, logical);
        }
    }
}

struct StagedPairMaps {
    CUtensorMap keys;
    CUtensorMap values;
};

static_assert(alignof(StagedPairMaps) >= 64 && sizeof(CUtensorMap) % 64 == 0);

constexpr int staged_planes = channels / tmem_swizzle;
constexpr int staged_head_elements = channels * mxfp8_page_size;
constexpr int staged_producers = tmem_producers / 2;
constexpr int staged_value_leader = flash_queries + staged_producers;
constexpr int staged_compact_threads = channels;
constexpr int staged_compact_value_leader = flash_queries + 32;
constexpr int staged_compact_issuer = staged_compact_value_leader + 32;
constexpr std::size_t staged_alignment = 128;
static_assert(staged_compact_issuer + 32 <= staged_compact_threads);
static_assert(tmem_swizzle * sizeof(std::uint16_t) == staged_alignment);
static_assert(staged_head_elements * sizeof(std::uint16_t) % staged_alignment == 0);

template <int Count>
struct StagedRequests {
    Attention items[Count];
    int rows[Count];
    int page_base[Count];

    __device__ int lane() const {
        if constexpr (Count == 1) {
            return 0;
        }
        return blockIdx.z;
    }
};

template <int Count, bool Partial = false>
__global__ void pair_predecode(const __grid_constant__ StagedRequests<Count> requests,
                               std::uint16_t* keys, std::uint16_t* values, int* invalid) {
    constexpr int vector_elements = 16;
    constexpr int vectors = channels / vector_elements;
    const Attention request = requests.items[requests.lane()];
    int page_base = requests.lane() * gridDim.x;
    const int visible = request.position + requests.rows[requests.lane()];
    if constexpr (Partial) {
        if (blockIdx.x >= (visible + mxfp8_page_size - 1) / mxfp8_page_size) return;
        page_base = requests.page_base[requests.lane()];
    }
    const int page = request.pages[blockIdx.x];
    const int page_head = (page_base + blockIdx.x) * request.cache.kv_heads +
                          blockIdx.y;
    const auto base = static_cast<std::size_t>(page_head) * staged_head_elements;
    const int channel = (threadIdx.x % vectors) * vector_elements;
    __shared__ int earliest;
    if (threadIdx.x == 0) {
        earliest = INT_MAX;
    }
    __syncthreads();
    for (int token = threadIdx.x / vectors; token < mxfp8_page_size;
         token += blockDim.x / vectors) {
        const auto offset = base +
            (channel / tmem_swizzle * mxfp8_page_size + token) * tmem_swizzle +
            channel % tmem_swizzle;
        const int logical = static_cast<int>(blockIdx.x) * mxfp8_page_size + token;
        if constexpr (Partial) {
            if (logical >= visible) {
                auto* key = reinterpret_cast<uint4*>(keys + offset);
                auto* value = reinterpret_cast<uint4*>(values + offset);
                key[0] = key[1] = uint4{};
                value[0] = value[1] = uint4{};
                continue;
            }
        }
        bool bad = false;
        pair_decode_tail<true>(keys + offset, request.cache, page, token, blockIdx.y, channel, bad);
        bool bad_value = false;
        pair_decode_tail<false>(values + offset, request.cache, page, token,
                                blockIdx.y, channel, bad_value);
        if (bad || bad_value) {
            atomicMin(&earliest, logical);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        invalid[page_head] = earliest;
    }
}

__device__ void pair_copy_decoded(const CUtensorMap* map, void* destination, Word barrier,
                                  int token, int plane) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {0, %2, %3}], [%4];"
                 :: "r"(tmem_smem(destination)), "l"(map), "r"(token), "r"(plane),
                    "r"(barrier) : "memory");
}

template <bool Key, bool PartialPage, bool Compact>
__device__ void pair_stage_decoded(StagedPairTile& tile, Attention request,
                                   const StagedPairMaps& maps, const int* invalid, int page_base,
                                   int first, int kv_head, int rank, int full_end,
                                   int end, int stage) {
    if (PartialPage && first >= full_end) {
        pair_stage_tail<Key>(tile, first, rank, request, kv_head, end, stage);
    } else if (Compact || threadIdx.x == (Key ? flash_queries : staged_value_leader)) {
        const int page_head =
            (page_base + first / mxfp8_page_size) * request.cache.kv_heads + kv_head;
        const int token = first % mxfp8_page_size + (Key ? rank * pair_key_rows : 0);
        const int plane = page_head * staged_planes + (Key ? 0 : rank * staged_planes / 2);
        const Word barrier = tmem_smem(Key ? &tile.copy_ready : &tile.value_copy_ready);
        constexpr Word bytes = Key ? sizeof(tile.key[0]) : sizeof(tile.value[0]);
        void* destination = Key ? static_cast<void*>(tile.key[stage]) : tile.value[stage];
        const CUtensorMap* map = Key ? &maps.keys : &maps.values;
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(barrier), "r"(bytes) : "memory");
        pair_copy_decoded(map, destination, barrier, token, plane);
        if constexpr (Key) {
            if (first % mxfp8_page_size == 0) {
                atomicMin(&tile.bad_key, invalid[page_head]);
            }
        }
        tmem_copy_wait(barrier, stage);
    }
}

template <bool Key, bool PartialPage, bool Compact>
__device__ void pair_stage_chunks(StagedPairTile& tile, Attention request,
                                  const StagedPairMaps& maps, const int* invalid, int page_base,
                                  int kv_head, int rank, int full_end, int end) {
    for (int chunk = Key ? 1 : 0; chunk * flash_keys < end; ++chunk) {
        const int stage = chunk & 1;
        if (chunk >= 2) {
            const Word retired = tmem_smem(Key ? &tile.scores_ready[stage] :
                                                 &tile.output_ready[stage]);
            tmem_mma_wait(retired, ((chunk - 2) >> 1) & 1);
        }
        pair_stage_decoded<Key, PartialPage, Compact>(tile, request, maps, invalid, page_base,
                                                     chunk * flash_keys, kv_head, rank,
                                                     full_end, end, stage);
        if constexpr (Compact) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            pair_arrive(tmem_smem(Key ? &tile.key_ready[stage] : &tile.operands_ready[stage]));
        } else if constexpr (Key) {
            pair_publish<1, staged_producers, flash_queries>(tmem_smem(&tile.key_ready[stage]));
        } else {
            pair_publish<3, staged_producers, staged_value_leader>(
                tmem_smem(&tile.operands_ready[stage]));
        }
    }
}

template <bool PartialPage, int Count>
__launch_bounds__(Count > 1 ? staged_compact_threads : pair_threads, 1)
__attribute__((cluster_dims(2, 1, 1)))
__global__ void tmem_staged_pair_attention(const __grid_constant__ StagedRequests<Count> requests,
                                           const __grid_constant__ StagedPairMaps maps,
                                           const int* invalid) {
    extern __shared__ StagedPairTile shared_staged_pair[];
    auto& tile = shared_staged_pair[0];
    const auto pair = cooperative_groups::this_cluster();
    constexpr bool compact = Count > 1;
    static_assert(!compact || !PartialPage);
    const int rank = pair.block_rank();
    const Attention request = requests.items[requests.lane()];
    const int page_base = requests.lane() * ((request.position + flash_chunk) / mxfp8_page_size);
    const int row = blockIdx.x * flash_rows;
    const int head = blockIdx.y * ordered_heads;
    const int kv_head = head / ratio;
    const int position = request.position + row;
    const int end = position + (2 - rank) * flash_rows;
    const int full_end = PartialPage ? end - end % mxfp8_page_size : end;
    const int warp = threadIdx.x / 32;
    const bool softmax = warp < flash_queries / 32;
    const bool key_producer = compact ? threadIdx.x == flash_queries :
        threadIdx.x >= flash_queries && threadIdx.x < staged_value_leader;
    const bool value_producer = compact ? threadIdx.x == staged_compact_value_leader :
        threadIdx.x >= staged_value_leader && threadIdx.x < tmem_threads;
    const int issuer_thread = compact ? staged_compact_issuer : tmem_threads;
    const bool issuer = rank == 0 && threadIdx.x == issuer_thread;
    const Word copy = tmem_smem(&tile.copy_ready);
    tmem_initialize<true, true>(tile, warp, copy, tmem_smem(&tile.scores_ready[0]));
    if constexpr (compact) {
        tmem_stage_query<flash_queries, staged_compact_threads / 32>(tile, request, row, head);
    } else if (threadIdx.x < tmem_threads) {
        tmem_stage_query(tile, request, row, head);
    }
    __syncthreads();
    if (key_producer) {
        pair_stage_decoded<true, PartialPage, compact>(tile, request, maps, invalid, page_base, 0,
                                                       kv_head, rank, full_end, end, 0);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    pair.sync();
    if (issuer) {
        pair_issue_mmas<true>(tile, end);
    }
    const Word row_address = tile.tmem + ((warp * 32) << 16);
    FlashSoftmax state{-CUDART_INF_F, 0.0f};
    if (softmax) {
        for (int first = 0, chunk = 0; first < end; first += flash_keys, ++chunk) {
            const int stage = chunk & 1;
            const Word phase = (chunk >> 1) & 1;
            tmem_mma_wait(tmem_smem(&tile.scores_ready[stage]), phase);
            if (chunk >= 2) {
                tmem_mma_wait(tmem_smem(&tile.output_ready[stage]), phase ^ 1);
            }
            const float history = tmem_softmax_chunk<true>(
                tile.probability[stage], row_address, position, first, state,
                tmem_scores + stage * flash_keys);
            if (chunk != 0) {
                tmem_mma_wait(tmem_smem(&tile.output_ready[1 - stage]), ((chunk - 1) >> 1) & 1);
                tmem_correct_output(row_address, history);
            }
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            pair_publish<2, flash_queries, 0>(tmem_smem(&tile.operands_ready[stage]));
        }
        const int final_chunk = (end - 1) / flash_keys;
        tmem_mma_wait(tmem_smem(&tile.output_ready[final_chunk & 1]), (final_chunk >> 1) & 1);
        tmem_store_output<compact>(row_address, state, request, row, head);
    }
    if (key_producer) {
        pair_stage_chunks<true, PartialPage, compact>(tile, request, maps, invalid, page_base,
                                                       kv_head, rank, full_end, end);
    }
    if (value_producer) {
        pair_stage_chunks<false, PartialPage, compact>(tile, request, maps, invalid, page_base,
                                                        kv_head, rank, full_end, end);
    }
    pair.sync();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                     :: "r"(tile.tmem), "r"(tmem_cols) : "memory");
    }
    pair_repair(tile, request, pair, row, head);
}

constexpr int staged_single_score_slots = 3;
constexpr int staged_single_probability = tmem_scores + staged_single_score_slots * flash_keys;
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

template <bool Key, class Tile>
__device__ void single_stage(Tile& tile, const StagedPairMaps& maps,
                             const int* invalid, int page_base, int kv_head, int kv_heads,
                             int end) {
    const auto* map = Key ? &maps.keys : &maps.values;
    auto& destination = [&]() -> auto& {
        if constexpr (Key) return tile.key;
        else return tile.value;
    }();
    auto& ready = Key ? tile.key_ready : tile.value_ready;
    for (int chunk = 0; chunk * flash_keys < end; ++chunk) {
        const int stage = chunk & 1;
        if (chunk >= 2) {
            const int old = chunk - 2;
            const Word retired = Key ? tmem_smem(&tile.scores_ready[old % 3]) :
                                        tmem_smem(&tile.output_ready[stage]);
            tmem_copy_wait(retired, Key ? (old / 3) & 1 : (old >> 1) & 1);
        }
        const int first = chunk * flash_keys;
        const int page_head = (page_base + first / mxfp8_page_size) * kv_heads + kv_head;
        if constexpr (Key) {
            if (stage == 0 && invalid[page_head] < end) {
                atomicMin(&tile.bad_key, invalid[page_head]);
            }
        }
        const Word barrier = tmem_smem(&ready[stage]);
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(barrier), "n"(sizeof(destination[0])) : "memory");
        pair_copy_decoded(map, destination[stage], barrier, first % mxfp8_page_size,
                          page_head * staged_planes);
    }
}

__device__ void single_issue_score(StagedSingleTile& tile, int chunk) {
    const int k_slot = chunk & 1;
    const int score_slot = chunk % 3;
    tmem_copy_wait(tmem_smem(&tile.key_ready[k_slot]), (chunk >> 1) & 1);
    tmem_query_key<false, true>(tile, tmem_smem(&tile.scores_ready[score_slot]),
                               tmem_scores + score_slot * flash_keys, k_slot);
}

template <class Tile>
__device__ void issue_staged_value(Tile& tile, int chunk, Word probability) {
    const int stage = chunk & 1;
    #pragma unroll
    for (int token = 0; token < flash_keys; token += 16) {
        constexpr Word format = (1u << 4) | (1u << 7) | (1u << 10);
        constexpr Word shape = ((flash_queries / 16) << 24) |
                               ((channels / 8) << 17) | (1u << 16);
        const Word p = tile.tmem + probability + token / 2;
        const auto v = tmem_descriptor<true>(&tile.value[stage][0][token][0]);
        asm volatile("{ .reg .pred accum;\n"
                     "setp.ne.u32 accum, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, "
                     "{%5,%5,%5,%5}, accum; }"
                     :: "r"(tile.tmem), "r"(p), "l"(v), "r"(shape | format),
                        "r"(static_cast<int>(chunk != 0 || token != 0)), "r"(0) : "memory");
    }
    tmem_commit(tmem_smem(&tile.output_ready[stage]));
}

__device__ void single_issue_mmas(StagedSingleTile& tile, int end) {
    single_issue_score(tile, 0);
    if (end > flash_keys) single_issue_score(tile, 1);
    for (int chunk = 0; chunk * flash_keys < end; ++chunk) {
        if ((chunk + 2) * flash_keys < end) single_issue_score(tile, chunk + 2);
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        tmem_copy_wait(tmem_smem(&tile.value_ready[stage]), phase);
        tmem_mma_wait(tmem_smem(&tile.probability_ready[stage]), phase);
        issue_staged_value(tile, chunk, staged_single_probability + stage * (flash_keys / 2));
    }
}

template <int Count>
__global__ __launch_bounds__(256, 1) void tmem_staged_single_attention(
    const __grid_constant__ StagedRequests<Count> requests,
    const __grid_constant__ StagedPairMaps maps, const int* invalid) {
    extern __shared__ StagedSingleTile single_shared[];
    auto& tile = single_shared[0];
    const Attention request = requests.items[requests.lane()];
    const int row = blockIdx.x * flash_rows;
    const int head = blockIdx.y * ordered_heads;
    const int position = request.position + row;
    constexpr int active_rows = flash_rows;
    const int end = position + active_rows;
    const int page_base = requests.lane() * ((request.position + flash_chunk) / mxfp8_page_size);
    const int warp = threadIdx.x / 32;
    if (threadIdx.x == 0) {
        tile.bad_key = INT_MAX;
        for (int stage = 0; stage < 2; ++stage) {
            pipeline_init(tile.key_ready[stage], 1);
            pipeline_init(tile.value_ready[stage], 1);
            pipeline_init(tile.output_ready[stage], 1);
            pipeline_init(tile.probability_ready[stage], flash_queries / 32);
        }
        for (auto& barrier : tile.scores_ready) pipeline_init(barrier, 1);
    }
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                     "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                     :: "r"(tmem_smem(&tile.tmem)), "r"(tmem_cols) : "memory");
    }
    tmem_stage_query<flash_queries, 8>(tile, request, row, head, active_rows);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x < flash_queries) {
        const Word row_address = tile.tmem + ((warp * 32) << 16);
        FlashSoftmax state{-CUDART_INF_F, 0.0f};
        for (int chunk = 0; chunk * flash_keys < end; ++chunk) {
            const int stage = chunk & 1;
            const int score_slot = chunk % 3;
            const Word phase = (chunk >> 1) & 1;
            tmem_mma_wait(tmem_smem(&tile.scores_ready[score_slot]), (chunk / 3) & 1);
            if (chunk >= 2) tmem_mma_wait(tmem_smem(&tile.output_ready[stage]), phase ^ 1);
            const int first = chunk * flash_keys;
            const int visible = position + threadIdx.x / ordered_heads - first + 1;
            const Word score = tmem_scores + score_slot * flash_keys;
            const Word p_column = staged_single_probability + stage * (flash_keys / 2);
            const float history = first + flash_keys <= position + 1 ?
                tmem_probabilities<true, true, true>(tile.tmem, row_address, threadIdx.x,
                    visible, state, score, p_column) :
                tmem_probabilities<false, true, true>(tile.tmem, row_address, threadIdx.x,
                    visible, state, score, p_column);
            if (chunk != 0) {
                tmem_mma_wait(tmem_smem(&tile.output_ready[1 - stage]), ((chunk - 1) >> 1) & 1);
                tmem_correct_output(row_address, history);
            }
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            __syncwarp();
            if ((threadIdx.x & 31) == 0) {
                asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                             :: "r"(tmem_smem(&tile.probability_ready[stage])) : "memory");
            }
        }
        const int final_chunk = (end - 1) / flash_keys;
        tmem_mma_wait(tmem_smem(&tile.output_ready[final_chunk & 1]), (final_chunk >> 1) & 1);
        tmem_store_output<true>(row_address, state, request, row, head, active_rows);
    } else if (threadIdx.x == 128) {
        single_stage<true>(tile, maps, invalid, page_base, head / ratio,
                           request.cache.kv_heads, end);
    } else if (threadIdx.x == 160) {
        single_stage<false>(tile, maps, invalid, page_base, head / ratio,
                            request.cache.kv_heads, end);
    } else if (threadIdx.x == 192) {
        single_issue_mmas(tile, end);
    }
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tile.tmem), "r"(tmem_cols) : "memory");
    }
    tmem_repair_ordered(tile, request, row, head, active_rows);
}

#include "sglang/text/mxfp8_gqa_prefill_page.cuh"

std::optional<cudaError_t> encode_staged_maps(int full_pages, int kv_heads,
                                           std::uint16_t* keys, std::uint16_t* values,
                                           StagedPairMaps& maps,
                                           bool single = false, bool page = false,
                                           bool strict = false) {
    PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
    if (const auto status = staged_driver_entry("cuTensorMapEncodeTiled", 12000, encode);
        !status || *status != cudaSuccess) {
        return status;
    }
    constexpr cuuint64_t width_bytes = tmem_swizzle * sizeof(std::uint16_t);
    constexpr cuuint64_t strides[] = {width_bytes, width_bytes * mxfp8_page_size};
    const cuuint64_t shape[] = {tmem_swizzle, mxfp8_page_size,
        static_cast<cuuint64_t>(full_pages) * kv_heads * staged_planes};
    constexpr cuuint32_t elements[] = {1, 1, 1};
    const cuuint32_t key_rows = page ? mxfp8_page_size : (single ? flash_keys : pair_key_rows);
    const cuuint32_t key_box[] = {tmem_swizzle, key_rows, staged_planes};
    const cuuint32_t value_box[] = {
        tmem_swizzle, flash_keys,
        static_cast<cuuint32_t>(single ? staged_planes : staged_planes / 2)};
    const auto encode_map = [&](CUtensorMap& map, void* source, const cuuint32_t* box) {
        return encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 3, source, shape, strides,
                      box, elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    auto result = encode_map(maps.keys, keys, key_box);
    if (result == CUDA_SUCCESS) {
        result = encode_map(maps.values, values, value_box);
    }
    switch (result) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE:
        if (strict) return cudaErrorInvalidValue;
        return std::nullopt;
    case CUDA_ERROR_NOT_SUPPORTED: return std::nullopt;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_DEINITIALIZED: return cudaErrorCudartUnloading;
    default: return cudaErrorUnknown;
    }
}

std::optional<cudaError_t> validate_paired_request(const Attention& request,
                                                  Mxfp8PageTable pages, int rows = flash_chunk) {
    const auto& cache = request.cache;
    const int position = request.position;
    const int required_pages = (position + rows + mxfp8_page_size - 1) / mxfp8_page_size;
    if (pages.page_count < required_pages || cache.page_count < required_pages ||
        !pages.device_pages || !cache.key_payload || !cache.value_payload ||
        !cache.key_scales || !cache.value_scales || !request.queries || !request.output) {
        return cudaErrorInvalidValue;
    }
    constexpr auto tma_alignment = 16;
    if (cache.page_count > INT_MAX / mxfp8_page_size ||
        reinterpret_cast<std::uintptr_t>(cache.key_payload) % tma_alignment != 0 ||
        reinterpret_cast<std::uintptr_t>(cache.value_payload) % tma_alignment != 0 ||
        reinterpret_cast<std::uintptr_t>(cache.key_scales) % tma_alignment != 0 ||
        reinterpret_cast<std::uintptr_t>(cache.value_scales) % tma_alignment != 0) {
        return std::nullopt;
    }
    return cudaSuccess;
}

cudaLaunchConfig_t paired_config(std::size_t shared, int head_pairs, cudaStream_t stream) {
    return {dim3(flash_chunk / flash_rows, head_pairs),
            dim3(pair_threads), shared, stream, nullptr, 0};
}

std::optional<cudaError_t> prepare_paired_kernel(const void* kernel,
                                                 const cudaLaunchConfig_t& config) {
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return status;
    }
    if (properties.major != 10 || properties.minor != 0 || !properties.clusterLaunch) {
        return std::nullopt;
    }
    cudaFuncAttributes attributes{};
    status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) {
        return flash_setup_error(status, paired_kernel_unavailable(status));
    }
    if (properties.sharedMemPerBlockOptin < config.dynamicSmemBytes + attributes.sharedSizeBytes ||
        attributes.maxThreadsPerBlock < config.blockDim.x || attributes.requiredClusterWidth != 2 ||
        attributes.requiredClusterHeight != 1 || attributes.requiredClusterDepth != 1) {
        return std::nullopt;
    }
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  config.dynamicSmemBytes);
    if (status != cudaSuccess) {
        return flash_setup_error(
            status, paired_kernel_unavailable(status) || status == cudaErrorInvalidValue);
    }
    int clusters = 0;
    status = cudaOccupancyMaxActiveClusters(&clusters, kernel, &config);
    if (status != cudaSuccess) {
        return flash_setup_error(status, paired_kernel_unavailable(status) ||
            status == cudaErrorInvalidClusterSize || status == cudaErrorInvalidConfiguration ||
            status == cudaErrorInvalidValue);
    }
    return clusters > 0 ? std::optional{cudaSuccess} : std::nullopt;
}

template <bool Strict = false>
std::optional<cudaError_t> prepare_staged_single_kernel(const void* kernel,
                                                       const cudaLaunchConfig_t& config) {
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) return status;
    if (properties.major != 10 || properties.minor != 0) return std::nullopt;
    const auto prepared = prepare_flash_kernel(
        kernel, config.dynamicSmemBytes, properties, !Strict);
    if (!prepared || *prepared != cudaSuccess) return prepared;
    int blocks = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, config.blockDim.x,
                                                          config.dynamicSmemBytes);
    if (status != cudaSuccess) {
        if constexpr (Strict) return status;
        return flash_setup_error(status, paired_kernel_unavailable(status) ||
            status == cudaErrorInvalidConfiguration || status == cudaErrorInvalidValue);
    }
    return blocks > 0 ? std::optional{cudaSuccess} : std::nullopt;
}

}

std::optional<cudaError_t> launch_paired_tmem_prefill(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
    const Bf16* queries, Bf16* output, cudaStream_t stream) {
    if (position < 0 || position > INT_MAX - flash_chunk - mxfp8_page_size) {
        return std::nullopt;
    }
    Attention request{cache, pages.device_pages, position, queries, output};
    if (const auto status = validate_paired_request(request, pages);
        !status || *status != cudaSuccess) {
        return status;
    }
    const auto kernel = position % mxfp8_page_size ? tmem_pair_attention<true> :
                                                     tmem_pair_attention<false>;
    const auto config = paired_config(sizeof(PairTile), gqa_head_pairs(cache), stream);
    if (const auto status = prepare_paired_kernel(reinterpret_cast<const void*>(kernel), config);
        !status || *status != cudaSuccess) {
        return status;
    }
    PairMaps maps{};
    const auto encoded = encode_pair_maps(cache, maps);
    if (!encoded || *encoded != cudaSuccess) {
        return encoded;
    }
    void* args[] = {&request, &maps};
    return cudaLaunchKernel(kernel, config.gridDim, config.blockDim, args,
                            config.dynamicSmemBytes, stream);
}

namespace {

constexpr int staged_causal_group_rows = 4 * flash_chunk;
constexpr int staged_causal_early_context = 32768;

template <int Count>
std::optional<cudaError_t> prepare_long_page(const StagedRequests<Count>& requests,
                                            std::uint16_t* keys, std::uint16_t* values,
                                            int total_pages, StagedPairMaps& maps,
                                            cudaLaunchConfig_t& config) {
    for (const auto& request : requests.items) {
        if (reinterpret_cast<std::uintptr_t>(request.output) % alignof(uint4) != 0) {
            return std::nullopt;
        }
    }
    const auto kernel = reinterpret_cast<const void*>(tmem_staged_page_attention<Count>);
    auto local = config;
    local.blockDim.x = staged_compact_threads;
    local.dynamicSmemBytes = sizeof(StagedPageTile);
    const auto prepared = prepare_staged_single_kernel<true>(kernel, local);
    if (const auto status = consume_prefill_submission(prepared.value_or(cudaSuccess));
        status != cudaSuccess) return status;
    if (!prepared) return std::nullopt;
    const auto encoded = encode_staged_maps(total_pages, requests.items[0].cache.kv_heads,
                                            keys, values, maps, true, true, true);
    if (const auto status = consume_prefill_submission(encoded.value_or(cudaSuccess));
        status != cudaSuccess) return status;
    if (encoded) config = local;
    return encoded;
}

template <int Count, bool Partial, bool CausalGroup, bool LongPage>
std::optional<cudaError_t> submit_staged_requests(StagedRequests<Count>& requests,
                                                  std::uint16_t* keys, std::uint16_t* values,
                                                  int* invalid, int full_pages, int total_pages,
                                                  cudaStream_t stream) {
    const int position = requests.items[0].position;
    auto paired = tmem_staged_pair_attention<false, Count>;
    if constexpr (Count == 1) {
        if (position % mxfp8_page_size) {
            paired = tmem_staged_pair_attention<true, Count>;
        }
    }
    auto config = paired_config(sizeof(StagedPairTile),
                                gqa_head_pairs(requests.items[0].cache), stream);
    if constexpr (Count > 1) {
        config.gridDim.z = Count;
        config.blockDim.x = staged_compact_threads;
    }
    if constexpr (CausalGroup) config.gridDim.x = requests.rows[0] / flash_rows;
    StagedPairMaps maps{};
    const void* kernel = reinterpret_cast<const void*>(paired);
    bool single = false;
    if constexpr ((Count == 4 || CausalGroup) && !LongPage) {
        if (Partial || position % flash_chunk == 0) {
            const auto local = Partial
                ? reinterpret_cast<const void*>(tmem_staged_page_attention<Count>)
                : reinterpret_cast<const void*>(tmem_staged_single_attention<Count>);
            auto local_config = config;
            local_config.dynamicSmemBytes = sizeof(StagedSingleTile);
            if constexpr (CausalGroup) {
                local_config.blockDim.x = staged_compact_threads;
            }
            if constexpr (Partial) {
                int rows = 0;
                for (const int active : requests.rows) rows = std::max(rows, active);
                local_config.gridDim.x = (rows + flash_rows - 1) / flash_rows;
            }
            const auto prepared = prepare_staged_single_kernel(local, local_config);
            if (prepared && *prepared != cudaSuccess) return prepared;
            if (prepared) {
                const auto encoded = encode_staged_maps(total_pages,
                                                        requests.items[0].cache.kv_heads,
                                                        keys, values, maps, true, Partial);
                if (encoded && *encoded != cudaSuccess && *encoded != cudaErrorMemoryAllocation) {
                    return encoded;
                }
                if (encoded && *encoded == cudaSuccess) {
                    kernel = local;
                    config = local_config;
                    single = true;
                }
            }
        }
    }
    if constexpr (LongPage) {
        const auto prepared = prepare_long_page(requests, keys, values, total_pages, maps, config);
        if (prepared && *prepared != cudaSuccess) return prepared;
        if (prepared) {
            kernel = reinterpret_cast<const void*>(tmem_staged_page_attention<Count>);
            single = true;
        }
    }
    if constexpr (Partial || CausalGroup) {
        if (!single) return std::nullopt;
    }
    if (!single) {
        if (const auto status = prepare_paired_kernel(kernel, config);
            !status || *status != cudaSuccess) {
            return status;
        }
        const auto encoded = encode_staged_maps(full_pages * Count,
                                                requests.items[0].cache.kv_heads,
                                                keys, values, maps);
        if (!encoded || *encoded == cudaErrorMemoryAllocation) return std::nullopt;
        if (*encoded != cudaSuccess) return encoded;
    }
    void* decode_args[] = {&requests, &keys, &values, &invalid};
    const auto decoded = cudaLaunchKernel(pair_predecode<Count, Partial>,
        dim3(full_pages, requests.items[0].cache.kv_heads, Count),
            channels, decode_args, 0, stream);
    if constexpr (LongPage) {
        if (single) {
            if (const auto status = consume_prefill_submission(decoded); status != cudaSuccess) {
                return status;
            }
            void* args[] = {&requests, &maps, &invalid};
            return consume_prefill_submission(cudaLaunchKernel(kernel, config.gridDim,
                config.blockDim, args, config.dynamicSmemBytes, stream));
        }
    }
    if (decoded != cudaSuccess) {
        return staged_submission(decoded);
    }
    void* args[] = {&requests, &maps, &invalid};
    return staged_submission(cudaLaunchKernel(kernel, config.gridDim, config.blockDim, args,
                                              config.dynamicSmemBytes, stream));
}

template <int Count, bool Partial = false, bool CausalGroup = false, bool LongPage = false>
std::optional<cudaError_t> launch_staged_requests(const decoder::PagedPrefillSequence* sequences,
                                                 cudaStream_t stream,
                                                 decoder::PairedPrefillScratch scratch) {
    const int position = sequences[0].position;
    constexpr int min_partial_rows = 256;
    if (!StagedRange(scratch.keys, scratch.key_bytes, 1).valid() ||
        !StagedRange(scratch.values, scratch.value_bytes, 1).valid()) return std::nullopt;
    if constexpr (!Partial && !CausalGroup) {
        if (position < flash_chunk || position > 131072 / Count - flash_chunk) {
            return std::nullopt;
        }
    }
    StagedRequests<Count> requests{};
    int full_pages = 0;
    int total_pages = 0;
    for (int lane = 0; lane < Count; ++lane) {
        const auto& sequence = sequences[lane];
        const int rows = sequence.rows;
        if constexpr (Partial) {
            if (rows < min_partial_rows || rows > flash_chunk || sequence.position < 0 ||
                sequence.position > 32768 - rows) return std::nullopt;
        } else if constexpr (CausalGroup) {
            constexpr int context = LongPage
                ? decoder::staged_prefill_batch_long_max_position + flash_chunk
                : staged_causal_early_context;
            if (rows != staged_causal_group_rows || position < 0 || position > context - rows ||
                position % flash_chunk != 0) return std::nullopt;
        } else if (rows != flash_chunk || sequence.position != position) {
            return std::nullopt;
        }
        const int visible = sequence.position + rows;
        const int pages = (visible + mxfp8_page_size - 1) / mxfp8_page_size;
        requests.rows[lane] = rows;
        requests.page_base[lane] = total_pages;
        total_pages += pages;
        full_pages = std::max(full_pages, pages);
        auto& request = requests.items[lane];
        request = {sequence.cache, sequence.pages.device_pages, position, sequence.queries,
                   sequence.output};
        request.position = sequence.position;
        if (const auto status = validate_paired_request(request, sequence.pages, rows);
            !status || *status != cudaSuccess) return status;
    }
    if (const auto pending = cudaPeekAtLastError(); pending != cudaSuccess) {
        return pending;
    }
    const auto page_heads = static_cast<std::size_t>(total_pages) * sequences[0].cache.kv_heads;
    const auto elements = page_heads * staged_head_elements;
    const auto decoded_bytes = elements * sizeof(std::uint16_t);
    const auto value_bytes = decoded_bytes + page_heads * sizeof(int);
    static_assert(staged_alignment % alignof(int) == 0);
    void* key_storage = scratch.keys;
    void* value_storage = scratch.values;
    auto* keys = static_cast<std::uint16_t*>(
        std::align(staged_alignment, decoded_bytes, key_storage, scratch.key_bytes));
    auto* values = static_cast<std::uint16_t*>(
        std::align(staged_alignment, value_bytes, value_storage, scratch.value_bytes));
    if (!keys || !values) {
        return std::nullopt;
    }
    cudaStreamCaptureStatus capture{};
    const auto capture_status = cudaStreamIsCapturing(stream, &capture);
    if (capture_status != cudaSuccess) {
        return capture_status;
    }
    if (capture != cudaStreamCaptureStatusNone) {
        return std::nullopt;
    }
    if constexpr (Count > 1) {
        if (const auto status = staged_stream_context(stream); !status || *status != cudaSuccess) {
            return status;
        }
    }
    if (!valid_staged_ranges<Count>(sequences, {keys, decoded_bytes, staged_alignment},
                                    {values, value_bytes, staged_alignment})) {
        return std::nullopt;
    }
    return submit_staged_requests<Count, Partial, CausalGroup, LongPage>(requests, keys, values,
        reinterpret_cast<int*>(values + elements), full_pages, total_pages, stream);
}

}

std::optional<cudaError_t> launch_staged_tmem_prefill(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
    const Bf16* queries, Bf16* output, cudaStream_t stream,
    decoder::PairedPrefillScratch scratch) {
    if (position > 31744) {
        return std::nullopt;
    }
    const decoder::PagedPrefillSequence sequence{cache, pages, position, flash_chunk, queries,
                                                output};
    return launch_staged_requests<1>(&sequence, stream, scratch);
}

std::optional<cudaError_t> launch_causal_group_staged_tmem_prefill(
    const decoder::PagedPrefillSequence& sequence, cudaStream_t stream,
    decoder::PairedPrefillScratch scratch) {
    const auto status = sequence.position <= staged_causal_early_context - staged_causal_group_rows
        ? launch_staged_requests<1, false, true>(&sequence, stream, scratch)
        : launch_staged_requests<1, false, true, true>(&sequence, stream, scratch);
    if (status && *status != cudaSuccess) return consume_prefill_submission(*status);
    return status;
}

std::optional<cudaError_t> decoder::try_batched_staged_paged_prefill(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream,
    PairedPrefillScratch scratch) {
    if (!sequences || count != staged_prefill_batch_size) {
        return std::nullopt;
    }
    const int position = sequences[0].position;
    if (position < staged_prefill_batch_min_position ||
        position > staged_prefill_batch_max_position || position % flash_chunk != 0) {
        return std::nullopt;
    }
    return launch_staged_requests<staged_prefill_batch_size>(sequences, stream, scratch);
}

std::optional<cudaError_t> decoder::try_long_staged_paged_prefill(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream,
    PairedPrefillScratch scratch) {
    if (!sequences || !(count == 1 || count == 2 || count == staged_prefill_batch_size)) {
        return std::nullopt;
    }
    const int position = sequences[0].position;
    if (position <= staged_prefill_batch_max_position ||
        position > staged_prefill_batch_long_max_position || position % flash_chunk != 0) {
        return std::nullopt;
    }
    if (count == staged_prefill_batch_size) {
        return launch_staged_requests<staged_prefill_batch_size>(sequences, stream, scratch);
    }
    if (count == 2) {
        return launch_staged_requests<2, false, false, true>(sequences, stream, scratch);
    }
    return launch_staged_requests<1, false, false, true>(sequences, stream, scratch);
}

std::optional<cudaError_t> launch_partial_staged_tmem_prefill(
    const decoder::PagedPrefillSequence* sequences, int count, cudaStream_t stream,
    decoder::PairedPrefillScratch scratch) {
    if (!sequences || count != decoder::staged_prefill_batch_size) return std::nullopt;
    return launch_staged_requests<decoder::staged_prefill_batch_size, true>(
        sequences, stream, scratch);
}

}
