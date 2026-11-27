#include "sglang/text/gqa/tmem_ops.cuh"

#include <climits>
#include <cute/arch/mma_sm100_umma.hpp>

namespace sglang::gqa_prefill::tmem {

template <int Count>
__device__ int StagedRequests<Count>::lane() const {
    if constexpr (Count == 1) {
        return 0;
    }
    return blockIdx.z;
}

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

template <int Count, bool Partial>
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

__device__ void stage_page_keys(StagedPageTile& tile, const StagedPairMaps& maps,
                               const int* invalid, int page_base, int kv_head, int pages,
                               int kv_heads) {
    for (int page = 0; page < pages; ++page) {
        if (page) {
            const int previous = page - 1;
            tmem_copy_wait(tmem_smem(&tile.scores_ready[previous & 1]), (previous >> 1) & 1);
        }
        const int page_head = (page_base + page) * kv_heads + kv_head;
        const int bad = invalid[page_head];
        if (bad != INT_MAX) atomicMin(&tile.bad_key, bad);
        const Word barrier = tmem_smem(&tile.key_ready[page & 1]);
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(barrier), "n"(sizeof(tile.key)) : "memory");
        pair_copy_decoded(&maps.keys, tile.key, barrier, 0, page_head * staged_planes);
    }
}

template <int N, cute::UMMA::Major B>
__device__ __forceinline__ std::uint64_t page_instruction() {
    return cute::UMMA::make_runtime_instr_desc<cute::bfloat16_t, cute::bfloat16_t, float,
        flash_queries, N, cute::UMMA::Major::K, B>();
}

__device__ void issue_page_scores(StagedPageTile& tile, int page) {
    using Atom = cute::SM100_MMA_F16BF16_SS<cute::bfloat16_t, cute::bfloat16_t, float,
        flash_queries, staged_page_keys, cute::UMMA::Major::K, cute::UMMA::Major::K>;
    const int slot = page & 1;
    const Word phase = (page >> 1) & 1;
    tmem_copy_wait(tmem_smem(&tile.key_ready[slot]), phase);
    const Word score = tile.tmem + tmem_scores + slot * staged_page_keys;
    #pragma unroll
    for (int dimension = 0; dimension < channels; dimension += 16) {
        const int plane = dimension / tmem_swizzle;
        const int offset = dimension % tmem_swizzle;
        const auto query = tmem_descriptor(&tile.query[plane][0][offset]);
        const auto key = tmem_descriptor(&tile.key[plane][0][offset]);
        Atom::fma(query, key, score, dimension != 0,
                  page_instruction<staged_page_keys, cute::UMMA::Major::K>());
    }
    if (cute::elect_one_sync()) tmem_commit(tmem_smem(&tile.scores_ready[slot]));
}

__device__ Word staged_page_score(int chunk) {
    return tmem_scores + ((chunk >> 1) & 1) * staged_page_keys + (chunk & 1) * flash_keys;
}

__device__ void issue_page_values(StagedPageTile& tile, int chunk) {
    using Atom = cute::SM100_MMA_F16BF16_TS<cute::bfloat16_t, cute::bfloat16_t, float,
        flash_queries, channels, cute::UMMA::Major::K, cute::UMMA::Major::MN>;
    const Word probability = staged_page_score(chunk);
    const int stage = chunk & 1;
    #pragma unroll
    for (int token = 0; token < flash_keys; token += 16) {
        const auto value = tmem_descriptor<true>(&tile.value[stage][0][token][0]);
        const Word p = tile.tmem + probability + token / 2;
        Atom::fma(p, value, tile.tmem, chunk != 0 || token != 0,
                  page_instruction<channels, cute::UMMA::Major::MN>());
    }
    if (cute::elect_one_sync()) tmem_commit(tmem_smem(&tile.output_ready[stage]));
}

template <bool ScoreLookahead>
__device__ void issue_page_mmas(StagedPageTile& tile, int chunks) {
    issue_page_scores(tile, 0);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        if constexpr (ScoreLookahead) {
            if (stage == 0 && chunk + 2 < chunks) issue_page_scores(tile, (chunk >> 1) + 1);
        }
        tmem_copy_wait(tmem_smem(&tile.value_ready[stage]), phase);
        tmem_mma_wait(tmem_smem(&tile.probability_ready[stage]), phase);
        issue_page_values(tile, chunk);
        if constexpr (!ScoreLookahead) {
            if (stage == 0 && chunk + 2 < chunks) issue_page_scores(tile, (chunk >> 1) + 1);
        }
    }
}

__device__ void staged_page_softmax(StagedPageTile& tile, const Attention& request, int row,
                                    int head, int position, int active_rows, int chunks) {
    const int warp = threadIdx.x / 32;
    const Word row_address = tile.tmem + ((warp * 32) << 16);
    FlashSoftmax state{-CUDART_INF_F, 0.0f};
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        if (stage == 0) {
            tmem_mma_wait(tmem_smem(&tile.scores_ready[phase]), (chunk >> 2) & 1);
        }
        const int first = chunk * flash_keys;
        const int visible = position + threadIdx.x / ordered_heads - first + 1;
        const Word score = staged_page_score(chunk);
        const float history = first + flash_keys <= position + 1 ?
            tmem_probabilities<true, true, true>(tile.tmem, row_address, threadIdx.x,
                visible, state, score, score) :
            tmem_probabilities<false, true, true>(tile.tmem, row_address, threadIdx.x,
                visible, state, score, score);
        if (chunk) {
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
    const int final = chunks - 1;
    tmem_mma_wait(tmem_smem(&tile.output_ready[final & 1]), (final >> 1) & 1);
    tmem_store_output<true>(row_address, state, request, row, head, active_rows);
}

template <int Count, bool ScoreLookahead>
__global__ __launch_bounds__(256, 1) void tmem_staged_page_attention(
    const __grid_constant__ StagedRequests<Count> requests,
    const __grid_constant__ StagedPairMaps maps, const int* invalid) {
    extern __shared__ StagedPageTile page_shared[];
    auto& tile = page_shared[0];
    const int lane = requests.lane();
    const Attention request = requests.items[lane];
    const int row = blockIdx.x * flash_rows;
    const int head = blockIdx.y * ordered_heads;
    const int position = request.position + row;
    const int active_rows = min(flash_rows, requests.rows[lane] - row);
    if (active_rows <= 0) return;
    const int end = position + active_rows;
    const int chunks = (end + flash_keys - 1) / flash_keys;
    const int page_base = requests.page_base[lane];
    const int warp = threadIdx.x / 32;
    if (threadIdx.x == 0) {
        tile.bad_key = INT_MAX;
        for (int stage = 0; stage < 2; ++stage) {
            pipeline_init(tile.key_ready[stage], 1);
            pipeline_init(tile.value_ready[stage], 1);
            pipeline_init(tile.scores_ready[stage], 1);
            pipeline_init(tile.output_ready[stage], 1);
            pipeline_init(tile.probability_ready[stage], flash_queries / 32);
        }
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
        staged_page_softmax(tile, request, row, head, position, active_rows, chunks);
    } else if (threadIdx.x == 128) {
        stage_page_keys(tile, maps, invalid, page_base, head / ratio, (chunks + 1) / 2,
                        request.cache.kv_heads);
    } else if (threadIdx.x == 160) {
        single_stage<false>(tile, maps, invalid, page_base, head / ratio,
                            request.cache.kv_heads, end);
    } else if (warp == 6) {
        issue_page_mmas<ScoreLookahead>(tile, chunks);
    }
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tile.tmem), "r"(tmem_cols) : "memory");
    }
    tmem_repair_ordered(tile, request, row, head, active_rows);
}

template __global__ void pair_predecode<1, false>(
    const __grid_constant__ StagedRequests<1>, std::uint16_t*, std::uint16_t*, int*);
template __global__ void pair_predecode<1, true>(
    const __grid_constant__ StagedRequests<1>, std::uint16_t*, std::uint16_t*, int*);
template __global__ void pair_predecode<2, false>(
    const __grid_constant__ StagedRequests<2>, std::uint16_t*, std::uint16_t*, int*);
template __global__ void pair_predecode<4, false>(
    const __grid_constant__ StagedRequests<4>, std::uint16_t*, std::uint16_t*, int*);
template __global__ void pair_predecode<4, true>(
    const __grid_constant__ StagedRequests<4>, std::uint16_t*, std::uint16_t*, int*);
template __global__ void tmem_staged_pair_attention<false, 1>(
    const __grid_constant__ StagedRequests<1>, const __grid_constant__ StagedPairMaps,
    const int*);
template __global__ void tmem_staged_pair_attention<true, 1>(
    const __grid_constant__ StagedRequests<1>, const __grid_constant__ StagedPairMaps,
    const int*);
template __global__ void tmem_staged_pair_attention<false, 2>(
    const __grid_constant__ StagedRequests<2>, const __grid_constant__ StagedPairMaps,
    const int*);
template __global__ void tmem_staged_pair_attention<false, 4>(
    const __grid_constant__ StagedRequests<4>, const __grid_constant__ StagedPairMaps,
    const int*);
template __global__ void tmem_staged_single_attention<1>(
    const __grid_constant__ StagedRequests<1>, const __grid_constant__ StagedPairMaps,
    const int*);
template __global__ void tmem_staged_single_attention<4>(
    const __grid_constant__ StagedRequests<4>, const __grid_constant__ StagedPairMaps,
    const int*);
template __global__ void tmem_staged_page_attention<1, false>(
    const __grid_constant__ StagedRequests<1>, const __grid_constant__ StagedPairMaps,
    const int*);
template __global__ void tmem_staged_page_attention<1, true>(
    const __grid_constant__ StagedRequests<1>, const __grid_constant__ StagedPairMaps,
    const int*);
template __global__ void tmem_staged_page_attention<2, false>(
    const __grid_constant__ StagedRequests<2>, const __grid_constant__ StagedPairMaps,
    const int*);
template __global__ void tmem_staged_page_attention<4, false>(
    const __grid_constant__ StagedRequests<4>, const __grid_constant__ StagedPairMaps,
    const int*);

}
