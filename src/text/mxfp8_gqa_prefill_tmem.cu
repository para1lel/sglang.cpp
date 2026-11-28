#include "sglang/text/gqa/tmem_ops.cuh"
#include "sglang/runtime/backend/cuda_resources.h"
#include <algorithm>
#include <climits>
#include <cudaTypedefs.h>

namespace sglang {

using namespace gqa_prefill;
using namespace gqa_prefill::tmem;

namespace {

__device__ void flash_tma_copy(const CUtensorMap* map, void* destination,
                              Word barrier, int channel, int row) {
    const auto shared = static_cast<Word>(__cvta_generic_to_shared(destination));
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3}], [%4];"
                 :: "r"(shared), "l"(map), "r"(channel), "r"(row), "r"(barrier) : "memory");
}

template <int Issuer = 0>
__device__ void flash_tma_prefetch(FlashPayload& raw, Attention request, int first,
                                   int kv_head, const CUtensorMap* keys, const CUtensorMap* values,
                                   Word barrier) {
    if (threadIdx.x == Issuer) {
        const int logical_page = first / mxfp8_page_size;
        const int row = request.pages[logical_page] * mxfp8_page_size + first % mxfp8_page_size;
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(barrier), "r"(static_cast<Word>(sizeof(FlashPayload))) : "memory");
        flash_tma_copy(keys, raw.key, barrier, kv_head * channels, row);
        flash_tma_copy(values, raw.value, barrier, kv_head * channels, row);
    }
}

template <bool Key, bool Partial = false, typename Decoded>
__device__ void tmem_stage_kv(Decoded& decoded, int& bad_key, const FlashPayload& payload,
                             Attention request, int first, int kv_head, int count) {
    constexpr int vector_elements = 16;
    constexpr int token_lanes = channels / vector_elements;
    const int producer = threadIdx.x - flash_queries;
    const int vector = producer % token_lanes;
    const int channel = vector * vector_elements;
    const int page = request.pages[first / mxfp8_page_size];
    const auto* scales = Key ? request.cache.key_scales : request.cache.value_scales;
    const auto* bytes = Key ? request.cache.key_payload : request.cache.value_payload;
    const auto& raw = Key ? payload.key : payload.value;
    for (int index = producer / token_lanes; index < flash_keys;
         index += tmem_producers / token_lanes) {
        const int sub = channel / tmem_swizzle;
        const int column = channel % tmem_swizzle;
        const int low = tmem_swizzled(index, column);
        const int high = tmem_swizzled(index, column + 8);
        const int token = (first + index) % mxfp8_page_size;
        const auto scale = scale_offset(request.cache.kv_heads, page, token, kv_head)
            + channel / mxfp8_group_size;
        bool bad = false;
        uint4 packed[2]{};
        if (!Partial || index < count) {
            const auto data = Partial ?
                *reinterpret_cast<const uint4*>(
                    bytes + payload_offset(request.cache.kv_heads, page, token, kv_head)
                    + channel) : raw[index][vector];
            tmem_decode(reinterpret_cast<std::uint16_t*>(packed), data, scales[scale], bad);
        }
        *reinterpret_cast<uint4*>(&decoded[sub][index][low]) = packed[0];
        *reinterpret_cast<uint4*>(&decoded[sub][index][high]) = packed[1];
        const auto invalid = __ballot_sync(warp_mask, bad);
        if (channel == 0 && ((invalid >> (producer & 16)) & 0xffffu)) {
            atomicMin(&bad_key, first + index);
        }
    }
}

__device__ __forceinline__ void tmem_softmax_tile(TmemFlashTile& tile, Word row_address, Word mma,
                                                  int position, int first, FlashSoftmax& state) {
    tmem_mma_wait(mma, 0);
    const float history =
        tmem_softmax_chunk<true>(tile.probability, row_address, position, first, state);
    if (first != 0) {
        tmem_correct_output(row_address, history);
    }
    asm volatile("tcgen05.fence::before_thread_sync;\n"
                 "fence.proxy.async.shared::cta;" ::: "memory");
}

template <bool Key, bool Ragged>
__device__ __forceinline__ void tmem_stage_tile(
    TmemFlashTile& tile, Attention request, int first, int end,
    int kv_head, int buffer, Word copy) {
    auto& decoded = Key ? tile.key : tile.value;
    if constexpr (Ragged) {
        if (end - first < flash_keys) {
            tmem_stage_kv<Key, true>(decoded, tile.bad_key, tile.raw[buffer], request, first,
                                     kv_head, end - first);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            return;
        }
    }
    if constexpr (Key) {
        tmem_copy_wait(copy, buffer);
    }
    tmem_stage_kv<Key>(decoded, tile.bad_key, tile.raw[buffer], request, first, kv_head,
                       flash_keys);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

template <int Barrier, int Count, int Leader>
__device__ __forceinline__ void pipeline_publish(unsigned long long& ready) {
    asm volatile("fence.proxy.async.shared::cta;\n"
                 "bar.sync %0, %1;" :: "n"(Barrier), "n"(Count) : "memory");
    if (threadIdx.x == Leader) {
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                     :: "r"(tmem_smem(&ready)) : "memory");
    }
}

__device__ void pipeline_issue(TmemPipelineTile& tile, int chunks) {
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        if (chunk + 1 < chunks) {
            tmem_copy_wait(tmem_smem(&tile.key_ready), stage ^ 1);
            tmem_query_key<false, false, pipeline_members>(
                tile, tmem_smem(&tile.scores_ready[1 - stage]),
                tmem_scores + (1 - stage) * flash_keys);
        }
        tmem_copy_wait(tmem_smem(&tile.operands_ready[stage]), phase);
        tmem_probability_value<false, pipeline_members>(
            tile.tmem, tile.probability[stage], tile.value[stage],
            tmem_smem(&tile.output_ready[stage]), chunk != 0);
    }
}

template <bool Key>
__device__ void pipeline_decode(TmemPipelineTile& tile, Attention request, int kv_head,
                                int chunk, int end, Word copy) {
    const int first = chunk * flash_keys;
    const int stage = chunk & 1;
    auto& decoded = Key ? tile.key : tile.value[stage];
    if (end - first < flash_keys) {
        tmem_stage_kv<Key, true>(decoded, tile.bad_key, tile.raw[stage], request, first,
                                kv_head, end - first);
        return;
    }
    if constexpr (Key) {
        tmem_copy_wait(copy, stage);
    }
    tmem_stage_kv<Key>(decoded, tile.bad_key, tile.raw[stage], request, first, kv_head, flash_keys);
}

__device__ void pipeline_produce(TmemPipelineTile& tile, Attention request, int kv_head, int end,
                                 const CUtensorMap* keys, const CUtensorMap* values) {
    const Word copy = tmem_smem(&tile.copy_ready);
    flash_tma_prefetch<flash_queries>(tile.raw[0], request, 0, kv_head, keys, values, copy);
    pipeline_decode<true>(tile, request, kv_head, 0, end, copy);
    pipeline_publish<1, tmem_producers, flash_queries>(tile.key_ready);
    if (2 * flash_keys <= end) {
        flash_tma_prefetch<flash_queries>(tile.raw[1], request, flash_keys, kv_head,
                                         keys, values, copy);
    }
    for (int first = 0, chunk = 0; first < end; first += flash_keys, ++chunk) {
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        if (chunk >= 2) {
            tmem_mma_wait(tmem_smem(&tile.output_ready[stage]), phase ^ 1);
        }
        pipeline_decode<false>(tile, request, kv_head, chunk, end, copy);
        pipeline_publish<1, tmem_producers, flash_queries>(tile.operands_ready[stage]);
        if (first + flash_keys < end) {
            tmem_mma_wait(tmem_smem(&tile.scores_ready[stage]), phase);
            pipeline_decode<true>(tile, request, kv_head, chunk + 1, end, copy);
            pipeline_publish<1, tmem_producers, flash_queries>(tile.key_ready);
            if (first + 3 * flash_keys <= end) {
                flash_tma_prefetch<flash_queries>(tile.raw[stage], request,
                    first + 2 * flash_keys, kv_head, keys, values, copy);
            }
        }
    }
}

__device__ void pipeline_softmax(TmemPipelineTile& tile, Attention request, int row, int head,
                                 int position, int end, int rows, int warp) {
    const Word address = tile.tmem + ((warp * 32) << 16);
    const int lane = threadIdx.x % 32;
    const int member = lane < 16 ? warp * 16 + lane : -1;
    FlashSoftmax state{-CUDART_INF_F, 0.0f};
    for (int first = 0, chunk = 0; first < end; first += flash_keys, ++chunk) {
        const int stage = chunk & 1;
        const Word phase = (chunk >> 1) & 1;
        tmem_mma_wait(tmem_smem(&tile.scores_ready[stage]), phase);
        if (chunk >= 2) {
            tmem_mma_wait(tmem_smem(&tile.output_ready[stage]), phase ^ 1);
        }
        const float history = tmem_softmax_chunk<true>(tile.probability[stage], address,
            position, first, state, tmem_scores + stage * flash_keys, member);
        if (chunk != 0) {
            tmem_mma_wait(tmem_smem(&tile.output_ready[1 - stage]), ((chunk - 1) >> 1) & 1);
            tmem_correct_output(address, history);
        }
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        pipeline_publish<2, flash_queries, 0>(tile.operands_ready[stage]);
    }
    const int last = (end - 1) / flash_keys;
    tmem_mma_wait(tmem_smem(&tile.output_ready[last & 1]), (last >> 1) & 1);
    tmem_store_output(address, state, request, row, head, rows, member);
}

__launch_bounds__(pipeline_threads, 1)
__global__ void tmem_pipeline_attention(Attention request,
                                        const __grid_constant__ CUtensorMap key_map,
                                        const __grid_constant__ CUtensorMap value_map, int rows) {
    extern __shared__ TmemPipelineTile shared_pipeline[];
    auto& tile = shared_pipeline[0];
    const int row = blockIdx.x * pipeline_rows;
    const int head = blockIdx.y * ordered_heads;
    const int position = request.position + row;
    const int active = min(pipeline_rows, rows - row);
    const int end = position + active;
    const int warp = threadIdx.x / 32;
    tmem_initialize(tile, warp, tmem_smem(&tile.copy_ready), tmem_smem(&tile.scores_ready[0]));
    if (threadIdx.x == 0) {
        pipeline_init(tile.key_ready, 1);
        pipeline_init(tile.scores_ready[1], 1);
        for (int stage = 0; stage < 2; ++stage) {
            pipeline_init(tile.output_ready[stage], 1);
            pipeline_init(tile.operands_ready[stage], 2);
        }
    }
    if (threadIdx.x < tmem_threads) {
        tmem_stage_query<pipeline_members>(tile, request, row, head, active);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (threadIdx.x < flash_queries) {
        pipeline_softmax(tile, request, row, head, position, end, active, warp);
    } else if (threadIdx.x < tmem_threads) {
        pipeline_produce(tile, request, head / ratio, end, &key_map, &value_map);
    } else if (threadIdx.x == tmem_threads) {
        tmem_copy_wait(tmem_smem(&tile.key_ready), 0);
        tmem_query_key<false, false, pipeline_members>(tile, tmem_smem(&tile.scores_ready[0]));
        pipeline_issue(tile, (end + flash_keys - 1) / flash_keys);
    }
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tile.tmem), "r"(tmem_cols) : "memory");
    }
    tmem_repair_ordered(tile, request, row, head, active);
}

cudaError_t launch_pipeline(Attention& request, CUtensorMap& keys, CUtensorMap& values,
                            int rows, cudaStream_t stream) {
    void* args[] = {&request, &keys, &values, &rows};
    const dim3 blocks((rows + pipeline_rows - 1) / pipeline_rows,
                      gqa_head_pairs(request.cache));
    return cudaLaunchKernel(tmem_pipeline_attention, blocks, pipeline_threads, args,
                            sizeof(TmemPipelineTile), stream);
}

template <bool Ragged>
__device__ __forceinline__ void tmem_flash_tile(TmemFlashTile& tile, Attention request,
                                                const CUtensorMap& key_map,
                                                const CUtensorMap& value_map, int rows, int row) {
    const int active_rows = Ragged ? min(flash_rows, rows - row) : flash_rows;
    const int head = blockIdx.y * ordered_heads;
    const int position = request.position + row;
    const int end = position + active_rows;
    const int kv_head = head / ratio;
    const int warp = threadIdx.x / 32;
    const bool softmax = warp < flash_queries / 32;
    const Word copy = tmem_smem(&tile.copy_ready);
    const Word mma = tmem_smem(&tile.mma_ready);
    tmem_initialize(tile, warp, copy, mma);
    tmem_stage_query(tile, request, row, head, active_rows);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    const Word row_address = tile.tmem + ((warp * 32) << 16);
    FlashSoftmax state{-CUDART_INF_F, 0.0f};
    if (!Ragged || end >= flash_keys) {
        flash_tma_prefetch(tile.raw[0], request, 0, kv_head, &key_map, &value_map, copy);
    }
    if (!softmax) {
        tmem_stage_tile<true, Ragged>(tile, request, 0, end, kv_head, 0, copy);
    }
    __syncthreads();
    for (int first = 0; first < end; first += flash_keys) {
        const int next = first + flash_keys;
        const int buffer = (first / flash_keys) % 2;
        if (next < end && (!Ragged || end - next >= flash_keys)) {
            flash_tma_prefetch(tile.raw[1 - buffer], request, next, kv_head,
                               &key_map, &value_map, copy);
        }
        if (threadIdx.x == 0) {
            tmem_query_key(tile, mma);
        }
        if (softmax) {
            tmem_softmax_tile(tile, row_address, mma, position, first, state);
        } else {
            tmem_stage_tile<false, Ragged>(tile, request, first, end, kv_head, buffer, copy);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            tmem_probability_value(tile.tmem, tile.probability, tile.value, mma, first != 0);
        }
        if (!softmax && next < end) {
            tmem_stage_tile<true, Ragged>(tile, request, next, end, kv_head, 1 - buffer, copy);
        }
        tmem_mma_wait(mma, 1);
        __syncthreads();
    }
    if (softmax) {
        tmem_store_output(row_address, state, request, row, head, active_rows);
    }
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tile.tmem), "r"(tmem_cols) : "memory");
    }
    tmem_repair_ordered(tile, request, row, head, active_rows);
}

template <bool Ordered, bool Ragged>
__launch_bounds__(tmem_threads, 1)
    __global__ void tmem_flash_attention(Attention request,
                                         const __grid_constant__ CUtensorMap key_map,
                                         const __grid_constant__ CUtensorMap value_map, int rows) {
    static_assert(Ordered);
    alignas(1024) extern __shared__ unsigned char shared_tmem[];
    tmem_flash_tile<Ragged>(*reinterpret_cast<TmemFlashTile*>(shared_tmem), request, key_map,
                            value_map, rows, blockIdx.x * flash_rows);
}

__device__ void pair_copy_payload(const CUtensorMap* map, void* destination, Word barrier,
                                  int channel, int token, int page) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(tmem_smem(destination)), "l"(map), "r"(channel), "r"(token),
                    "r"(page * pair_page_halves), "r"(barrier) : "memory");
}

template <int Issuer = 0>
__device__ void pair_prefetch(PairPayload& raw, Attention request, const PairMaps& maps,
                              int first, int kv_head, int rank, Word barrier) {
    if (threadIdx.x == Issuer) {
        const int page = request.pages[first / mxfp8_page_size];
        const int head = kv_head * channels;
        const int scales = page * request.cache.kv_heads + kv_head;
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(barrier), "r"(static_cast<Word>(sizeof(PairPayload))) : "memory");
        pair_copy_payload(&maps.keys, raw.key, barrier, head, rank * pair_key_rows, page);
        pair_copy_payload(&maps.values, raw.value, barrier, head + rank * pair_value_channels,
                          0, page);
        flash_tma_copy(&maps.key_scales, raw.key_scales, barrier, 0, scales);
        flash_tma_copy(&maps.value_scales, raw.value_scales, barrier, 0, scales);
    }
}

template <bool Key, bool PartialPage>
__device__ void pair_stage_kv(PairTile& tile, const PairPayload& payload, int first, int rank,
                             const Attention& request, int kv_head, int full_end, int end,
                             int stage = 0) {
    constexpr int vector_elements = 16;
    constexpr int width = Key ? channels : pair_value_channels;
    constexpr int row_count = Key ? pair_key_rows : flash_keys;
    constexpr int token_lanes = width / vector_elements;
    constexpr int vectors_per_subtile = tmem_swizzle / vector_elements;
    constexpr int paired_row_bits = vectors_per_subtile | 1;
    constexpr int value_swizzle_rows = pair_value_swizzle / sizeof(uint4);
    const int producer = threadIdx.x - flash_queries;
    const int vector_lane = producer % token_lanes;
    const int row = producer / token_lanes;
    const int first_row = Key ? row : row ^ ((vector_lane / vectors_per_subtile) * paired_row_bits);
    const int vector = Key ? vector_lane :
        vector_lane % vectors_per_subtile + (row & 1) * vectors_per_subtile;
    const int channel = vector * vector_elements;
    const int scale_channel = channel + (Key ? 0 : rank * pair_value_channels);
    const int half = (first / flash_keys) % pair_page_halves;
    const auto* scales = Key ? payload.key_scales : payload.value_scales;
    const bool tail = PartialPage && first >= full_end;
    const int page = tail ? request.pages[first / mxfp8_page_size] : 0;
    for (int index = first_row; index < row_count;
         index += tmem_producers / token_lanes) {
        const int sub = channel / tmem_swizzle;
        const int column = channel % tmem_swizzle;
        const int low = tmem_swizzled(index, column);
        const int high = tmem_swizzled(index, column + 8);
        const int logical = first + index + (Key ? rank * pair_key_rows : 0);
        const int token = logical % mxfp8_page_size;
        bool bad = false;
        uint4 packed[2];
        auto* decoded = reinterpret_cast<std::uint16_t*>(packed);
        if (tail) {
            if (logical < end) {
                pair_decode_tail<Key>(decoded, request.cache, page, token, kv_head,
                                      scale_channel, bad);
            } else {
                packed[0] = {};
                packed[1] = {};
            }
        } else {
            const int scale = (token % mxfp8_group_size) * (pair_scale_bytes / mxfp8_group_size)
                             + (token / mxfp8_group_size) * mxfp8_groups
                             + scale_channel / mxfp8_group_size;
            const int value_column = vector ^ (index % value_swizzle_rows);
            const uint4 data = Key ? payload.key[half][index][vector] :
                                     payload.value[half][index][value_column];
            tmem_decode(decoded, data, scales[scale], bad);
        }
        if constexpr (Key) {
            *reinterpret_cast<uint4*>(&tile.key[sub][index][low]) = packed[0];
            *reinterpret_cast<uint4*>(&tile.key[sub][index][high]) = packed[1];
            const Word active = __ballot_sync(warp_mask, bad);
            const int lane_base = (producer % 32) & ~(token_lanes - 1);
            if (vector == 0 && ((active >> lane_base) & ((1u << token_lanes) - 1))) {
                atomicMin(&tile.bad_key, logical);
            }
        } else {
            *reinterpret_cast<uint4*>(&tile.value[stage][sub][index][low]) = packed[0];
            *reinterpret_cast<uint4*>(&tile.value[stage][sub][index][high]) = packed[1];
            if (bad) {
                atomicMin(&tile.bad_key, logical);
            }
        }
    }
}

template <bool PartialPage>
__launch_bounds__(pair_threads, 1) __attribute__((cluster_dims(2, 1, 1)))
__global__ void tmem_pair_attention(Attention request, const __grid_constant__ PairMaps maps) {
    extern __shared__ PairTile shared_pair[];
    auto& tile = shared_pair[0];
    const auto pair = cooperative_groups::this_cluster();
    const int rank = pair.block_rank();
    const int row = blockIdx.x * flash_rows;
    const int head = blockIdx.y * ordered_heads;
    const int kv_head = head / ratio;
    const int position = request.position + row;
    const int end = position + (2 - rank) * flash_rows;
    const int full_end = PartialPage ? end - end % mxfp8_page_size : end;
    const int warp = threadIdx.x / 32;
    const bool softmax = warp < flash_queries / 32;
    const bool producer = threadIdx.x >= flash_queries && threadIdx.x < tmem_threads;
    const bool issuer = rank == 0 && threadIdx.x == tmem_threads;
    const Word copy = tmem_smem(&tile.copy_ready);
    const Word key = tmem_smem(&tile.key_ready);
    tmem_initialize<true>(tile, warp, copy, tmem_smem(&tile.scores_ready[0]));
    if (threadIdx.x < tmem_threads) {
        tmem_stage_query(tile, request, row, head);
    }
    __syncthreads();
    pair_prefetch(tile.raw[0], request, maps, 0, kv_head, rank, copy);
    if (producer) {
        tmem_copy_wait(copy, 0);
        pair_stage_kv<true, PartialPage>(tile, tile.raw[0], 0, rank, request, kv_head,
                                        full_end, end);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    pair.sync();
    if (mxfp8_page_size < full_end) {
        pair_prefetch(tile.raw[1], request, maps, mxfp8_page_size, kv_head, rank, copy);
    }
    if (issuer) {
        pair_issue_mmas(tile, end);
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
        tmem_store_output(row_address, state, request, row, head);
    }
    if (producer) {
        for (int first = 0, chunk = 0; first < end; first += flash_keys, ++chunk) {
            const int stage = chunk & 1;
            const Word phase = (chunk >> 1) & 1;
            const int buffer = (first / mxfp8_page_size) & 1;
            if (chunk >= 2) {
                tmem_mma_wait(tmem_smem(&tile.output_ready[stage]), phase ^ 1);
            }
            pair_stage_kv<false, PartialPage>(tile, tile.raw[buffer], first, rank, request,
                                              kv_head, full_end, end, stage);
            pair_publish<1, tmem_producers, flash_queries>(tmem_smem(&tile.operands_ready[stage]));
            const int next = first + flash_keys;
            if (next >= end) {
                break;
            }
            const bool next_page = next % mxfp8_page_size == 0;
            if (next_page && next < full_end) {
                tmem_copy_wait(copy, 1 - buffer);
            }
            tmem_mma_wait(tmem_smem(&tile.scores_ready[stage]), phase);
            pair_stage_kv<true, PartialPage>(tile, tile.raw[buffer ^ static_cast<int>(next_page)],
                                            next, rank, request, kv_head, full_end, end);
            pair_publish<1, tmem_producers, flash_queries>(key);
            if (next_page && next + mxfp8_page_size < full_end) {
                pair_prefetch<flash_queries>(tile.raw[buffer], request, maps,
                                             next + mxfp8_page_size, kv_head, rank, copy);
            }
        }
    }
    pair.sync();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                     :: "r"(tile.tmem), "r"(tmem_cols) : "memory");
    }
    pair_repair(tile, request, pair, row, head);
}

struct TmemAttentionBatch {
    PagedAttentionBatch tiles;
    CUtensorMap key_maps[paged_batch_sequences];
    CUtensorMap value_maps[paged_batch_sequences];
};

__launch_bounds__(tmem_threads, 1)
__global__ void batched_ragged_tmem_attention(const __grid_constant__ TmemAttentionBatch batch) {
    alignas(1024) extern __shared__ unsigned char shared_tmem[];
    const int tile = blockIdx.x;
    const int index = batch.tiles.locate(tile);
    tmem_flash_tile<true>(*reinterpret_cast<TmemFlashTile*>(shared_tmem),
                          batch.tiles.requests[index], batch.key_maps[index],
                          batch.value_maps[index], batch.tiles.rows[index],
                          batch.tiles.row(tile, index, flash_rows));
}

constexpr int max_physical_pages = INT_MAX / mxfp8_page_size + 1;

std::optional<cudaError_t> ragged_device(cudaDeviceProp& properties) {
    int device = 0;
    if (const auto status = cudaGetDevice(&device); status != cudaSuccess) {
        return status;
    }
    if (const auto status = cudaGetDeviceProperties(&properties, device); status != cudaSuccess) {
        return status;
    }
    if (properties.major != 10 || properties.minor != 0 || properties.multiProcessorCount != 148) {
        return std::nullopt;
    }
    return cudaPeekAtLastError();
}
}

namespace gqa_prefill::tmem {

std::optional<cudaError_t> prepare_flash_kernel(
    const void* kernel, std::size_t dynamic_shared, const cudaDeviceProp& properties,
    bool automatic) {
    cudaFuncAttributes attributes{};
    auto status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) {
        return flash_setup_error(status, automatic && paired_kernel_unavailable(status));
    }
    if (properties.sharedMemPerBlockOptin < dynamic_shared + attributes.sharedSizeBytes) {
        return std::nullopt;
    }
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  dynamic_shared);
    return status == cudaSuccess ? cudaSuccess : flash_setup_error(status, automatic &&
        (status == cudaErrorInvalidValue || paired_kernel_unavailable(status)));
}

std::optional<cudaError_t> validate_paired_request(const Attention& request,
                                                  Mxfp8PageTable pages, int rows) {
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

}

std::optional<cudaError_t> launch_tmem_prefill(Mxfp8KvCache cache, Mxfp8PageTable pages,
                                               int position, const Bf16* queries, Bf16* output,
                                               cudaStream_t stream, int rows) {
    constexpr auto dynamic_shared = sizeof(TmemFlashTile);
    if (position < 0 || rows < 1 || rows > flash_chunk || position > INT_MAX - rows) {
        return cudaErrorInvalidValue;
    }
    const int visible = position + rows;
    const int required_pages = visible / mxfp8_page_size + (visible % mxfp8_page_size != 0);
    if (pages.page_count < required_pages || cache.page_count < required_pages ||
        !pages.device_pages || !cache.key_payload || !cache.value_payload ||
        !cache.key_scales || !cache.value_scales || !queries || !output) {
        return cudaErrorInvalidValue;
    }
    if (cache.page_count > max_physical_pages ||
        reinterpret_cast<std::uintptr_t>(cache.key_payload) % alignof(uint4) ||
        reinterpret_cast<std::uintptr_t>(cache.value_payload) % alignof(uint4)) {
        return std::nullopt;
    }
    cudaDeviceProp properties{};
    if (const auto status = ragged_device(properties); !status || *status != cudaSuccess) {
        return status;
    }
    const bool tune_pipeline = visible <= max_multirow_visible_tokens;
    const bool before_flash = tune_pipeline && raw_pipeline_before_flash(cache, rows, visible);
    const bool one_wave = (rows + pipeline_rows - 1) / pipeline_rows
        * gqa_head_pairs(cache) <= properties.multiProcessorCount;
    bool pipeline =
        tune_pipeline &&
        (before_flash || rows == pipeline_query_rows ||
         (rows > pipeline_query_rows && one_wave && default_ragged_prefill(cache, position, rows)));
    const void* const stock_kernel =
        reinterpret_cast<const void*>(tmem_flash_attention<true, true>);
    const auto pipeline_kernel = reinterpret_cast<const void*>(tmem_pipeline_attention);
    const void* kernel = pipeline ? pipeline_kernel : stock_kernel;
    auto ready = prepare_flash_kernel(kernel, dynamic_shared, properties, true);
    if (pipeline && !ready) {
        if (before_flash) return std::nullopt;
        pipeline = false;
        kernel = stock_kernel;
        ready = prepare_flash_kernel(kernel, dynamic_shared, properties, true);
    }
    if (!ready || *ready != cudaSuccess) {
        return ready;
    }
    CUtensorMap key_map{}, value_map{};
    const auto encoded = encode_flash_maps(cache, key_map, value_map, true);
    if (!encoded || *encoded != cudaSuccess) {
        return encoded;
    }
    Attention request{cache, pages.device_pages, position, queries, output};
    if (pipeline) {
        return consume_prefill_submission(
            launch_pipeline(request, key_map, value_map, rows, stream));
    }
    int active = rows;
    void* args[] = {&request, &key_map, &value_map, &active};
    const dim3 blocks((rows + flash_rows - 1) / flash_rows, gqa_head_pairs(cache));
    return cudaLaunchKernel(kernel, blocks, tmem_threads, args, dynamic_shared, stream);
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

std::optional<cudaError_t> decoder::launch_batched_ragged_tmem(
    const PagedPrefillSequence* sequences, int count, cudaStream_t stream) {
    cudaDeviceProp properties{};
    if (const auto status = ragged_device(properties); !status || *status != cudaSuccess) {
        return status;
    }
    constexpr auto dynamic_shared = sizeof(TmemFlashTile);
    const auto* const kernel = reinterpret_cast<const void*>(batched_ragged_tmem_attention);
    const auto prepared = prepare_flash_kernel(kernel, dynamic_shared, properties, true);
    if (!prepared || *prepared != cudaSuccess) {
        return prepared;
    }
    TmemAttentionBatch batch{};
    batch.tiles.count = count;
    const auto& first_cache = sequences[0].cache;
    int tiles = 0;
    for (int index = 0; index < count; ++index) {
        const auto& sequence = sequences[index];
        if (sequence.cache.kv_heads != first_cache.kv_heads ||
            sequence.cache.page_count > max_physical_pages) {
            return std::nullopt;
        }
        const auto encoded = encode_flash_maps(sequence.cache, batch.key_maps[index],
                                               batch.value_maps[index], true);
        if (!encoded || *encoded != cudaSuccess) {
            return encoded;
        }
        batch.tiles.requests[index] = {sequence.cache, sequence.pages.device_pages,
                                       sequence.position, sequence.queries, sequence.output};
        batch.tiles.rows[index] = sequence.rows;
        tiles += (sequence.rows + flash_rows - 1) / flash_rows;
        batch.tiles.tile_end[index] = tiles;
    }
    void* args[] = {&batch};
    return cudaLaunchKernel(kernel, dim3(tiles, gqa_head_pairs(first_cache)), tmem_threads, args,
                            dynamic_shared, stream);
}

}
