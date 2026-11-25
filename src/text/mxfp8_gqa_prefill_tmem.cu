#include "sglang/decoder_internal.h"
#include "sglang/text/mxfp8_gqa_prefill_tmem_kv.cuh"
#include <cute/arch/mma_sm100_umma.hpp>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <climits>
#include <cstddef>
#include <cooperative_groups.h>
#include <memory>

namespace sglang {
namespace {

struct alignas(128) FlashPayload {
    alignas(16) uint4 key[flash_keys][channels / sizeof(uint4)];
    alignas(16) uint4 value[flash_keys][channels / sizeof(uint4)];
};

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

constexpr int tmem_cols = 512;
constexpr int tmem_scores = channels;
constexpr int tmem_threads = flash_queries + tmem_producers;
constexpr int pair_threads = tmem_threads + 32;

static_assert(tmem_scores + 2 * flash_keys <= tmem_cols);
constexpr int tmem_parts = 3;

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

__device__ __forceinline__ Word tmem_smem(const void* pointer) {
    return static_cast<Word>(__cvta_generic_to_shared(pointer));
}

template <int Members = flash_queries, int LoadWarps = tmem_threads / 32, typename Tile>
__device__ void tmem_stage_query(Tile& tile, Attention request, int row, int head,
                                 int rows = flash_rows) {
    const int lane = threadIdx.x % 32;
    for (int member = threadIdx.x / 32; member < Members; member += LoadWarps) {
        const auto offset = row_offset(gqa_query_heads(request.cache.kv_heads),
                                       row + member / ordered_heads,
                                       head + member % ordered_heads);
        const bool valid = member / ordered_heads < rows;
        bool bad = false;
        #pragma unroll
        for (int channel = lane; channel < channels; channel += 32) {
            const auto bits = valid ? __bfloat16_as_ushort(request.queries[offset + channel]) : 0;
            const unsigned magnitude = bits & 0x7fffu;
            const bool invalid = magnitude > 0x4e80u || (magnitude && magnitude < 0x80u);
            tile.query[channel / tmem_swizzle][member]
                      [tmem_swizzled(member, channel % tmem_swizzle)] = invalid ? 0 : bits;
            bad |= invalid;
        }
        const auto invalid = __any_sync(warp_mask, bad);
        if (lane == 0) {
            tile.bad_query[member] = invalid;
        }
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

template <bool MnMajor = false>
__device__ __forceinline__ std::uint64_t tmem_descriptor(const void* pointer) {
    constexpr auto swizzle = std::uint64_t{2} << 61;
    constexpr auto version = std::uint64_t{1} << 46;
    constexpr auto stride = std::uint64_t{64} << 32;
    constexpr auto leading = std::uint64_t{MnMajor ? 512 : 1} << 16;
    return swizzle | version | stride | leading | (tmem_smem(pointer) >> 4);
}

template <int Columns, bool MnMajor = false, bool Paired = false, int Members = flash_queries>
__device__ __forceinline__ void tmem_mma(Word destination, std::uint64_t lhs,
                                      std::uint64_t rhs, bool accumulate) {
    constexpr Word format = (1u << 4) | (1u << 7) | (1u << 10);
    constexpr Word instruction = ((Members * (Paired ? 2 : 1) / 16) << 24) |
                                 ((Columns / 8) << 17) | (MnMajor << 16) | format;
    if constexpr (Paired) {
        asm volatile("{ .reg .pred accum;\n"
                     "setp.ne.u32 accum, %4, 0;\n"
                     "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, "
                     "{%5,%5,%5,%5,%5,%5,%5,%5}, accum; }"
                     :: "r"(destination), "l"(lhs), "l"(rhs), "r"(instruction),
                        "r"(static_cast<int>(accumulate)), "r"(0) : "memory");
    } else {
        asm volatile("{ .reg .pred accum;\n"
                     "setp.ne.u32 accum, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
                     "{%5,%5,%5,%5}, accum; }"
                     :: "r"(destination), "l"(lhs), "l"(rhs), "r"(instruction),
                        "r"(static_cast<int>(accumulate)), "r"(0) : "memory");
    }
}

template <bool Paired = false>
__device__ __forceinline__ void tmem_commit(Word barrier) {
    if constexpr (Paired) {
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
                     "multicast::cluster.b64 [%0], %1;"
                     :: "r"(barrier), "h"(static_cast<unsigned short>(3)) : "memory");
    } else {
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                     :: "r"(barrier) : "memory");
    }
}

__device__ __forceinline__ void tmem_mma_wait(Word barrier, Word phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "MMA_WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready,"
                 " [%0], %1;\n"
                 "@!ready bra.uni MMA_WAIT; }\n"
                 "tcgen05.fence::after_thread_sync;"
                 :: "r"(barrier), "r"(phase) : "memory");
}

__device__ __forceinline__ void tmem_copy_wait(Word barrier, Word phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "COPY_WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready,"
                 " [%0], %1;\n"
                 "@!ready bra.uni COPY_WAIT; }\n"
                 :: "r"(barrier), "r"(phase) : "memory");
}

__device__ __forceinline__ std::size_t tmem_partial_row(int part, int row, int head,
                                                        int query_heads) {
    return static_cast<std::size_t>(part) * flash_chunk * query_heads
        + row * query_heads + head;
}

__device__ __forceinline__ void tmem_load(Word pointer, float (&values)[32]) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,"
                 "%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
                 "%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
                 "tcgen05.wait::ld.sync.aligned;"
                 : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]),
                   "=f"(values[4]), "=f"(values[5]), "=f"(values[6]), "=f"(values[7]),
                   "=f"(values[8]), "=f"(values[9]), "=f"(values[10]), "=f"(values[11]),
                   "=f"(values[12]), "=f"(values[13]), "=f"(values[14]), "=f"(values[15]),
                   "=f"(values[16]), "=f"(values[17]), "=f"(values[18]), "=f"(values[19]),
                   "=f"(values[20]), "=f"(values[21]), "=f"(values[22]), "=f"(values[23]),
                   "=f"(values[24]), "=f"(values[25]), "=f"(values[26]), "=f"(values[27]),
                   "=f"(values[28]), "=f"(values[29]), "=f"(values[30]), "=f"(values[31])
                 : "r"(pointer) : "memory");
}

__device__ __forceinline__ void tmem_store(Word pointer, const float (&values)[32]) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,"
                 "%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,"
                 "%25,%26,%27,%28,%29,%30,%31,%32};"
                 :: "r"(pointer), "f"(values[0]), "f"(values[1]), "f"(values[2]),
                    "f"(values[3]), "f"(values[4]), "f"(values[5]), "f"(values[6]),
                    "f"(values[7]), "f"(values[8]), "f"(values[9]), "f"(values[10]),
                    "f"(values[11]), "f"(values[12]), "f"(values[13]), "f"(values[14]),
                    "f"(values[15]), "f"(values[16]), "f"(values[17]), "f"(values[18]),
                    "f"(values[19]), "f"(values[20]), "f"(values[21]), "f"(values[22]),
                    "f"(values[23]), "f"(values[24]), "f"(values[25]), "f"(values[26]),
                    "f"(values[27]), "f"(values[28]), "f"(values[29]), "f"(values[30]),
                    "f"(values[31]) : "memory");
}

__device__ __forceinline__ Word tmem_pack(float low, float high) {
    Word packed;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(packed) : "f"(low), "f"(high));
    return packed;
}

__device__ __forceinline__ void tmem_store_probability(Word destination, const Word (&bits)[16]) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x16.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,"
                 "%9,%10,%11,%12,%13,%14,%15,%16};\n"
                 "tcgen05.wait::st.sync.aligned;"
                 :: "r"(destination), "r"(bits[0]), "r"(bits[1]), "r"(bits[2]), "r"(bits[3]),
                    "r"(bits[4]), "r"(bits[5]), "r"(bits[6]), "r"(bits[7]), "r"(bits[8]),
                    "r"(bits[9]), "r"(bits[10]), "r"(bits[11]), "r"(bits[12]), "r"(bits[13]),
                    "r"(bits[14]), "r"(bits[15]) : "memory");
}

template <bool Historical, bool Ordered, bool TmemProbability = false, typename Probability>
__device__ float tmem_probabilities(Probability& probability_tile, Word row_address, int member,
                                    int visible, FlashSoftmax& state,
                                    Word score_column = tmem_scores, Word probability_column = 0) {
    float scores[2][32];
    tmem_load(row_address + score_column, scores[0]);
    tmem_load(row_address + score_column + 32, scores[1]);
    if (member < 0) {
        return 1.0f;
    }
    float maximum = state.maximum;
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        #pragma unroll
        for (int column = 0; column < 32; ++column) {
            auto& score = scores[half][column];
            score = Historical || half * 32 + column < visible ?
                score * softmax_scale : -CUDART_INF_F;
            maximum = fmaxf(maximum, score);
        }
    }
    const float history = state.maximum == maximum ? 1.0f : __expf(state.maximum - maximum);
    float mass[4]{};
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        Word packed[16];
        #pragma unroll
        for (int vector = 0; vector < 32; vector += 8) {
            float probability[8];
            #pragma unroll
            for (int item = 0; item < 8; ++item) {
                probability[item] = __expf(scores[half][vector + item] - maximum);
                mass[Ordered ? item / 2 : item % 4] += probability[item];
            }
            const auto bits = make_uint4(tmem_pack(probability[0], probability[1]),
                                         tmem_pack(probability[2], probability[3]),
                                         tmem_pack(probability[4], probability[5]),
                                         tmem_pack(probability[6], probability[7]));
            if constexpr (TmemProbability) {
                packed[vector / 2] = bits.x;
                packed[vector / 2 + 1] = bits.y;
                packed[vector / 2 + 2] = bits.z;
                packed[vector / 2 + 3] = bits.w;
            } else {
                const int column = tmem_swizzled(member, half * 32 + vector);
                *reinterpret_cast<uint4*>(&probability_tile[member][column]) = bits;
            }
        }
        if constexpr (TmemProbability) {
            tmem_store_probability(row_address + probability_column + 16 * half, packed);
        }
    }
    const float tile_mass = Ordered ? (mass[0] + mass[2]) + (mass[1] + mass[3]) :
                                     (mass[0] + mass[1]) + (mass[2] + mass[3]);
    state.mass = fmaf(state.mass, history, tile_mass);
    state.maximum = maximum;
    return history;
}

template <bool Paired = false, bool Staged = false, int Members = flash_queries, typename Tile>
__device__ __forceinline__ void tmem_query_key(Tile& tile, Word mma,
                                              Word score_column = tmem_scores, int key_slot = -1) {
    const auto& keys = [&]() -> auto& {
        if constexpr (Staged) {
            return tile.key[key_slot < 0 ? (score_column - tmem_scores) / flash_keys : key_slot];
        } else {
            return tile.key;
        }
    }();
    #pragma unroll
    for (int dimension = 0; dimension < channels; dimension += 16) {
        const int sub = dimension / tmem_swizzle;
        const int offset = dimension % tmem_swizzle;
        tmem_mma<flash_keys, false, Paired, Members>(
            tile.tmem + score_column, tmem_descriptor(&tile.query[sub][0][offset]),
            tmem_descriptor(&keys[sub][0][offset]), dimension != 0);
    }
    tmem_commit<Paired>(mma);
}

template <bool Paired = false, int Members = flash_queries, typename Probability, typename Value>
__device__ __forceinline__ void tmem_probability_value(
    Word tmem, const Probability& probability, const Value& value, Word mma, bool history) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    #pragma unroll
    for (int token = 0; token < flash_keys; token += 16) {
        tmem_mma<channels, true, Paired, Members>(
            tmem, tmem_descriptor(&probability[0][token]),
            tmem_descriptor<true>(&value[0][token][0]), history || token != 0);
    }
    tmem_commit<Paired>(mma);
}

__device__ __forceinline__ void tmem_store_partial(
    const TmemFlashTile& tile, Word row_address, FlashSoftmax state, int position,
    float* partial, float2* statistics, int query_heads) {
    const int member = threadIdx.x;
    const int offset = member / ordered_heads;
    const auto index = tmem_partial_row(blockIdx.z, blockIdx.x * flash_rows + offset,
                                        blockIdx.y * ordered_heads + member % ordered_heads,
                                        query_heads);
    auto* output = partial + index * channels;
    const bool bad = tile.bad_query[member] || tile.bad_key <= position + offset;
    statistics[index] = make_float2(state.maximum, bad ? -1.0f : state.mass);
    #pragma unroll 1
    for (int column = 0; column < channels; column += 32) {
        float accumulator[32];
        tmem_load(row_address + column, accumulator);
        #pragma unroll
        for (int index = 0; index < 32; ++index) {
            output[column + index] = accumulator[index];
        }
    }
}

#include "sglang/text/mxfp8_gqa_prefill_staged.cuh"

template <bool Paired = false, bool Staged = false, typename Tile>
__device__ __forceinline__ void tmem_initialize(Tile& tile, int warp,
                                               Word copy, Word mma) {
    if (threadIdx.x == 0) {
        tile.bad_key = INT_MAX;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                     "mbarrier.init.shared::cta.b64 [%1], 1;\n"
                     "fence.proxy.async.shared::cta;"
                     :: "r"(copy), "r"(mma) : "memory");
        if constexpr (Paired) {
            if constexpr (Staged) {
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                             "mbarrier.init.shared::cta.b64 [%1], 2;"
                             :: "r"(tmem_smem(&tile.value_copy_ready)),
                                "r"(tmem_smem(&tile.key_ready[1])) : "memory");
            }
            const Word key_ready = [&] {
                if constexpr (Staged) {
                    return tmem_smem(&tile.key_ready[0]);
                } else {
                    return tmem_smem(&tile.key_ready);
                }
            }();
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                         "mbarrier.init.shared::cta.b64 [%1], 1;\n"
                         "mbarrier.init.shared::cta.b64 [%2], 1;\n"
                         "mbarrier.init.shared::cta.b64 [%3], 4;\n"
                         "mbarrier.init.shared::cta.b64 [%4], 4;\n"
                         "mbarrier.init.shared::cta.b64 [%5], 2;\n"
                         "fence.proxy.async.shared::cta;\n"
                         "fence.mbarrier_init.release.cluster;"
                         :: "r"(tmem_smem(&tile.scores_ready[1])),
                            "r"(tmem_smem(&tile.output_ready[0])),
                            "r"(tmem_smem(&tile.output_ready[1])),
                            "r"(tmem_smem(&tile.operands_ready[0])),
                            "r"(tmem_smem(&tile.operands_ready[1])),
                            "r"(key_ready) : "memory");
        }
    }
    if constexpr (Paired) {
        cooperative_groups::this_cluster().sync();
        if (warp == 0) {
            asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;\n"
                         "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"
                         :: "r"(tmem_smem(&tile.tmem)), "r"(tmem_cols) : "memory");
        }
    } else if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                     "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                     :: "r"(tmem_smem(&tile.tmem)), "r"(tmem_cols) : "memory");
    }
}

__device__ __forceinline__ void tmem_correct_output(Word row_address, float history) {
    if (__any_sync(warp_mask, history != 1.0f)) {
        #pragma unroll 1
        for (int column = 0; column < channels; column += 32) {
            float accumulator[32];
            tmem_load(row_address + column, accumulator);
            #pragma unroll
            for (auto& value : accumulator) {
                value *= history;
            }
            tmem_store(row_address + column, accumulator);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
}

template <bool Ordered, typename Probability>
__device__ __forceinline__ float tmem_softmax_chunk(Probability& probability, Word row_address,
                                                    int position, int first, FlashSoftmax& state,
                                                    Word score_column = tmem_scores,
                                                    int member = threadIdx.x) {
    const int visible = position + max(0, member) / ordered_heads - first + 1;
    return first + flash_keys <= position + 1 ?
        tmem_probabilities<true, Ordered>(probability, row_address, member, visible, state,
                                         score_column) :
        tmem_probabilities<false, Ordered>(probability, row_address, member, visible, state,
                                          score_column);
}

template <bool Ordered>
__device__ __forceinline__ void tmem_softmax_tile(TmemFlashTile& tile, Word row_address, Word mma,
                                                 int position, int first, int begin,
                                                 FlashSoftmax& state) {
    tmem_mma_wait(mma, 0);
    const float history = tmem_softmax_chunk<Ordered>(tile.probability, row_address, position,
                                                      first, state);
    if (first != begin) {
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

#include "sglang/text/mxfp8_gqa_prefill_pipeline.cuh"

template <bool Ordered, bool Ragged>
__device__ __forceinline__ void tmem_flash_tile(
    TmemFlashTile& tile, Attention request, const CUtensorMap& key_map,
    const CUtensorMap& value_map, float* partial, float2* statistics, int rows, int row) {
    const int active_rows = Ragged ? min(flash_rows, rows - row) : flash_rows;
    const int head = blockIdx.y * ordered_heads;
    const int position = request.position + row;
    const int part = blockIdx.z;
    const int tiles = (position + flash_rows) / flash_keys;
    const int begin = Ordered ? 0 : (tiles * part / tmem_parts) * flash_keys;
    const int end = Ordered ? position + active_rows :
                              (tiles * (part + 1) / tmem_parts) * flash_keys;
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
    if (!Ragged || end - begin >= flash_keys) {
        flash_tma_prefetch(tile.raw[0], request, begin, kv_head, &key_map, &value_map, copy);
    }
    if (!softmax) {
        tmem_stage_tile<true, Ragged>(tile, request, begin, end, kv_head, 0, copy);
    }
    __syncthreads();
    for (int first = begin; first < end; first += flash_keys) {
        const int next = first + flash_keys;
        const int buffer = ((first - begin) / flash_keys) & 1;
        if (next < end && (!Ragged || end - next >= flash_keys)) {
            flash_tma_prefetch(tile.raw[1 - buffer], request, next, kv_head,
                               &key_map, &value_map, copy);
        }
        if (threadIdx.x == 0) {
            tmem_query_key(tile, mma);
        }
        if (softmax) {
            tmem_softmax_tile<Ordered>(tile, row_address, mma, position, first, begin, state);
        } else {
            tmem_stage_tile<false, Ragged>(tile, request, first, end, kv_head, buffer, copy);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            tmem_probability_value(tile.tmem, tile.probability, tile.value, mma, first != begin);
        }
        if (!softmax && next < end) {
            tmem_stage_tile<true, Ragged>(tile, request, next, end, kv_head, 1 - buffer, copy);
        }
        tmem_mma_wait(mma, 1);
        __syncthreads();
    }
    if (softmax) {
        if constexpr (Ordered) {
            tmem_store_output(row_address, state, request, row, head, active_rows);
        } else {
            tmem_store_partial(tile, row_address, state, position, partial, statistics,
                               gqa_query_heads(request.cache.kv_heads));
        }
    }
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tile.tmem), "r"(tmem_cols) : "memory");
    }
    if constexpr (Ordered) {
        tmem_repair_ordered(tile, request, row, head, active_rows);
    }
}

template <bool Ordered, bool Ragged = false>
__launch_bounds__(tmem_threads, 1)
__global__ void tmem_flash_attention(Attention request,
                                     const __grid_constant__ CUtensorMap key_map,
                                     const __grid_constant__ CUtensorMap value_map,
                                     float* partial, float2* statistics, int rows) {
    alignas(1024) extern __shared__ unsigned char shared_tmem[];
    tmem_flash_tile<Ordered, Ragged>(*reinterpret_cast<TmemFlashTile*>(shared_tmem),
                                     request, key_map, value_map, partial, statistics, rows,
                                     blockIdx.x * flash_rows);
}

struct PairMaps {
    CUtensorMap keys;
    CUtensorMap values;
    CUtensorMap key_scales;
    CUtensorMap value_scales;
};

static_assert(alignof(PairMaps) >= 64 && sizeof(CUtensorMap) % 64 == 0);

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

template <typename Tile>
__device__ void pair_repair(Tile& tile, Attention request,
                            cooperative_groups::cluster_group pair, int row, int head) {
    __shared__ OrderedScratch fallback;
    const int peer_key = *pair.map_shared_rank(&tile.bad_key, 1 - pair.block_rank());
    const int bad_key = min(tile.bad_key, peer_key);
    pair.sync();
    const int position = request.position + row;
    if (!__syncthreads_or(threadIdx.x < flash_queries && tile.bad_query[threadIdx.x]) &&
        bad_key >= position + flash_rows) {
        return;
    }
    if (threadIdx.x >= channels) {
        return;
    }
    for (int offset = 0; offset < flash_rows; ++offset) {
        if (position + offset >= bad_key || tile.bad_query[offset * ordered_heads] ||
            tile.bad_query[offset * ordered_heads + 1]) {
            protected_attention(fallback, request, row + offset, head);
        }
    }
}

template <int Barrier, int Count, int Leader>
__device__ __forceinline__ void pair_publish(Word ready) {
    asm volatile("fence.proxy.async.shared::cta;\n"
                 "bar.sync %0, %1;" :: "n"(Barrier), "n"(Count) : "memory");
    if (threadIdx.x == Leader) {
        pair_arrive(ready);
    }
}

__device__ __forceinline__ void pair_wait(Word ready, Word phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "PAIR_WAIT: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready,"
                 " [%0], %1;\n"
                 "@!ready bra.uni PAIR_WAIT; }"
                 :: "r"(ready), "r"(phase) : "memory");
}

template <bool Staged = false, typename Tile>
__device__ __forceinline__ void pair_issue_mmas(Tile& tile, int end) {
    tmem_query_key<true, Staged>(tile, tmem_smem(&tile.scores_ready[0]));
    for (int chunk = 0; chunk * flash_keys < end; ++chunk) {
        const int stage = chunk & 1;
        if ((chunk + 1) * flash_keys < end) {
            if constexpr (Staged) {
                pair_wait(tmem_smem(&tile.key_ready[1 - stage]), (chunk >> 1) & 1);
            } else {
                pair_wait(tmem_smem(&tile.key_ready), stage);
            }
            tmem_query_key<true, Staged>(tile, tmem_smem(&tile.scores_ready[1 - stage]),
                                         tmem_scores + (1 - stage) * flash_keys);
        }
        pair_wait(tmem_smem(&tile.operands_ready[stage]), (chunk >> 1) & 1);
        tmem_probability_value<true>(tile.tmem, tile.probability[stage], tile.value[stage],
                                      tmem_smem(&tile.output_ready[stage]), chunk != 0);
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

__global__ void tmem_merge(Attention request, const float* partial, const float2* statistics) {
    __shared__ OrderedScratch fallback;
    const int row = blockIdx.x;
    const int first_head = blockIdx.y * ordered_heads;
    float2 stats[ordered_heads][tmem_parts];
    bool bad = false;
    #pragma unroll
    for (int head = 0; head < ordered_heads; ++head) {
        #pragma unroll
        for (int part = 0; part < tmem_parts; ++part) {
            stats[head][part] = statistics[tmem_partial_row(
                part, row, first_head + head, gqa_query_heads(request.cache.kv_heads))];
            bad |= stats[head][part].y < 0.0f;
        }
    }
    if (bad) {
        protected_attention(fallback, request, row, first_head);
        return;
    }
    const int channel = threadIdx.x;
    #pragma unroll
    for (int head = 0; head < ordered_heads; ++head) {
        float maximum = -CUDART_INF_F;
        #pragma unroll
        for (int part = 0; part < tmem_parts; ++part) {
            maximum = fmaxf(maximum, stats[head][part].x);
        }
        float mass = 0.0f;
        float numerator = 0.0f;
        #pragma unroll
        for (int part = 0; part < tmem_parts; ++part) {
            const float weight = __expf(stats[head][part].x - maximum);
            mass = fmaf(stats[head][part].y, weight, mass);
            const auto index = tmem_partial_row(part, row, first_head + head,
                                                gqa_query_heads(request.cache.kv_heads));
            numerator = fmaf(partial[index * channels + channel], weight, numerator);
        }
        request.output[row_offset(gqa_query_heads(request.cache.kv_heads),
                                  row, first_head + head) + channel] =
            __float2bfloat16_rn(numerator / mass);
    }
}

}

#include "sglang/text/gqa/tmem_maps.cuh"

std::optional<cudaError_t> launch_tmem_prefill(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
    const Bf16* queries, Bf16* output, cudaStream_t stream, bool ordered,
    int ragged_rows, bool automatic) {
    constexpr auto dynamic_shared = sizeof(TmemFlashTile);
    constexpr int max_physical_pages = INT_MAX / mxfp8_page_size + 1;
    const int rows = ragged_rows ? ragged_rows : flash_chunk;
    if (position < 0 || rows < 1 || rows > flash_chunk || position > INT_MAX - rows ||
        (ragged_rows && !ordered)) {
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
    int device = 0;
    if (auto status = cudaGetDevice(&device); status != cudaSuccess) {
        return status;
    }
    cudaDeviceProp properties{};
    if (auto status = cudaGetDeviceProperties(&properties, device); status != cudaSuccess) {
        return status;
    }
    if (properties.major != 10 || properties.minor != 0 ||
        (automatic && properties.multiProcessorCount != 148) ||
        (!ordered && !properties.memoryPoolsSupported)) {
        return std::nullopt;
    }
    if (const auto pending = automatic ? cudaPeekAtLastError() : cudaSuccess;
        pending != cudaSuccess) return pending;
    const auto stock = ragged_rows ? tmem_flash_attention<true, true> :
        ordered ? tmem_flash_attention<true> : tmem_flash_attention<false>;
    const bool before_flash = automatic && raw_pipeline_before_flash(cache, ragged_rows, visible);
    // A 32-row pipeline tile doubles row parallelism. Keep it within one resident wave.
    const bool one_wave = (rows + pipeline_rows - 1) / pipeline_rows
        * gqa_head_pairs(cache) <= properties.multiProcessorCount;
    bool pipeline = automatic && (before_flash || ragged_rows == pipeline_query_rows ||
        (ragged_rows > pipeline_query_rows && one_wave &&
         default_ragged_prefill(cache, position, rows)));
    const auto stock_kernel = reinterpret_cast<const void*>(stock);
    const auto pipeline_kernel = reinterpret_cast<const void*>(tmem_pipeline_attention);
    const void* kernel = pipeline ? pipeline_kernel : stock_kernel;
    auto ready = prepare_flash_kernel(kernel, dynamic_shared, properties, automatic);
    if (pipeline && !ready) {
        if (before_flash) return std::nullopt;
        pipeline = false;
        kernel = stock_kernel;
        ready = prepare_flash_kernel(kernel, dynamic_shared, properties, automatic);
    }
    if (!ready || *ready != cudaSuccess) {
        return ready;
    }
    CUtensorMap key_map{}, value_map{};
    const auto encoded = encode_flash_maps(cache, key_map, value_map, automatic);
    if (!encoded || *encoded != cudaSuccess) {
        return encoded;
    }
    Attention request{cache, pages.device_pages, position, queries, output};
    if (pipeline) {
        return consume_prefill_submission(
            launch_pipeline(request, key_map, value_map, rows, stream));
    }
    if (ordered) {
        float* partial = nullptr;
        float2* statistics = nullptr;
        int active = rows;
        void* args[] = {&request, &key_map, &value_map, &partial, &statistics, &active};
        const dim3 blocks((rows + flash_rows - 1) / flash_rows, gqa_head_pairs(cache));
        return cudaLaunchKernel(kernel, blocks, tmem_threads, args, dynamic_shared, stream);
    }
    const std::size_t partial_rows = tmem_parts * flash_chunk * gqa_query_heads(cache.kv_heads);
    const std::size_t partial_values = partial_rows * channels;
    void* scratch = nullptr;
    auto status = cudaMallocAsync(&scratch,
        partial_values * sizeof(float) + partial_rows * sizeof(float2), stream);
    if (status != cudaSuccess) {
        return status;
    }
    auto* partial = static_cast<float*>(scratch);
    auto* statistics = reinterpret_cast<float2*>(partial + partial_values);
    int active = rows;
    void* args[] = {&request, &key_map, &value_map, &partial, &statistics, &active};
    const dim3 blocks(flash_chunk / flash_rows, gqa_head_pairs(cache), tmem_parts);
    status = cudaLaunchKernel(kernel, blocks, tmem_threads, args, dynamic_shared, stream);
    if (status == cudaSuccess) {
        void* merge_args[] = {&request, &partial, &statistics};
        const dim3 merge_blocks(flash_chunk, gqa_head_pairs(cache));
        status = cudaLaunchKernel(tmem_merge, merge_blocks, channels, merge_args, 0, stream);
    }
    const auto release = cudaFreeAsync(scratch, stream);
    return status == cudaSuccess ? release : status;
}
}

#include "sglang/text/mxfp8_gqa_prefill_paired.cuh"
#include "sglang/text/mxfp8_gqa_prefill_tmem_batch.cuh"
