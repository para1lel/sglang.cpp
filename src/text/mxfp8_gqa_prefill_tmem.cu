#include "sglang/text/mxfp8_gqa_prefill_common.cuh"

#include <cuda.h>
#include <cudaTypedefs.h>
#include <climits>

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

__device__ void flash_tma_prefetch(FlashPayload& raw, Attention request, int first,
                                   int kv_head, const CUtensorMap* keys, const CUtensorMap* values,
                                   Word barrier) {
    if (threadIdx.x == 0) {
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
constexpr int tmem_swizzle = 64;
constexpr int tmem_producers = 256;
constexpr int tmem_threads = flash_queries + tmem_producers;
constexpr int tmem_parts = 3;
constexpr int tmem_partial_rows = flash_chunk * mxfp8_query_heads;

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

__device__ __forceinline__ int tmem_swizzled(int row, int channel) {
    return channel ^ ((row & 7) * 8);
}

__device__ void tmem_stage_query(TmemFlashTile& tile, Attention request, int row, int head) {
    const int lane = threadIdx.x % 32;
    for (int member = threadIdx.x / 32; member < flash_queries; member += tmem_threads / 32) {
        const auto offset = row_offset(row + member / ordered_heads, head + member % ordered_heads);
        bool bad = false;
        #pragma unroll
        for (int channel = lane; channel < channels; channel += 32) {
            const auto bits = __bfloat16_as_ushort(request.queries[offset + channel]);
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

__device__ __forceinline__ Word tmem_fp8_pair(Word packed, Word scale) {
    Word result;
    asm("{ .reg .b32 pair;\n"
        "cvt.rn.bf16x2.e4m3x2 pair, %1;\n"
        "mul.rn.bf16x2 %0, pair, %2; }"
        : "=r"(result) : "h"(static_cast<short>(packed)), "r"(scale));
    return result;
}

__device__ __forceinline__ Word tmem_fp8_nan(Word packed) {
    return ((packed & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u;
}

__device__ __forceinline__ void tmem_decode(std::uint16_t* destination, uint4 payload,
                                             int scale, bool& bad) {
    const Word nan = tmem_fp8_nan(payload.x) | tmem_fp8_nan(payload.y) |
                     tmem_fp8_nan(payload.z) | tmem_fp8_nan(payload.w);
    if (scale < 15 || scale > 148 || nan) {
        flash_store_vector<false>(destination, payload, scale, bad);
        return;
    }
    const Word gain = static_cast<Word>(scale) * 0x00800080u;
    auto* output = reinterpret_cast<uint4*>(destination);
    output[0] = make_uint4(tmem_fp8_pair(payload.x, gain), tmem_fp8_pair(payload.x >> 16, gain),
                           tmem_fp8_pair(payload.y, gain), tmem_fp8_pair(payload.y >> 16, gain));
    output[1] = make_uint4(tmem_fp8_pair(payload.z, gain), tmem_fp8_pair(payload.z >> 16, gain),
                           tmem_fp8_pair(payload.w, gain), tmem_fp8_pair(payload.w >> 16, gain));
}

template <bool Key>
__device__ void tmem_stage_kv(TmemFlashTile& tile, const FlashPayload& payload,
                             Attention request, int first, int kv_head) {
    constexpr int vector_elements = 16;
    constexpr int token_lanes = channels / vector_elements;
    const int producer = threadIdx.x - flash_queries;
    const int vector = producer % token_lanes;
    const int channel = vector * vector_elements;
    const int page = request.pages[first / mxfp8_page_size];
    const auto* scales = Key ? request.cache.key_scales : request.cache.value_scales;
    const auto& raw = Key ? payload.key : payload.value;
    auto& decoded = Key ? tile.key : tile.value;
    for (int index = producer / token_lanes; index < flash_keys;
         index += tmem_producers / token_lanes) {
        const int sub = channel / tmem_swizzle;
        const int column = channel % tmem_swizzle;
        const int low = tmem_swizzled(index, column);
        const int high = tmem_swizzled(index, column + 8);
        const int token = (first + index) % mxfp8_page_size;
        const auto scale = scale_offset(page, token, kv_head) + channel / mxfp8_group_size;
        bool bad = false;
        uint4 packed[2];
        tmem_decode(reinterpret_cast<std::uint16_t*>(packed), raw[index][vector],
                    scales[scale], bad);
        *reinterpret_cast<uint4*>(&decoded[sub][index][low]) = packed[0];
        *reinterpret_cast<uint4*>(&decoded[sub][index][high]) = packed[1];
        const auto invalid = __ballot_sync(warp_mask, bad);
        if (channel == 0 && ((invalid >> (producer & 16)) & 0xffffu)) {
            atomicMin(&tile.bad_key, first + index);
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

template <int Columns, bool MnMajor = false>
__device__ __forceinline__ void tmem_mma(Word destination, std::uint64_t lhs,
                                      std::uint64_t rhs, bool accumulate) {
    constexpr Word format = (1u << 4) | (1u << 7) | (1u << 10);
    constexpr Word instruction = ((flash_queries / 16) << 24) |
                                 ((Columns / 8) << 17) | (MnMajor << 16) | format;
    asm volatile("{ .reg .pred accum;\n"
                 "setp.ne.u32 accum, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5,%5,%5,%5}, accum; }"
                 :: "r"(destination), "l"(lhs), "l"(rhs), "r"(instruction),
                    "r"(static_cast<int>(accumulate)), "r"(0) : "memory");
}

__device__ __forceinline__ void tmem_commit(Word barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"(barrier) : "memory");
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

__device__ __forceinline__ std::size_t tmem_partial_row(int part, int row, int head) {
    return static_cast<std::size_t>(part) * tmem_partial_rows + row * mxfp8_query_heads + head;
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

template <bool Historical, bool Ordered>
__device__ float tmem_probabilities(TmemFlashTile& tile, Word row_address, int member,
                                     int visible, FlashSoftmax& state) {
    float scores[2][32];
    tmem_load(row_address + tmem_scores, scores[0]);
    tmem_load(row_address + tmem_scores + 32, scores[1]);
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
        #pragma unroll
        for (int vector = 0; vector < 32; vector += 8) {
            float probability[8];
            #pragma unroll
            for (int item = 0; item < 8; ++item) {
                probability[item] = __expf(scores[half][vector + item] - maximum);
                mass[Ordered ? item / 2 : item % 4] += probability[item];
            }
            const int column = tmem_swizzled(member, half * 32 + vector);
            *reinterpret_cast<uint4*>(&tile.probability[member][column]) =
                make_uint4(tmem_pack(probability[0], probability[1]),
                           tmem_pack(probability[2], probability[3]),
                           tmem_pack(probability[4], probability[5]),
                           tmem_pack(probability[6], probability[7]));
        }
    }
    const float tile_mass = Ordered ? (mass[0] + mass[2]) + (mass[1] + mass[3]) :
                                     (mass[0] + mass[1]) + (mass[2] + mass[3]);
    state.mass = fmaf(state.mass, history, tile_mass);
    state.maximum = maximum;
    return history;
}

__device__ __forceinline__ void tmem_query_key(TmemFlashTile& tile, Word mma) {
    #pragma unroll
    for (int dimension = 0; dimension < channels; dimension += 16) {
        const int sub = dimension / tmem_swizzle;
        const int offset = dimension % tmem_swizzle;
        tmem_mma<flash_keys>(tile.tmem + tmem_scores,
                             tmem_descriptor(&tile.query[sub][0][offset]),
                             tmem_descriptor(&tile.key[sub][0][offset]), dimension != 0);
    }
    tmem_commit(mma);
}

__device__ __forceinline__ void tmem_probability_value(
    TmemFlashTile& tile, Word mma, bool history) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    #pragma unroll
    for (int token = 0; token < flash_keys; token += 16) {
        tmem_mma<channels, true>(tile.tmem, tmem_descriptor(&tile.probability[0][token]),
                                tmem_descriptor<true>(&tile.value[0][token][0]),
                                history || token != 0);
    }
    tmem_commit(mma);
}

__device__ __forceinline__ void tmem_store_partial(
    const TmemFlashTile& tile, Word row_address, FlashSoftmax state, int position,
    float* partial, float2* statistics) {
    const int member = threadIdx.x;
    const int offset = member / ordered_heads;
    const auto index = tmem_partial_row(blockIdx.z, blockIdx.x * flash_rows + offset,
                                        blockIdx.y * ordered_heads + member % ordered_heads);
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

__device__ __forceinline__ void tmem_store_output(
    Word row_address, FlashSoftmax state, Attention request, int row, int head) {
    const int member = threadIdx.x;
    auto* output = request.output + row_offset(row + member / ordered_heads,
                                               head + member % ordered_heads);
    const float inverse = __frcp_rn(state.mass);
    #pragma unroll 1
    for (int column = 0; column < channels; column += 32) {
        float accumulator[32];
        tmem_load(row_address + column, accumulator);
        #pragma unroll
        for (int index = 0; index < 32; ++index) {
            output[column + index] = __float2bfloat16_rn(accumulator[index] * inverse);
        }
    }
}

__device__ __forceinline__ void tmem_initialize(TmemFlashTile& tile, int warp,
                                               Word copy, Word mma) {
    if (threadIdx.x == 0) {
        tile.bad_key = INT_MAX;
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                     "mbarrier.init.shared::cta.b64 [%1], 1;\n"
                     "fence.proxy.async.shared::cta;"
                     :: "r"(copy), "r"(mma) : "memory");
    }
    if (warp == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                     "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                     :: "r"(tmem_smem(&tile.tmem)), "r"(tmem_cols) : "memory");
    }
}

template <bool Ordered>
__device__ __forceinline__ void tmem_softmax_tile(TmemFlashTile& tile, Word row_address, Word mma,
                                                 int position, int first, int begin,
                                                 FlashSoftmax& state) {
    tmem_mma_wait(mma, 0);
    const int member = threadIdx.x;
    const int visible = position + member / ordered_heads - first + 1;
    const float history = first + flash_keys <= position + 1 ?
        tmem_probabilities<true, Ordered>(tile, row_address, member, visible, state) :
        tmem_probabilities<false, Ordered>(tile, row_address, member, visible, state);
    if (first != begin && __any_sync(warp_mask, history != 1.0f)) {
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
    asm volatile("tcgen05.fence::before_thread_sync;\n"
                 "fence.proxy.async.shared::cta;" ::: "memory");
}

template <bool Ordered>
__launch_bounds__(tmem_threads, 1)
__global__ void tmem_flash_attention(Attention request,
                                     const __grid_constant__ CUtensorMap key_map,
                                     const __grid_constant__ CUtensorMap value_map,
                                     float* partial, float2* statistics) {
    extern __shared__ TmemFlashTile shared_tmem[];
    auto& tile = shared_tmem[0];
    const int row = blockIdx.x * flash_rows;
    const int head = blockIdx.y * ordered_heads;
    const int position = request.position + row;
    const int part = blockIdx.z;
    const int tiles = (position + flash_rows) / flash_keys;
    const int begin = Ordered ? 0 : (tiles * part / tmem_parts) * flash_keys;
    const int end = Ordered ? position + flash_rows :
                              (tiles * (part + 1) / tmem_parts) * flash_keys;
    const int kv_head = head / ratio;
    const int warp = threadIdx.x / 32;
    const bool softmax = warp < flash_queries / 32;
    const Word copy = tmem_smem(&tile.copy_ready);
    const Word mma = tmem_smem(&tile.mma_ready);
    tmem_initialize(tile, warp, copy, mma);
    tmem_stage_query(tile, request, row, head);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    const Word row_address = tile.tmem + ((warp * 32) << 16);
    FlashSoftmax state{-CUDART_INF_F, 0.0f};
    flash_tma_prefetch(tile.raw[0], request, begin, kv_head, &key_map, &value_map, copy);
    if (!softmax) {
        tmem_copy_wait(copy, 0);
        tmem_stage_kv<true>(tile, tile.raw[0], request, begin, kv_head);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();
    for (int first = begin; first < end; first += flash_keys) {
        const int next = first + flash_keys;
        const int buffer = ((first - begin) / flash_keys) & 1;
        if (next < end) {
            flash_tma_prefetch(tile.raw[1 - buffer], request, next, kv_head,
                               &key_map, &value_map, copy);
        }
        if (threadIdx.x == 0) {
            tmem_query_key(tile, mma);
        }
        if (softmax) {
            tmem_softmax_tile<Ordered>(tile, row_address, mma, position, first, begin, state);
        } else {
            tmem_stage_kv<false>(tile, tile.raw[buffer], request, first, kv_head);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            tmem_probability_value(tile, mma, first != begin);
        }
        if (!softmax && next < end) {
            tmem_copy_wait(copy, 1 - buffer);
            tmem_stage_kv<true>(tile, tile.raw[1 - buffer], request, next, kv_head);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        }
        tmem_mma_wait(mma, 1);
        __syncthreads();
    }
    if (softmax) {
        if constexpr (Ordered) {
            tmem_store_output(row_address, state, request, row, head);
        } else {
            tmem_store_partial(tile, row_address, state, position, partial, statistics);
        }
    }
    __syncthreads();
    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tile.tmem), "r"(tmem_cols) : "memory");
    }
    if constexpr (Ordered) {
        __shared__ OrderedScratch fallback;
        if (__syncthreads_or(softmax && tile.bad_query[threadIdx.x]) ||
            tile.bad_key < position + flash_rows) {
            if (threadIdx.x >= channels) {
                return;
            }
            for (int offset = 0; offset < flash_rows; ++offset) {
                if (position + offset >= tile.bad_key ||
                    tile.bad_query[offset * ordered_heads] ||
                    tile.bad_query[offset * ordered_heads + 1]) {
                    protected_attention(fallback, request, row + offset, head);
                }
            }
        }
    }
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
            stats[head][part] = statistics[tmem_partial_row(part, row, first_head + head)];
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
            const auto index = tmem_partial_row(part, row, first_head + head);
            numerator = fmaf(partial[index * channels + channel], weight, numerator);
        }
        request.output[row_offset(row, first_head + head) + channel] =
            __float2bfloat16_rn(numerator / mass);
    }
}

}

std::optional<cudaError_t> launch_tmem_prefill(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
    const Bf16* queries, Bf16* output, cudaStream_t stream, bool ordered) {
    constexpr auto dynamic_shared = sizeof(TmemFlashTile);
    constexpr int max_physical_pages = INT_MAX / mxfp8_page_size + 1;
    const int required_pages = (position + flash_chunk) / mxfp8_page_size +
                               ((position + flash_chunk) % mxfp8_page_size != 0);
    if (pages.page_count < required_pages || cache.page_count < required_pages ||
        !pages.device_pages || !cache.key_payload || !cache.value_payload ||
        !cache.key_scales || !cache.value_scales || !queries || !output) {
        return cudaErrorInvalidValue;
    }
    if (cache.page_count > max_physical_pages) {
        return std::nullopt;
    }
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
    if (properties.major != 10 || properties.minor != 0 ||
        (!ordered && !properties.memoryPoolsSupported)) {
        return std::nullopt;
    }
    const auto kernel = ordered ? tmem_flash_attention<true> : tmem_flash_attention<false>;
    cudaFuncAttributes attributes{};
    status = cudaFuncGetAttributes(&attributes, kernel);
    if (status != cudaSuccess) {
        return status;
    }
    if (properties.sharedMemPerBlockOptin < dynamic_shared + attributes.sharedSizeBytes) {
        return std::nullopt;
    }
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  dynamic_shared);
    if (status != cudaSuccess) {
        return status;
    }
    CUtensorMap key_map{}, value_map{};
    void* entry = nullptr;
    status = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &entry, 12000,
                                              cudaEnableDefault, nullptr);
    if (status != cudaSuccess) {
        return status;
    }
    if (!entry) {
        return std::nullopt;
    }
    const auto encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(entry);
    constexpr cuuint32_t tile[] = {channels, flash_keys};
    constexpr cuuint32_t elements[] = {1, 1};
    constexpr cuuint64_t stride[] = {mxfp8_kv_heads * channels};
    const cuuint64_t shape[] = {
        stride[0], static_cast<cuuint64_t>(cache.page_count) * mxfp8_page_size};
    const auto encode_payload = [&](CUtensorMap& map, std::uint8_t* data) {
        return encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, data, shape, stride, tile,
                      elements, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    };
    if (encode_payload(key_map, cache.key_payload) != CUDA_SUCCESS ||
        encode_payload(value_map, cache.value_payload) != CUDA_SUCCESS) {
        return cudaErrorInvalidValue;
    }
    Attention request{cache, pages.device_pages, position, queries, output};
    if (ordered) {
        float* partial = nullptr;
        float2* statistics = nullptr;
        void* args[] = {&request, &key_map, &value_map, &partial, &statistics};
        const dim3 blocks(flash_chunk / flash_rows, mxfp8_query_heads / ordered_heads);
        return cudaLaunchKernel(kernel, blocks, tmem_threads, args, dynamic_shared, stream);
    }
    constexpr std::size_t partial_rows = tmem_parts * tmem_partial_rows;
    constexpr std::size_t partial_values = partial_rows * channels;
    void* scratch = nullptr;
    status = cudaMallocAsync(&scratch,
        partial_values * sizeof(float) + partial_rows * sizeof(float2), stream);
    if (status != cudaSuccess) {
        return status;
    }
    auto* partial = static_cast<float*>(scratch);
    auto* statistics = reinterpret_cast<float2*>(partial + partial_values);
    void* args[] = {&request, &key_map, &value_map, &partial, &statistics};
    const dim3 blocks(flash_chunk / flash_rows, mxfp8_query_heads / ordered_heads, tmem_parts);
    status = cudaLaunchKernel(kernel, blocks, tmem_threads, args, dynamic_shared, stream);
    if (status == cudaSuccess) {
        void* merge_args[] = {&request, &partial, &statistics};
        const dim3 merge_blocks(flash_chunk, mxfp8_query_heads / ordered_heads);
        status = cudaLaunchKernel(tmem_merge, merge_blocks, channels, merge_args, 0, stream);
    }
    const auto release = cudaFreeAsync(scratch, stream);
    return status == cudaSuccess ? release : status;
}

}
