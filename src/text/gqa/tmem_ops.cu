#include "sglang/text/gqa/tmem_ops.cuh"

#include <climits>

namespace sglang::gqa_prefill::tmem {

__device__ __inline_hint__ Word tmem_smem(const void* pointer) {
    return static_cast<Word>(__cvta_generic_to_shared(pointer));
}

__device__ __inline_hint__ int tmem_swizzled(int row, int channel) {
    return channel ^ ((row & 7) * 8);
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

__device__ __inline_hint__ void tmem_decode(std::uint16_t* destination, uint4 payload,
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
__device__ __inline_hint__ void pair_decode_tail(
    std::uint16_t* output, const Mxfp8KvCache& cache, int page, int token, int kv_head,
    int channel, bool& bad) {
    const auto* payload = Key ? cache.key_payload : cache.value_payload;
    const auto* scales = Key ? cache.key_scales : cache.value_scales;
    const auto offset = payload_offset(cache.kv_heads, page, token, kv_head) + channel;
    const auto scale = scale_offset(cache.kv_heads, page, token, kv_head)
        + channel / mxfp8_group_size;
    tmem_decode(output, *reinterpret_cast<const uint4*>(payload + offset), scales[scale], bad);
}

template <int Members, int LoadWarps, typename Tile>
__device__ __inline_hint__ void tmem_stage_query(Tile& tile, Attention request, int row, int head,
                                 int rows) {
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

template <bool MnMajor>
__device__ __inline_hint__ std::uint64_t tmem_descriptor(const void* pointer) {
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

template <bool Paired>
__device__ __inline_hint__ void tmem_commit(Word barrier) {
    if constexpr (Paired) {
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
                     "multicast::cluster.b64 [%0], %1;"
                     :: "r"(barrier), "h"(static_cast<unsigned short>(3)) : "memory");
    } else {
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                     :: "r"(barrier) : "memory");
    }
}

__device__ __inline_hint__ void tmem_mma_wait(Word barrier, Word phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "MMA_WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready,"
                 " [%0], %1;\n"
                 "@!ready bra.uni MMA_WAIT; }\n"
                 "tcgen05.fence::after_thread_sync;"
                 :: "r"(barrier), "r"(phase) : "memory");
}

__device__ __inline_hint__ void tmem_copy_wait(Word barrier, Word phase) {
    asm volatile("{ .reg .pred ready;\n"
                 "COPY_WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ready,"
                 " [%0], %1;\n"
                 "@!ready bra.uni COPY_WAIT; }\n"
                 :: "r"(barrier), "r"(phase) : "memory");
}

__device__ __inline_hint__ void pipeline_init(unsigned long long& barrier, Word arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(tmem_smem(&barrier)), "r"(arrivals) : "memory");
}

__device__ __inline_hint__ void tmem_load(Word pointer, float (&values)[32]) {
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

template <bool Historical, bool Ordered, bool TmemProbability, typename Probability>
__device__ __inline_hint__ float tmem_probabilities(Probability& probability_tile, Word row_address,
                                                   int member, int visible, FlashSoftmax& state,
                                    Word score_column, Word probability_column) {
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

template <bool Paired, bool Staged, int Members, typename Tile>
__device__ __inline_hint__ void tmem_query_key(Tile& tile, Word mma,
                                              Word score_column, int key_slot) {
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

template <bool Paired, int Members, typename Probability, typename Value>
__device__ __inline_hint__ void tmem_probability_value(
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

template <bool Paired, bool Staged, typename Tile>
__device__ __inline_hint__ void tmem_initialize(Tile& tile, int warp,
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

__device__ __inline_hint__ void tmem_correct_output(Word row_address, float history) {
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
__device__ __inline_hint__ float tmem_softmax_chunk(Probability& probability, Word row_address,
                                                    int position, int first, FlashSoftmax& state,
                                                    Word score_column,
                                                    int member) {
    const int visible = position + max(0, member) / ordered_heads - first + 1;
    return first + flash_keys <= position + 1 ?
        tmem_probabilities<true, Ordered>(probability, row_address, member, visible, state,
                                         score_column) :
        tmem_probabilities<false, Ordered>(probability, row_address, member, visible, state,
                                          score_column);
}

template <bool Packed>
__device__ __inline_hint__ void tmem_store_output(
    Word row_address, FlashSoftmax state, Attention request, int row, int head,
    int rows, int member) {
    const auto output = row_offset(gqa_query_heads(request.cache.kv_heads),
                                   row + max(0, member) / ordered_heads,
                                   head + max(0, member) % ordered_heads);
    const float inverse = __frcp_rn(state.mass);
    #pragma unroll 1
    for (int column = 0; column < channels; column += 32) {
        float accumulator[32];
        tmem_load(row_address + column, accumulator);
        if (member >= 0 && member / ordered_heads < rows) {
            if constexpr (Packed) {
                #pragma unroll
                for (int index = 0; index < 32; index += 8) {
                    const uint4 packed = make_uint4(
                        tmem_pack(accumulator[index] * inverse,
                                  accumulator[index + 1] * inverse),
                        tmem_pack(accumulator[index + 2] * inverse,
                                  accumulator[index + 3] * inverse),
                        tmem_pack(accumulator[index + 4] * inverse,
                                  accumulator[index + 5] * inverse),
                        tmem_pack(accumulator[index + 6] * inverse,
                                  accumulator[index + 7] * inverse));
                    *reinterpret_cast<uint4*>(request.output + output + column + index) = packed;
                }
            } else {
                #pragma unroll
                for (int index = 0; index < 32; ++index) {
                    request.output[output + column + index] =
                        __float2bfloat16_rn(accumulator[index] * inverse);
                }
            }
        }
    }
}

__device__ __inline_hint__ void pair_arrive(Word ready) {
    asm volatile("{ .reg .u32 leader;\n"
                 "mapa.shared::cluster.u32 leader, %0, 0;\n"
                 "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [leader]; }"
                 :: "r"(ready) : "memory");
}

template <typename Tile>
__device__ __inline_hint__ void pair_repair(Tile& tile, Attention request,
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
__device__ __inline_hint__ void pair_publish(Word ready) {
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

template <bool Staged, typename Tile>
__device__ __inline_hint__ void pair_issue_mmas(Tile& tile, int end) {
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

template <typename Tile>
__device__ __inline_hint__ void tmem_repair_ordered(const Tile& tile, Attention request,
                                                    int row, int head, int rows) {
    __shared__ OrderedScratch fallback;
    constexpr int members = sizeof(tile.bad_query) / sizeof(int);
    const int position = request.position + row;
    if (!__syncthreads_or(threadIdx.x < members && tile.bad_query[threadIdx.x]) &&
        tile.bad_key >= position + rows) {
        return;
    }
    if (threadIdx.x >= channels) {
        return;
    }
    for (int offset = 0; offset < rows; ++offset) {
        if (position + offset >= tile.bad_key || tile.bad_query[offset * ordered_heads] ||
            tile.bad_query[offset * ordered_heads + 1]) {
            protected_attention(fallback, request, row + offset, head);
        }
    }
}

template __device__ void tmem_stage_query<128, 12>(TmemFlashTile&, Attention, int, int, int);
template __device__ void tmem_stage_query<64, 12>(TmemPipelineTile&, Attention, int, int, int);
template __device__ void tmem_stage_query<128, 12>(PairTile&, Attention, int, int, int);
template __device__ void tmem_stage_query<128, 12>(StagedPairTile&, Attention, int, int, int);
template __device__ void tmem_stage_query<128, 8>(StagedPairTile&, Attention, int, int, int);
template __device__ void tmem_stage_query<128, 8>(StagedSingleTile&, Attention, int, int, int);
template __device__ void tmem_stage_query<128, 8>(StagedPageTile&, Attention, int, int, int);
template __device__ void tmem_query_key<false, false, 128>(TmemFlashTile&, Word, Word, int);
template __device__ void tmem_query_key<false, false, 64>(TmemPipelineTile&, Word, Word, int);
template __device__ void tmem_query_key<true, false, 128>(PairTile&, Word, Word, int);
template __device__ void tmem_query_key<true, true, 128>(StagedPairTile&, Word, Word, int);
template __device__ void tmem_query_key<false, true, 128>(StagedSingleTile&, Word, Word, int);
using FlashProbability = std::uint16_t[flash_queries][flash_keys];
using PipelineProbability = std::uint16_t[pipeline_members][flash_keys];
using FullValue = std::uint16_t[channels / tmem_swizzle][flash_keys][tmem_swizzle];
using PairValue = std::uint16_t[pair_value_channels / tmem_swizzle][flash_keys][tmem_swizzle];
template __device__ void tmem_probability_value<false, flash_queries>(
    Word, const FlashProbability&, const FullValue&, Word, bool);
template __device__ void tmem_probability_value<false, pipeline_members>(
    Word, const PipelineProbability&, const FullValue&, Word, bool);
template __device__ void tmem_probability_value<true, flash_queries>(
    Word, const FlashProbability&, const PairValue&, Word, bool);
template __device__ void tmem_initialize<false, false>(TmemFlashTile&, int, Word, Word);
template __device__ void tmem_initialize<false, false>(TmemPipelineTile&, int, Word, Word);
template __device__ void tmem_initialize<true, false>(PairTile&, int, Word, Word);
template __device__ void tmem_initialize<true, true>(StagedPairTile&, int, Word, Word);
template __device__ float tmem_probabilities<true, true, true>(
    Word&, Word, int, int, FlashSoftmax&, Word, Word);
template __device__ float tmem_probabilities<false, true, true>(
    Word&, Word, int, int, FlashSoftmax&, Word, Word);
template __device__ float tmem_softmax_chunk<true>(
    FlashProbability&, Word, int, int, FlashSoftmax&, Word, int);
template __device__ float tmem_softmax_chunk<false>(
    FlashProbability&, Word, int, int, FlashSoftmax&, Word, int);
template __device__ float tmem_softmax_chunk<true>(
    PipelineProbability&, Word, int, int, FlashSoftmax&, Word, int);
template __device__ void tmem_store_output<true>(
    Word, FlashSoftmax, Attention, int, int, int, int);
template __device__ void tmem_store_output<false>(
    Word, FlashSoftmax, Attention, int, int, int, int);
template __device__ void pair_publish<2, flash_queries, 0>(Word);
template __device__ void pair_publish<1, tmem_producers, flash_queries>(Word);
template __device__ void pair_publish<1, staged_producers, flash_queries>(Word);
template __device__ void pair_publish<3, staged_producers, staged_value_leader>(Word);
template __device__ void pair_issue_mmas<false>(PairTile&, int);
template __device__ void pair_issue_mmas<true>(StagedPairTile&, int);
template __device__ void pair_repair(PairTile&, Attention, cooperative_groups::cluster_group,
                                     int, int);
template __device__ void pair_repair(StagedPairTile&, Attention, cooperative_groups::cluster_group,
                                     int, int);
template __device__ void tmem_repair_ordered(const TmemFlashTile&, Attention, int, int, int);
template __device__ void tmem_repair_ordered(const TmemPipelineTile&, Attention, int, int, int);
template __device__ void tmem_repair_ordered(const StagedSingleTile&, Attention, int, int, int);
template __device__ void tmem_repair_ordered(const StagedPageTile&, Attention, int, int, int);
template __device__ void pair_decode_tail<true>(
    std::uint16_t*, const Mxfp8KvCache&, int, int, int, int, bool&);
template __device__ void pair_decode_tail<false>(
    std::uint16_t*, const Mxfp8KvCache&, int, int, int, int, bool&);
template __device__ std::uint64_t tmem_descriptor<true>(const void*);
template __device__ std::uint64_t tmem_descriptor<false>(const void*);
template __device__ void tmem_commit<false>(Word);

}
