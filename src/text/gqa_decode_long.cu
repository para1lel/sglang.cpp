#include "sglang/text/gqa/gqa_decode_long.cuh"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/text/gqa/tmem_ops.cuh"

#include <cuda.h>
#include <cudaTypedefs.h>
#include <type_traits>

namespace sglang::gqa_decode {
namespace {

using gqa_prefill::tmem::tmem_decode;
using gqa_prefill::tmem::tmem_swizzle;
using gqa_prefill::tmem::tmem_swizzled;

using Bits = std::uint32_t;

__device__ __forceinline__ Bits shared_address(const void* pointer) {
    return static_cast<Bits>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void barrier_init(Bits barrier, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(barrier), "r"(count) : "memory");
}

__device__ __forceinline__ void barrier_arrive(Bits barrier) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                 :: "r"(barrier) : "memory");
}

__device__ __forceinline__ void tensor_alloc(Bits destination, int columns) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 :: "r"(destination), "r"(columns) : "memory");
}

__device__ __forceinline__ void tensor_dealloc(Bits address, int columns) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(address), "r"(columns) : "memory");
}

__device__ __forceinline__ std::uint64_t smem_descriptor(const void* pointer, Bits leading) {
    constexpr auto swizzle = std::uint64_t{2} << 61;
    constexpr auto version = std::uint64_t{1} << 46;
    constexpr auto stride = std::uint64_t{1024 >> 4} << 32;
    return swizzle | version | stride | (std::uint64_t{leading} << 16) |
           (shared_address(pointer) >> 4);
}

template <int Columns, bool ValueMajor, int Rows = 128>
__host__ __device__ constexpr Bits mma_instruction() {
    constexpr Bits format = (1u << 4) | (1u << 7) | (1u << 10);
    return (Bits{Rows / 16} << 24) | (Bits{Columns / 8} << 17) |
           (Bits{ValueMajor} << 16) | format;
}

__device__ __forceinline__ void mma_shared(Bits destination, std::uint64_t lhs,
                                           std::uint64_t rhs, Bits instruction,
                                           bool accumulate) {
    asm volatile("{ .reg .pred accum;\n"
                 "setp.ne.u32 accum, %4, 0;\n"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, accum; }"
                 :: "r"(destination), "l"(lhs), "l"(rhs), "r"(instruction),
                    "r"(static_cast<Bits>(accumulate)) : "memory");
}

__device__ __forceinline__ void mma_commit(Bits barrier) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                 :: "r"(barrier) : "memory");
}

__device__ __forceinline__ void tensor_load(Bits address, float* values) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,"
                 "%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,"
                 "%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
                 : "=f"(values[0]), "=f"(values[1]), "=f"(values[2]), "=f"(values[3]),
                   "=f"(values[4]), "=f"(values[5]), "=f"(values[6]), "=f"(values[7]),
                   "=f"(values[8]), "=f"(values[9]), "=f"(values[10]), "=f"(values[11]),
                   "=f"(values[12]), "=f"(values[13]), "=f"(values[14]), "=f"(values[15]),
                   "=f"(values[16]), "=f"(values[17]), "=f"(values[18]), "=f"(values[19]),
                   "=f"(values[20]), "=f"(values[21]), "=f"(values[22]), "=f"(values[23]),
                   "=f"(values[24]), "=f"(values[25]), "=f"(values[26]), "=f"(values[27]),
                   "=f"(values[28]), "=f"(values[29]), "=f"(values[30]), "=f"(values[31])
                 : "r"(address) : "memory");
}

__device__ __forceinline__ void tensor_load_wait() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tensor_store(Bits address, const Bits* bits) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,"
                 "%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,"
                 "%25,%26,%27,%28,%29,%30,%31,%32};"
                 :: "r"(address), "r"(bits[0]), "r"(bits[1]), "r"(bits[2]), "r"(bits[3]),
                    "r"(bits[4]), "r"(bits[5]), "r"(bits[6]), "r"(bits[7]), "r"(bits[8]),
                    "r"(bits[9]), "r"(bits[10]), "r"(bits[11]), "r"(bits[12]),
                    "r"(bits[13]), "r"(bits[14]), "r"(bits[15]), "r"(bits[16]),
                    "r"(bits[17]), "r"(bits[18]), "r"(bits[19]), "r"(bits[20]),
                    "r"(bits[21]), "r"(bits[22]), "r"(bits[23]), "r"(bits[24]),
                    "r"(bits[25]), "r"(bits[26]), "r"(bits[27]), "r"(bits[28]),
                    "r"(bits[29]), "r"(bits[30]), "r"(bits[31]) : "memory");
}

__device__ __forceinline__ void tensor_store_release() {
    asm volatile("tcgen05.wait::st.sync.aligned;\n"
                 "tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void init(std::uint64_t* barrier, int count) {
    barrier_init(shared_address(barrier), count);
}

__device__ __forceinline__ void arrive(std::uint64_t* barrier) {
    barrier_arrive(shared_address(barrier));
}

__device__ __forceinline__ void wait(std::uint64_t* barrier, unsigned phase) {
    const auto address = shared_address(barrier);
    constexpr unsigned ticks = 10000000;
    asm volatile("{ .reg .pred ready;\n"
                 "WAIT: mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1, %2;\n"
                 "@!ready bra WAIT; }" :: "r"(address), "r"(phase), "r"(ticks) : "memory");
}

__device__ __forceinline__ bool elect_warp_lane() {
    unsigned elected;
    asm("{ .reg .pred leader; elect.sync _|leader, 0xffffffff;"
        " selp.u32 %0, 1, 0, leader; }" : "=r"(elected));
    return elected;
}

__device__ __forceinline__ void commit(std::uint64_t* barrier) {
    if (elect_warp_lane()) mma_commit(shared_address(barrier));
}

__device__ __forceinline__ void mma(unsigned output, std::uint64_t lhs, std::uint64_t rhs,
                                    unsigned instruction, bool accumulate) {
    if (elect_warp_lane()) mma_shared(output, lhs, rhs, instruction, accumulate);
}

constexpr int members = 32;
constexpr int key_rows_per_warp = 16;
constexpr int stripe = 128;
constexpr int group_heads = mxfp8_query_heads / mxfp8_kv_heads;
constexpr int keys = split_chunk_tokens;
constexpr int stages = 2;
constexpr int key_slots = 1;
constexpr int value_slots = stages;
constexpr int raw_tokens = keys;
constexpr int raw_stages = 2;
constexpr int columns = mxfp8_head_dim;
constexpr int copy_warps = 4;
constexpr int key_warp = 4;
constexpr int value_warp = key_warp + copy_warps;
constexpr int mma_warp = value_warp + copy_warps;
constexpr int raw_warp = mma_warp + 1;
constexpr int threads = 32 * (raw_warp + 2);
constexpr int tensor_columns = 128;
constexpr int score_columns = 2 * members;
constexpr float log2_scale = 1.44269504088896341f / 16.0f;
constexpr unsigned full = 0xffffffffu;

constexpr int raw_scale_width = raw_tokens / mxfp8_group_size * mxfp8_groups;

struct alignas(128) RawTile {
    uint4 payload[2][raw_tokens][columns / 16];
    std::uint8_t scale[2][mxfp8_group_size][raw_scale_width];
};

struct BatchMaps {
    LongTensorMaps requests[long_requests];
};

struct alignas(1024) Shared {
    std::uint16_t query[columns / tmem_swizzle][members][tmem_swizzle];
    std::uint16_t key[key_slots][columns / tmem_swizzle][keys][tmem_swizzle];
    std::uint16_t value[value_slots][columns / tmem_swizzle][keys][tmem_swizzle];
    alignas(128) std::uint16_t probability[stages][members][keys];
    alignas(128) std::uint16_t residual[stages][members][keys];
    float transposed[stages][keys][members + 1];
    RawTile raw[raw_stages];
    std::uint64_t raw_ready[2][raw_stages], raw_released[2][raw_stages];
    std::uint64_t key_ready[stages], value_ready[stages], scores[stages], probabilities[stages];
    std::uint64_t score_read[stages], corrected[stages], outputs[stages];
    unsigned tensor;
    int pages[1 + text_attention_max_tokens / max_splits / mxfp8_page_size];
    int bad_key;
    int bad_query[members];
    float history[stages][members];
};
static_assert(long_rows * group_heads <= members);
static_assert(score_columns + stages * members == tensor_columns);
static_assert(sizeof(LongBatch) + sizeof(BatchMaps) + alignof(BatchMaps) <= 32764);
static_assert(sizeof(RawTile) == 2 * raw_tokens * (columns + mxfp8_groups));

struct Online {
    float maximum;
    float mass;
    bool bad;
};

__device__ void initialize(Shared& shared) {
    if (threadIdx.x == 0) {
        shared.bad_key = text_attention_max_tokens;
        for (int operand = 0; operand < 2; ++operand) {
            for (int slot = 0; slot < raw_stages; ++slot) {
                init(&shared.raw_ready[operand][slot], 1);
                init(&shared.raw_released[operand][slot], copy_warps);
            }
        }
        for (int slot = 0; slot < stages; ++slot) {
            init(&shared.key_ready[slot], copy_warps);
            init(&shared.value_ready[slot], copy_warps);
            init(&shared.scores[slot], 1);
            init(&shared.score_read[slot], 4);
            init(&shared.probabilities[slot], 4);
            init(&shared.corrected[slot], 4);
            init(&shared.outputs[slot], 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (threadIdx.x / 32 == mma_warp) {
        tensor_alloc(shared_address(&shared.tensor), tensor_columns);
    }
}

__device__ void load_queries(Shared& shared, const LongRequest& request, int kv_head) {
    for (int index = threadIdx.x; index < members; index += threads) {
        shared.bad_query[index] = 0;
    }
    __syncthreads();
    for (int vector = threadIdx.x; vector < members * columns / 8; vector += threads) {
        const int member = vector / (columns / 8);
        const int channel = (vector % (columns / 8)) * 8;
        uint4 bits{};
        if (member < request.count * group_heads) {
            const auto* query = request.rows[member / group_heads].queries +
                                (kv_head * group_heads + member % group_heads) * columns;
            bits = *reinterpret_cast<const uint4*>(query + channel);
            const unsigned words[] = {bits.x, bits.y, bits.z, bits.w};
            bool bad = false;
            #pragma unroll
            for (const unsigned word : words) {
                const unsigned low = word & 0x7fffu;
                const unsigned high = (word >> 16) & 0x7fffu;
                bad |= low >= 0x4e80u || high >= 0x4e80u;
                bad |= (low && low < 0x0d80u) || (high && high < 0x0d80u);
            }
            if (bad) atomicExch(&shared.bad_query[member], 1);
        }
        auto* target = shared.query[channel / tmem_swizzle][member];
        *reinterpret_cast<uint4*>(target + tmem_swizzled(member, channel % tmem_swizzle)) = bits;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ bool expand(std::uint16_t* target, int swizzled,
                                       uint4 payload, unsigned scale, bool live) {
    alignas(16) uint4 decoded[2]{};
    bool bad = false;
    if (live) tmem_decode(reinterpret_cast<std::uint16_t*>(decoded), payload, scale, bad);
    *reinterpret_cast<uint4*>(target + swizzled) = decoded[0];
    *reinterpret_cast<uint4*>(target + (swizzled ^ 8)) = decoded[1];
    return bad;
}

template <int Operand>
__device__ void issue_raw(Shared& shared, const LongTensorMaps& maps, int kv_head, int kv_heads,
                          int begin, int chunks) {
    constexpr int tiles_per_page = mxfp8_page_size / raw_tokens;
    for (int tile = 0; tile < chunks; ++tile) {
        const int slot = tile % raw_stages;
        const int epoch = tile / raw_stages & 1;
        if (tile >= raw_stages) wait(&shared.raw_released[Operand][slot], epoch ^ 1);
        const int logical = begin + tile;
        const int page = shared.pages[logical / tiles_per_page - begin / tiles_per_page];
        const int page_tile = logical % tiles_per_page;
        const auto barrier = shared_address(&shared.raw_ready[Operand][slot]);
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                     :: "r"(barrier), "r"(static_cast<unsigned>(sizeof(RawTile) / 2)) : "memory");
        auto& raw = shared.raw[slot];
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global."
                     "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                     :: "r"(shared_address(raw.payload[Operand])),
                        "l"(&maps.payload[Operand]), "r"(kv_head * columns),
                        "r"(page * mxfp8_page_size + page_tile * raw_tokens), "r"(barrier)
                     : "memory");
        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global."
                     "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                     :: "r"(shared_address(raw.scale[Operand])),
                        "l"(&maps.scale[Operand]), "r"(page_tile * raw_scale_width), "r"(0),
                        "r"(page * kv_heads + kv_head), "r"(barrier) : "memory");
    }
}

template <int Operand, bool RemapRows, typename Decoded>
__device__ __forceinline__ void decode_operand(Shared& shared, Decoded& decoded,
                                                int slot, int first, int end, int producer) {
    constexpr int vector_width = 16;
    constexpr int vectors = columns / vector_width;
    const int vector = producer % vectors;
    const int channel = vector * vector_width;
    const int group = channel / tmem_swizzle;
    const int column = channel % tmem_swizzle;
    const auto& raw = shared.raw[slot];
    int first_invalid = text_attention_max_tokens;
    int first_row = producer / vectors;
    if constexpr (RemapRows) first_row ^= group & 1;
    #pragma unroll 2
    for (int index = first_row; index < keys; index += copy_warps * 32 / vectors) {
        const int raw_index = first % raw_tokens + index;
        const int scale_row = raw_index % mxfp8_group_size;
        const int scale_column = raw_index / mxfp8_group_size * mxfp8_groups +
                                 channel / mxfp8_group_size;
        const int swizzled = tmem_swizzled(index, column);
        const auto scale = raw.scale[Operand][scale_row][scale_column];
        const bool visible = first + index < end;
        const bool invalid = expand(decoded[group][index], swizzled,
                                    raw.payload[Operand][raw_index][vector], scale, visible);
        if (invalid || (Operand == 1 && visible && scale > 129)) {
            first_invalid = min(first_invalid, first + index);
        }
    }
    if (__any_sync(full, first_invalid < text_attention_max_tokens)) {
        for (int offset = 16; offset > 0; offset /= 2) {
            first_invalid = min(first_invalid, __shfl_xor_sync(full, first_invalid, offset));
        }
        if (!(producer % 32)) atomicMin(&shared.bad_key, first_invalid);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
}

template <int Operand, bool RemapRows>
__device__ void load_operands(Shared& shared, const LongRequest& request, int begin,
                              int chunks, int producer) {
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int slot = chunk % stages;
        const int raw_slot = chunk % raw_stages;
        const int raw_epoch = chunk / raw_stages & 1;
        const int first = (begin + chunk) * keys;
        const int end = request.first + request.count;
        wait(&shared.raw_ready[Operand][raw_slot], raw_epoch);
        constexpr int decoded_slots = Operand == 0 ? key_slots : value_slots;
        const int previous = chunk - decoded_slots;
        if (previous >= 0) {
            auto* consumed = Operand == 0 ? shared.scores : shared.outputs;
            wait(&consumed[previous % stages], previous / stages & 1);
        }
        if constexpr (Operand == 0) {
            decode_operand<0, RemapRows>(shared, shared.key[chunk % decoded_slots], raw_slot,
                                         first, end, producer);
        } else {
            decode_operand<1, RemapRows>(shared, shared.value[chunk % decoded_slots], raw_slot,
                                         first, end, producer);
        }
        if (!(producer % 32)) {
            arrive(Operand == 0 ? &shared.key_ready[slot] : &shared.value_ready[slot]);
            arrive(&shared.raw_released[Operand][raw_slot]);
        }
    }
}

template <bool MnMajor = false>
__device__ __forceinline__ std::uint64_t descriptor(const void* pointer) {
    return smem_descriptor(pointer, MnMajor ? 512 : 1);
}

__device__ __forceinline__ void issue_scores(Shared& shared, int chunk) {
    const int slot = chunk % stages;
    wait(&shared.key_ready[slot], chunk / stages & 1);
    const auto score = shared.tensor + score_columns + slot * members;
    #pragma unroll
    for (int dim = 0; dim < columns; dim += 16) {
        const int group = dim / tmem_swizzle;
        const int column = dim % tmem_swizzle;
        mma(score, descriptor(&shared.key[chunk % key_slots][group][0][column]),
            descriptor(&shared.query[group][0][column]),
            mma_instruction<members, false, keys>(), dim != 0);
    }
    commit(&shared.scores[slot]);
}

__device__ void issue_mmas(Shared& shared, int chunks) {
    issue_scores(shared, 0);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int slot = chunk % stages;
        const int epoch = chunk / stages & 1;
        if (chunk + 1 < chunks) issue_scores(shared, chunk + 1);
        wait(&shared.probabilities[slot], epoch);
        wait(&shared.corrected[slot], epoch);
        wait(&shared.value_ready[slot], epoch);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        #pragma unroll
        for (int token = 0; token < keys; token += 16) {
            const auto probability = descriptor(&shared.probability[slot][0][token]);
            const auto residual = descriptor(&shared.residual[slot][0][token]);
            #pragma unroll
            for (int part = 0; part < 2; ++part) {
                const auto value = descriptor<true>(
                    &shared.value[chunk % value_slots][part * stripe / tmem_swizzle][token][0]);
                const unsigned output = shared.tensor + part * members;
                constexpr unsigned a_mn_major = 1u << 15;
                constexpr auto op = mma_instruction<members, false, stripe>() | a_mn_major;
                mma(output, value, probability, op, chunk || token);
                mma(output, value, residual, op, true);
            }
        }
        commit(&shared.outputs[slot]);
    }
}

__device__ __forceinline__ void store_tensor(unsigned address, const float* values) {
    unsigned bits[members];
    #pragma unroll
    for (int member = 0; member < members; ++member) bits[member] = __float_as_uint(values[member]);
    tensor_store(address, bits);
}

__device__ void soften_chunk(Shared& shared, int slot, int visible, int member,
                             int phase, Online& state) {
    float values[keys / 4];
    float maximum = state.maximum;
    #pragma unroll
    for (int index = 0; index < keys / 4; ++index) {
        const int token = index * 4 + phase;
        const float value = shared.transposed[slot][token][member] * log2_scale;
        state.bad |= token < visible && !isfinite(value);
        values[index] = token < visible ? value : -CUDART_INF_F;
        maximum = fmaxf(maximum, values[index]);
    }
    maximum = fmaxf(maximum, __shfl_xor_sync(full, maximum, 8));
    maximum = fmaxf(maximum, __shfl_xor_sync(full, maximum, 16));
    const float delta = state.maximum == maximum ? 0.0f : state.maximum - maximum;
    const float history = exp2f(delta);
    state.bad |= isfinite(state.maximum) && delta < -126.0f;
    if (!phase) shared.history[slot][member] = history;
    float sum = 0.0f;
    const float base = isfinite(maximum) ? maximum : 0.0f;
    #pragma unroll
    for (int index = 0; index < keys / 4; ++index) {
        const int token = index * 4 + phase;
        const float relative = values[index] - base;
        const float weight = exp2f(relative);
        state.bad |= token < visible && relative < -126.0f;
        sum += weight;
        const auto high = __float2bfloat16_rn(weight);
        const auto low = __float2bfloat16_rn(weight - __bfloat162float(high));
        const int col = tmem_swizzled(member, token);
        shared.probability[slot][member][col] = __bfloat16_as_ushort(high);
        shared.residual[slot][member][col] = __bfloat16_as_ushort(low);
    }
    const int query = member % 8;
    const float sum0 = __shfl_sync(full, sum, query);
    const float sum1 = __shfl_sync(full, sum, query + 8);
    const float sum2 = __shfl_sync(full, sum, query + 16);
    const float sum3 = __shfl_sync(full, sum, query + 24);
    state.mass = fmaf(state.mass, history, (sum0 + sum2) + (sum1 + sum3));
    state.maximum = maximum;
}

__device__ void correct(Shared& shared, int chunk, unsigned tensor_row) {
    const int slot = chunk % stages;
    const int previous = chunk - 1;
    wait(&shared.probabilities[slot], chunk / stages & 1);
    wait(&shared.outputs[previous % stages], previous / stages & 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const bool changed = __any_sync(full, shared.history[slot][threadIdx.x % 32] != 1.0f);
    if (!changed) return;
    #pragma unroll 1
    for (int part = 0; part < 2; ++part) {
        float values[members];
        tensor_load(tensor_row + part * members, values);
        tensor_load_wait();
        #pragma unroll
        for (int member = 0; member < members; ++member) {
            values[member] *= shared.history[slot][member];
        }
        store_tensor(tensor_row + part * members, values);
    }
    tensor_store_release();
}

template <int Splits>
__device__ void softmax(Shared& shared, const LongRequest& request, int begin, int chunks,
                        int warp, int split, int kv_head) {
    const int lane = threadIdx.x % 32;
    const int member = warp * 8 + (lane % 8);
    const int position = member < request.count * group_heads
        ? request.first + member / group_heads : -1;
    const unsigned tensor_row = shared.tensor + ((warp * 32) << 16);
    Online state{-CUDART_INF_F, 0.0f, false};
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int slot = chunk % stages;
        const int epoch = chunk / stages & 1;
        wait(&shared.scores[slot], epoch);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        float row[members];
        tensor_load(tensor_row + score_columns + slot * members, row);
        tensor_load_wait();
        if (lane < key_rows_per_warp) {
            const int token = warp * key_rows_per_warp + lane;
            #pragma unroll
            for (int query = 0; query < members; ++query) {
                shared.transposed[slot][token][query] = row[query];
            }
        }
        __syncwarp();
        if (!lane) arrive(&shared.score_read[slot]);
        wait(&shared.score_read[slot], epoch);
        soften_chunk(shared, slot, position + 1 - (begin + chunk) * keys, member, lane / 8, state);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (!lane) arrive(&shared.probabilities[slot]);
        if (chunk) correct(shared, chunk, tensor_row);
        __syncwarp();
        if (!lane) arrive(&shared.corrected[slot]);
    }
    const int final = chunks - 1;
    wait(&shared.outputs[final % stages], final / stages & 1);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    int bad = state.bad;
    bad |= __shfl_xor_sync(full, bad, 8);
    bad |= __shfl_xor_sync(full, bad, 16);
    if (lane < 8 && position >= 0) {
        auto& out = partials<Splits>(request.rows[member / group_heads].scratch,
                                     kv_head * group_heads + member % group_heads)[split];
        out.mass = state.mass;
        const int repair = (bad ? 1 : 0) | (shared.bad_query[member] ? 2 : 0) |
                           (shared.bad_key <= position ? 4 : 0);
        out.maximum = repair ? __int_as_float(0x7fc00000 | repair) : state.maximum;
    }
    #pragma unroll 1
    for (int part = 0; part < 2; ++part) {
        float values[members];
        tensor_load(tensor_row + part * members, values);
        tensor_load_wait();
        const int channel = part * stripe + warp * 32 + lane;
        #pragma unroll 1
        for (int query = 0; query < request.count * group_heads; ++query) {
            auto& out = partials<Splits>(request.rows[query / group_heads].scratch,
                                         kv_head * group_heads + query % group_heads)[split];
            out.value[channel] = values[query];
        }
    }
}

template <int Splits, bool RemapRows>
__device__ __forceinline__ void decode_long_body(Shared& shared, const LongRequest& request,
                                                 const LongTensorMaps& maps) {
    const int last_position = request.first + request.count - 1;
    const int split = blockIdx.x;
    const int splits = Splits == max_splits ? split_count(last_position) : Splits;
    if (split >= splits) return;
    const int total = request.first / keys + 1;
    const int first = split * total / splits;
    const int last = split + 1 == splits ? last_position / keys + 1 : (split + 1) * total / splits;
    const int first_page = first * keys / mxfp8_page_size;
    const int last_page = (last - 1) * keys / mxfp8_page_size;
    for (int page = first_page + threadIdx.x; page <= last_page; page += threads) {
        shared.pages[page - first_page] = request.pages.device_pages[page];
    }
    const int warp = threadIdx.x / 32;
    const int kv_head = blockIdx.y;
    initialize(shared);
    load_queries(shared, request, kv_head);
    __syncthreads();
    if (warp < 4) {
        softmax<Splits>(shared, request, first, last - first, warp, split, kv_head);
    } else if (warp == mma_warp) {
        issue_mmas(shared, last - first);
    } else if (warp >= raw_warp) {
        if (threadIdx.x % 32 == 0) {
            if (warp == raw_warp) {
                issue_raw<0>(shared, maps, kv_head, request.cache.kv_heads, first, last - first);
            } else {
                issue_raw<1>(shared, maps, kv_head, request.cache.kv_heads, first, last - first);
            }
        }
    } else if (warp < value_warp) {
        load_operands<0, RemapRows>(shared, request, first, last - first,
                                    threadIdx.x - key_warp * 32);
    } else {
        load_operands<1, RemapRows>(shared, request, first, last - first,
                                    threadIdx.x - value_warp * 32);
    }
    __syncthreads();
    if (warp == mma_warp) tensor_dealloc(shared.tensor, tensor_columns);
}

__device__ __forceinline__ Shared& long_decode_shared() {
    alignas(1024) extern __shared__ unsigned char storage[];
    return *reinterpret_cast<Shared*>(storage);
}

template <int Splits, bool RemapRows = false>
__global__ __launch_bounds__(threads, 1) void decode_long(const __grid_constant__ LongBatch batch,
                                                          const __grid_constant__ BatchMaps maps) {
    decode_long_body<Splits, RemapRows>(long_decode_shared(), batch.requests[blockIdx.z],
                                        maps.requests[blockIdx.z]);
}

template <int Splits, bool RemapRows = false>
__global__
__launch_bounds__(threads, 1) void decode_bound_long(const __grid_constant__ BoundLongBatch batch) {
    const auto& binding = *batch.requests[blockIdx.z];
    const int warp = static_cast<int>(threadIdx.x) / 32;
    if ((warp == raw_warp || warp == raw_warp + 1) && threadIdx.x % 32 == 0) {
        const int operand = warp - raw_warp;
        asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(
                         &binding.maps.payload[operand])
                     : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(
                         &binding.maps.scale[operand])
                     : "memory");
    }
    decode_long_body<Splits, RemapRows>(long_decode_shared(), binding.request, binding.maps);
}

std::optional<cudaError_t> encode_maps(const LongRequest* batch, int requests,
                                       LongTensorMaps* maps) {
    void* symbol = nullptr;
    cudaDriverEntryPointQueryResult query{};
    const auto status = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &symbol, 12000, cudaEnableDefault, &query);
    if (status != cudaSuccess) {
        const auto failure = long_decode_status(status);
        if (failure == cudaErrorNotSupported || failure == cudaErrorSymbolNotFound) {
            return std::nullopt;
        }
        return failure;
    }
    if (!symbol || query != cudaDriverEntryPointSuccess) return std::nullopt;
    const auto encode = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(symbol);
    constexpr cuuint64_t scale_width = mxfp8_page_size / mxfp8_group_size * mxfp8_groups;
    constexpr cuuint64_t scale_strides[]{scale_width, scale_width * mxfp8_group_size};
    constexpr cuuint32_t payload_box[]{columns, raw_tokens};
    constexpr cuuint32_t scale_box[]{raw_scale_width, mxfp8_group_size, 1};
    constexpr cuuint32_t elements[]{1, 1, 1};
    for (int request = 0; request < requests; ++request) {
        const auto& cache = batch[request].cache;
        const auto pages = static_cast<cuuint64_t>(cache.page_count);
        const cuuint64_t payload_width =
            static_cast<cuuint64_t>(cache.kv_heads) * columns;
        const cuuint64_t payload_strides[]{payload_width};
        const cuuint64_t payload_shape[]{payload_width, pages * mxfp8_page_size};
        const cuuint64_t scale_shape[]{scale_width, mxfp8_group_size,
                                       pages * cache.kv_heads};
        void* payload[]{cache.key_payload, cache.value_payload};
        void* scale[]{cache.key_scales, cache.value_scales};
        auto& map = maps[request];
        for (int operand = 0; operand < 2; ++operand) {
            auto result = encode(&map.payload[operand], CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
                                 payload[operand], payload_shape, payload_strides, payload_box,
                                 elements, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                 CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (result == CUDA_SUCCESS) {
                result = encode(&map.scale[operand], CU_TENSOR_MAP_DATA_TYPE_UINT8, 3,
                                scale[operand], scale_shape, scale_strides, scale_box, elements,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            }
            if (result == CUDA_ERROR_INVALID_VALUE || result == CUDA_ERROR_NOT_SUPPORTED) {
                return std::nullopt;
            }
            if (result != CUDA_SUCCESS) {
                const auto status = decoder::CudaError::from_driver(result);
                return long_decode_status(decoder::CudaError::terminal(status)
                    ? status : cudaErrorUnknown);
            }
        }
    }
    return cudaSuccess;
}

}

cudaError_t long_decode_status(cudaError_t status) {
    const auto pending = cudaGetLastError();
    if (status == cudaSuccess) return pending;
    return !decoder::CudaError::terminal(status) && decoder::CudaError::terminal(pending)
        ? pending : status;
}

LongDecodeShape long_decode_shape(const LongBatch& batch, int requests) {
    bool rank_wide = requests == long_rank_wide_requests;
    bool deep_rank = requests == decoder::wide_mtp_lanes;
    for (int index = 0; index < requests; ++index) {
        const auto& request = batch.requests[index];
        if (request.count < 1 || request.count > long_rows)
            return {};
        rank_wide &= request.count == long_rows &&
                     request.cache.kv_heads == gqa_rank_geometry.kv_heads &&
                     request.first >= long_min_position && request.first < long_rank_wide_end;
        deep_rank &= request.count == long_rows &&
                     request.cache.kv_heads == gqa_rank_geometry.kv_heads &&
                     request.first >= 1024 * mxfp8_page_size;
    }
    if (rank_wide)
        return {long_rank_wide_splits, true};
    if (requests == 1 && batch.requests[0].cache.kv_heads == gqa_rank_geometry.kv_heads &&
        batch.requests[0].first >= long_min_position)
        return {long_single_rank_splits, false};
    return {max_splits, deep_rank};
}

namespace {
std::optional<cudaError_t> long_decode_supported() {
    if (const auto pending = cudaPeekAtLastError(); pending != cudaSuccess) return pending;
    int device;
    auto status = long_decode_status(cudaGetDevice(&device));
    if (status != cudaSuccess) return status;
    cudaDeviceProp prop;
    status = long_decode_status(cudaGetDeviceProperties(&prop, device));
    if (status != cudaSuccess) return status;
    if (prop.major != 10 || prop.minor != 0 || prop.sharedMemPerBlockOptin < sizeof(Shared)) {
        return std::nullopt;
    }
    return cudaSuccess;
}

template <auto Regular, auto Remapped, auto Rank, auto Wide, typename Batch>
cudaError_t submit_long_decode(const Batch& batch, int requests, int kv_heads,
                               LongDecodeShape shape, void** arguments, cudaStream_t stream) {
    auto kernel = shape.remap_rows ? Remapped : Regular;
    if (shape.splits == long_rank_wide_splits) {
        kernel = Wide;
    } else if (shape.splits == long_single_rank_splits) {
        kernel = Rank;
    }
    auto status = long_decode_status(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared)));
    if (status != cudaSuccess) return status;
    status = long_decode_status(cudaLaunchKernel(kernel, dim3(shape.splits, kv_heads, requests),
                                                 threads, arguments, sizeof(Shared), stream));
    if (status != cudaSuccess) return status;
    if constexpr (std::is_same_v<Batch, LongBatch>) {
        return long_decode_status(combine_long_decode(batch, requests, shape.splits, stream));
    } else {
        return long_decode_status(
            combine_long_decode(batch, requests, kv_heads, shape.splits, stream));
    }
}
}

std::optional<cudaError_t> prepare_long_decode_binding(LongGraphBinding& binding) {
    const auto supported = long_decode_supported();
    if (!supported || *supported != cudaSuccess)
        return supported;
    return encode_maps(&binding.request, 1, &binding.maps);
}

std::optional<cudaError_t> launch_long_decode(LongBatch& batch, int requests, cudaStream_t stream) {
    if (requests < 1 || requests > long_requests)
        return std::nullopt;
    const auto shape = long_decode_shape(batch, requests);
    if (shape.splits == 0)
        return std::nullopt;
    const auto supported = long_decode_supported();
    if (!supported || *supported != cudaSuccess)
        return supported;
    BatchMaps maps{};
    const auto encoded = encode_maps(batch.requests, requests, maps.requests);
    if (const auto pending = cudaPeekAtLastError(); pending != cudaSuccess)
        return pending;
    if (!encoded || *encoded != cudaSuccess)
        return encoded;
    void* arguments[] = {&batch, &maps};
    return submit_long_decode<decode_long<max_splits>, decode_long<max_splits, true>,
                              decode_long<long_single_rank_splits>,
                              decode_long<long_rank_wide_splits, true>>(
        batch, requests, batch.requests[0].cache.kv_heads, shape, arguments, stream);
}

std::optional<cudaError_t> launch_long_decode(BoundLongBatch& batch, int requests, int kv_heads,
                                              LongDecodeShape shape, cudaStream_t stream) {
    if (requests < 1 || requests > long_requests || shape.splits == 0)
        return std::nullopt;
    void* arguments[] = {&batch};
    return submit_long_decode<decode_bound_long<max_splits>, decode_bound_long<max_splits, true>,
                              decode_bound_long<long_single_rank_splits>,
                              decode_bound_long<long_rank_wide_splits, true>>(
        batch, requests, kv_heads, shape, arguments, stream);
}
}
