#include "sglang/text/prefill_attn_sm100.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/text/mxfp8_gqa_prefill_common.cuh"
#include "sglang/text/prefill_attn_tcgen05.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <queue>
#include <string_view>
#include <vector>

namespace sglang {
namespace {

using namespace prefill_attn;

constexpr int members = 128;
constexpr int tile_keys = mxfp8_page_size;
constexpr int slot_count = 8;
constexpr int slot_bytes = 16384;
constexpr int chunk_bytes = members * 128;
constexpr int softmax_warps = 8;
constexpr int convert_threads = 256;
constexpr int mma_warp = softmax_warps + convert_threads / 32;
constexpr int attn_threads = (mma_warp + 1) * 32;
constexpr int load_depth = 4;
constexpr int score_column = 256;
constexpr int tensor_columns = 512;
constexpr float score_scale = softmax_scale * 1.4426950408889634f;
constexpr std::size_t max_partial_bytes = std::size_t{128} << 20;
constexpr int max_tmem_rows = 8192;

static_assert(tile_keys == 128 && channels == 256);

struct alignas(1024) AttnShared {
    std::uint8_t query[4][chunk_bytes];
    std::uint8_t slot[slot_count][slot_bytes];
    float row_max[2][2][members];
    float row_mass[2][members];
    unsigned long long slot_full[slot_count];
    unsigned long long slot_empty[slot_count];
    unsigned long long score_full[2];
    unsigned long long prob_ready[2];
    unsigned long long output_done;
    Bits tensor;
};

constexpr std::size_t attn_smem_bytes = sizeof(AttnShared) + 1024;

struct AttnPlan {
    Attention request;
    int rows;
    int ctas_per_kv;
    int splits;
    float* partial;
    float2* stats;
    std::uint8_t* bad_query;
    int* bad_key;
};

struct Unit {
    int kv;
    int first_member;
    int first_tile;
    int tiles;
    int limit;
};

__host__ __device__ inline int cta_limit(int position, int rows, int local) {
    const int end = (local + 1) * members;
    const int last_member = (rows * ratio < end ? rows * ratio : end) - 1;
    return position + last_member / ratio;
}

__host__ __device__ inline int unit_index(int ctas_per_kv, int splits, int local, int kv,
                                          int split) {
    return ((ctas_per_kv - 1 - local) * mxfp8_kv_heads + kv) * splits + split;
}

__host__ __device__ inline int split_tile(int tiles, int split, int splits) {
    return static_cast<int>(static_cast<long long>(tiles) * split / splits);
}

__device__ Unit make_unit(const AttnPlan& plan) {
    const int split = blockIdx.x % plan.splits;
    const int order = blockIdx.x / plan.splits;
    const int local = plan.ctas_per_kv - 1 - order / mxfp8_kv_heads;
    Unit unit;
    unit.kv = order % mxfp8_kv_heads;
    unit.first_member = local * members;
    unit.limit = cta_limit(plan.request.position, plan.rows, local);
    const int tiles = unit.limit / tile_keys + 1;
    unit.first_tile = split_tile(tiles, split, plan.splits);
    unit.tiles = split_tile(tiles, split + 1, plan.splits) - unit.first_tile;
    return unit;
}

__device__ __forceinline__ int swizzled(int row, int unit) {
    return row * 128 + ((unit ^ (row & 7)) * 16);
}

__device__ void stage_query(AttnShared& shared, const AttnPlan& plan, const Unit& unit) {
    const int lane = threadIdx.x % 32;
    const int chunk = lane / 8;
    for (int member = threadIdx.x / 32; member < members; member += attn_threads / 32) {
        const int flat = unit.first_member + member;
        uint4 bits = make_uint4(0, 0, 0, 0);
        bool bad = false;
        if (flat < plan.rows * ratio) {
            const int head = unit.kv * ratio + flat % ratio;
            const auto* source = reinterpret_cast<const uint4*>(
                plan.request.queries + row_offset(flat / ratio, head));
            bits = source[lane];
            Word words[4] = {bits.x, bits.y, bits.z, bits.w};
            #pragma unroll
            for (int index = 0; index < 4; ++index) {
                #pragma unroll
                for (int part = 0; part < 2; ++part) {
                    const Word magnitude = (words[index] >> (16 * part)) & 0x7fffu;
                    const bool invalid = magnitude > 0x4e80u || (magnitude && magnitude < 0x80u);
                    if (invalid) {
                        words[index] &= ~(0xffffu << (16 * part));
                    }
                    bad |= invalid;
                }
            }
            bits = make_uint4(words[0], words[1], words[2], words[3]);
            const bool any = __any_sync(warp_mask, bad);
            if (lane == 0 && unit.first_tile == 0) {
                plan.bad_query[static_cast<std::size_t>(flat / ratio) * mxfp8_query_heads + head] =
                    any;
            }
        }
        *reinterpret_cast<uint4*>(&shared.query[chunk][swizzled(member, lane % 8)]) = bits;
    }
}

struct Payload {
    uint4 low;
    uint4 high;
    int scale;
    int position;
};

struct Item {
    bool value;
    int tile;
    int part;
};

__device__ __forceinline__ Item item_of(int sequence, int tiles) {
    const int group = sequence / 4;
    Item item{false, 0, sequence % 4};
    if (group == 2 * tiles - 1) {
        item.value = true;
        item.tile = tiles - 1;
    } else if (group > 0) {
        item.value = group % 2 == 0;
        item.tile = (group + 1) / 2 - item.value;
    }
    return item;
}

__device__ __forceinline__ void slot_coordinates(const Item& item, int thread, int& key,
                                                 int& group) {
    if (item.value) {
        key = item.part * 32 + thread / mxfp8_groups;
        group = thread % mxfp8_groups;
    } else {
        key = thread / 2;
        group = item.part * 2 + thread % 2;
    }
}

__device__ void issue_load(Payload& payload, const AttnPlan& plan, const Unit& unit,
                           int sequence, int thread) {
    const Item item = item_of(sequence, unit.tiles);
    int key = 0;
    int group = 0;
    slot_coordinates(item, thread, key, group);
    const int tile = unit.first_tile + item.tile;
    payload.position = tile * tile_keys + key;
    if (payload.position > unit.limit) {
        return;
    }
    const int page = __ldg(plan.request.pages + tile);
    const auto& cache = plan.request.cache;
    const auto* bytes = item.value ? cache.value_payload : cache.key_payload;
    const auto* scales = item.value ? cache.value_scales : cache.key_scales;
    const auto* source = bytes + payload_offset(page, key, unit.kv) + group * mxfp8_group_size;
    asm("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=r"(payload.low.x), "=r"(payload.low.y), "=r"(payload.low.z), "=r"(payload.low.w),
          "=r"(payload.high.x), "=r"(payload.high.y), "=r"(payload.high.z), "=r"(payload.high.w)
        : "l"(source));
    payload.scale = __ldg(scales + scale_offset(page, key, unit.kv) + group);
}

__device__ __forceinline__ Word fp8_pair(Word packed, Word gain) {
    Word result;
    asm("{ .reg .b32 pair;\n"
        "cvt.rn.bf16x2.e4m3x2 pair, %1;\n"
        "mul.rn.bf16x2 %0, pair, %2; }"
        : "=r"(result) : "h"(static_cast<short>(packed)), "r"(gain));
    return result;
}

__device__ __forceinline__ Word fp8_nan(Word packed) {
    return ((packed & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u;
}

__device__ __forceinline__ void decode16(uint4 payload, int scale, bool& bad, uint4* output) {
    const Word nan = fp8_nan(payload.x) | fp8_nan(payload.y) | fp8_nan(payload.z) |
                     fp8_nan(payload.w);
    if (scale < 15 || scale > 148 || nan) {
        alignas(16) std::uint16_t values[16];
        flash_store_vector<false>(values, payload, scale, bad);
        output[0] = reinterpret_cast<const uint4*>(values)[0];
        output[1] = reinterpret_cast<const uint4*>(values)[1];
        return;
    }
    const Word gain = static_cast<Word>(scale) * 0x00800080u;
    output[0] = make_uint4(fp8_pair(payload.x, gain), fp8_pair(payload.x >> 16, gain),
                           fp8_pair(payload.y, gain), fp8_pair(payload.y >> 16, gain));
    output[1] = make_uint4(fp8_pair(payload.z, gain), fp8_pair(payload.z >> 16, gain),
                           fp8_pair(payload.w, gain), fp8_pair(payload.w >> 16, gain));
}

__device__ void write_slot(std::uint8_t* slot, const Payload& payload, const AttnPlan& plan,
                           const Unit& unit, int sequence, int thread) {
    const Item item = item_of(sequence, unit.tiles);
    uint4 values[4] = {};
    if (payload.position <= unit.limit) {
        bool bad = false;
        decode16(payload.low, payload.scale, bad, values);
        decode16(payload.high, payload.scale, bad, values + 2);
        if (bad) {
            atomicMin(plan.bad_key + unit.kv, payload.position);
        }
    }
    std::uint8_t* base = slot;
    int row = thread / 2;
    int first = (thread % 2) * 4;
    if (item.value) {
        const int group = thread % mxfp8_groups;
        row = thread / mxfp8_groups;
        first = (group % 2) * 4;
        base += (group / 2) * 32 * 128;
    }
    #pragma unroll
    for (int index = 0; index < 4; ++index) {
        *reinterpret_cast<uint4*>(base + swizzled(row, first + index)) = values[index];
    }
}

__device__ void convert_loop(AttnShared& shared, const AttnPlan& plan, const Unit& unit) {
    const int thread = threadIdx.x - softmax_warps * 32;
    const int total = 8 * unit.tiles;
    Payload ring[load_depth];
    #pragma unroll
    for (int index = 0; index < load_depth; ++index) {
        if (index < total) {
            issue_load(ring[index], plan, unit, index, thread);
        }
    }
    for (int base = 0; base < total; base += load_depth) {
        #pragma unroll
        for (int index = 0; index < load_depth; ++index) {
            const int sequence = base + index;
            if (sequence >= total) {
                break;
            }
            const int slot = sequence % slot_count;
            barrier_wait(shared_address(&shared.slot_empty[slot]),
                         ((sequence / slot_count) & 1) ^ 1);
            write_slot(shared.slot[slot], ring[index], plan, unit, sequence, thread);
            proxy_fence();
            __syncwarp();
            if (threadIdx.x % 32 == 0) {
                barrier_arrive(shared_address(&shared.slot_full[slot]));
            }
            if (sequence + load_depth < total) {
                issue_load(ring[index], plan, unit, sequence + load_depth, thread);
            }
        }
    }
}

struct MmaState {
    AttnShared& shared;
    Bits tensor;
    int sequence;

    __device__ int acquire_slot() {
        const int slot = sequence % slot_count;
        tensor_wait(shared_address(&shared.slot_full[slot]), (sequence / slot_count) & 1);
        return slot;
    }

    __device__ void release_slot(int slot) {
        mma_commit(shared_address(&shared.slot_empty[slot]));
        ++sequence;
    }

    __device__ void scores(int tile) {
        constexpr Bits instruction = mma_instruction<tile_keys, false>();
        const Bits destination = tensor + score_column + (tile & 1) * tile_keys;
        for (int chunk = 0; chunk < 4; ++chunk) {
            const int slot = acquire_slot();
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                const auto lhs = smem_descriptor(shared.query[chunk] + step * 32, 1);
                const auto rhs = smem_descriptor(shared.slot[slot] + step * 32, 1);
                mma_shared(destination, lhs, rhs, instruction, (chunk | step) != 0);
            }
            release_slot(slot);
        }
        mma_commit(shared_address(&shared.score_full[tile & 1]));
    }

    __device__ void values(int tile) {
        constexpr Bits instruction = mma_instruction<channels, true>();
        constexpr Bits value_chunk = 32 * 128 >> 4;
        const Bits probabilities = tensor + score_column + (tile & 1) * tile_keys;
        tensor_wait(shared_address(&shared.prob_ready[tile & 1]), (tile >> 1) & 1);
        for (int part = 0; part < 4; ++part) {
            const int slot = acquire_slot();
            #pragma unroll
            for (int step = 0; step < 2; ++step) {
                const int keys = part * 2 + step;
                const Bits lhs = probabilities + (keys / 4) * 64 + (keys % 4) * 8;
                const auto rhs = smem_descriptor(shared.slot[slot] + step * 16 * 128, value_chunk);
                mma_tensor(tensor, lhs, rhs, instruction, (tile | part | step) != 0);
            }
            release_slot(slot);
        }
        mma_commit(shared_address(&shared.output_done));
    }
};

__device__ void mma_loop(AttnShared& shared, const Unit& unit, Bits tensor) {
    MmaState state{shared, tensor, 0};
    state.scores(0);
    if (unit.tiles > 1) {
        state.scores(1);
    }
    for (int tile = 0; tile < unit.tiles; ++tile) {
        state.values(tile);
        if (tile + 2 < unit.tiles) {
            tensor_wait(shared_address(&shared.output_done), tile & 1);
            state.scores(tile + 2);
        }
    }
}

struct SoftmaxState {
    float used;
    float mass;
};

__device__ void rescale_output(Bits address, float factor) {
    #pragma unroll 1
    for (int chunk = 0; chunk < 4; ++chunk) {
        float values[32];
        tensor_load(address + chunk * 32, values);
        tensor_load_wait();
        Bits bits[32];
        #pragma unroll
        for (int index = 0; index < 32; ++index) {
            bits[index] = __float_as_uint(values[index] * factor);
        }
        tensor_store(address + chunk * 32, bits);
    }
}

__device__ float masked_scores(float* values, Bits address, int first_key, int limit) {
    tensor_load(address, values);
    tensor_load(address + 32, values + 32);
    tensor_load_wait();
    float maximum = -CUDART_INF_F;
    #pragma unroll
    for (int index = 0; index < 64; ++index) {
        const float value = first_key + index <= limit ? values[index] * score_scale :
                                                         -CUDART_INF_F;
        values[index] = value;
        maximum = fmaxf(maximum, value);
    }
    return maximum;
}

__device__ float store_probabilities(const float* values, Bits address, float base) {
    Bits packed[32];
    float sum = 0.0f;
    #pragma unroll
    for (int index = 0; index < 32; ++index) {
        const float low = fast_exp2(values[2 * index] - base);
        const float high = fast_exp2(values[2 * index + 1] - base);
        sum += low + high;
        packed[index] = pack_bf16(low, high);
    }
    tensor_store(address, packed);
    return sum;
}

__device__ void store_output(const AttnPlan& plan, const Unit& unit, Bits address,
                             int member, int half, float used, float mass) {
    const int flat = unit.first_member + member;
    const bool valid = flat < plan.rows * ratio;
    const float inverse = mass > 0.0f ? 1.0f / mass : 0.0f;
    const std::size_t part = static_cast<std::size_t>(blockIdx.x) * members + member;
    #pragma unroll 1
    for (int chunk = 0; chunk < 4; ++chunk) {
        float values[32];
        tensor_load(address + chunk * 32, values);
        tensor_load_wait();
        const int channel = half * 128 + chunk * 32;
        if (plan.splits > 1) {
            auto* destination = reinterpret_cast<float4*>(plan.partial + part * channels + channel);
            #pragma unroll
            for (int index = 0; index < 8; ++index) {
                destination[index] = make_float4(values[4 * index], values[4 * index + 1],
                                                 values[4 * index + 2], values[4 * index + 3]);
            }
        } else if (valid) {
            const int head = unit.kv * ratio + flat % ratio;
            auto* destination = reinterpret_cast<uint4*>(
                plan.request.output + row_offset(flat / ratio, head) + channel);
            #pragma unroll
            for (int index = 0; index < 4; ++index) {
                const float* source = values + 8 * index;
                destination[index] = make_uint4(
                    pack_bf16(source[0] * inverse, source[1] * inverse),
                    pack_bf16(source[2] * inverse, source[3] * inverse),
                    pack_bf16(source[4] * inverse, source[5] * inverse),
                    pack_bf16(source[6] * inverse, source[7] * inverse));
            }
        }
    }
    if (plan.splits > 1 && half == 0) {
        plan.stats[part] = make_float2(used, mass);
    }
}

__device__ void softmax_loop(AttnShared& shared, const AttnPlan& plan, const Unit& unit,
                             Bits tensor) {
    const int warp = threadIdx.x / 32;
    const int quarter = warp % 4;
    const int half = warp / 4;
    const int member = quarter * 32 + threadIdx.x % 32;
    const int flat = unit.first_member + member;
    const int limit = flat < plan.rows * ratio ? plan.request.position + flat / ratio : -1;
    const Bits lanes = tensor + (static_cast<Bits>(quarter * 32) << 16);
    SoftmaxState state{-CUDART_INF_F, 0.0f};
    for (int tile = 0; tile < unit.tiles; ++tile) {
        const int buffer = tile & 1;
        tensor_wait(shared_address(&shared.score_full[buffer]), (tile >> 1) & 1);
        const Bits scores = lanes + score_column + buffer * tile_keys + half * 64;
        float values[64];
        const int first_key = (unit.first_tile + tile) * tile_keys + half * 64;
        float maximum = masked_scores(values, scores, first_key, limit);
        shared.row_max[buffer][half][member] = maximum;
        named_sync(1 + quarter, 64);
        maximum = fmaxf(maximum, shared.row_max[buffer][1 - half][member]);
        float factor = 1.0f;
        if (maximum > state.used) {
            factor = fast_exp2(state.used - maximum);
            state.used = maximum;
        }
        const float base = state.used == -CUDART_INF_F ? 0.0f : state.used;
        state.mass = state.mass * factor + store_probabilities(values, scores, base);
        if (tile > 0) {
            tensor_wait(shared_address(&shared.output_done), (tile - 1) & 1);
            if (__any_sync(warp_mask, factor != 1.0f)) {
                rescale_output(lanes + half * 128, factor);
            }
        }
        tensor_store_release();
        __syncwarp();
        if (threadIdx.x % 32 == 0) {
            barrier_arrive(shared_address(&shared.prob_ready[buffer]));
        }
    }
    tensor_wait(shared_address(&shared.output_done), (unit.tiles - 1) & 1);
    shared.row_mass[half][member] = state.mass;
    named_sync(1 + quarter, 64);
    const float mass = state.mass + shared.row_mass[1 - half][member];
    store_output(plan, unit, lanes + half * 128, member, half, state.used, mass);
}

__device__ void init_barriers(AttnShared& shared) {
    for (int slot = 0; slot < slot_count; ++slot) {
        barrier_init(shared_address(&shared.slot_full[slot]), convert_threads / 32);
        barrier_init(shared_address(&shared.slot_empty[slot]), 1);
    }
    for (int buffer = 0; buffer < 2; ++buffer) {
        barrier_init(shared_address(&shared.score_full[buffer]), 1);
        barrier_init(shared_address(&shared.prob_ready[buffer]), softmax_warps);
    }
    barrier_init(shared_address(&shared.output_done), 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__global__ void __launch_bounds__(attn_threads, 1)
prefill_attn_kernel(const __grid_constant__ AttnPlan plan) {
    extern __shared__ std::uint8_t raw[];
    auto& shared = *reinterpret_cast<AttnShared*>(
        (reinterpret_cast<std::uintptr_t>(raw) + 1023) & ~std::uintptr_t{1023});
    const Unit unit = make_unit(plan);
    const int warp = threadIdx.x / 32;
    if (unit.tiles == 0) {
        if (threadIdx.x < members) {
            plan.stats[static_cast<std::size_t>(blockIdx.x) * members + threadIdx.x] =
                make_float2(-CUDART_INF_F, 0.0f);
        }
        return;
    }
    if (warp == 0) {
        tensor_alloc(shared_address(&shared.tensor), tensor_columns);
    }
    if (threadIdx.x == 32) {
        init_barriers(shared);
    }
    stage_query(shared, plan, unit);
    proxy_fence();
    tensor_before_sync();
    __syncthreads();
    tensor_after_sync();
    const Bits tensor = shared.tensor;
    if (warp < softmax_warps) {
        softmax_loop(shared, plan, unit, tensor);
    } else if (warp < mma_warp) {
        convert_loop(shared, plan, unit);
    } else if (threadIdx.x % 32 == 0) {
        mma_loop(shared, unit, tensor);
    }
    tensor_before_sync();
    __syncthreads();
    if (warp == 0) {
        tensor_after_sync();
        tensor_dealloc(tensor, tensor_columns);
    }
}

__device__ void merge_head(const AttnPlan& plan, int token, int head) {
    const int flat = token * ratio + head % ratio;
    const int local = flat / members;
    const int member = flat % members;
    const int kv = head / ratio;
    float maximum = -CUDART_INF_F;
    for (int split = 0; split < plan.splits; ++split) {
        const int unit = unit_index(plan.ctas_per_kv, plan.splits, local, kv, split);
        const float2 stats = plan.stats[static_cast<std::size_t>(unit) * members + member];
        maximum = stats.y > 0.0f ? fmaxf(maximum, stats.x) : maximum;
    }
    const int channel = (threadIdx.x % (channels / ordered_heads)) * ordered_heads;
    float2 numerator = make_float2(0.0f, 0.0f);
    float denominator = 0.0f;
    for (int split = 0; split < plan.splits; ++split) {
        const int unit = unit_index(plan.ctas_per_kv, plan.splits, local, kv, split);
        const auto part = static_cast<std::size_t>(unit) * members + member;
        const float2 stats = plan.stats[part];
        if (stats.y > 0.0f) {
            const float weight = exp2f(stats.x - maximum);
            const auto* partial = reinterpret_cast<const float2*>(
                plan.partial + part * channels + channel);
            const float2 values = *partial;
            numerator.x += weight * values.x;
            numerator.y += weight * values.y;
            denominator += weight * stats.y;
        }
    }
    auto* output = reinterpret_cast<__nv_bfloat162*>(
        plan.request.output + row_offset(token, head) + channel);
    *output = denominator > 0.0f ?
        __floats2bfloat162_rn(numerator.x / denominator, numerator.y / denominator) :
        __floats2bfloat162_rn(0.0f, 0.0f);
}

__global__ void __launch_bounds__(channels)
prefill_attn_finish(const __grid_constant__ AttnPlan plan) {
    __shared__ OrderedScratch scratch;
    const int token = blockIdx.x;
    const int head = blockIdx.y * ordered_heads;
    const auto flags = static_cast<std::size_t>(token) * mxfp8_query_heads + head;
    const bool repair = plan.bad_query[flags] || plan.bad_query[flags + 1] ||
                        plan.request.position + token >= plan.bad_key[head / ratio];
    if (repair) {
        protected_attention(scratch, plan.request, token, head);
    } else if (plan.splits > 1) {
        merge_head(plan, token, head + threadIdx.x / (channels / ordered_heads));
    }
}

int simulated_makespan(int position, int rows, int ctas_per_kv, int splits, int sms) {
    constexpr int unit_overhead = 4;
    std::priority_queue<long long, std::vector<long long>, std::greater<>> finish;
    for (int sm = 0; sm < sms; ++sm) {
        finish.push(0);
    }
    long long makespan = 0;
    for (int order = 0; order < ctas_per_kv * mxfp8_kv_heads; ++order) {
        const int local = ctas_per_kv - 1 - order / mxfp8_kv_heads;
        const int tiles = cta_limit(position, rows, local) / tile_keys + 1;
        for (int split = 0; split < splits; ++split) {
            const int length = split_tile(tiles, split + 1, splits) -
                               split_tile(tiles, split, splits);
            const long long start = finish.top();
            finish.pop();
            const long long end = start + length + unit_overhead;
            finish.push(end);
            makespan = std::max(makespan, end);
        }
    }
    return static_cast<int>(std::min<long long>(makespan, INT_MAX));
}

int choose_splits(int position, int rows, int ctas_per_kv, int sms) {
    const auto ctas = static_cast<std::size_t>(ctas_per_kv) * mxfp8_kv_heads;
    const std::size_t unit_bytes = members * channels * sizeof(float);
    int best = 1;
    double best_cost = simulated_makespan(position, rows, ctas_per_kv, 1, sms);
    for (int splits = 2; splits <= 16; ++splits) {
        if (ctas * splits * unit_bytes > max_partial_bytes) {
            break;
        }
        const double cost =
            simulated_makespan(position, rows, ctas_per_kv, splits, sms) * 1.03;
        if (cost < best_cost) {
            best_cost = cost;
            best = splits;
        }
    }
    return best;
}

struct SplitChoice {
    int position;
    int rows;
    int sms;
    int splits;
};

int cached_splits(int position, int rows, int ctas_per_kv, int sms) {
    constexpr std::uint32_t capacity = 1024;
    thread_local std::array<SplitChoice, capacity> choices{};
    std::uint32_t hash = static_cast<std::uint32_t>(position) * 0x9e3779b1u;
    hash ^= static_cast<std::uint32_t>(rows) * 0x85ebca77u;
    hash ^= static_cast<std::uint32_t>(sms) * 0xc2b2ae3du;
    hash ^= hash >> 16;
    auto& choice = choices[hash % capacity];
    if (choice.position != position || choice.rows != rows || choice.sms != sms) {
        choice = {position, rows, sms, choose_splits(position, rows, ctas_per_kv, sms)};
    }
    return choice.splits;
}

std::array<std::atomic<int>, 64> cached_sms{};

cudaError_t prepare_device(int device, int& sms) {
    if (device < 0 || static_cast<std::size_t>(device) >= cached_sms.size()) {
        return cudaSuccess;
    }
    const int cached = cached_sms[device].load(std::memory_order_acquire);
    if (cached != 0) {
        sms = std::max(cached, 0);
        return cudaSuccess;
    }

    int major = 0;
    int minor = 0;
    int optin = 0;
    int pools = 0;
    int multiprocessors = 0;
    const std::pair<cudaDeviceAttr, int*> attributes[] = {
        {cudaDevAttrComputeCapabilityMajor, &major},
        {cudaDevAttrComputeCapabilityMinor, &minor},
        {cudaDevAttrMaxSharedMemoryPerBlockOptin, &optin},
        {cudaDevAttrMemoryPoolsSupported, &pools},
        {cudaDevAttrMultiProcessorCount, &multiprocessors},
    };
    for (const auto& attribute : attributes) {
        const auto status = cudaDeviceGetAttribute(attribute.second, attribute.first, device);
        if (status != cudaSuccess) {
            return status;
        }
    }
    const bool eligible = major == 10 && minor == 0 && pools && multiprocessors > 0 &&
                          optin >= static_cast<int>(attn_smem_bytes);
    if (eligible) {
        const auto status = cudaFuncSetAttribute(prefill_attn_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(attn_smem_bytes));
        if (status != cudaSuccess) {
            return status;
        }
        sms = multiprocessors;
    }
    cached_sms[device].store(eligible ? multiprocessors : -1, std::memory_order_release);
    return cudaSuccess;
}

bool aligned(const void* pointer, std::size_t alignment) {
    return pointer && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

}

bool sm100_prefill_enabled() {
    const char* setting = std::getenv("SGLANG_PREFILL_ATTN");
    return setting && std::string_view(setting) == "sm100";
}

std::optional<cudaError_t> launch_prefill_attn_sm100(
    Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
    const __nv_bfloat16* queries, __nv_bfloat16* output, cudaStream_t stream) {
    const char* setting = std::getenv("SGLANG_PREFILL_ATTN");
    if (!setting || std::string_view(setting) != "sm100" || rows == 0) {
        return std::nullopt;
    }
    if (position < 0 || rows < 0 || position > INT_MAX - rows) {
        return cudaErrorInvalidValue;
    }
    const int visible = position + rows;
    if (rows > max_tmem_rows || visible > text_attention_max_tokens ||
        (rows < 144 && visible <= 32768)) {
        return std::nullopt;
    }
    const int required_pages = (visible - 1) / mxfp8_page_size + 1;
    if (!queries || !output || !cache.key_payload || !cache.value_payload ||
        !cache.key_scales || !cache.value_scales || !pages.device_pages ||
        required_pages > pages.page_count || required_pages > cache.page_count) {
        return cudaErrorInvalidValue;
    }
    if (!aligned(queries, alignof(uint4)) || !aligned(output, alignof(uint4)) ||
        !aligned(cache.key_payload, mxfp8_group_size) ||
        !aligned(cache.value_payload, mxfp8_group_size)) {
        return std::nullopt;
    }
    int device = 0;
    if (const auto status = cudaGetDevice(&device); status != cudaSuccess) {
        return status;
    }
    int sms = 0;
    if (const auto status = prepare_device(device, sms); status != cudaSuccess) {
        return status;
    }
    if (sms == 0) {
        return std::nullopt;
    }
    AttnPlan plan{};
    plan.request = Attention{cache, pages.device_pages, position, queries, output};
    plan.rows = rows;
    plan.ctas_per_kv = (rows * ratio + members - 1) / members;
    plan.splits = cached_splits(position, rows, plan.ctas_per_kv, sms);
    const auto units = static_cast<std::size_t>(plan.ctas_per_kv) * mxfp8_kv_heads * plan.splits;
    const std::size_t flag_bytes = (static_cast<std::size_t>(rows) * mxfp8_query_heads + 255) /
                                   256 * 256;
    const std::size_t stats_bytes = plan.splits > 1 ? units * members * sizeof(float2) : 0;
    const std::size_t partial_bytes =
        plan.splits > 1 ? units * members * channels * sizeof(float) : 0;
    void* scratch = nullptr;
    auto status = cudaMallocAsync(&scratch, 256 + flag_bytes + stats_bytes + partial_bytes, stream);
    if (status != cudaSuccess) {
        return status;
    }
    auto* bytes = static_cast<std::uint8_t*>(scratch);
    plan.bad_key = reinterpret_cast<int*>(bytes);
    plan.bad_query = bytes + 256;
    plan.stats = plan.splits > 1 ? reinterpret_cast<float2*>(bytes + 256 + flag_bytes) : nullptr;
    plan.partial = plan.splits > 1 ?
        reinterpret_cast<float*>(bytes + 256 + flag_bytes + stats_bytes) : nullptr;
    status = cudaMemsetAsync(plan.bad_key, 0x7f, mxfp8_kv_heads * sizeof(int), stream);
    void* args[] = {&plan};
    if (status == cudaSuccess) {
        status = cudaLaunchKernel(reinterpret_cast<const void*>(prefill_attn_kernel),
                                  dim3(static_cast<unsigned>(units)), dim3(attn_threads),
                                  args, attn_smem_bytes, stream);
    }
    if (status == cudaSuccess) {
        status = cudaLaunchKernel(reinterpret_cast<const void*>(prefill_attn_finish),
                                  dim3(rows, mxfp8_query_heads / ordered_heads), dim3(channels),
                                  args, 0, stream);
    }
    const auto release = cudaFreeAsync(scratch, stream);
    return status != cudaSuccess ? status : release;
}

}
