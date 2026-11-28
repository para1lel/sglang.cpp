#include "sglang/decoder_sampling.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>

namespace sglang::decoder {

double sampling_uniform(std::uint64_t seed, int position) {
    auto bits = seed + (0x9e3779b97f4a7c15ULL * (static_cast<std::uint64_t>(position) + 1));
    bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
    bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<double>((bits ^ (bits >> 31)) >> 11) * 0x1.0p-53;
}

namespace {

constexpr int group_size = 256;
constexpr int group_count = 256;
constexpr int key_count = group_count * group_size;
constexpr int stripe_size = 2048;
constexpr int max_stripes = 128;
constexpr int max_sampling_rows = 20;
constexpr unsigned warp_mask = 0xffffffffU;

struct Histogram {
    int counts[key_count];
    int highest;
};

struct MassGroup {
    double mass;
    int count;
};

struct Scratch {
    Histogram histogram;
    double weights[key_count];
    MassGroup groups[group_count];
    int stripe_counts[max_stripes];
    int key;
    int ordinal;
};

struct SamplingDraw {
    const __nv_bfloat16* scores;
    int* token;
    SamplingParameters parameters;
    double uniform;
};

SamplingDraw sampling_draw(const SamplingRow& row) {
    return {
        row.scores,
        row.token,
        row.parameters,
        sampling_uniform(row.parameters.seed, row.position),
    };
}

template <int Capacity> struct SamplingRows {
    SamplingDraw entries[Capacity];

    template <typename RowScratch>
    __device__ __forceinline__ const SamplingDraw& bind(RowScratch*& scratch) const {
        const unsigned index = Capacity == 1 ? 0U : blockIdx.y;
        scratch += index;
        return entries[index];
    }
};

__device__ int score_key(__nv_bfloat16 value) {
    unsigned bits = __bfloat16_as_ushort(value);
    if ((bits & 0x7f80U) == 0x7f80U) {
        return bits == 0xff80U ? 0 : key_count;
    }
    if ((bits & 0x7fffU) == 0) {
        bits = 0;
    }
    return static_cast<int>((bits & 0x8000U) != 0U ? (bits ^ 0xffffU) : (bits ^ 0x8000U));
}

__device__ double key_score(int key) {
    const auto ordered = static_cast<unsigned>(key);
    const auto bits = static_cast<unsigned short>((ordered & 0x8000U) != 0U ? (ordered ^ 0x8000U)
                                                                            : (ordered ^ 0xffffU));
    return __bfloat162float(__ushort_as_bfloat16(bits));
}

template <typename Value> __device__ Value block_sum(Value value) {
    __shared__ Value warps[group_size / 32];
    const unsigned lane = threadIdx.x % 32;
    const int warp = static_cast<int>(threadIdx.x / 32);
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(warp_mask, value, offset);
    }
    if (lane == 0) {
        warps[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < group_size / 32 ? warps[lane] : Value{};
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(warp_mask, value, offset);
        }
    }
    return value;
}

template <int Capacity>
__global__ void histogram_scores(const __grid_constant__ SamplingRows<Capacity> rows,
                                 int vocabulary, Scratch* scratch) {
    const auto* scores = rows.bind(scratch).scores;
    auto* histogram = &scratch->histogram;
    int highest = 0;
    for (int start = static_cast<int>(blockIdx.x * blockDim.x); start < vocabulary;
         start += static_cast<int>(gridDim.x * blockDim.x)) {
        const int token = start + static_cast<int>(threadIdx.x);
        const int key = token < vocabulary ? score_key(scores[token]) : 0;
        highest = max(highest, key);
        const unsigned mask = __ballot_sync(warp_mask, key > 0 && key < key_count);
        if (key > 0 && key < key_count) {
            const unsigned matches = __match_any_sync(mask, key);
            if (static_cast<int>(threadIdx.x % 32) == __ffs(matches) - 1) {
                atomicAdd(&histogram->counts[key], __popc(matches));
            }
        }
    }
    __shared__ int warp_max[group_size / 32];
    highest = __reduce_max_sync(warp_mask, highest);
    if (threadIdx.x % 32 == 0) {
        warp_max[threadIdx.x / 32] = highest;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        highest = threadIdx.x < group_size / 32 ? warp_max[threadIdx.x] : 0;
        highest = __reduce_max_sync(warp_mask, highest);
        if (threadIdx.x == 0) {
            atomicMax(&histogram->highest, highest);
        }
    }
}

template <int Capacity>
__global__ void measure_groups(const __grid_constant__ SamplingRows<Capacity> rows,
                               Scratch* scratch) {
    const double temperature = rows.bind(scratch).parameters.temperature;
    const int highest = scratch->histogram.highest;
    if (highest == 0 || highest == key_count)
        return;
    const int key = static_cast<int>((blockIdx.x * group_size) + threadIdx.x);
    const int count = scratch->histogram.counts[key];
    const double weight =
        count != 0 ? exp((key_score(key) - key_score(highest)) / temperature) : 0.0;
    scratch->weights[key] = weight;
    const double mass = block_sum(count * weight);
    const int tokens = block_sum(count);
    if (threadIdx.x == 0)
        scratch->groups[blockIdx.x] = {mass, tokens};
}

template <typename Reached>
__device__ __forceinline__ int first_reached(int first, int last, Reached reached) {
    while (first < last) {
        const int middle = first + ((last - first) / 2);
        if (reached(middle)) {
            last = middle;
        } else {
            first = middle + 1;
        }
    }
    return first;
}

__device__ int minimum_key(int highest, const SamplingParameters& parameters) {
    if (parameters.min_p == 0.0)
        return 0;
    const int first_finite_key = 0x0080;
    const double peak = key_score(highest);
    return first_reached(first_finite_key, highest, [=](int key) {
        return exp((key_score(key) - peak) / parameters.temperature) >= parameters.min_p;
    });
}

__device__ MassGroup retain(const Scratch& scratch, int top_k, double cutoff, int lowest) {
    MassGroup kept{};
    for (int group = scratch.histogram.highest / group_size; group >= lowest / group_size;
         --group) {
        const auto current = scratch.groups[group];
        const int start = group * group_size;
        if (start >= lowest && kept.count + current.count <= top_k &&
            kept.mass + current.mass <= cutoff) {
            kept.mass += current.mass;
            kept.count += current.count;
            continue;
        }
        for (int key = start + group_size - 1; key >= max(start, lowest); --key) {
            if (kept.count == top_k || kept.mass > cutoff) {
                return kept;
            }
            int count = min(scratch.histogram.counts[key], top_k - kept.count);
            const double weight = scratch.weights[key];
            if ((count != 0) && kept.mass + (count * weight) > cutoff) {
                const double prefix = kept.mass;
                count = first_reached(
                    1, count, [=](int index) { return prefix + (index * weight) > cutoff; });
                return {kept.mass + (count * weight), kept.count + count};
            }
            kept.mass += count * weight;
            kept.count += count;
        }
    }
    return kept;
}

template <int Capacity>
__global__ void choose_score(const __grid_constant__ SamplingRows<Capacity> rows,
                             Scratch* scratch) {
    const auto& row = rows.bind(scratch);
    const auto& parameters = row.parameters;
    scratch->key = 0;
    const int highest = scratch->histogram.highest;
    if (highest == 0 || highest == key_count)
        return;
    double cutoff = CUDART_INF;
    if (parameters.top_p < 1.0) {
        double full_mass = 0;
        for (int group = highest / group_size; group >= 0; --group) {
            full_mass += scratch->groups[group].mass;
        }
        cutoff = parameters.top_p * full_mass;
    }
    const int lowest = minimum_key(highest, parameters);
    const auto kept = retain(*scratch, parameters.top_k, cutoff, lowest);
    const double draw = min(row.uniform * kept.mass, nextafter(kept.mass, 0.0));
    double mass = 0;
    int remaining = kept.count;
    for (int group = highest / group_size; group >= 0 && remaining > 0; --group) {
        const auto current = scratch->groups[group];
        if (current.count <= remaining && mass + current.mass <= draw) {
            mass += current.mass;
            remaining -= current.count;
            continue;
        }
        const int start = group * group_size;
        for (int key = start + group_size - 1; key >= start && remaining > 0; --key) {
            const int count = min(scratch->histogram.counts[key], remaining);
            const double weight = scratch->weights[key];
            if (count != 0 && mass + (count * weight) > draw) {
                scratch->key = key;
                scratch->ordinal =
                    first_reached(1, count,
                                  [=](int index) { return mass + (index * weight) > draw; }) -
                    1;
                return;
            }
            if (count != 0 && weight > 0.0) {
                scratch->key = key;
                scratch->ordinal = count - 1;
            }
            mass += count * weight;
            remaining -= count;
        }
    }
}

template <int Capacity>
__global__ void count_stripes(const __grid_constant__ SamplingRows<Capacity> rows, int vocabulary,
                              Scratch* scratch) {
    const auto* scores = rows.bind(scratch).scores;
    if (scratch->key == 0)
        return;
    const int start = static_cast<int>(blockIdx.x * stripe_size);
    const int end = min(start + stripe_size, vocabulary);
    int count = 0;
    for (int token = start + static_cast<int>(threadIdx.x); token < end;
         token += static_cast<int>(blockDim.x)) {
        count += score_key(scores[token]) == scratch->key;
    }
    const int stripe = block_sum(count);
    if (threadIdx.x == 0)
        scratch->stripe_counts[blockIdx.x] = stripe;
}

template <int Capacity>
__global__ void choose_token(const __grid_constant__ SamplingRows<Capacity> rows, int vocabulary,
                             const Scratch* scratch, int stripes) {
    const auto& row = rows.bind(scratch);
    auto* token = row.token;
    __shared__ int begin;
    __shared__ int ordinal;
    __shared__ int warp_counts[group_size / 32];
    if (threadIdx.x == 0) {
        *token = -1;
        begin = vocabulary;
        if (scratch->key != 0) {
            int target = scratch->ordinal;
            for (int stripe = 0; stripe < stripes; ++stripe) {
                if (target < scratch->stripe_counts[stripe]) {
                    begin = stripe * stripe_size;
                    ordinal = target;
                    break;
                }
                target -= scratch->stripe_counts[stripe];
            }
        }
    }
    __syncthreads();
    const int end = min(begin + stripe_size, vocabulary);
    const unsigned lane = threadIdx.x % 32;
    const int warp = static_cast<int>(threadIdx.x / 32);
    int target = begin < end ? ordinal : 0;
    for (int offset = begin; offset < end; offset += static_cast<int>(blockDim.x)) {
        const int index = offset + static_cast<int>(threadIdx.x);
        const unsigned matches =
            __ballot_sync(warp_mask, index < end && score_key(row.scores[index]) == scratch->key);
        if (lane == 0) {
            warp_counts[warp] = __popc(matches);
        }
        __syncthreads();
        int before = 0;
        for (int preceding = 0; preceding < warp; ++preceding) {
            before += warp_counts[preceding];
        }
        if ((matches & (1U << lane)) != 0U &&
            before + __popc(matches & ((1U << lane) - 1)) == target) {
            *token = index;
        }
        for (int following = warp; following < group_size / 32; ++following) {
            before += warp_counts[following];
        }
        __syncthreads();
        target -= before;
        if (target < 0) {
            return;
        }
    }
}

template <int Capacity>
cudaError_t launch_sampling(const SamplingRows<Capacity>& rows, int count, int vocabulary,
                            Scratch* scratch, cudaStream_t stream) {
    const auto status = count == 1 ? cudaMemsetAsync(scratch, 0, sizeof(Histogram), stream)
                                   : cudaMemset2DAsync(scratch, sizeof(Scratch), 0,
                                                       sizeof(Histogram), count, stream);
    if (status != cudaSuccess)
        return status;
    const int stripes = (vocabulary + stripe_size - 1) / stripe_size;
    histogram_scores<<<dim3(128, count), group_size, 0, stream>>>(rows, vocabulary, scratch);
    measure_groups<<<dim3(group_count, count), group_size, 0, stream>>>(rows, scratch);
    choose_score<<<dim3(1, count), 1, 0, stream>>>(rows, scratch);
    count_stripes<<<dim3(stripes, count), group_size, 0, stream>>>(rows, vocabulary, scratch);
    choose_token<<<dim3(1, count), group_size, 0, stream>>>(rows, vocabulary, scratch, stripes);
    return cudaGetLastError();
}
}

std::size_t sampling_workspace_bytes() {
    return sizeof(Scratch);
}

cudaError_t sample_logits(const __nv_bfloat16* scores, int vocabulary, int* token,
                          SamplingParameters parameters, int position, void* workspace,
                          cudaStream_t stream) {
    if (vocabulary < 1 || vocabulary > max_stripes * stripe_size)
        return cudaErrorInvalidValue;
    return launch_sampling(SamplingRows<1>{{sampling_draw({scores, token, parameters, position})}},
                           1, vocabulary, static_cast<Scratch*>(workspace), stream);
}

cudaError_t sample_logits_rows(const SamplingRow* rows, int count, int vocabulary,
                               void* single_workspace, std::size_t batch_bytes,
                               void* batch_workspace, cudaStream_t stream) {
    if (count < 0 || vocabulary < 1 || vocabulary > max_stripes * stripe_size)
        return cudaErrorInvalidValue;
    const int capacity = batch_workspace != nullptr
                             ? static_cast<int>(std::min<std::size_t>(
                                   max_sampling_rows, batch_bytes / sizeof(Scratch)))
                             : 0;
    for (int first = 0; first < count;) {
        const int batch_count = std::max(1, std::min(capacity, count - first));
        cudaError_t status = cudaSuccess;
        if (batch_count == 1) {
            status = launch_sampling(SamplingRows<1>{{sampling_draw(rows[first])}}, 1, vocabulary,
                                     static_cast<Scratch*>(single_workspace), stream);
        } else {
            SamplingRows<max_sampling_rows> batch{};
            for (int index = 0; index < batch_count; ++index) {
                batch.entries[index] = sampling_draw(rows[first + index]);
            }
            status = launch_sampling(batch, batch_count, vocabulary,
                                     static_cast<Scratch*>(batch_workspace), stream);
        }
        if (status != cudaSuccess)
            return status;
        first += batch_count;
    }
    return cudaSuccess;
}
}
