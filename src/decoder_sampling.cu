#include "sglang/decoder_sampling.h"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace sglang::decoder {
namespace {

constexpr int group_size = 256;
constexpr int group_count = 256;
constexpr int key_count = group_count * group_size;
constexpr int stripe_size = 2048;
constexpr int max_stripes = 128;
constexpr unsigned warp_mask = 0xffffffffu;

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

__device__ int score_key(__nv_bfloat16 value) {
    unsigned bits = __bfloat16_as_ushort(value);
    if ((bits & 0x7f80u) == 0x7f80u) {
        return bits == 0xff80u ? 0 : key_count;
    }
    if ((bits & 0x7fffu) == 0) {
        bits = 0;
    }
    return bits & 0x8000u ? (bits ^ 0xffffu) : (bits ^ 0x8000u);
}

__device__ double key_score(int key) {
    const auto bits = static_cast<unsigned short>(key & 0x8000 ? key ^ 0x8000 : key ^ 0xffff);
    return __bfloat162float(__ushort_as_bfloat16(bits));
}

template <typename Value>
__device__ Value block_sum(Value value) {
    __shared__ Value warps[group_size / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
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

__global__ void histogram_scores(const __nv_bfloat16* scores, int vocabulary,
                                 Histogram* histogram) {
    int highest = 0;
    for (int start = blockIdx.x * blockDim.x; start < vocabulary;
         start += gridDim.x * blockDim.x) {
        const int token = start + threadIdx.x;
        const int key = token < vocabulary ? score_key(scores[token]) : 0;
        highest = max(highest, key);
        const unsigned valid = __ballot_sync(warp_mask, key > 0 && key < key_count);
        if (key > 0 && key < key_count) {
            const unsigned matches = __match_any_sync(valid, key);
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

__global__ void measure_groups(Scratch* scratch, double temperature) {
    const int highest = scratch->histogram.highest;
    if (highest == 0 || highest == key_count) {
        return;
    }
    const int key = blockIdx.x * group_size + threadIdx.x;
    const int count = scratch->histogram.counts[key];
    const double weight = count ? exp((key_score(key) - key_score(highest)) / temperature) : 0.0;
    scratch->weights[key] = weight;
    const double mass = block_sum(count * weight);
    const int tokens = block_sum(count);
    if (threadIdx.x == 0) {
        scratch->groups[blockIdx.x] = {mass, tokens};
    }
}

__device__ double uniform(std::uint64_t seed, int position) {
    auto bits = seed + 0x9e3779b97f4a7c15ULL * (static_cast<std::uint64_t>(position) + 1);
    bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
    bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<double>((bits ^ (bits >> 31)) >> 11) * 0x1.0p-53;
}

__device__ int crossing(double prefix, double weight, double limit, int count) {
    int first = 1;
    int last = count;
    while (first < last) {
        const int middle = first + (last - first) / 2;
        if (prefix + middle * weight > limit) {
            last = middle;
        } else {
            first = middle + 1;
        }
    }
    return first;
}

__device__ int minimum_key(int highest, double temperature, double min_p) {
    if (min_p == 0.0) {
        return 0;
    }
    int first = 0x0080;
    int last = highest;
    const double peak = key_score(highest);
    while (first < last) {
        const int middle = first + (last - first) / 2;
        if (exp((key_score(middle) - peak) / temperature) >= min_p) {
            last = middle;
        } else {
            first = middle + 1;
        }
    }
    return first;
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
            if (count && kept.mass + count * weight > cutoff) {
                count = crossing(kept.mass, weight, cutoff, count);
                return {kept.mass + count * weight, kept.count + count};
            }
            kept.mass += count * weight;
            kept.count += count;
        }
    }
    return kept;
}

__global__ void choose_score(Scratch* scratch, SamplingParameters parameters, int position) {
    scratch->key = 0;
    const int highest = scratch->histogram.highest;
    if (highest == 0 || highest == key_count) {
        return;
    }
    double cutoff = CUDART_INF;
    if (parameters.top_p < 1.0) {
        double full_mass = 0;
        for (int group = highest / group_size; group >= 0; --group) {
            full_mass += scratch->groups[group].mass;
        }
        cutoff = parameters.top_p * full_mass;
    }
    const int lowest = minimum_key(highest, parameters.temperature, parameters.min_p);
    const auto kept = retain(*scratch, parameters.top_k, cutoff, lowest);
    const double draw = min(uniform(parameters.seed, position) * kept.mass,
                            nextafter(kept.mass, 0.0));
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
            if (count && mass + count * weight > draw) {
                scratch->key = key;
                scratch->ordinal = crossing(mass, weight, draw, count) - 1;
                return;
            }
            if (count && weight > 0.0) {
                scratch->key = key;
                scratch->ordinal = count - 1;
            }
            mass += count * weight;
            remaining -= count;
        }
    }
}

__global__ void count_stripes(const __nv_bfloat16* scores, int vocabulary, Scratch* scratch) {
    if (scratch->key == 0) {
        return;
    }
    const int start = blockIdx.x * stripe_size;
    const int end = min(start + stripe_size, vocabulary);
    int count = 0;
    for (int token = start + threadIdx.x; token < end; token += blockDim.x) {
        count += score_key(scores[token]) == scratch->key;
    }
    const int stripe = block_sum(count);
    if (threadIdx.x == 0) {
        scratch->stripe_counts[blockIdx.x] = stripe;
    }
}

__global__ void choose_token(const __nv_bfloat16* scores, int vocabulary, int stripes,
                             const Scratch* scratch, int* token) {
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
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    int target = begin < end ? ordinal : 0;
    for (int offset = begin; offset < end; offset += blockDim.x) {
        const int index = offset + threadIdx.x;
        const unsigned matches = __ballot_sync(
            warp_mask, index < end && score_key(scores[index]) == scratch->key);
        if (lane == 0) {
            warp_counts[warp] = __popc(matches);
        }
        __syncthreads();
        int before = 0;
        for (int preceding = 0; preceding < warp; ++preceding) {
            before += warp_counts[preceding];
        }
        if ((matches & (1u << lane)) && before + __popc(matches & ((1u << lane) - 1)) == target) {
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

}

std::size_t sampling_workspace_bytes() { return sizeof(Scratch); }

cudaError_t sample_logits(const __nv_bfloat16* scores, int vocabulary, int* token,
                          SamplingParameters parameters, int position, void* workspace,
                          cudaStream_t stream) {
    if (vocabulary < 1 || vocabulary > max_stripes * stripe_size) {
        return cudaErrorInvalidValue;
    }
    const int stripes = (vocabulary + stripe_size - 1) / stripe_size;
    auto* scratch = static_cast<Scratch*>(workspace);
    const auto status = cudaMemsetAsync(&scratch->histogram, 0, sizeof(Histogram), stream);
    if (status != cudaSuccess) {
        return status;
    }
    histogram_scores<<<128, group_size, 0, stream>>>(scores, vocabulary, &scratch->histogram);
    measure_groups<<<group_count, group_size, 0, stream>>>(scratch, parameters.temperature);
    choose_score<<<1, 1, 0, stream>>>(scratch, parameters, position);
    count_stripes<<<stripes, group_size, 0, stream>>>(scores, vocabulary, scratch);
    choose_token<<<1, group_size, 0, stream>>>(scores, vocabulary, stripes, scratch, token);
    return cudaGetLastError();
}

}
