#include "sglang/decoder_sampling.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>

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

constexpr int token_bits = 18;
constexpr std::uint64_t token_mask = (1u << token_bits) - 1;
constexpr std::uint64_t invalid_key = (std::uint64_t{1} << (32 + token_bits)) - 1;
using ScoreKey = std::uint64_t;

struct BiasSamplingScratch {
    ScoreKey* keys;
    ScoreKey* sorted;
    double* weights;
    double* prefix;
    void* temporary;
};

__global__ void rank_scores(const float* scores, int vocabulary, ScoreKey* keys) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token < vocabulary) {
        const float score = scores[token];
        const auto bits = __float_as_uint(score == 0 ? 0.0f : score);
        const auto ordered = bits ^ (bits & 0x80000000u ? 0xffffffffu : 0x80000000u);
        const auto key = (ScoreKey{ordered} << token_bits) | (token_mask - token);
        keys[token] = isfinite(score) ? key : score == -CUDART_INF_F ? 0 : invalid_key;
    }
}

__device__ double ranked_weight(ScoreKey key, ScoreKey peak, double temperature) {
    const auto score = [](ScoreKey value) {
        const auto ordered = static_cast<unsigned>(value >> token_bits);
        return static_cast<double>(
            __uint_as_float(ordered ^ (ordered & 0x80000000u ? 0x80000000u : 0xffffffffu)));
    };
    return key != 0 && peak != invalid_key ? exp((score(key) - score(peak)) / temperature) : 0.0;
}

__global__ void measure_weights(const ScoreKey* keys, int vocabulary, double temperature,
                                double* weights) {
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank < vocabulary) {
        weights[rank] = ranked_weight(keys[rank], keys[0], temperature);
    }
}

__device__ int first_crossing(const double* prefix, int count, double cutoff) {
    int first = 0;
    while (first < count) {
        const int middle = first + (count - first) / 2;
        if (prefix[middle] > cutoff) {
            count = middle;
        } else {
            first = middle + 1;
        }
    }
    return first;
}

__global__ void choose_biased_token(BiasSamplingScratch scratch, int vocabulary,
                                    SamplingParameters parameters, int position, int* token) {
    const auto peak = scratch.sorted[0];
    *token = -1;
    if (peak == 0 || peak == invalid_key) {
        return;
    }
    int keep = min(vocabulary, parameters.top_k);
    if (parameters.top_p < 1.0) {
        const auto cutoff = parameters.top_p * scratch.prefix[vocabulary - 1];
        keep = min(keep, first_crossing(scratch.prefix, keep, cutoff) + 1);
    }
    if (parameters.min_p > 0.0) {
        int first = 1;
        while (first < keep) {
            const int middle = first + (keep - first) / 2;
            if (scratch.weights[middle] >= parameters.min_p) {
                first = middle + 1;
            } else {
                keep = middle;
            }
        }
    }
    const double mass = scratch.prefix[keep - 1];
    const double draw = min(uniform(parameters.seed, position) * mass, nextafter(mass, 0.0));
    const int rank = first_crossing(scratch.prefix, keep, draw);
    *token = static_cast<int>(token_mask - (scratch.sorted[rank] & token_mask));
}

std::size_t biased_sampling_temporary_bytes(int vocabulary) {
    std::size_t sort = 0;
    std::size_t scan = 0;
    cub::DeviceRadixSort::SortKeysDescending(
        nullptr, sort, static_cast<ScoreKey*>(nullptr), static_cast<ScoreKey*>(nullptr),
        vocabulary, 0, 32 + token_bits);
    cub::DeviceScan::InclusiveSum(
        nullptr, scan, static_cast<double*>(nullptr), static_cast<double*>(nullptr), vocabulary);
    return std::max(sort, scan);
}

__device__ bool is_allowed(const std::uint32_t* mask, int token) {
    return !mask || (mask[token / 32] >> (token % 32) & 1u);
}

__global__ void initialize_scores(const __nv_bfloat16* source, float* scores, int vocabulary,
                                   const std::uint32_t* mask) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token < vocabulary) {
        scores[token] = is_allowed(mask, token) ? __bfloat162float(source[token]) : -CUDART_INF_F;
    }
}

__global__ void add_bias(float* scores, const DecoderLogitBias* bias, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const auto entry = bias[index];
        scores[entry.token_id] += entry.bias;
    }
}

__global__ void validate_allowed(const float* scores, int* token) {
    if (*token >= 0 && !isfinite(scores[*token])) {
        *token = -1;
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

std::size_t biased_sampling_workspace_bytes(int vocabulary) {
    return 2 * vocabulary * (sizeof(ScoreKey) + sizeof(double)) +
        biased_sampling_temporary_bytes(vocabulary) + 255;
}

cudaError_t sample_logits(const float* scores, int vocabulary, int* token,
                          SamplingParameters parameters, int position, void* workspace,
                          std::size_t workspace_bytes, cudaStream_t stream) {
    if (vocabulary < 1 || vocabulary > token_mask + 1 || parameters.top_k < 1) {
        return cudaErrorInvalidValue;
    }
    auto* keys = static_cast<ScoreKey*>(workspace);
    auto* weights = reinterpret_cast<double*>(keys + 2 * vocabulary);
    const auto end = reinterpret_cast<std::uintptr_t>(weights + 2 * vocabulary);
    const BiasSamplingScratch scratch{
        keys, keys + vocabulary, weights, weights + vocabulary,
        reinterpret_cast<void*>((end + 255) & ~std::uintptr_t{255})};
    const int blocks = (vocabulary + group_size - 1) / group_size;
    const auto offset = reinterpret_cast<std::uintptr_t>(scratch.temporary) -
        reinterpret_cast<std::uintptr_t>(workspace);
    if (workspace_bytes <= offset) {
        return cudaErrorInvalidValue;
    }
    auto bytes = workspace_bytes - offset;
    rank_scores<<<blocks, group_size, 0, stream>>>(scores, vocabulary, scratch.keys);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    status = cub::DeviceRadixSort::SortKeysDescending(
        scratch.temporary, bytes, scratch.keys, scratch.sorted, vocabulary, 0, 32 + token_bits,
        stream);
    if (status != cudaSuccess) return status;
    measure_weights<<<blocks, group_size, 0, stream>>>(
        scratch.sorted, vocabulary, parameters.temperature, scratch.weights);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    bytes = workspace_bytes - offset;
    status = cub::DeviceScan::InclusiveSum(
        scratch.temporary, bytes, scratch.weights, scratch.prefix, vocabulary, stream);
    if (status != cudaSuccess) return status;
    choose_biased_token<<<1, 1, 0, stream>>>(scratch, vocabulary, parameters, position, token);
    return cudaGetLastError();
}

cudaError_t apply_logit_bias(const __nv_bfloat16* source, float* scores, int vocabulary,
                            const DecoderLogitBias* bias, int count, const std::uint32_t* mask,
                            cudaStream_t stream) {
    initialize_scores<<<(vocabulary + group_size - 1) / group_size, group_size, 0, stream>>>(
        source, scores, vocabulary, mask);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    add_bias<<<(count + group_size - 1) / group_size, group_size, 0, stream>>>(scores, bias, count);
    return cudaGetLastError();
}

cudaError_t check_allowed_logit(const float* scores, int* token, cudaStream_t stream) {
    validate_allowed<<<1, 1, 0, stream>>>(scores, token);
    return cudaGetLastError();
}

}
