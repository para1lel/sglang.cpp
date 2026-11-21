#include "sglang/decoder_sampling.h"
#include "sglang/text/decoder_logprobs.h"

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cfloat>

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

__device__ int choose_rank(const double* weights, const double* prefix, int vocabulary,
                          SamplingParameters parameters, int position) {
    int keep = min(vocabulary, parameters.top_k);
    if (parameters.top_p < 1.0) {
        const auto cutoff = parameters.top_p * prefix[vocabulary - 1];
        keep = min(keep, first_crossing(prefix, keep, cutoff) + 1);
    }
    if (parameters.min_p > 0.0) {
        int first = 1;
        while (first < keep) {
            const int middle = first + (keep - first) / 2;
            if (weights[middle] >= parameters.min_p) {
                first = middle + 1;
            } else {
                keep = middle;
            }
        }
    }
    const double mass = prefix[keep - 1];
    const double draw = min(uniform(parameters.seed, position) * mass, nextafter(mass, 0.0));
    return first_crossing(prefix, keep, draw);
}

__global__ void choose_biased_token(BiasSamplingScratch scratch, int vocabulary,
                                    SamplingParameters parameters, int position, int* token) {
    const auto peak = scratch.sorted[0];
    *token = -1;
    if (peak != 0 && peak != invalid_key) {
        const int rank = choose_rank(scratch.weights, scratch.prefix, vocabulary,
                                     parameters, position);
        *token = static_cast<int>(token_mask - (scratch.sorted[rank] & token_mask));
    }
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

__device__ float subtract_history(float score, int token, OutputPenalties penalties,
                                  int previous_token, bool& seen) {
    float total = 0.0f;
    if (penalties.frequency_totals) {
        total = penalties.frequency_totals[token];
        if (token == previous_token) {
            total += penalties.frequency;
            penalties.frequency_totals[token] = total;
        }
    }
    if (penalties.seen) {
        if (token == previous_token) {
            penalties.seen[token] = 1;
        }
        seen = penalties.seen[token] != 0;
        if (seen) {
            total += penalties.presence;
        }
    }
    return score - total;
}

__global__ void initialize_scores(const __nv_bfloat16* source, float* scores, int vocabulary,
                                  const std::uint32_t* mask, OutputPenalties penalties,
                                  int previous_token) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token < vocabulary) {
        bool seen = false;
        const float score = subtract_history(__bfloat162float(source[token]), token,
                                              penalties, previous_token, seen);
        scores[token] = is_allowed(mask, token) ? score : -CUDART_INF_F;
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

struct ScaledScore {
    double fraction;
    int exponent;
};

__device__ ScaledScore scaled_score(double value, int exponent = 0) {
    if (value == 0 || !isfinite(value)) return {value, 0};
    int shift;
    const double fraction = frexp(value, &shift);
    return {fraction, exponent + shift};
}

struct WideScore {
    ScaledScore high;
    ScaledScore low;
};

__device__ WideScore add_scaled(ScaledScore left, ScaledScore right) {
    if (left.fraction == 0) return {right, {}};
    if (right.fraction == 0 || !isfinite(left.fraction)) return {left, {}};
    if (left.exponent < right.exponent) {
        const auto swap = left;
        left = right;
        right = swap;
    }
    if (left.exponent - right.exponent > 60) return {left, right};
    const double aligned = scalbn(right.fraction, right.exponent - left.exponent);
    const double sum = left.fraction + aligned;
    const double rounded = sum - left.fraction;
    const double remainder = (left.fraction - (sum - rounded)) + (aligned - rounded);
    return {scaled_score(sum, left.exponent), scaled_score(remainder, left.exponent)};
}

__device__ WideScore scale_repetition(ScaledScore score, double repetition) {
    int exponent;
    const double fraction = frexp(repetition, &exponent);
    if (score.fraction < 0) {
        const double product = score.fraction * fraction;
        return {scaled_score(product, score.exponent + exponent),
                scaled_score(fma(score.fraction, fraction, -product), score.exponent + exponent)};
    }
    const double quotient = score.fraction / fraction;
    const double remainder = fma(-quotient, fraction, score.fraction) / fraction;
    return {scaled_score(quotient, score.exponent - exponent),
            scaled_score(remainder, score.exponent - exponent)};
}

__device__ int compare_scaled(ScaledScore left, ScaledScore right) {
    if (left.fraction == right.fraction && left.exponent == right.exponent) return 0;
    const bool same_sign = (left.fraction > 0 && right.fraction > 0) ||
        (left.fraction < 0 && right.fraction < 0);
    if (same_sign && left.exponent != right.exponent) {
        return (left.exponent < right.exponent) == (left.fraction > 0) ? -1 : 1;
    }
    return left.fraction < right.fraction ? -1 : 1;
}

struct RepetitionEntry {
    float base;
    float bias;
    bool repeated;
    WideScore score;
};

struct RepetitionState {
    unsigned wide;
    unsigned invalid;
    int tiles[1024];
    GeneratedLogprobReadback logprobs;
};

struct RepetitionScratch {
    RepetitionState* state;
    RepetitionEntry* entries;
    int* ranked;
    double* weights;
    double* prefix;
    DecoderLogit* debug;
    void* temporary;
    std::size_t overhead;
};

RepetitionScratch repetition_scratch(void* workspace, int vocabulary, bool ranked) {
    auto* state = static_cast<RepetitionState*>(workspace);
    auto offset = (sizeof(RepetitionState) + 255) & ~std::size_t{255};
    const auto address = reinterpret_cast<std::uintptr_t>(workspace);
    auto* entries = reinterpret_cast<RepetitionEntry*>(address + offset);
    offset += vocabulary * sizeof(RepetitionEntry);
    auto* order = reinterpret_cast<int*>(address + offset);
    offset = (offset + vocabulary * sizeof(int) + 7) & ~std::size_t{7};
    auto* weights = reinterpret_cast<double*>(address + offset);
    auto* prefix = reinterpret_cast<double*>(address + offset + vocabulary * sizeof(double));
    if (ranked) offset += 2 * vocabulary * sizeof(double);
    auto* debug = reinterpret_cast<DecoderLogit*>(address + offset);
    if (ranked) offset += vocabulary * sizeof(DecoderLogit);
    offset = (offset + 255) & ~std::size_t{255};
    return {state, entries, order, weights, prefix, debug,
            reinterpret_cast<void*>(address + offset), offset};
}

__global__ void prepare_repetition(const __nv_bfloat16* source, int vocabulary,
                                   const std::uint32_t* mask, OutputPenalties penalties,
                                   int previous, RepetitionScratch scratch) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    bool wide = false;
    bool invalid = false;
    if (token < vocabulary) {
        bool seen = false;
        float score = subtract_history(__bfloat162float(source[token]), token, penalties,
                                        previous, seen);
        if (!is_allowed(mask, token)) score = -CUDART_INF_F;
        scratch.entries[token].base = score;
        scratch.entries[token].bias = 0;
        scratch.entries[token].repeated = seen;
        scratch.ranked[token] = token;
        invalid = isnan(score) || score == CUDART_INF_F;
        if (seen && score != 0 && isfinite(score)) {
            const auto base = static_cast<double>(score);
            const double scaled = base < 0 ? base * penalties.repetition
                                          : base / penalties.repetition;
            wide = fabs(scaled) > 0x1p24 || fabs(scaled) < FLT_MIN ||
                penalties.repetition < FLT_MIN;
        }
    }
    if (__syncthreads_or(wide) && threadIdx.x == 0) atomicOr(&scratch.state->wide, 1u);
    if (__syncthreads_or(invalid) && threadIdx.x == 0) atomicOr(&scratch.state->invalid, 1u);
}

__global__ void add_repetition_bias(RepetitionEntry* entries, const DecoderLogitBias* bias,
                                     int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) entries[bias[index].token_id].bias = bias[index].bias;
}

__global__ void encode_repetition(RepetitionScratch scratch, int vocabulary,
                                   double repetition) {
    const int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= vocabulary) return;
    auto& entry = scratch.entries[token];
    const auto base = static_cast<double>(entry.base);
    if (!scratch.state->wide || !isfinite(base)) {
        const float value = entry.repeated && isfinite(base)
            ? static_cast<float>(base < 0 ? base * repetition : base / repetition)
            : entry.base;
        entry.score = {scaled_score(entry.bias ? value + entry.bias : value), {}};
        entry.bias = 0;
        return;
    }
    WideScore score{scaled_score(base), {}};
    if (entry.repeated && base != 0) {
        score = scale_repetition(score.high, repetition);
    }
    entry.score = score;
}

__device__ __noinline__ ScaledScore score_difference(const RepetitionEntry& left,
                                                     const RepetitionEntry& right) {
    const auto lhs = left.score;
    const auto rhs = right.score;
    const auto high = add_scaled(lhs.high, {-rhs.high.fraction, rhs.high.exponent});
    const auto low = add_scaled(lhs.low, {-rhs.low.fraction, rhs.low.exponent});
    if (left.bias == right.bias) {
        const auto residual = add_scaled(high.low, low.high);
        return add_scaled(high.high, residual.high).high;
    }

    const ScaledScore terms[]{high.high, high.low, low.high, low.low,
                              scaled_score(left.bias), scaled_score(-right.bias)};
    ScaledScore partials[6];
    int count = 0;
    for (auto term : terms) {
        int retained = 0;
        for (int index = 0; index < count; ++index) {
            const auto sum = add_scaled(term, partials[index]);
            if (sum.low.fraction != 0) partials[retained++] = sum.low;
            term = sum.high;
        }
        if (term.fraction != 0) partials[retained++] = term;
        count = retained;
    }
    return count ? partials[count - 1] : ScaledScore{};
}

struct RepetitionOrder {
    const RepetitionEntry* entries;

    __device__ bool operator()(int left, int right) const {
        if (left < 0) return false;
        if (right < 0) return true;
        const auto& left_entry = entries[left];
        const auto& right_entry = entries[right];
        const auto lhs = left_entry.score;
        const auto rhs = right_entry.score;
        const double first = lhs.high.fraction;
        const double second = rhs.high.fraction;
        if (isnan(first) != isnan(second)) return isnan(second);
        if (isnan(first) || first == CUDART_INF || second == CUDART_INF ||
            first == -CUDART_INF || second == -CUDART_INF) {
            return first == second || isnan(first) ? left < right : first > second;
        }
        if (left_entry.bias != right_entry.bias) {
            const double difference = score_difference(left_entry, right_entry).fraction;
            return difference ? difference > 0 : left < right;
        }
        const int high = compare_scaled(lhs.high, rhs.high);
        if (high) return high > 0;
        const int low = compare_scaled(lhs.low, rhs.low);
        return low ? low > 0 : left < right;
    }
};

struct RepetitionWinner {
    RepetitionOrder order;
    bool constrained;

    __device__ int operator()(int left, int right) const {
        return order(left, right) ? left : right;
    }

    __device__ int valid(int token) const {
        const double value = order.entries[token].score.high.fraction;
        return isnan(value) || (constrained && !isfinite(value)) ? -1 : token;
    }
};

template <bool Final>
__global__ void repetition_argmax(RepetitionScratch scratch, int count, bool constrained,
                                  int* winner) {
    using Reduce = cub::BlockReduce<int, group_size>;
    __shared__ typename Reduce::TempStorage temp;
    const RepetitionWinner best{{scratch.entries}, constrained};
    int token = -1;
    for (int index = blockIdx.x * group_size + threadIdx.x; index < count;
         index += gridDim.x * group_size) {
        token = best(token, Final ? scratch.state->tiles[index] : best.valid(index));
    }
    const int chosen = Reduce(temp).Reduce(token, best);
    if (threadIdx.x == 0) {
        if constexpr (Final) *winner = chosen;
        else scratch.state->tiles[blockIdx.x] = chosen;
    }
}

__device__ double score_distance(const RepetitionEntry& score, const RepetitionEntry& maximum,
                                  double temperature) {
    if (!isfinite(score.score.high.fraction)) return -CUDART_INF;
    const auto distance = score_difference(score, maximum);
    int exponent;
    const double fraction = frexp(temperature, &exponent);
    return fmin(0.0, scalbn(distance.fraction / fraction, distance.exponent - exponent));
}

__global__ void repetition_weights(RepetitionScratch scratch, int count, double temperature) {
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank < count) {
        const auto& maximum = scratch.entries[scratch.ranked[0]];
        const auto& score = scratch.entries[scratch.ranked[rank]];
        scratch.weights[rank] = !scratch.state->invalid && isfinite(maximum.score.high.fraction)
            ? exp(score_distance(score, maximum, temperature)) : 0.0;
    }
}

__global__ void sample_repetition(RepetitionScratch scratch, int vocabulary,
                                   SamplingParameters parameters, int position, int* token) {
    *token = -1;
    if (scratch.prefix[vocabulary - 1] > 0) {
        const int rank = choose_rank(scratch.weights, scratch.prefix, vocabulary,
                                     parameters, position);
        *token = scratch.ranked[rank];
    }
}

__global__ void score_repetition(RepetitionScratch scratch, int vocabulary, int top, int* token) {
    const int rank = threadIdx.x;
    const int winner = scratch.ranked[0];
    const double mass = scratch.prefix[vocabulary - 1];
    if (rank == 0) {
        *token = mass > 0 ? winner : -1;
        scratch.state->logprobs.selected = {*token, mass > 0 ? float(-log(mass)) : 0.0f};
    }
    if (rank < top && mass > 0) {
        const int id = scratch.ranked[rank];
        const auto& maximum = scratch.entries[winner];
        const auto& score = scratch.entries[id];
        const double logprob = score_distance(score, maximum, 1.0) - log(mass);
        scratch.state->logprobs.alternatives[rank] = {
            id, static_cast<float>(fmax(-static_cast<double>(FLT_MAX), logprob))};
    }
}

__global__ void export_repetition(RepetitionScratch scratch, DecoderLogit* output, int count) {
    const int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank < count) {
        const int id = scratch.ranked[rank];
        const auto& score = scratch.entries[id];
        double display = scalbn(score.score.high.fraction, score.score.high.exponent);
        const auto& maximum = scratch.entries[scratch.ranked[0]];
        if (scratch.state->wide && isfinite(score.score.high.fraction) &&
            isfinite(maximum.score.high.fraction)) {
            display = fmax(-static_cast<double>(FLT_MAX), score_distance(score, maximum, 1.0));
        }
        output[rank] = {isnan(display) ? -1 : id, static_cast<float>(display)};
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

cudaError_t adjust_logits(const __nv_bfloat16* source, float* scores, int vocabulary,
                          const DecoderLogitBias* bias, int count, const std::uint32_t* mask,
                          OutputPenalties penalties, int previous_token, cudaStream_t stream) {
    if (penalties.repetition != 1.0) return cudaErrorInvalidValue;
    initialize_scores<<<(vocabulary + group_size - 1) / group_size, group_size, 0, stream>>>(
        source, scores, vocabulary, mask, penalties, previous_token);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess || count == 0) return status;
    add_bias<<<(count + group_size - 1) / group_size, group_size, 0, stream>>>(scores, bias, count);
    return cudaGetLastError();
}

std::size_t repetition_workspace_bytes(int vocabulary, bool ranked) {
    std::size_t sort = 0;
    std::size_t scan = 0;
    if (ranked) {
        cub::DeviceMergeSort::SortKeys(
            nullptr, sort, static_cast<int*>(nullptr), vocabulary, RepetitionOrder{nullptr});
        cub::DeviceScan::InclusiveSum(
            nullptr, scan, static_cast<double*>(nullptr), static_cast<double*>(nullptr),
            vocabulary);
    }
    return repetition_scratch(nullptr, vocabulary, ranked).overhead + std::max(sort, scan);
}

cudaError_t select_repetition_logits(
    const __nv_bfloat16* source, int vocabulary, const DecoderLogitBias* bias, int bias_count,
    const std::uint32_t* mask, OutputPenalties penalties, int previous_token,
    RepetitionSelection selection, void* workspace, std::size_t bytes, cudaStream_t stream) {
    if (!workspace || vocabulary < 1 || vocabulary > token_mask + 1 ||
        (!selection.token && !selection.debug) || !(penalties.repetition > 0) ||
        selection.top > decoder_max_generated_top_logprobs || selection.top > vocabulary ||
        (selection.top >= 0 && !selection.logprobs) || selection.debug_count > vocabulary) {
        return cudaErrorInvalidValue;
    }
    const bool ranked = selection.sampling || selection.logprobs || selection.debug;
    const auto scratch = repetition_scratch(workspace, vocabulary, ranked);
    if (bytes < scratch.overhead) return cudaErrorInvalidValue;
    auto status = cudaMemsetAsync(scratch.state, 0, offsetof(RepetitionState, tiles), stream);
    if (status != cudaSuccess) return status;
    const int blocks = (vocabulary + group_size - 1) / group_size;
    prepare_repetition<<<blocks, group_size, 0, stream>>>(
        source, vocabulary, mask, penalties, previous_token, scratch);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (bias_count) {
        add_repetition_bias<<<(bias_count + group_size - 1) / group_size, group_size, 0, stream>>>(
            scratch.entries, bias, bias_count);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
    }
    encode_repetition<<<blocks, group_size, 0, stream>>>(scratch, vocabulary, penalties.repetition);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (!ranked) {
        repetition_argmax<false><<<blocks, group_size, 0, stream>>>(
            scratch, vocabulary, mask != nullptr, nullptr);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        repetition_argmax<true><<<1, group_size, 0, stream>>>(
            scratch, blocks, mask != nullptr, selection.token);
        return cudaGetLastError();
    }
    auto temp_bytes = bytes - scratch.overhead;
    status = cub::DeviceMergeSort::SortKeys(scratch.temporary, temp_bytes, scratch.ranked,
                                           vocabulary, RepetitionOrder{scratch.entries}, stream);
    if (status != cudaSuccess) return status;
    if (selection.debug) {
        const int count = selection.debug_count;
        export_repetition<<<(count + group_size - 1) / group_size, group_size, 0, stream>>>(
            scratch, scratch.debug, count);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        return cudaMemcpyAsync(selection.debug, scratch.debug, count * sizeof(DecoderLogit),
                               cudaMemcpyDeviceToHost, stream);
    }
    const double temperature = selection.sampling ? selection.sampling->temperature : 1.0;
    repetition_weights<<<blocks, group_size, 0, stream>>>(scratch, vocabulary, temperature);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    temp_bytes = bytes - scratch.overhead;
    status = cub::DeviceScan::InclusiveSum(scratch.temporary, temp_bytes, scratch.weights,
                                          scratch.prefix, vocabulary, stream);
    if (status != cudaSuccess) return status;
    if (selection.sampling) {
        sample_repetition<<<1, 1, 0, stream>>>(
            scratch, vocabulary, *selection.sampling, selection.position, selection.token);
        return cudaGetLastError();
    }
    score_repetition<<<1, group_size, 0, stream>>>(
        scratch, vocabulary, selection.top, selection.token);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    return cudaMemcpyAsync(selection.logprobs, &scratch.state->logprobs,
                           offsetof(GeneratedLogprobReadback, alternatives) +
                               selection.top * sizeof(DecoderLogprob),
                           cudaMemcpyDeviceToHost, stream);
}

cudaError_t check_allowed_logit(const float* scores, int* token, cudaStream_t stream) {
    validate_allowed<<<1, 1, 0, stream>>>(scores, token);
    return cudaGetLastError();
}

}
