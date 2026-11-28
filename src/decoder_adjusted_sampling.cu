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

constexpr unsigned token_bits = 18;
constexpr int max_vocabulary = static_cast<int>(1U << token_bits);
constexpr std::uint64_t token_mask = max_vocabulary - 1;
constexpr std::uint64_t invalid_key = (std::uint64_t{1} << (32U + token_bits)) - 1U;
using ScoreKey = std::uint64_t;

struct BiasSamplingScratch {
    ScoreKey* keys;
    ScoreKey* sorted;
    double* weights;
    double* prefix;
    void* temporary;
};

__global__ void rank_scores(const float* scores, int vocabulary, ScoreKey* keys) {
    const int token = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (token < vocabulary) {
        const float score = scores[token];
        const auto bits = __float_as_uint(score == 0 ? 0.0F : score);
        const auto ordered = bits ^ (((bits & 0x80000000U) != 0U) ? 0xffffffffU : 0x80000000U);
        const auto key =
            (ScoreKey{ordered} << token_bits) | (token_mask - static_cast<unsigned>(token));
        if (isfinite(score)) {
            keys[token] = key;
        } else {
            keys[token] = score == -CUDART_INF_F ? 0 : invalid_key;
        }
    }
}

__device__ double ranked_weight(ScoreKey key, ScoreKey peak, double temperature) {
    const auto score = [](ScoreKey value) {
        const auto ordered = static_cast<unsigned>(value >> token_bits);
        return static_cast<double>(
            __uint_as_float(ordered ^ ((ordered & 0x80000000U) != 0U ? 0x80000000U : 0xffffffffU)));
    };
    return key != 0 && peak != invalid_key ? exp((score(key) - score(peak)) / temperature) : 0.0;
}

__global__ void measure_weights(const ScoreKey* keys, int vocabulary, double* weights,
                                double temperature) {
    const int rank = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (rank < vocabulary) {
        weights[rank] = ranked_weight(keys[rank], keys[0], temperature);
    }
}

__device__ int first_crossing(int count, const double* prefix, double cutoff) {
    return first_reached(0, count, [=](int index) { return prefix[index] > cutoff; });
}

template <typename Scratch>
__device__ int choose_rank(const Scratch& scratch, int vocabulary, SamplingParameters parameters,
                           double uniform) {
    const auto* weights = scratch.weights;
    const auto* prefix = scratch.prefix;
    int keep = min(vocabulary, parameters.top_k);
    if (parameters.top_p < 1.0) {
        const auto cutoff = parameters.top_p * prefix[vocabulary - 1];
        keep = min(keep, first_crossing(keep, prefix, cutoff) + 1);
    }
    if (parameters.min_p > 0.0) {
        keep = first_reached(1, keep,
                             [=](int index) { return !(weights[index] >= parameters.min_p); });
    }
    const double mass = prefix[keep - 1];
    const double draw = min(uniform * mass, nextafter(mass, 0.0));
    return first_crossing(keep, prefix, draw);
}

__global__ void choose_biased_token(BiasSamplingScratch scratch, int vocabulary,
                                    SamplingParameters parameters, double uniform, int* token) {
    const auto peak = scratch.sorted[0];
    *token = -1;
    if (peak != 0 && peak != invalid_key) {
        const int rank = choose_rank(scratch, vocabulary, parameters, uniform);
        *token = static_cast<int>(token_mask - (scratch.sorted[rank] & token_mask));
    }
}

std::size_t biased_sampling_temporary_bytes(int vocabulary) {
    std::size_t sort = 0;
    std::size_t scan = 0;
    cub::DeviceRadixSort::SortKeysDescending(nullptr, sort, static_cast<ScoreKey*>(nullptr),
                                             static_cast<ScoreKey*>(nullptr), vocabulary, 0,
                                             32 + token_bits);
    cub::DeviceScan::InclusiveSum(nullptr, scan, static_cast<double*>(nullptr),
                                  static_cast<double*>(nullptr), vocabulary);
    return std::max(sort, scan);
}

__device__ bool is_allowed(LogitMasks masks, int token) {
    const auto bit = std::uint32_t{1} << (static_cast<unsigned>(token) % 32U);
    const int word = token / 32;
    return ((masks.allowed == nullptr) || ((masks.allowed[word] & bit) != 0U)) &&
           ((masks.denied == nullptr) || ((masks.denied[word] & bit) == 0U));
}

__device__ float subtract_history(float score, OutputPenalties penalties, int token,
                                  int previous_token, bool& seen) {
    float total = 0.0F;
    if (penalties.frequency_totals != nullptr) {
        total = penalties.frequency_totals[token];
        if (token == previous_token) {
            total += penalties.frequency;
            penalties.frequency_totals[token] = total;
        }
    }
    if (penalties.seen != nullptr) {
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
                                  LogitMasks masks, OutputPenalties penalties, int previous_token) {
    const int token = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (token < vocabulary) {
        bool seen = false;
        const float score = subtract_history(__bfloat162float(source[token]), penalties, token,
                                             previous_token, seen);
        scores[token] = is_allowed(masks, token) ? score : -CUDART_INF_F;
    }
}

__global__ void add_bias(float* scores, const DecoderLogitBias* bias, int count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
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

__device__ ScaledScore scaled_score(ScaledScore score) {
    if (score.fraction == 0 || !isfinite(score.fraction))
        return {score.fraction, 0};
    int shift = 0;
    const double fraction = frexp(score.fraction, &shift);
    return {fraction, score.exponent + shift};
}

struct WideScore {
    ScaledScore high;
    ScaledScore low;
};

__device__ WideScore add_scaled(ScaledScore left, ScaledScore right) {
    if (left.fraction == 0)
        return {right, {}};
    if (right.fraction == 0 || !isfinite(left.fraction))
        return {left, {}};
    if (left.exponent < right.exponent) {
        const auto swap = left;
        left = right;
        right = swap;
    }
    if (left.exponent - right.exponent > 60)
        return {left, right};
    const double aligned = scalbn(right.fraction, right.exponent - left.exponent);
    const double sum = left.fraction + aligned;
    const double rounded = sum - left.fraction;
    const double remainder = (left.fraction - (sum - rounded)) + (aligned - rounded);
    return {scaled_score({sum, left.exponent}), scaled_score({remainder, left.exponent})};
}

__device__ WideScore scale_repetition(ScaledScore score, double repetition) {
    int exponent = 0;
    const double fraction = frexp(repetition, &exponent);
    if (score.fraction < 0) {
        const double product = score.fraction * fraction;
        return {
            scaled_score({product, score.exponent + exponent}),
            scaled_score({fma(score.fraction, fraction, -product), score.exponent + exponent}),
        };
    }
    const double quotient = score.fraction / fraction;
    const double remainder = fma(-quotient, fraction, score.fraction) / fraction;
    return {
        scaled_score({quotient, score.exponent - exponent}),
        scaled_score({remainder, score.exponent - exponent}),
    };
}

__device__ int compare_scaled(ScaledScore left, ScaledScore right) {
    if (left.fraction == right.fraction && left.exponent == right.exponent)
        return 0;
    const bool same_sign =
        (left.fraction > 0 && right.fraction > 0) || (left.fraction < 0 && right.fraction < 0);
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
    offset = (offset + (vocabulary * sizeof(int)) + 7) & ~std::size_t{7};
    auto* weights = reinterpret_cast<double*>(address + offset);
    auto* prefix = reinterpret_cast<double*>(address + offset + (vocabulary * sizeof(double)));
    if (ranked)
        offset += 2 * static_cast<std::size_t>(vocabulary) * sizeof(double);
    auto* debug = reinterpret_cast<DecoderLogit*>(address + offset);
    if (ranked)
        offset += vocabulary * sizeof(DecoderLogit);
    offset = (offset + 255) & ~std::size_t{255};
    return {
        state,  entries, order, weights, prefix, debug, reinterpret_cast<void*>(address + offset),
        offset,
    };
}

__global__ void prepare_repetition(const __nv_bfloat16* source, int vocabulary, LogitMasks masks,
                                   OutputPenalties penalties, int previous,
                                   RepetitionScratch scratch) {
    const int token = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    bool wide = false;
    bool invalid = false;
    if (token < vocabulary) {
        bool seen = false;
        float score =
            subtract_history(__bfloat162float(source[token]), penalties, token, previous, seen);
        if (!is_allowed(masks, token))
            score = -CUDART_INF_F;
        scratch.entries[token].base = score;
        scratch.entries[token].bias = 0;
        scratch.entries[token].repeated = seen;
        scratch.ranked[token] = token;
        invalid = isnan(score) || score == CUDART_INF_F;
        if (seen && score != 0 && isfinite(score)) {
            const auto base = static_cast<double>(score);
            const double scaled =
                base < 0 ? base * penalties.repetition : base / penalties.repetition;
            wide =
                fabs(scaled) > 0x1p24 || fabs(scaled) < FLT_MIN || penalties.repetition < FLT_MIN;
        }
    }
    if ((__syncthreads_or(static_cast<int>(wide)) != 0) && threadIdx.x == 0)
        atomicOr(&scratch.state->wide, 1U);
    if ((__syncthreads_or(static_cast<int>(invalid)) != 0) && threadIdx.x == 0)
        atomicOr(&scratch.state->invalid, 1U);
}

__global__ void add_repetition_bias(RepetitionEntry* entries, const DecoderLogitBias* bias,
                                    int count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index < count)
        entries[bias[index].token_id].bias = bias[index].bias;
}

__global__ void encode_repetition(int vocabulary, RepetitionScratch scratch, double repetition) {
    const int token = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (token >= vocabulary)
        return;
    auto& entry = scratch.entries[token];
    const auto base = static_cast<double>(entry.base);
    if ((scratch.state->wide == 0U) || !isfinite(base)) {
        float value = entry.base;
        if (entry.repeated && isfinite(base)) {
            value = static_cast<float>(base < 0 ? base * repetition : base / repetition);
        }
        entry.base = (entry.bias != 0.0F) ? value + entry.bias : value;
        entry.score = {scaled_score({entry.base, 0}), {}};
        entry.bias = 0;
        return;
    }
    WideScore score{scaled_score({base, 0}), {}};
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

    const ScaledScore terms[]{
        high.high,
        high.low,
        low.high,
        low.low,
        scaled_score({left.bias, 0}),
        scaled_score({-right.bias, 0}),
    };
    ScaledScore partials[6];
    int count = 0;
    for (auto term : terms) {
        int retained = 0;
        for (int index = 0; index < count; ++index) {
            const auto sum = add_scaled(term, partials[index]);
            if (sum.low.fraction != 0)
                partials[retained++] = sum.low;
            term = sum.high;
        }
        if (term.fraction != 0)
            partials[retained++] = term;
        count = retained;
    }
    return (count != 0) ? partials[count - 1] : ScaledScore{};
}

struct RepetitionOrder {
    const RepetitionEntry* entries;
    const unsigned* wide;

    __device__ bool operator()(int left, int right) const {
        if (left < 0)
            return false;
        if (right < 0)
            return true;
        const auto& left_entry = entries[left];
        const auto& right_entry = entries[right];
        if ((*wide) == 0U) {
            const float first = left_entry.base;
            const float second = right_entry.base;
            if (isnan(first) != isnan(second))
                return isnan(second);
            return first == second || isnan(first) ? left < right : first > second;
        }
        const auto lhs = left_entry.score;
        const auto rhs = right_entry.score;
        const double first = lhs.high.fraction;
        const double second = rhs.high.fraction;
        if (isnan(first) != isnan(second))
            return isnan(second);
        if (isnan(first) || first == CUDART_INF || second == CUDART_INF || first == -CUDART_INF ||
            second == -CUDART_INF) {
            return first == second || isnan(first) ? left < right : first > second;
        }
        if (left_entry.bias != right_entry.bias) {
            const double difference = score_difference(left_entry, right_entry).fraction;
            return (difference != 0.0) ? difference > 0 : left < right;
        }
        const int high = compare_scaled(lhs.high, rhs.high);
        if (high != 0)
            return high > 0;
        const int low = compare_scaled(lhs.low, rhs.low);
        return (low != 0) ? low > 0 : left < right;
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
    const RepetitionWinner best{{scratch.entries, &scratch.state->wide}, constrained};
    int token = -1;
    for (int index = static_cast<int>((blockIdx.x * group_size) + threadIdx.x); index < count;
         index += static_cast<int>(gridDim.x * group_size)) {
        token = best(token, Final ? scratch.state->tiles[index] : best.valid(index));
    }
    const int chosen = Reduce(temp).Reduce(token, best);
    if (threadIdx.x == 0) {
        if constexpr (Final)
            *winner = chosen;
        else
            scratch.state->tiles[blockIdx.x] = chosen;
    }
}

__device__ double score_distance(const RepetitionEntry& score, const RepetitionEntry& maximum,
                                 double temperature, bool wide) {
    if (!isfinite(score.score.high.fraction))
        return -CUDART_INF;
    const auto distance =
        wide ? score_difference(score, maximum)
             : scaled_score(
                   {static_cast<double>(score.base) - static_cast<double>(maximum.base), 0});
    int exponent = 0;
    const double fraction = frexp(temperature, &exponent);
    return fmin(0.0, scalbn(distance.fraction / fraction, distance.exponent - exponent));
}

__global__ void repetition_weights(int count, RepetitionScratch scratch, double temperature) {
    const int rank = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (rank < count) {
        const auto& maximum = scratch.entries[scratch.ranked[0]];
        const auto& score = scratch.entries[scratch.ranked[rank]];
        scratch.weights[rank] =
            (scratch.state->invalid == 0U) && isfinite(maximum.score.high.fraction)
                ? exp(score_distance(score, maximum, temperature, scratch.state->wide != 0U))
                : 0.0;
    }
}

__global__ void sample_repetition(RepetitionScratch scratch, int vocabulary,
                                  SamplingParameters parameters, double uniform, int* token) {
    *token = -1;
    if (scratch.prefix[vocabulary - 1] > 0) {
        const int rank = choose_rank(scratch, vocabulary, parameters, uniform);
        *token = scratch.ranked[rank];
    }
}

__global__ void score_repetition(int vocabulary, RepetitionScratch scratch, int top, int* token) {
    const int rank = static_cast<int>(threadIdx.x);
    const int winner = scratch.ranked[0];
    const double mass = scratch.prefix[vocabulary - 1];
    if (rank == 0) {
        *token = mass > 0 ? winner : -1;
        scratch.state->logprobs.selected = {
            *token,
            mass > 0 ? static_cast<float>(-log(mass)) : 0.0F,
        };
    }
    if (rank < top && mass > 0) {
        const int id = scratch.ranked[rank];
        const auto& maximum = scratch.entries[winner];
        const auto& score = scratch.entries[id];
        const double logprob =
            score_distance(score, maximum, 1.0, scratch.state->wide != 0U) - log(mass);
        scratch.state->logprobs.alternatives[rank] = {
            id,
            static_cast<float>(fmax(-static_cast<double>(FLT_MAX), logprob)),
        };
    }
}

__global__ void export_repetition(RepetitionScratch scratch, DecoderLogit* output, int count) {
    const int rank = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (rank < count) {
        const int id = scratch.ranked[rank];
        const auto& score = scratch.entries[id];
        double display = scalbn(score.score.high.fraction, score.score.high.exponent);
        const auto& maximum = scratch.entries[scratch.ranked[0]];
        if ((scratch.state->wide != 0U) && isfinite(score.score.high.fraction) &&
            isfinite(maximum.score.high.fraction)) {
            display =
                fmax(-static_cast<double>(FLT_MAX), score_distance(score, maximum, 1.0, true));
        }
        output[rank] = {isnan(display) ? -1 : id, static_cast<float>(display)};
    }
}

}

std::size_t biased_sampling_workspace_bytes(int vocabulary) {
    return (2 * static_cast<std::size_t>(vocabulary) * (sizeof(ScoreKey) + sizeof(double))) +
           biased_sampling_temporary_bytes(vocabulary) + 255;
}

cudaError_t sample_logits(const float* scores, int vocabulary, int* token,
                          SamplingParameters parameters, int position, void* workspace,
                          std::size_t workspace_bytes, cudaStream_t stream) {
    if (vocabulary < 1 || vocabulary > max_vocabulary || parameters.top_k < 1) {
        return cudaErrorInvalidValue;
    }
    auto* keys = static_cast<ScoreKey*>(workspace);
    auto* weights = reinterpret_cast<double*>(keys + (2 * static_cast<std::ptrdiff_t>(vocabulary)));
    const auto end =
        reinterpret_cast<std::uintptr_t>(weights + (2 * static_cast<std::ptrdiff_t>(vocabulary)));
    const BiasSamplingScratch scratch{
        keys,
        keys + vocabulary,
        weights,
        weights + vocabulary,
        reinterpret_cast<void*>((end + 255) & ~std::uintptr_t{255}),
    };
    const int blocks = (vocabulary + group_size - 1) / group_size;
    const auto offset = reinterpret_cast<std::uintptr_t>(scratch.temporary) -
                        reinterpret_cast<std::uintptr_t>(workspace);
    if (workspace_bytes <= offset) {
        return cudaErrorInvalidValue;
    }
    auto bytes = workspace_bytes - offset;
    rank_scores<<<blocks, group_size, 0, stream>>>(scores, vocabulary, scratch.keys);
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    status = cub::DeviceRadixSort::SortKeysDescending(scratch.temporary, bytes, scratch.keys,
                                                      scratch.sorted, vocabulary, 0,
                                                      32 + token_bits, stream);
    if (status != cudaSuccess)
        return status;
    measure_weights<<<blocks, group_size, 0, stream>>>(scratch.sorted, vocabulary, scratch.weights,
                                                       parameters.temperature);
    status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    bytes = workspace_bytes - offset;
    status = cub::DeviceScan::InclusiveSum(scratch.temporary, bytes, scratch.weights,
                                           scratch.prefix, vocabulary, stream);
    if (status != cudaSuccess)
        return status;
    choose_biased_token<<<1, 1, 0, stream>>>(scratch, vocabulary, parameters,
                                             sampling_uniform(parameters.seed, position), token);
    return cudaGetLastError();
}

cudaError_t adjust_logits(const __nv_bfloat16* source, float* scores, int vocabulary,
                          const DecoderLogitBias* bias, int count, LogitMasks masks,
                          OutputPenalties penalties, int previous_token, cudaStream_t stream) {
    if (penalties.repetition != 1.0)
        return cudaErrorInvalidValue;
    initialize_scores<<<(vocabulary + group_size - 1) / group_size, group_size, 0, stream>>>(
        source, scores, vocabulary, masks, penalties, previous_token);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess || count == 0)
        return status;
    add_bias<<<(count + group_size - 1) / group_size, group_size, 0, stream>>>(scores, bias, count);
    return cudaGetLastError();
}

std::size_t repetition_workspace_bytes(int vocabulary, bool ranked) {
    std::size_t sort = 0;
    std::size_t scan = 0;
    if (ranked) {
        cub::DeviceMergeSort::SortKeys(nullptr, sort, static_cast<int*>(nullptr), vocabulary,
                                       RepetitionOrder{nullptr, nullptr});
        cub::DeviceScan::InclusiveSum(nullptr, scan, static_cast<double*>(nullptr),
                                      static_cast<double*>(nullptr), vocabulary);
    }
    return repetition_scratch(nullptr, vocabulary, ranked).overhead + std::max(sort, scan);
}

cudaError_t select_repetition_logits(const __nv_bfloat16* source, int vocabulary,
                                     const DecoderLogitBias* bias, int bias_count, LogitMasks masks,
                                     OutputPenalties penalties, int previous_token,
                                     RepetitionSelection selection, void* workspace,
                                     std::size_t bytes, cudaStream_t stream) {
    if ((workspace == nullptr) || vocabulary < 1 || vocabulary > max_vocabulary ||
        ((selection.token == nullptr) && (selection.debug == nullptr)) ||
        !(penalties.repetition > 0) || selection.top > decoder_max_generated_top_logprobs ||
        selection.top > vocabulary || (selection.top >= 0 && (selection.logprobs == nullptr)) ||
        selection.debug_count > vocabulary) {
        return cudaErrorInvalidValue;
    }
    const bool ranked = (selection.sampling != nullptr) || (selection.logprobs != nullptr) ||
                        (selection.debug != nullptr);
    const auto scratch = repetition_scratch(workspace, vocabulary, ranked);
    if (bytes < scratch.overhead)
        return cudaErrorInvalidValue;
    auto status = cudaMemsetAsync(scratch.state, 0, offsetof(RepetitionState, tiles), stream);
    if (status != cudaSuccess)
        return status;
    const int blocks = (vocabulary + group_size - 1) / group_size;
    prepare_repetition<<<blocks, group_size, 0, stream>>>(source, vocabulary, masks, penalties,
                                                          previous_token, scratch);
    status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    if (bias_count != 0) {
        add_repetition_bias<<<(bias_count + group_size - 1) / group_size, group_size, 0, stream>>>(
            scratch.entries, bias, bias_count);
        status = cudaGetLastError();
        if (status != cudaSuccess)
            return status;
    }
    encode_repetition<<<blocks, group_size, 0, stream>>>(vocabulary, scratch, penalties.repetition);
    status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    if (!ranked) {
        const bool masked = (masks.allowed != nullptr) || (masks.denied != nullptr);
        repetition_argmax<false>
            <<<blocks, group_size, 0, stream>>>(scratch, vocabulary, masked, nullptr);
        status = cudaGetLastError();
        if (status != cudaSuccess)
            return status;
        repetition_argmax<true>
            <<<1, group_size, 0, stream>>>(scratch, blocks, masked, selection.token);
        return cudaGetLastError();
    }
    auto temp_bytes = bytes - scratch.overhead;
    status = cub::DeviceMergeSort::SortKeys(
        scratch.temporary, temp_bytes, scratch.ranked, vocabulary,
        RepetitionOrder{scratch.entries, &scratch.state->wide}, stream);
    if (status != cudaSuccess)
        return status;
    if (selection.debug != nullptr) {
        const int count = selection.debug_count;
        export_repetition<<<(count + group_size - 1) / group_size, group_size, 0, stream>>>(
            scratch, scratch.debug, count);
        status = cudaGetLastError();
        if (status != cudaSuccess)
            return status;
        return cudaMemcpyAsync(selection.debug, scratch.debug, count * sizeof(DecoderLogit),
                               cudaMemcpyDeviceToHost, stream);
    }
    const double temperature =
        (selection.sampling != nullptr) ? selection.sampling->temperature : 1.0;
    repetition_weights<<<blocks, group_size, 0, stream>>>(vocabulary, scratch, temperature);
    status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    temp_bytes = bytes - scratch.overhead;
    status = cub::DeviceScan::InclusiveSum(scratch.temporary, temp_bytes, scratch.weights,
                                           scratch.prefix, vocabulary, stream);
    if (status != cudaSuccess)
        return status;
    if (selection.sampling != nullptr) {
        sample_repetition<<<1, 1, 0, stream>>>(
            scratch, vocabulary, *selection.sampling,
            sampling_uniform(selection.sampling->seed, selection.position), selection.token);
        return cudaGetLastError();
    }
    score_repetition<<<1, group_size, 0, stream>>>(vocabulary, scratch, selection.top,
                                                   selection.token);
    status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    return cudaMemcpyAsync(selection.logprobs, &scratch.state->logprobs,
                           offsetof(GeneratedLogprobReadback, alternatives) +
                               (selection.top * sizeof(DecoderLogprob)),
                           cudaMemcpyDeviceToHost, stream);
}

cudaError_t check_allowed_logit(const float* scores, int* token, cudaStream_t stream) {
    validate_allowed<<<1, 1, 0, stream>>>(scores, token);
    return cudaGetLastError();
}

}
