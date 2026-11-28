#include "sglang/decoder_batch.h"
#include "sglang/decoder_sampling.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using sglang::Decoder;
using sglang::DecoderBatch;
using sglang::DecoderBatchStepStats;
using sglang::DecoderFinishReason;
using sglang::DecoderOptions;
using sglang::decoder::SamplingParameters;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check(cudaError_t status) {
    require(status == cudaSuccess, cudaGetErrorString(status));
}

template <typename Value>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) { check(cudaMalloc(&data, sizeof(Value) * count)); }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    Value* data = nullptr;
};

using Law = std::vector<std::pair<int, long double>>;

Law probabilities(const std::vector<__nv_bfloat16>& encoded, SamplingParameters options) {
    Law ranked;
    for (std::size_t index = 0; index < encoded.size(); ++index) {
        const auto logit = __bfloat162float(encoded[index]);
        if (std::isnan(logit) || logit == std::numeric_limits<float>::infinity()) {
            return {};
        }
        if (std::isfinite(logit)) {
            ranked.emplace_back(index, logit);
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
        return left.second > right.second;
    });
    if (ranked.empty()) {
        return {};
    }
    const auto largest = ranked.front().second;
    long double total = 0;
    for (auto& [token, weight] : ranked) {
        weight = std::exp((weight - largest) / static_cast<long double>(options.temperature));
        total += weight;
    }
    long double retained = 0;
    auto last = ranked.begin();
    for (; last != ranked.end() && last - ranked.begin() < options.top_k; ++last) {
        if ((options.top_p < 1 && retained > total * options.top_p) ||
            last->second < options.min_p) {
            break;
        }
        retained += last->second;
    }
    ranked.erase(last, ranked.end());
    for (auto& [token, weight] : ranked) {
        weight /= retained;
    }
    return ranked;
}

std::vector<int> draws(const std::vector<float>& scores, SamplingParameters options, int samples) {
    const std::vector<__nv_bfloat16> encoded(scores.begin(), scores.end());
    DeviceBuffer<__nv_bfloat16> device_scores(scores.size());
    DeviceBuffer<int> selected(samples);
    DeviceBuffer<std::byte> scratch(sglang::decoder::sampling_workspace_bytes());
    check(cudaMemcpy(device_scores.data, encoded.data(), scores.size() * sizeof(__nv_bfloat16),
                     cudaMemcpyHostToDevice));
    for (int position = 0; position < samples; ++position) {
        check(sglang::decoder::sample_logits(device_scores.data, scores.size(),
                                           selected.data + position, options, position,
                                           scratch.data, nullptr));
    }
    std::vector<int> tokens(samples);
    check(cudaMemcpy(tokens.data(), selected.data, tokens.size() * sizeof(int),
                     cudaMemcpyDeviceToHost));
    return tokens;
}

void check_law(const char* name, const std::vector<float>& scores, SamplingParameters options) {
    constexpr int samples = 4096;
    const Law law = probabilities({scores.begin(), scores.end()}, options);
    require(!law.empty(), "fixture did not produce a probability distribution");
    const auto tokens = draws(scores, options, samples);
    std::map<int, int> counts;
    for (const int token : tokens) {
        ++counts[token];
    }
    std::cout << name;
    for (const auto& [token, expected] : law) {
        const int seen = counts[token];
        counts.erase(token);
        const auto observed = static_cast<long double>(seen) / samples;
        const auto margin = 6 * std::sqrt(expected * (1 - expected) / samples) + 0.002;
        std::cout << " " << token << "=" << seen << "/" << samples << "(" << expected << ")";
        require(std::abs(observed - expected) <= margin, "sample frequency outside tolerance");
    }
    std::cout << '\n';
    require(counts.empty(), "sampler returned an excluded token");
}

void check_kernel() {
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    check_law("full mass before cap", {0, 1, 0, 0, 0}, {1, 0.5, 2, 84});
    check_law("cold", {0, 2, -1, 1, -inf, 0}, {0.75, 0.9, 5, 93});
    check_law("warm", {0, 2, -1, 1, -inf, 0}, {2.5, 0.9, 5, 93});
    check_law("equal prefix cutoff", {-0.0f, -inf, 0, 0, 0}, {1, 0.5, 3, 178});
    check_law("below equal prefix cutoff", {-0.0f, -inf, 0, 0, 0}, {1, 0.499, 3, 178});
    check_law("minimum mass", {0, 1, -2, 0, -inf}, {1, 0.99, 4, 840, 0.3});
    check_law("minimum mass and nucleus", {0, 2, 1, 0, -inf}, {2, 0.6, 3, 841, 0.5});
    check_law("minimum mass tied peak", {0, 1, 1, 0}, {1, 1, 4, 842, 1});
    check_law("negative BF16 score keys", {-300, -299, -298, -298}, {1, 1, 4, 843, 0.5});
    std::vector<float> stripes(248320, -inf);
    for (const int token : {31, 32, 2047, 2048, 248319}) {
        stripes[token] = 1;
    }
    check_law("ties across stripe boundaries", stripes, {1e-6, 1, 5, 351});
    check_law("clipped ties", stripes, {1e-6, 1, 4, 351});
    for (const auto& values : {std::vector<float>{-inf, -inf}, {0, inf}, {0, nan}}) {
        require(draws(values, {1, 1, 2, 42}, 1) == std::vector<int>{-1},
                "invalid or all-masked row produced a token");
    }
}

const std::vector<int> prompt{48, 25, 220, 16, 10, 16, 28};

std::vector<int> generate(const Decoder& decoder, DecoderOptions options) {
    const auto result = decoder.generate(prompt, options);
    require(result.finish_reason == DecoderFinishReason::length, "unexpected finish reason");
    require(result.output_token_ids.size() == static_cast<std::size_t>(options.max_new_tokens),
            "unexpected serial output length");
    if (options.mtp_steps != 0) {
        require(result.mtp_rounds > 0, "serial MTP verification did not run");
    }
    return result.output_token_ids;
}

void validation(const Decoder& decoder, DecoderOptions options) {
    const auto invalid = [&](const DecoderOptions& rejected) {
        try {
            decoder.generate(prompt, rejected);
        } catch (const std::invalid_argument&) {
            return;
        }
        throw std::runtime_error("invalid direct request accepted");
    };
    auto rejected = options;
    rejected.mtp_steps = sglang::decoder_max_mtp_steps + 1;
    invalid(rejected);
    rejected = options;
    rejected.temperature = std::numeric_limits<double>::quiet_NaN();
    invalid(rejected);
    rejected.temperature = std::numeric_limits<double>::infinity();
    invalid(rejected);
    rejected = options;
    rejected.top_p = std::numeric_limits<double>::quiet_NaN();
    invalid(rejected);
    rejected = options;
    rejected.top_k = 0;
    invalid(rejected);
    rejected = options;
    rejected.min_p = std::numeric_limits<double>::quiet_NaN();
    invalid(rejected);
}

struct Lane {
    int slot;
    DecoderOptions options;
    std::vector<int> tokens{};
    int accepted = 0;
    int drafted = 0;
    int discarded = 0;
    int boundaries = 0;
};

std::pair<std::uint32_t, int> drain(DecoderBatch& batch, std::vector<Lane>& lanes) {
    std::array<bool, sglang::decoder_batch_max_lanes> selected{};
    for (const auto& lane : lanes) {
        batch.admit(lane.slot, prompt, lane.options);
        selected[lane.slot] = true;
    }
    std::uint32_t graph_slots = 0;
    int verification = 0;
    int slices = 0;
    while (std::any_of(selected.begin(), selected.end(), [](bool active) { return active; })) {
        require(++slices <= 32, "batch failed to retire requests");
        DecoderBatchStepStats stats;
        const auto steps = batch.advance_chunked(selected, &stats);
        require(stats.verification_lanes == 0 || stats.graph_slots == 0,
                "target graph replay overlapped MTP verification");
        graph_slots |= stats.graph_slots;
        verification += stats.verification_lanes;
        for (auto& lane : lanes) {
            const auto& step = steps[lane.slot];
            require(!step.sampling_failed, "valid batch sampled row failed");
            if (step.token) {
                lane.tokens.push_back(*step.token);
            }
            lane.tokens.insert(lane.tokens.end(), step.extra_tokens.begin(),
                               step.extra_tokens.begin() + step.extra_token_count);
            lane.accepted += step.mtp_accepted_tokens;
            lane.drafted += step.mtp_drafted_tokens;
            require(step.mtp_replayed_rows == 0, "Sampled MTP replayed committed GDN rows");
            lane.discarded +=
                std::max(0, step.mtp_verified_rows - static_cast<int>(step.token.has_value()) -
                                step.extra_token_count);
            lane.boundaries += step.mtp_boundary;
            if (step.finish_reason) {
                require(*step.finish_reason == DecoderFinishReason::length,
                        "unexpected batch finish reason");
                selected[lane.slot] = false;
            }
        }
    }
    require(batch.available() && batch.request_device_allocation_bytes() == 0,
            "batch retained a retired request");
    return {graph_slots, verification};
}

void check_sampled_mtp(const Decoder& decoder, DecoderOptions sampled) {
    std::vector<Lane> lanes;
    std::vector<std::vector<int>> expected;
    sampled.top_logits = 8;
    for (int depth = 1; depth <= 4; ++depth) {
        sampled.seed = -100 + depth;
        sampled.max_new_tokens = 6 + depth * 3;
        sampled.min_p = depth % 2 ? 0.04 : 0.0;
        const auto control = decoder.generate(prompt, sampled);
        auto options = sampled;
        options.mtp_steps = depth;
        const auto candidate = decoder.generate(prompt, options);
        require(candidate.output_token_ids == control.output_token_ids &&
                    candidate.finish_reason == control.finish_reason,
                "serial sampled MTP changed the K0 trace");
        require(candidate.step_top_logits.size() == control.step_top_logits.size(),
                "serial sampled MTP changed the logit row count");
        for (std::size_t row = 0; row < control.step_top_logits.size(); ++row) {
            const auto& reference = control.step_top_logits[row];
            const auto& verified = candidate.step_top_logits[row];
            require(std::equal(reference.begin(), reference.end(), verified.begin(), verified.end(),
                               [](const auto& first, const auto& second) {
                                   return first.token_id == second.token_id &&
                                          first.logit == second.logit;
                               }), "serial sampled MTP changed ordered BF16 top logits");
        }
        require(candidate.mtp_drafted_tokens > 0, "serial sampled MTP did not draft");
        options.top_logits = 0;
        lanes.push_back({depth - 1, options});
        expected.push_back(control.output_token_ids);
    }
    DecoderBatch batch(decoder, {true, 4});
    const auto mixed = drain(batch, lanes);
    require(mixed.second > 0, "heterogeneous sampled MTP batch did not verify");
    int discarded = 0;
    for (std::size_t index = 0; index < lanes.size(); ++index) {
        const auto& lane = lanes[index];
        require(lane.tokens == expected[index] && lane.drafted > 0,
                "heterogeneous sampled MTP changed a K0 trace");
        discarded += lane.discarded;
        std::cout << "sampled K" << lane.options.mtp_steps << " drafted=" << lane.drafted
                  << " accepted=" << lane.accepted << " discarded=" << lane.discarded
                  << " boundary=" << lane.boundaries << '\n';
    }
    require(discarded > 0, "heterogeneous fixture did not exercise target rollback");

    auto stopped = lanes.back().options;
    stopped.stop_token_id = expected.back()[2];
    const auto candidate_stop = decoder.generate(prompt, stopped);
    stopped.mtp_steps = 0;
    const auto control_stop = decoder.generate(prompt, stopped);
    require(candidate_stop.finish_reason == DecoderFinishReason::stop &&
                candidate_stop.output_token_ids == control_stop.output_token_ids,
            "sampled MTP changed stop behavior");

    auto cancelled = lanes.back().options;
    int emitted = 0;
    sglang::DecoderCallbacks callbacks;
    callbacks.on_token = [&](int) { return ++emitted < 3; };
    const auto partial = decoder.generate(prompt, cancelled, callbacks);
    require(partial.finish_reason == DecoderFinishReason::cancelled &&
                partial.output_token_ids.size() == 3 &&
                std::equal(partial.output_token_ids.begin(), partial.output_token_ids.end(),
                           expected.back().begin()), "sampled MTP changed cancellation prefix");
    batch.admit(3, prompt, cancelled);
    std::array<bool, sglang::decoder_batch_max_lanes> active{};
    active[3] = true;
    const auto first = batch.advance_chunked(active);
    require(first[3].token == expected.back().front(), "sampled MTP prefill stream changed");
    const auto draft = batch.advance_chunked(active);
    require(draft[3].mtp_drafted_tokens > 0, "cancel fixture did not verify");
    batch.cancel(3);
    require(batch.request_device_allocation_bytes() == 0, "cancel retained sampled MTP state");
    lanes = {{3, cancelled}};
    drain(batch, lanes);
    require(lanes[0].tokens == expected.back(), "re-admission changed the sample stream");
}

void check_decoder(const char* checkpoint) {
    Decoder decoder(checkpoint, 0, true);
    DecoderOptions sampled;
    sampled.max_new_tokens = 9;
    sampled.temperature = 1.8;
    sampled.top_k = 80;
    sampled.top_p = 0.9;
    sampled.seed = std::numeric_limits<std::int64_t>::min();
    const auto wanted = generate(decoder, sampled);
    sampled.eager = true;
    require(generate(decoder, sampled) == wanted, "serial graph and eager sampling differ");
    sampled.eager = false;
    auto other = sampled;
    other.seed = std::numeric_limits<std::int64_t>::max();
    const auto wanted_other = generate(decoder, other);
    require(wanted != wanted_other, "signed seed extremes yielded identical fixture streams");
    DecoderOptions greedy;
    greedy.max_new_tokens = sampled.max_new_tokens;
    const auto wanted_greedy = generate(decoder, greedy);
    auto near_zero = sampled;
    near_zero.temperature = std::nextafter(1e-6, 0);
    require(generate(decoder, near_zero) == wanted_greedy, "sub-threshold path did not use greedy");
    auto rank_one = sampled;
    rank_one.top_k = 1;
    require(generate(decoder, rank_one) == wanted_greedy, "rank-one path did not use greedy");
    auto speculative = greedy;
    speculative.mtp_steps = 4;
    require(generate(decoder, speculative) == wanted_greedy, "greedy MTP changed IDs");
    validation(decoder, sampled);

    DecoderBatch batch(decoder, {true, 32});
    std::vector<Lane> lanes{{31, sampled}, {4, greedy}, {1, other}};
    const auto graph = drain(batch, lanes);
    if (decoder.uses_b200_operators()) {
        require(graph.first == 0 && graph.second == 0,
                "B200 ordered mixed batch used a graph");
    } else {
        require(graph.first == (0x80000000u | (1u << 4) | (1u << 1)) && graph.second == 0,
                "mixed batch graph did not replay with the sampled high slot");
    }
    const auto compare = [&] {
        require(lanes[0].tokens == wanted && lanes[1].tokens == wanted_greedy &&
                    lanes[2].tokens == wanted_other, "request RNG changed under slot reassignment");
    };
    compare();
    lanes = {{31, sampled}};
    const auto serial_graph = drain(batch, lanes);
    require(serial_graph.first == (std::uint32_t{1} << 31) && serial_graph.second == 0,
            "graph-eligible serial decode did not replay");
    require(lanes[0].tokens == wanted, "serial graph replay changed the sampled stream");
    lanes = {{1, sampled}, {0, greedy}, {31, other}, {4, greedy}, {6, greedy}};
    const auto fallback = drain(batch, lanes);
    require(fallback.first == 0 && fallback.second == 0,
            "width-five graph fallback was not eager");
    compare();
    DecoderBatch mtp_batch(decoder, {true, 4});
    lanes = {{1, sampled}, {0, speculative}, {3, other}};
    const auto mixed_mtp = drain(mtp_batch, lanes);
    require(mixed_mtp.second > 0, "mixed MTP verification did not run");
    compare();

    batch.admit(31, prompt, sampled);
    batch.cancel(31);
    require(batch.request_device_allocation_bytes() == 0, "cancelled batch retained allocations");
    lanes = {{31, sampled}};
    drain(batch, lanes);
    require(lanes[0].tokens == wanted, "seeded result changed after a cancelled slot");
    sglang::DecoderCallbacks callbacks;
    callbacks.on_token = [](int) { return false; };
    const auto cancelled = decoder.generate(prompt, sampled, callbacks);
    require(cancelled.finish_reason == DecoderFinishReason::cancelled &&
                cancelled.output_token_ids == std::vector<int>{wanted.front()},
            "serial cancellation changed the first sample");
    require(generate(decoder, sampled) == wanted, "serial seed leaked across cancellation");
    std::cout << "decoder: eager, graph, mixed MTP, slot reassignment, and cleanup passed\n";
    check_sampled_mtp(decoder, sampled);
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 1 || argc == 2, "expected an optional checkpoint path");
        check_kernel();
        if (argc == 2) {
            check_decoder(argv[1]);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
