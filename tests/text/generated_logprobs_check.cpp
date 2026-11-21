#include "sglang/decoder_batch.h"
#include "sglang/text/decoder_logprobs.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using namespace sglang;
using namespace sglang::decoder;
constexpr int vocabulary = 248320;
constexpr int top_limit = decoder_max_generated_top_logprobs;
constexpr int maximum_vocabulary = 262144;
constexpr float infinity = std::numeric_limits<float>::infinity();

void require(bool condition, const char* description) {
    if (!condition) {
        throw std::runtime_error(description);
    }
}

void check(cudaError_t status) {
    require(status == cudaSuccess, cudaGetErrorString(status));
}

struct Reducer {
    cudaStream_t stream = nullptr;
    __nv_bfloat16* logits = nullptr;
    TextLogitCandidate* winner = nullptr;
    GeneratedLogprobReadback* output = nullptr;
    void* scratch = nullptr;

    Reducer() {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaMalloc(reinterpret_cast<void**>(&logits), maximum_vocabulary * sizeof(*logits)));
        check(cudaMalloc(reinterpret_cast<void**>(&winner), sizeof(*winner)));
        check(cudaMalloc(&scratch, generated_logprob_workspace_bytes()));
        check(cudaMallocHost(reinterpret_cast<void**>(&output), sizeof(*output)));
    }

    ~Reducer() {
        cudaFreeHost(output);
        cudaFree(scratch);
        cudaFree(winner);
        cudaFree(logits);
        cudaStreamDestroy(stream);
    }

    void verify(const std::vector<float>& scores, int top, int selected, bool valid = true) {
        std::vector<__nv_bfloat16> raw;
        for (const float score : scores) {
            raw.push_back(__float2bfloat16(score));
        }
        const TextLogitCandidate chosen{0, selected};
        std::memset(output, 0x5a, sizeof(*output));
        check(cudaMemcpyAsync(logits, raw.data(), raw.size() * sizeof(raw.front()),
                              cudaMemcpyHostToDevice, stream));
        check(cudaMemcpyAsync(winner, &chosen, sizeof(chosen), cudaMemcpyHostToDevice, stream));
        check(enqueue_generated_logprobs(logits, raw.size(), winner, top, scratch, output, stream));
        check(cudaStreamSynchronize(stream));
        const auto* bytes = reinterpret_cast<const unsigned char*>(output);
        const auto received = offsetof(GeneratedLogprobReadback, alternatives) +
                              top * sizeof(DecoderLogprob);
        require(std::all_of(bytes + received, bytes + sizeof(*output),
                            [](unsigned char byte) { return byte == 0x5a; }),
                "Readback wrote unrequested alternatives");
        if (!valid) {
            require(output->selected.token_id == -1, "Invalid row returned a selected ID");
            return;
        }

        std::vector<double> values;
        for (const auto score : raw) {
            values.push_back(__bfloat162float(score));
        }
        const double maximum = *std::max_element(values.begin(), values.end());
        const double mass = std::accumulate(values.begin(), values.end(), 0.0,
            [maximum](double sum, double score) { return sum + std::exp(score - maximum); });
        const double log_mass = std::log(mass);
        const auto verify_score = [&](const DecoderLogprob& score, int token) {
            const auto expected = std::max(-double{FLT_MAX}, (values[token] - maximum) - log_mass);
            require(score.token_id == token, "Selected ID or top tie order changed");
            require(std::abs(score.value - expected) <= 4e-6,
                    "Score disagrees with the full-vocabulary FP64 oracle");
        };
        verify_score(output->selected, selected);
        std::vector<int> ranks(raw.size());
        std::iota(ranks.begin(), ranks.end(), 0);
        const auto higher_score = [&](int left, int right) {
            return values[left] == values[right] ? left < right : values[left] > values[right];
        };
        std::partial_sort(ranks.begin(), ranks.begin() + top, ranks.end(), higher_score);
        for (int rank = 0; rank < top; ++rank) {
            verify_score(output->alternatives[rank], ranks[rank]);
        }
    }
};

void check_reducer() {
    Reducer reducer;
    reducer.verify({-0.0f, 0.0f, -0.0f, -9.0f, -infinity}, 5, 0);
    reducer.verify({7.0f}, 0, 0);
    std::vector<float> scores(vocabulary, -8.0f);
    for (const int token : {0, 255, 256, 1023, 1024, 2048, vocabulary - 1}) {
        scores[token] = 2.0f;
    }
    reducer.verify(scores, top_limit, 0);
    reducer.verify(scores, 0, 0);
    std::fill(scores.begin(), scores.end(), -infinity);
    scores.front() = 3.0f;
    scores.back() = 7.0f;
    reducer.verify(scores, top_limit, 0);
    scores.resize(1025);
    reducer.verify(scores, top_limit, 0);
    scores.assign(maximum_vocabulary, 3.0f);
    reducer.verify(scores, top_limit, 0);
    reducer.verify({1.0f, 2.0f}, 2, -1, false);
    reducer.verify({1.0f, 2.0f}, 2, 2, false);
    reducer.verify({1.0f, infinity}, 2, 1, false);
    reducer.verify({1.0f, std::numeric_limits<float>::quiet_NaN()}, 2, 0, false);
    reducer.verify({-infinity, -infinity}, 2, 0, false);

    for (const auto invalid : {std::array{0, 0}, {maximum_vocabulary + 1, 0}, {1, 2},
                               {vocabulary, -1}, {vocabulary, top_limit + 1}}) {
        require(enqueue_generated_logprobs(reducer.logits, invalid[0], reducer.winner, invalid[1],
                    reducer.scratch, reducer.output, reducer.stream) == cudaErrorInvalidValue,
                "Reducer accepted invalid arguments");
    }
    std::cout << "12 BF16 reducer rows and five argument rejections passed\n";
}

struct Lane {
    int slot;
    int phrase;
    int top;
    int budget;
    int stop = -1;
    std::vector<int> output{};
    std::optional<DecoderFinishReason> finish{};
};

std::vector<int> make_prompt(int phrase) {
    const std::array unit{12675 + phrase, 220, 96748, 198, 785, 279, 1495, 13};
    std::vector<int> prompt;
    for (int repeat = 0; repeat < 16; ++repeat) {
        prompt.insert(prompt.end(), unit.begin(), unit.end());
    }
    return prompt;
}

void drain(DecoderBatch& batch, std::vector<Lane>& lanes, int cached, bool graphs) {
    std::array<bool, decoder_batch_max_lanes> selected{};
    for (const auto& lane : lanes) {
        DecoderOptions options;
        options.generated_top_logprobs = lane.top;
        options.max_new_tokens = lane.budget;
        if (lane.stop != -1) {
            options.stop_token_id = lane.stop;
        }
        const int restored = batch.admit(lane.slot, make_prompt(lane.phrase), options);
        require(restored == (lane.budget ? cached : 0), "Unexpected prefix restore");
        selected[lane.slot] = true;
    }
    std::uint32_t graph_slots = 0;
    int round = 0;
    while (std::any_of(selected.begin(), selected.end(), [](bool live) { return live; })) {
        require(round++ < 20, "Batch did not retire");
        DecoderBatchStepStats stats;
        const auto steps = batch.advance_chunked(selected, &stats);
        graph_slots |= stats.graph_slots;
        for (auto& lane : lanes) {
            if (!selected[lane.slot]) {
                continue;
            }
            const auto& step = steps[lane.slot];
            require(!step.sampling_failed && !step.constraint_failed && step.extra_token_count == 0,
                    "Greedy K0 batch step failed");
            if (step.token) {
                lane.output.push_back(*step.token);
            }
            require(static_cast<bool>(step.generated_logprobs) == (step.token && lane.top >= 0),
                    "Unexpected score attachment");
            if (step.generated_logprobs) {
                const auto& score = *step.generated_logprobs;
                require(score.selected.token_id == *step.token &&
                            std::isfinite(score.selected.value) && score.selected.value <= 0 &&
                            score.alternatives.size() == static_cast<std::size_t>(lane.top),
                        "Invalid scored step");
                if (lane.top) {
                    const auto& best = score.alternatives.front();
                    require(best.token_id == score.selected.token_id &&
                                best.value == score.selected.value,
                            "First alternative disagrees with greedy selection");
                }
            }
            if (step.finish_reason) {
                selected[lane.slot] = false;
                lane.finish = step.finish_reason;
            }
        }
    }
    const auto graph_lane = std::uint32_t{1} << lanes.front().slot;
    require(graphs ? (graph_slots & graph_lane) != 0 : graph_slots == 0,
            "Expected scored graph replay or eager route was not observed");
    require(batch.available() && batch.request_device_allocation_bytes() == 0,
            "Batch retained a retired request");
}

void check_checkpoint(const char* checkpoint) {
    Decoder decoder(checkpoint, 0, false, Fp8PrefillPolicy::exact);
    for (const bool graphs : {false, true}) {
        DecoderBatch batch(decoder, {graphs, 4, 4});
        std::vector<Lane> reference{{0, 0, -1, 8}, {1, 1, -1, 8}, {3, 2, -1, 8}};
        drain(batch, reference, 0, graphs);
        std::vector<Lane> replay{{0, 0, top_limit, 8}, {1, 1, -1, 8}, {3, 2, 0, 8}};
        drain(batch, replay, 128, graphs);
        batch.reset_prefix_cache();
        std::vector<Lane> fresh{{0, 0, top_limit, 8}, {1, 1, -1, 8}, {3, 2, 0, 8}};
        drain(batch, fresh, 0, graphs);
        for (std::size_t index = 0; index < replay.size(); ++index) {
            require(reference[index].output == replay[index].output &&
                        reference[index].output == fresh[index].output &&
                        replay[index].finish == DecoderFinishReason::length &&
                        fresh[index].finish == DecoderFinishReason::length,
                    "Scores or prefix restore changed greedy output");
        }
        std::vector<Lane> boundary{{2, 2, 1, 8, replay.back().output.front()}, {0, 0, 20, 0}};
        drain(batch, boundary, 128, false);
        require(boundary[0].output.size() == 1 && boundary[0].finish == DecoderFinishReason::stop &&
                    boundary[1].output.empty() && boundary[1].finish == DecoderFinishReason::length,
                "Stop or empty budget returned unexpected output");
        std::cout << (graphs ? "graph" : "eager")
                  << " greedy, mixed scores, prefix restore, ID stop and zero budget passed\n";
    }
}

}

int main(int argc, char** argv) {
    try {
        check_reducer();
        if (argc > 1) {
            check_checkpoint(argv[1]);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
