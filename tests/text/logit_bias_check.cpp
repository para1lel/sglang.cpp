#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/decoder_constraint.h"
#include "sglang/decoder_sampling.h"
#include "sglang/text/decoder_logprobs.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace {

using namespace sglang;
using Bias = std::vector<DecoderLogitBias>;
using Row = std::vector<DecoderLogit>;
const std::vector<int> prompt{48, 25, 220, 16, 10, 16, 28};

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool exact(const DecoderLogit& left, const DecoderLogit& right) {
    return left.token_id == right.token_id &&
        std::memcmp(&left.logit, &right.logit, sizeof(float)) == 0;
}

std::shared_ptr<const Bias> biases(Bias entries) {
    std::sort(entries.begin(), entries.end(), [](const auto& left, const auto& right) {
        return left.token_id < right.token_id;
    });
    return std::make_shared<const Bias>(std::move(entries));
}

Row biased_row(Row row, const Bias& entries) {
    for (const auto& entry : entries) {
        auto token = std::find_if(row.begin(), row.end(), [&](const auto& score) {
            return score.token_id == entry.token_id;
        });
        require(token != row.end(), "oracle missed a token");
        token->logit += entry.bias;
    }
    std::sort(row.begin(), row.end(), [](const auto& left, const auto& right) {
        return left.logit == right.logit ? left.token_id < right.token_id
                                        : left.logit > right.logit;
    });
    return row;
}

std::vector<double> weights(const Row& row, double temperature) {
    std::vector<double> result;
    for (const auto& entry : row) {
        result.push_back(std::exp((double(entry.logit) - row.front().logit) / temperature));
    }
    return result;
}

double first_draw(std::uint64_t seed) {
    auto word = seed + 0x9e3779b97f4a7c15ULL;
    word = (word ^ (word >> 30)) * 0xbf58476d1ce4e5b9ULL;
    word = (word ^ (word >> 27)) * 0x94d049bb133111ebULL;
    return std::ldexp(static_cast<double>((word ^ (word >> 31)) >> 11), -53);
}

int sample(const Row& row, const DecoderOptions& options) {
    const auto mass = weights(row, options.temperature);
    const double cutoff = options.top_p * std::accumulate(mass.begin(), mass.end(), 0.0);
    const auto limit = std::min(row.size(), static_cast<std::size_t>(options.top_k));
    double kept_mass = 0;
    std::size_t kept = 0;
    do {
        kept_mass += mass[kept++];
    } while (kept < limit && mass[kept] >= options.min_p && kept_mass <= cutoff);
    const auto draw = first_draw(static_cast<std::uint64_t>(*options.seed)) * kept_mass;
    double cumulative = 0;
    for (std::size_t rank = 0; rank < kept; ++rank) {
        cumulative += mass[rank];
        if (cumulative > draw) {
            return row[rank].token_id;
        }
    }
    throw std::runtime_error("oracle draw fell outside retained mass");
}

void verify_scores(const DecoderTokenLogprobs& receipt, const Row& row, std::size_t top) {
    const auto mass = weights(row, 1);
    const auto log_mass = std::log(std::accumulate(mass.begin(), mass.end(), 0.0));
    const auto check = [&](const DecoderLogprob& value, const DecoderLogit& entry) {
        const double expected = double(entry.logit) - row.front().logit - log_mass;
        require(value.token_id == entry.token_id && std::abs(value.value - expected) < 6e-6,
                "generated score differs from full-vocabulary log-softmax");
    };
    check(receipt.selected, row.front());
    require(receipt.alternatives.size() == top, "scored top count differs");
    for (std::size_t rank = 0; rank < receipt.alternatives.size(); ++rank) {
        check(receipt.alternatives[rank], row[rank]);
    }
}

void verify_invalid(const Decoder& decoder) {
    const float infinity = std::numeric_limits<float>::infinity();
    const std::array<Bias, 8> invalid{{
        {{-1, 1}}, {{int(model_vocab_size), 1}}, {{32, 0}}, {{32, infinity}},
        {{32, std::numeric_limits<float>::quiet_NaN()}}, {{32, 2e30f}},
        {{32, 1}, {32, 2}}, {{33, 1}, {32, 2}},
    }};
    for (const auto& value : invalid) {
        DecoderOptions options;
        options.logit_bias = std::make_shared<const Bias>(value);
        options.max_new_tokens = 0;
        try {
            decoder.generate(prompt, options);
        } catch (const std::invalid_argument&) {
            continue;
        }
        throw std::runtime_error("invalid direct bias accepted");
    }
}

void verify_singleton(const Decoder& decoder, const Tokenizer& tokenizer) {
    DecoderOptions options;
    options.top_logits = model_vocab_size;
    options.eager = true;
    const auto control = decoder.generate(prompt, options);
    const auto& raw = control.step_top_logits.at(0);
    require(raw.size() == model_vocab_size, "oracle requires all vocabulary scores");
    options.logit_bias = biases({});
    const auto empty = decoder.generate(prompt, options);
    require(empty.output_token_ids == control.output_token_ids &&
            std::equal(raw.begin(), raw.end(), empty.step_top_logits.at(0).begin(),
                       empty.step_top_logits.at(0).end(), exact),
            "empty direct bias changed logits");
    options.logit_bias = biases({{raw[1].token_id, raw[0].logit - raw[1].logit + 0.00390625f},
                                 {raw[4].token_id, -0.1875f}});
    const auto row = biased_row(raw, *options.logit_bias);
    require(row.front().token_id != raw.front().token_id, "fixture did not change the winner");
    const auto result = decoder.generate(prompt, options);
    require(result.output_token_ids == std::vector<int>{row.front().token_id} &&
            std::equal(row.begin(), row.end(), result.step_top_logits.at(0).begin(),
                       result.step_top_logits.at(0).end(), exact),
            "direct biased row differs from exact FP32 oracle");
    auto scored = options;
    scored.top_logits = 0;
    scored.generated_top_logprobs = 8;
    DecoderBatch batch(decoder, {false, 1});
    batch.admit(0, prompt, scored);
    const auto step = batch.advance_chunked({true})[0];
    require(step.token == row.front().token_id && step.generated_logprobs,
            "scored selection differs from oracle");
    verify_scores(*step.generated_logprobs, row, 8);
    options.top_logits = 0;
    options.temperature = 0.83;
    options.top_k = 7;
    options.top_p = 0.74;
    options.min_p = 0.16;
    for (const auto seed : {-41, 0, 19, 247}) {
        options.seed = seed;
        require(decoder.generate(prompt, options).output_token_ids ==
                std::vector<int>{sample(row, options)}, "seeded direct sample differs");
    }

    options.constraint = TokenConstraint::compile_regex(tokenizer, "^(A|B)$");
    options.logit_bias = biases({{32, 0.5f}, {33, 0.25f}, {34, 1000.0f}});
    auto allowed = biased_row(raw, *options.logit_bias);
    allowed.erase(std::remove_if(allowed.begin(), allowed.end(), [](const auto& entry) {
        return entry.token_id != 32 && entry.token_id != 33;
    }), allowed.end());
    require(decoder.generate(prompt, options).output_token_ids ==
            std::vector<int>{sample(allowed, options)}, "constrained sample differs");
    options.temperature = 0;
    require(decoder.generate(prompt, options).output_token_ids ==
            std::vector<int>{allowed.front().token_id}, "constrained greedy differs");
    verify_invalid(decoder);
    std::cout << "singleton: full exact FP32 rows, generated log-softmax, four seeds, "
                 "constraint and eight direct rejections passed\n";
}

std::vector<Row> minimum_rows(const Decoder& decoder, const std::vector<int>& input,
                              const DecoderOptions& options) {
    constexpr float forcing_bias = 40;
    DecoderOptions control;
    control.max_new_tokens = options.max_new_tokens;
    control.top_logits = model_vocab_size;
    control.logit_bias = biases({{32, forcing_bias}});
    const auto raw = decoder.generate(input, control);
    require(raw.output_token_ids == std::vector<int>(control.max_new_tokens, 32),
            "minimum oracle did not retain the forced history");
    std::vector<Row> expected;
    for (auto row : raw.step_top_logits) {
        require(row.size() == model_vocab_size && row.front().token_id == 32,
                "minimum oracle did not export the full vocabulary");
        row.front().logit -= forcing_bias;
        if (expected.size() < static_cast<std::size_t>(options.min_new_tokens)) {
            for (auto& score : row) {
                if (score.token_id == end_of_text || score.token_id == end_of_message ||
                    score.token_id == 33 || score.token_id == 34) {
                    score.logit = -std::numeric_limits<float>::infinity();
                }
            }
        }
        expected.push_back(biased_row(std::move(row), *options.logit_bias));
    }
    return expected;
}

void verify_minimum_batch(const Decoder& decoder, const std::vector<int>& input,
                          const DecoderOptions& options, const std::vector<Row>& expected) {
    std::vector<DecoderTokenLogprobs> eager;
    for (bool graphs : {false, true}) {
        DecoderBatch batch(decoder, {graphs, 2});
        auto scored = options;
        scored.top_logits = 0;
        scored.generated_top_logprobs = 5;
        scored.mtp_steps = 2;
        auto short_lane = scored;
        short_lane.min_new_tokens = 2;
        batch.admit(0, input, scored);
        batch.admit(1, input, short_lane);
        std::uint32_t graph_slots = 0;
        for (int index = 0; index < scored.max_new_tokens; ++index) {
            DecoderBatchStepStats stats;
            const auto steps = batch.advance_chunked({true, index <= 2}, &stats);
            graph_slots |= stats.graph_slots;
            const auto& step = steps[0];
            const bool final = index + 1 == scored.max_new_tokens;
            require(step.token == expected[index].front().token_id && step.generated_logprobs &&
                        step.mtp_drafted_tokens == 0 && !step.mtp_boundary &&
                        step.finish_reason.has_value() == final,
                    "minimum lane token, finish, or MTP route differs");
            const auto& scores = *step.generated_logprobs;
            require(scores.selected.token_id == *step.token && scores.alternatives.size() == 5,
                    "minimum generated scores do not describe the selected token");
            if (index < scored.min_new_tokens) {
                for (const auto& candidate : scores.alternatives) {
                    require(!scored.stops_at(candidate.token_id),
                            "minimum returned a suppressed stop among the top scores");
                }
            }
            if (graphs) {
                const auto equal = [](const DecoderLogprob& left, const DecoderLogprob& right) {
                    return left.token_id == right.token_id && left.value == right.value;
                };
                require(equal(scores.selected, eager[index].selected) &&
                            std::equal(scores.alternatives.begin(), scores.alternatives.end(),
                                       eager[index].alternatives.begin(),
                                       eager[index].alternatives.end(), equal),
                        "minimum eager and captured generated scores differ");
            } else {
                eager.push_back(scores);
            }
            if (index <= 2) {
                const auto& peer = steps[1];
                require(peer.token == (index < 2 ? 32 : end_of_text) &&
                            peer.finish_reason.has_value() == (index == 2),
                        "minimum mask leaked across lanes");
            }
        }
        require(graph_slots == (graphs ? 3u : 0u), "minimum batch graph route differs");
    }
}

void verify_minimum(const Decoder& decoder) {
    const std::vector<int> input{32, 33, end_of_text, 220, 32};
    DecoderOptions options;
    options.min_new_tokens = 3;
    options.max_new_tokens = 4;
    options.stop_token_id = 33;
    options.additional_stop_token_ids = {34, end_of_text};
    options.top_logits = model_vocab_size;
    options.logit_bias = biases({{32, 40}, {33, 80}, {34, 60}, {end_of_text, 120},
                                 {end_of_message, 100}});
    const auto expected = minimum_rows(decoder, input, options);
    for (const auto repetition : {1.0, 0.5}) {
        options.repetition_penalty = repetition;
        auto oracle = expected;
        if (repetition != 1) {
            for (int index = 1; index < options.max_new_tokens; ++index) {
                const auto entry = std::find_if(oracle[index].begin(), oracle[index].end(),
                    [](const auto& score) { return score.token_id == 32; });
                const float value = entry->logit - 40;
                entry->logit = 40 + (value < 0 ? value * 0.5f : value / 0.5f);
                std::sort(oracle[index].begin(), oracle[index].end(),
                          [](const auto& left, const auto& right) {
                    return left.logit == right.logit ? left.token_id < right.token_id
                                                    : left.logit > right.logit;
                });
            }
        }
        for (const double temperature : {0.0, 0.8}) {
            options.temperature = temperature;
            options.top_k = 1;
            options.seed = 101;
            const auto actual = decoder.generate(input, options);
            require(actual.output_token_ids == std::vector<int>({32, 32, 32, end_of_text}),
                    "minimum selection admitted a stop early or suppressed it after the minimum");
            for (int index = 0; index < options.max_new_tokens; ++index) {
                const auto& row = actual.step_top_logits.at(index);
                require(std::equal(row.begin(), row.end(), oracle[index].begin(),
                                   oracle[index].end(), exact),
                        "minimum-adjusted full logits differ from oracle");
            }
        }
    }
    options.temperature = 0;
    options.repetition_penalty = 1;
    verify_minimum_batch(decoder, input, options, expected);
    options.min_new_tokens = options.max_new_tokens;
    options.top_logits = 0;
    const auto capped = decoder.generate(input, options);
    require(capped.output_token_ids == std::vector<int>(options.max_new_tokens, 32),
            "minimum equal to maximum allowed an early stop");
    std::cout << "minimum: 16 exact full-vocabulary rows, greedy and seeded selection, "
                 "repetition, mixed graph lanes and min=max passed\n";
}

void verify_minimum_sampling(const Decoder& decoder) {
    DecoderOptions options;
    options.max_new_tokens = 1;
    options.min_new_tokens = 1;
    options.stop_token_id = 33;
    options.additional_stop_token_ids = {34};
    options.logit_bias = biases({{33, 90}, {34, 80}, {end_of_text, 110}, {end_of_message, 100}});
    const std::vector<int> input{32, 220, 33, end_of_message};
    auto expected = minimum_rows(decoder, input, options).front();
    Bias balanced;
    const float peak = expected.front().logit + 4;
    for (const auto& score : {expected[0], expected[1], expected[2]}) {
        balanced.push_back({score.token_id, peak - score.logit});
    }
    expected = biased_row(std::move(expected), balanced);
    balanced.insert(balanced.end(), options.logit_bias->begin(), options.logit_bias->end());
    options.logit_bias = biases(std::move(balanced));
    options.temperature = 0.85;
    options.top_k = 12;
    options.top_p = 0.9;
    options.min_p = 0.01;
    std::unordered_set<int> winners;
    for (const auto seed : {0, 4, 11, 42}) {
        options.seed = seed;
        const auto winner = sample(expected, options);
        winners.insert(winner);
        require(decoder.generate(input, options).output_token_ids == std::vector<int>{winner},
                "minimum sampling differs from the masked distribution");
    }
    require(winners.size() > 1, "minimum sampling fixture did not exercise multiple winners");

    for (int token = 0; token < static_cast<int>(model_vocab_size); ++token) {
        options.additional_stop_token_ids.insert(token);
    }
    for (const auto temperature : {0.0, 0.85}) {
        options.temperature = temperature;
        try {
            decoder.generate(input, options);
            require(false, "fully masked minimum unexpectedly emitted an ID");
        } catch (const DecoderSamplingError&) {
        }
    }
    std::cout << "minimum: four seeded full-distribution decisions and all-masked errors passed\n";
}

void verify_output_history(const Decoder& decoder) {
    constexpr int repeated = 32;
    constexpr int last_token = model_vocab_size - 1;
    constexpr float forcing_bias = 24.25f;
    constexpr int budget = 7;
    constexpr int top = 5;
    const std::vector<int> repeated_prompt{repeated, repeated, 220, repeated};
    DecoderOptions control;
    control.eager = true;
    control.max_new_tokens = budget;
    control.top_logits = model_vocab_size;
    control.logit_bias = biases({{repeated, forcing_bias}});
    const auto raw = decoder.generate(repeated_prompt, control);
    require(raw.output_token_ids == std::vector<int>(budget, repeated),
            "zero-penalty control did not repeat the prompt token");

    auto options = control;
    options.eager = false;
    options.top_logits = 0;
    options.generated_top_logprobs = top;
    options.frequency_penalty = 0.29;
    options.presence_penalty = 1.1;
    require(raw.step_top_logits.size() == budget &&
                raw.step_top_logits[1].size() == model_vocab_size,
            "zero-penalty control did not expose the full cutoff row");
    const auto& cutoff = raw.step_top_logits[1];
    const auto tied_token = std::find_if(cutoff.begin(), cutoff.end(), [](const auto& entry) {
        return entry.token_id == last_token;
    });
    const auto cutoff_winner = cutoff[top - 1];
    require(tied_token != cutoff.end() && cutoff_winner.token_id < last_token &&
                tied_token->logit < cutoff_winner.logit,
            "zero-penalty control has no lower-ranked token to lift to the cutoff");
    const float tie_bias = cutoff_winner.logit - tied_token->logit;
    options.logit_bias = biases({{repeated, forcing_bias}, {last_token, tie_bias}});

    std::vector<Row> expected;
    float frequency = 0;
    for (const auto& scores : raw.step_top_logits) {
        require(scores.size() == model_vocab_size && scores.front().token_id == repeated,
                "zero-penalty control did not expose the full forced row");
        auto row = scores;
        row.front().logit -= forcing_bias;
        if (!expected.empty()) {
            row.front().logit -= frequency + static_cast<float>(options.presence_penalty);
        }
        expected.push_back(biased_row(std::move(row), *options.logit_bias));
        frequency += static_cast<float>(options.frequency_penalty);
    }
    require(expected.size() == budget, "zero-penalty control row count differs");
    const auto& tied = expected[1];
    require(tied[top - 1].token_id == cutoff_winner.token_id && tied[top].token_id == last_token &&
                tied[top - 1].logit == tied[top].logit,
            "penalty fixture did not put equal logits across the top cutoff");
    for (bool graphs : {false, true}) {
        DecoderBatch batch(decoder, {graphs, 1});
        batch.admit(0, repeated_prompt, options);
        std::uint32_t graph_slots = 0;
        for (int index = 0; index < budget; ++index) {
            DecoderBatchStepStats stats;
            const auto step = batch.advance_chunked({true}, &stats)[0];
            graph_slots |= stats.graph_slots;
            require(step.token == repeated && step.generated_logprobs &&
                        static_cast<bool>(step.finish_reason) == (index + 1 == budget),
                    "penalty history step did not emit the forced token and scores");
            verify_scores(*step.generated_logprobs, expected[index], top);
        }
        require(graph_slots == (graphs ? 1u : 0u), "penalty history graph route differs");
    }
    std::cout << "penalties: seven full-row log-softmax checks per route, empty and repeated "
                 "output history, and a tie at the top cutoff passed (retained "
              << cutoff_winner.token_id << ", excluded " << last_token << ", logit "
              << tied[top].logit << ", bias " << tie_bias << ")\n";
}

template <class Buffer, class Select>
void verify_extreme_repetition(Buffer& buffer, const Select& select) {
    using namespace sglang::decoder;
    const float lowest = -std::numeric_limits<float>::max();
    const float infinity = std::numeric_limits<float>::infinity();
    const auto set_logits = [&](const std::array<float, 5>& scores) {
        std::transform(scores.begin(), scores.end(), buffer.source,
                       [](float score) { return __float2bfloat16(score); });
    };
    const std::array factors{1e-40, 1e-46, std::numeric_limits<double>::denorm_min()};
    const SamplingParameters sampling{1.0, 1.0, 1, 42};
    GeneratedLogprobReadback scores{};
    std::fill_n(buffer.seen, 5, 1);
    for (const double factor : factors) {
        const OutputPenalties penalties{nullptr, buffer.seen, 0, 0, factor};
        set_logits({1, 2, 0, -2, infinity});
        select(penalties, -1, 0, {nullptr, 0, &buffer.token});
        require(buffer.token == 1, "distinct huge logits must preserve the greedy winner");
        select(penalties, -1, 0, {&sampling, 0, &buffer.token});
        require(buffer.token == 1, "distinct huge logits must preserve sampling rank");
        select(penalties, -1, 0, {nullptr, 0, &buffer.token, 5, &scores});
        require(buffer.token == 1 && scores.selected.token_id == 1 &&
                scores.selected.value == 0 &&
                scores.alternatives[1].token_id == 0 && scores.alternatives[1].value == lowest &&
                scores.alternatives[2].token_id == 2 && scores.alternatives[3].token_id == 3 &&
                scores.alternatives[4].token_id == 4, "extreme logprob ranking or mask differs");

        set_logits({2, 2, 0, -2, infinity});
        buffer.bias[0] = {1, static_cast<float>(std::log(3.0))};
        select(penalties, -1, 1, {nullptr, 0, &buffer.token});
        require(buffer.token == 1, "log(3) must break the equal repeated greedy tie");
        select(penalties, -1, 1, {&sampling, 0, &buffer.token});
        require(buffer.token == 1, "log(3) must break the equal repeated top-k tie");
        select(penalties, -1, 1, {nullptr, 0, &buffer.token, 2, &scores});
        require(buffer.token == 1 && scores.alternatives[1].token_id == 0 &&
                std::abs(std::exp(scores.selected.value) - 0.75) < 1e-7 &&
                std::abs(std::exp(scores.alternatives[1].value) - 0.25) < 1e-7,
                "log(3) must retain the 75/25 distribution for equal repeated logits");

        buffer.bias[0] = {1, 2};
        select(penalties, -1, 1, {nullptr, 0, &buffer.token, 2, &scores});
        const double log_mass = std::log1p(std::exp(-2.0));
        require(buffer.token == 1 && scores.alternatives[1].token_id == 0 &&
                std::abs(scores.selected.value + log_mass) < 1e-7 &&
                std::abs(scores.alternatives[1].value + 2 + log_mass) < 1e-7,
                "centered logprobs lost the finite difference between huge scores");

        set_logits({-2, -1, -infinity, -infinity, infinity});
        select(penalties, -1, 0, {nullptr, 0, &buffer.token});
        require(buffer.token == 1, "negative underflow must preserve the greedy winner");
    }
    set_logits({3, 0x1p-130f, 0, -infinity, infinity});
    std::fill_n(buffer.seen, 5, 0);
    buffer.seen[1] = 1;
    const OutputPenalties boundary{nullptr, buffer.seen, 0, 0, 0x1.5555555555555p-132};
    select(boundary, -1, 0, {nullptr, 0, &buffer.token});
    require(buffer.token == 1, "rounded quotient must retain the ranking residual");
    select(boundary, -1, 0, {&sampling, 0, &buffer.token});
    require(buffer.token == 1, "rounded quotient must retain the sampling rank");
    buffer.bias[0] = {1, -0x1p-54f};
    select(boundary, -1, 1, {nullptr, 0, &buffer.token});
    require(buffer.token == 1, "bias smaller than the quotient residual must retain rank");
    buffer.bias[0] = {1, -0x1p-52f};
    select(boundary, -1, 1, {nullptr, 0, &buffer.token});
    require(buffer.token == 0, "bias larger than the quotient residual must change rank");
    std::cout << "repetition: exact signs/history, extreme ranking and centered scores passed\n";
}

void verify_repetition_kernel() {
    constexpr int vocabulary = 5;
    struct Buffer {
        __nv_bfloat16 source[vocabulary];
        float frequency[vocabulary];
        std::uint8_t seen[vocabulary];
        DecoderLogitBias bias[2];
        std::uint32_t mask;
        int token;
    };
    void* storage = nullptr;
    require(cudaMallocManaged(&storage, sizeof(Buffer)) == cudaSuccess, "allocate test scores");
    const std::unique_ptr<Buffer, decltype(&cudaFree)> buffer(static_cast<Buffer*>(storage),
                                                              &cudaFree);
    using sglang::decoder::OutputPenalties;
    using sglang::decoder::RepetitionSelection;
    using sglang::decoder::select_repetition_logits;
    const auto bytes = sglang::decoder::repetition_workspace_bytes(vocabulary, true);
    void* scratch = nullptr;
    require(cudaMalloc(&scratch, bytes) == cudaSuccess, "allocate repetition workspace");
    const std::unique_ptr<void, decltype(&cudaFree)> workspace(scratch, &cudaFree);
    std::array<DecoderLogit, vocabulary> ranked{};
    const auto select = [&](OutputPenalties penalties, int previous, int bias_count,
                            RepetitionSelection selection) {
        require(select_repetition_logits(buffer->source, vocabulary, buffer->bias, bias_count,
                                         {&buffer->mask}, penalties, previous, selection,
                                         workspace.get(), bytes, nullptr) == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess, "select repetition scores");
    };
    RepetitionSelection debug;
    debug.debug = ranked.data();
    debug.debug_count = ranked.size();
    const std::array raw{8.0f, -8.0f, 1.0f, 0.0f, 5.0f};
    for (int token = 0; token < vocabulary; ++token) {
        buffer->source[token] = __float2bfloat16(raw[token]);
    }
    std::copy_n(std::array<DecoderLogitBias, 2>{{{0, 0.75f}, {4, 100.0f}}}.begin(),
                2, buffer->bias);
    buffer->mask = 0b01111u;
    for (const float repetition : {0.5f, 2.0f}) {
        std::fill_n(buffer->frequency, vocabulary, 0.0f);
        std::fill_n(buffer->seen, vocabulary, 0);
        std::array<int, vocabulary> counts{};
        const sglang::decoder::OutputPenalties penalties{
            buffer->frequency, buffer->seen, 1.0f, 0.5f, repetition,
        };
        for (const int previous : {-1, 0, 1, 0, 2}) {
            select(penalties, previous, 2, debug);
            if (previous >= 0) ++counts[previous];
            std::array<DecoderLogit, vocabulary> expected{};
            for (int token = 0; token < vocabulary; ++token) {
                float score = raw[token];
                if (counts[token]) {
                    score -= static_cast<float>(counts[token]) + 0.5f;
                    score = score < 0 ? score * repetition : score / repetition;
                }
                if (token == 0) score += 0.75f;
                if (token == 4) score = -std::numeric_limits<float>::infinity();
                expected[token] = {token, score};
            }
            std::sort(expected.begin(), expected.end(), [](const auto& left, const auto& right) {
                return left.logit == right.logit ? left.token_id < right.token_id
                                                 : left.logit > right.logit;
            });
            require(std::equal(expected.begin(), expected.end(), ranked.begin(), exact),
                    "repetition kernel order or sign differs");
        }
    }
    verify_extreme_repetition(*buffer, select);
}

void verify_repetition_model(const Decoder& decoder, const Tokenizer& tokenizer) {
    constexpr int repeated = 32;
    constexpr float forcing_bias = 24.25f;
    constexpr int budget = 4;
    const std::vector<int> tokens{repeated, repeated, 220, repeated};
    DecoderOptions options;
    options.eager = true;
    options.max_new_tokens = budget;
    const auto ordinary = decoder.generate(tokens, options).output_token_ids;
    options.repetition_penalty = 1;
    require(decoder.generate(tokens, options).output_token_ids == ordinary,
            "neutral repetition changed ordinary generated IDs");
    options.logit_bias = biases({{repeated, forcing_bias}});
    options.top_logits = model_vocab_size;
    const auto control = decoder.generate(tokens, options);
    require(control.output_token_ids == std::vector<int>(budget, repeated),
            "control did not repeat");

    for (const double repetition : {0.5, 2.0}) {
        options.repetition_penalty = repetition;
        options.frequency_penalty = 0.5;
        options.presence_penalty = 0.75;
        const auto adjusted = decoder.generate(tokens, options);
        require(adjusted.output_token_ids == control.output_token_ids, "forced history diverged");
        std::vector<Row> expected;
        for (int index = 0; index < budget; ++index) {
            auto row = control.step_top_logits.at(index);
            auto& winner = row.front().logit;
            if (index != 0) {
                winner -= forcing_bias;
                winner -= 0.5f * static_cast<float>(index) + 0.75f;
                const auto factor = static_cast<float>(repetition);
                winner = winner < 0 ? winner * factor : winner / factor;
                winner += forcing_bias;
            }
            const auto& actual = adjusted.step_top_logits.at(index);
            require(std::equal(row.begin(), row.end(), actual.begin(), actual.end(), exact),
                    "direct repetition row differs from full-vocabulary oracle");
            expected.push_back(std::move(row));
        }
        auto scored = options;
        scored.eager = false;
        scored.top_logits = 0;
        scored.generated_top_logprobs = 5;
        scored.mtp_steps = 2;
        DecoderBatch batch(decoder, {true, 1});
        batch.admit(0, tokens, scored);
        std::uint32_t graph_slots = 0;
        for (int index = 0; index < budget; ++index) {
            DecoderBatchStepStats stats;
            const auto step = batch.advance_chunked({true}, &stats)[0];
            graph_slots |= stats.graph_slots;
            require(step.token == repeated && step.generated_logprobs &&
                    step.mtp_drafted_tokens == 0 && !step.mtp_boundary, "scored repetition route");
            verify_scores(*step.generated_logprobs, expected[index], 5);
        }
        require(graph_slots == 1, "repetition did not use the eligible K0 graph");
    }

    auto extreme = options;
    extreme.top_logits = 0;
    extreme.seed = 42;
    for (const double factor : {1e-40, 1e-46}) {
        extreme.repetition_penalty = factor;
        for (const double temperature : {0.0, 1.0}) {
            extreme.temperature = temperature;
            require(decoder.generate(tokens, extreme).output_token_ids.size() == budget,
                    "tiny repetition factor must complete greedy and sampled generation");
        }
    }

    options.top_logits = 0;
    options.constraint = TokenConstraint::compile_regex(tokenizer, "^AA$");
    options.logit_bias = biases({{33, 100.0f}});
    require(decoder.generate(tokens, options).output_token_ids == std::vector<int>({32, 32}),
            "constraint or repetition allowed a prohibited token");
    std::cout << "repetition: neutral IDs, eight exact checkpoint rows, K0 graph logprobs, "
                 "and combined constraints passed\n";
}

using TokenStreams = std::array<std::vector<int>, 2>;

TokenStreams run_batch(const Decoder& decoder, bool graphs, bool mtp) {
    DecoderOptions biased;
    biased.max_new_tokens = 8;
    biased.logit_bias = biases({{33, 100.0f}});
    auto plain = biased;
    plain.logit_bias.reset();
    biased.mtp_steps = plain.mtp_steps = mtp ? 2 : 0;
    DecoderBatch batch(decoder, {graphs, 2});
    batch.admit(0, prompt, biased);
    batch.admit(1, prompt, plain);
    std::array<bool, decoder_batch_max_lanes> selected{true, true};
    std::uint32_t graph_slots = 0;
    int drafted = 0;
    TokenStreams output;
    for (int tick = 0; tick < 10 && (selected[0] || selected[1]); ++tick) {
        DecoderBatchStepStats stats;
        const auto steps = batch.advance_chunked(selected, &stats);
        graph_slots |= stats.graph_slots;
        require(steps[0].mtp_drafted_tokens == 0 && !steps[0].mtp_boundary,
                "biased lane used MTP");
        drafted += steps[1].mtp_drafted_tokens;
        for (std::size_t slot = 0; slot < output.size(); ++slot) {
            const auto& step = steps[slot];
            if (step.token) output[slot].push_back(*step.token);
            output[slot].insert(output[slot].end(), step.extra_tokens.begin(),
                                step.extra_tokens.begin() + step.extra_token_count);
            if (step.finish_reason) selected[slot] = false;
        }
    }
    require(output[0] == std::vector<int>(8, 33) && output[1].size() == 8,
            "packed biased or plain lane output mismatch");
    if (graphs && !mtp) require(graph_slots == 3, "both packed lanes must replay");
    if (!graphs) require(graph_slots == 0, "eager unexpectedly replayed");
    require(mtp ? drafted > 0 : drafted == 0, "plain lane MTP route mismatch");
    std::cout << "batch: graphs=" << graphs << " mtp=" << mtp
              << " graph_slots=" << graph_slots << " plain_drafted=" << drafted << '\n';
    return output;
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Usage: logit-bias-check CHECKPOINT");
        const std::filesystem::path checkpoint(argv[1]);
        Decoder decoder(checkpoint, 0, true, Fp8PrefillPolicy::tensor_core);
        const Tokenizer tokenizer(checkpoint / "tokenizer.json");
        verify_singleton(decoder, tokenizer);
        verify_output_history(decoder);
        verify_minimum(decoder);
        verify_minimum_sampling(decoder);
        verify_repetition_kernel();
        verify_repetition_model(decoder, tokenizer);
        const auto eager = run_batch(decoder, false, false);
        require(run_batch(decoder, true, false) == eager, "packed parity differs");
        require(run_batch(decoder, true, true) == eager, "mixed MTP parity differs");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
