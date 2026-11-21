#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/decoder_constraint.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

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

void verify_scores(const Decoder& decoder, DecoderOptions options, const Row& row) {
    options.top_logits = 0;
    options.generated_top_logprobs = 8;
    DecoderBatch batch(decoder, {false, 1});
    batch.admit(0, prompt, options);
    std::array<bool, decoder_batch_max_lanes> selected{true};
    const auto step = batch.advance_chunked(selected)[0];
    require(step.token == row.front().token_id && step.generated_logprobs,
            "scored selection differs from oracle");
    const auto mass = weights(row, 1);
    const auto log_mass = std::log(std::accumulate(mass.begin(), mass.end(), 0.0));
    const auto check = [&](const DecoderLogprob& value, const DecoderLogit& entry) {
        const double expected = double(entry.logit) - row.front().logit - log_mass;
        require(value.token_id == entry.token_id && std::abs(value.value - expected) < 6e-6,
                "generated score differs from full-vocabulary log-softmax");
    };
    const auto& receipt = *step.generated_logprobs;
    check(receipt.selected, row.front());
    require(receipt.alternatives.size() == 8, "scored top count differs");
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
    verify_scores(decoder, options, row);
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
        const auto eager = run_batch(decoder, false, false);
        require(run_batch(decoder, true, false) == eager, "packed parity differs");
        require(run_batch(decoder, true, true) == eager, "mixed MTP parity differs");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
