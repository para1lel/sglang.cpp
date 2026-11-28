#include "sglang/decoder_batch.h"
#include "sglang/decoder_constraint.h"
#include "sglang/json.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using sglang::Decoder;
using sglang::DecoderBatch;
using sglang::DecoderBatchStepStats;
using sglang::DecoderFinishReason;
using sglang::DecoderOptions;
using sglang::DecoderResult;
using sglang::TokenConstraint;
using sglang::Tokenizer;

void require(bool condition, const char* failure) {
    if (!condition) {
        throw std::runtime_error(failure);
    }
}

DecoderOptions sampled(DecoderOptions options) {
    options.temperature = 1.8;
    options.top_k = 80;
    options.top_p = 0.95;
    options.min_p = 0.015;
    options.seed = -123019;
    return options;
}

struct Case {
    const char* name;
    DecoderOptions options;
};

std::array<Case, 4> fixtures(const Tokenizer& tokenizer) {
    DecoderOptions ebnf;
    ebnf.max_new_tokens = 32;
    ebnf.constraint = TokenConstraint::compile(tokenizer,
        R"(root ::= "The colors are red, orange, yellow, green, blue, and violet.")");
    DecoderOptions regex = sampled(ebnf);
    regex.constraint = TokenConstraint::compile_regex(tokenizer, "^[A-Za-z ,.!]{23,36}$");
    require(regex.constraint->uses_wide_mask(), "regex fixture did not reach a dense mask");
    DecoderOptions schema = sampled(ebnf);
    const auto array = sglang::json::parse(
        R"({"type":"array","items":{"enum":["red","green","blue"]},"minItems":7,"maxItems":8})",
        sglang::json::Syntax::ordered);
    schema.constraint = TokenConstraint::compile_json_schema(tokenizer, array);
    DecoderOptions ordinary = sampled(ebnf);
    ordinary.constraint.reset();
    ordinary.max_new_tokens = 20;
    return {{{"ebnf", ebnf}, {"regex", regex}, {"schema", schema}, {"ordinary", ordinary}}};
}

void compare(const DecoderResult& row, const DecoderResult& control) {
    require(row.output_token_ids == control.output_token_ids &&
                row.finish_reason == control.finish_reason, "native result changed from K0");
}

void compare_logits(const DecoderResult& row, const DecoderResult& control) {
    require(row.step_top_logits.size() == row.output_token_ids.size(),
            "Missing or extra raw target rows");
    require(row.step_top_logits.size() <= control.step_top_logits.size(),
            "Raw target rows exceeded reference");
    for (std::size_t index = 0; index < row.step_top_logits.size(); ++index) {
        const auto& first = row.step_top_logits[index];
        const auto& second = control.step_top_logits[index];
        require(!first.empty() &&
                std::equal(first.begin(), first.end(), second.begin(), second.end(),
                           [](const auto& a, const auto& b) {
                               return a.token_id == b.token_id && a.logit == b.logit;
                           }), "Raw target logits differ at the committed token ordinal");
    }
}

void print_ids(const std::vector<int>& ids) {
    for (int token : ids) {
        std::cout << ' ' << token;
    }
    std::cout << '\n';
}

DecoderResult serial(const Decoder& decoder, const std::vector<int>& prompt, const Case& test) {
    std::vector<int> emitted;
    sglang::DecoderCallbacks callbacks;
    callbacks.on_token = [&](int token) {
        emitted.push_back(token);
        return true;
    };
    auto result = decoder.generate(prompt, test.options, callbacks);
    require(emitted == result.output_token_ids, "serial callback emitted uncommitted tokens");
    std::cout << "serial " << test.name << " K" << test.options.mtp_steps
              << " drafted=" << result.mtp_drafted_tokens
              << " accepted=" << result.mtp_accepted_tokens << " rounds=" << result.mtp_rounds
              << " finish=" << static_cast<int>(result.finish_reason) << " ids=";
    print_ids(result.output_token_ids);
    return result;
}

struct BatchRow {
    DecoderResult result;
    int verified = 0;
    int discarded = 0;
    int boundary = 0;
    int terminal_drafts = 0;
    int full = 0;
    int partial = 0;
    int zero = 0;
};

void consume(BatchRow& lane, const sglang::DecoderBatchStep& step) {
    require(!step.sampling_failed && !step.constraint_failed, "native lane failed selection");
    auto& result = lane.result;
    if (step.token) {
        result.output_token_ids.push_back(*step.token);
    }
    result.output_token_ids.insert(result.output_token_ids.end(), step.extra_tokens.begin(),
                                  step.extra_tokens.begin() + step.extra_token_count);
    result.mtp_drafted_tokens += step.mtp_drafted_tokens;
    result.mtp_accepted_tokens += step.mtp_accepted_tokens;
    result.mtp_rounds += step.mtp_verified_rows != 0;
    lane.verified += step.mtp_verified_rows;
    require(step.mtp_replayed_rows == 0, "Constrained MTP replayed committed GDN rows");
    lane.discarded +=
        std::max(0, step.mtp_verified_rows - static_cast<int>(step.token.has_value()) -
                        step.extra_token_count);
    lane.boundary += step.mtp_boundary;
    if (step.mtp_drafted_tokens) {
        lane.full += step.mtp_accepted_tokens == step.mtp_drafted_tokens;
        lane.partial += step.mtp_accepted_tokens > 0 &&
                        step.mtp_accepted_tokens < step.mtp_drafted_tokens;
        lane.zero += step.mtp_accepted_tokens == 0;
        lane.terminal_drafts += step.mtp_verified_rows == step.mtp_drafted_tokens;
    }
    if (step.finish_reason) {
        result.finish_reason = *step.finish_reason;
    }
}

void print_batch(const Case& test, const BatchRow& lane) {
    const auto& result = lane.result;
    std::cout << "batch " << test.name << " K" << test.options.mtp_steps
              << " drafted=" << result.mtp_drafted_tokens
              << " accepted=" << result.mtp_accepted_tokens << " verified=" << lane.verified
              << " discarded=" << lane.discarded << " boundary=" << lane.boundary
              << " full=" << lane.full << " partial=" << lane.partial << " zero=" << lane.zero
              << " terminal_draft=" << lane.terminal_drafts
              << " finish=" << static_cast<int>(result.finish_reason) << " ids=";
    print_ids(result.output_token_ids);
}

std::array<BatchRow, 4> batch(DecoderBatch& decoder, const std::vector<int>& prompt,
                             const std::array<Case, 4>& cases, bool require_mixed = false) {
    std::array<BatchRow, 4> results{};
    std::array<bool, sglang::decoder_batch_max_lanes> selected{};
    for (std::size_t slot = 0; slot < cases.size(); ++slot) {
        decoder.admit(slot, prompt, cases[slot].options);
        selected[slot] = true;
    }
    bool mixed = false;
    std::array<bool, 5> widths{};
    int iterations = 0;
    while (std::any_of(selected.begin(), selected.end(), [](bool active) { return active; })) {
        require(++iterations <= 64, "native batch did not retire");
        widths[std::count(selected.begin(), selected.end(), true)] = true;
        DecoderBatchStepStats stats;
        const auto steps = decoder.advance_chunked(selected, &stats);
        mixed |= stats.verification_lanes >= 2 && steps[0].mtp_drafted_tokens > 0 &&
                 steps[2].mtp_drafted_tokens > 0 && steps[1].decode && steps[3].decode;
        for (std::size_t slot = 0; slot < cases.size(); ++slot) {
            if (selected[slot]) {
                consume(results[slot], steps[slot]);
                selected[slot] = !steps[slot].finish_reason;
            }
        }
    }
    widths[0] = true;
    require(!require_mixed || (mixed && widths[4] && (widths[1] || widths[2] || widths[3])),
            "batch did not overlap constrained and K0 lanes and retire at different steps");
    std::cout << "active widths:";
    for (std::size_t width = 0; width < widths.size(); ++width) {
        if (widths[width]) {
            std::cout << ' ' << width;
        }
    }
    std::cout << '\n';
    require(decoder.available() && decoder.request_device_allocation_bytes() == 0,
            "batch retained retired request storage");
    for (std::size_t slot = 0; slot < cases.size(); ++slot) {
        print_batch(cases[slot], results[slot]);
    }
    return results;
}

void check_budget(const Decoder& decoder, const Tokenizer& tokenizer, DecoderBatch& cohort,
                  const std::vector<int>& prompt) {
    DecoderOptions empty;
    empty.constraint = TokenConstraint::compile_regex(tokenizer, "^(?:)$");
    empty.max_new_tokens = 0;
    auto eos = empty;
    eos.max_new_tokens = 1;
    auto exact = eos;
    exact.constraint = TokenConstraint::compile(tokenizer, R"(root ::= "A")");
    auto long_exact = exact;
    long_exact.constraint = TokenConstraint::compile(tokenizer,
        R"(root ::= "The colors are red, orange, yellow, green, blue, and violet.")");
    long_exact.max_new_tokens = long_exact.constraint->minimum_tokens();
    std::array<Case, 4> cases{{{"empty", empty}, {"eos", eos}, {"exact", exact},
                               {"long-exact", long_exact}}};
    const auto control = batch(cohort, prompt, cases);
    for (auto& test : cases) {
        test.options.mtp_steps = 4;
    }
    const auto result = batch(cohort, prompt, cases);
    for (std::size_t index = 0; index < cases.size(); ++index) {
        compare(result[index].result, control[index].result);
        compare(serial(decoder, prompt, cases[index]), control[index].result);
    }
    const auto& ids = result[1].result.output_token_ids;
    require(result[0].result.output_token_ids.empty() &&
                result[0].result.finish_reason == DecoderFinishReason::length &&
                ids.size() == 1 && sglang::constraint_eos(ids.front()) &&
                result[1].result.finish_reason == DecoderFinishReason::constraint &&
                result[2].result.output_token_ids.size() == 1 &&
                !sglang::constraint_eos(result[2].result.output_token_ids.front()) &&
                result[2].result.finish_reason == DecoderFinishReason::constraint &&
                result[3].result.finish_reason == DecoderFinishReason::constraint &&
                result[3].result.output_token_ids.size() ==
                    static_cast<std::size_t>(long_exact.max_new_tokens), "unexpected budget exit");
    require(result[3].verified > 0, "exact-budget case never verified a proposal");
}

struct CallbackFailure {};

void check_callbacks(const Decoder& decoder, const std::vector<int>& prompt, const Case& test,
                     const DecoderResult& control) {
    std::vector<int> emitted;
    sglang::DecoderCallbacks callbacks;
    callbacks.on_token = [&](int token) {
        emitted.push_back(token);
        return emitted.size() < 3;
    };
    const auto stopped = decoder.generate(prompt, test.options, callbacks);
    require(stopped.finish_reason == DecoderFinishReason::cancelled && emitted.size() == 3 &&
                stopped.output_token_ids == emitted &&
                std::equal(emitted.begin(), emitted.end(), control.output_token_ids.begin()),
            "Callback stop did not return the committed prefix");
    compare_logits(stopped, control);
    emitted.clear();
    callbacks.on_token = [&](int token) {
        emitted.push_back(token);
        return true;
    };
    callbacks.is_cancelled = [&] { return emitted.size() >= 4; };
    const auto cancelled = decoder.generate(prompt, test.options, callbacks);
    require(cancelled.finish_reason == DecoderFinishReason::cancelled &&
                cancelled.output_token_ids == emitted && emitted.size() >= 4 &&
                emitted.size() < control.output_token_ids.size() &&
                std::equal(emitted.begin(), emitted.end(), control.output_token_ids.begin()),
            "Cancellation predicate did not return the committed prefix");
    compare_logits(cancelled, control);
    int calls = 0;
    callbacks.is_cancelled = {};
    callbacks.on_token = [&](int) -> bool {
        if (++calls == 3) {
            throw CallbackFailure{};
        }
        return true;
    };
    bool propagated = false;
    try {
        decoder.generate(prompt, test.options, callbacks);
    } catch (const CallbackFailure&) {
        propagated = true;
    }
    require(propagated && calls == 3, "Callback exception did not propagate synchronously");
    const auto replay = decoder.generate(prompt, test.options);
    compare(replay, control);
    compare_logits(replay, control);
    std::cout << "callback stop/predicate/exception and subsequent replay matched K0\n";
}

void check_mtp(const Decoder& decoder, const Tokenizer& tokenizer, const std::vector<int>& prompt) {
    auto cases = fixtures(tokenizer);
    DecoderBatch cohort(decoder, {true, 4});
    const auto controls = batch(cohort, prompt, cases);
    std::array<DecoderResult, 4> serial_controls;
    for (std::size_t index = 0; index < cases.size(); ++index) {
        cases[index].options.top_logits = 8;
        serial_controls[index] = serial(decoder, prompt, cases[index]);
        compare(serial_controls[index], controls[index].result);
        compare_logits(serial_controls[index], serial_controls[index]);
        cases[index].options.top_logits = 0;
    }
    int corrections = 0;
    int partial = 0;
    int rejected = 0;
    int terminal = 0;
    for (int depth = 1; depth <= sglang::decoder_max_mtp_steps; ++depth) {
        for (std::size_t index = 0; index < cases.size(); ++index) {
            auto& test = cases[index];
            test.options.mtp_steps = depth;
            test.options.top_logits = 8;
            const auto result = serial(decoder, prompt, test);
            test.options.top_logits = 0;
            compare(result, serial_controls[index]);
            compare_logits(result, serial_controls[index]);
            require(result.mtp_drafted_tokens > 0 && result.mtp_rounds > 0,
                    "serial request did not execute a speculative round");
        }
        const auto results = batch(cohort, prompt, cases);
        for (std::size_t index = 0; index < results.size(); ++index) {
            const auto& lane = results[index];
            compare(lane.result, controls[index].result);
            require(lane.result.mtp_drafted_tokens > 0 && lane.verified > 0,
                    "batch lane did not execute a speculative round");
            if (index < 3) {
                corrections += lane.discarded;
                partial += lane.partial;
                rejected += lane.zero;
                terminal += lane.terminal_drafts;
            }
        }
        require(results[0].full > 0, "greedy EBNF never accepted a full proposal");
    }
    require(corrections > 0 && partial > 0 && rejected > 0,
            "fixtures did not exercise speculative rollback after zero and partial acceptance");
    require(terminal > 0, "fixtures did not verify a terminal proposal without a bonus row");
    cases[1].options.mtp_steps = 0;
    cases[3].options.mtp_steps = 0;
    const auto mixed = batch(cohort, prompt, cases, true);
    for (std::size_t index = 0; index < mixed.size(); ++index) {
        compare(mixed[index].result, controls[index].result);
    }
    require(mixed[1].verified == 0 && mixed[3].verified == 0, "K0 lanes verified in mixed batch");
    cases[2].options.top_logits = 8;
    check_callbacks(decoder, prompt, cases[2], serial_controls[2]);
    check_budget(decoder, tokenizer, cohort, prompt);
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected the checkpoint directory");
        const std::filesystem::path checkpoint = argv[1];
        const Tokenizer tokenizer(checkpoint / "tokenizer.json");
        const Decoder decoder(checkpoint, 0, true);
        const auto tokens = tokenizer.encode("Complete the requested output.");
        check_mtp(decoder, tokenizer, {tokens.begin(), tokens.end()});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
