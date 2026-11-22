#include "sglang/json.h"
#include "sglang/server/score.h"
#include "sglang/text/label_logprobs.h"
#include "sglang/tokenizer/tokenizer.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

using sglang::json::get;
using sglang::server::parse_score_request;
using sglang::server::score_response;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected) {
    require(std::abs(actual - expected) < 1e-12, "Score differs");
}

void check_budget_arithmetic() {
    using sglang::server::score_detail::consume;
    std::size_t remaining = 7;
    require(consume(remaining, 3, 2) && remaining == 1, "Product was not charged");
    require(!consume(remaining, 2) && remaining == 1, "Failed charge changed the budget");
    require(consume(remaining, 1) && remaining == 0, "Exact budget was rejected");
    require(consume(remaining, 0, 3) && consume(remaining, 3, 0),
            "Zero-sized product was rejected");
    const auto maximum = std::numeric_limits<std::size_t>::max();
    remaining = maximum;
    require(!consume(remaining, maximum, 2) && remaining == maximum,
            "Product overflow was accepted");
    require(consume(remaining, maximum) && !consume(remaining, 1),
            "Cumulative overflow was accepted");
}

template <class Error = std::invalid_argument>
void reject(const sglang::Tokenizer& tokenizer, std::string_view body, std::string_view message) {
    try {
        parse_score_request(body, tokenizer);
    } catch (const Error& error) {
        if constexpr (std::is_same_v<Error, std::invalid_argument>) {
            require(typeid(error) == typeid(Error), "Unexpected validation error class");
        }
        require(error.what() == message, "Unexpected error message");
        return;
    }
    throw std::runtime_error("Invalid score request accepted");
}

void check_inputs(const sglang::Tokenizer& tokenizer) {
    const auto request = parse_score_request(
        R"({"query":"hel","items":["lo","p"],"label_token_ids":[[33,17],[7]],"model":"alias"})",
        tokenizer);
    require(request.model == "alias" && request.prompts.size() == 2, "Parsed request differs");
    const auto hello = tokenizer.encode("hello");
    const auto help = tokenizer.encode("help");
    require(request.prompts[0].prompt == std::vector<int>(hello.begin(), hello.end()) &&
            request.prompts[1].prompt == std::vector<int>(help.begin(), help.end()),
            "Text was not concatenated before tokenization");
    require(*request.prompts[0].options.label_token_ids == std::vector<int>{33, 17} &&
            *request.prompts[1].options.label_token_ids == std::vector<int>{7},
            "Candidate order differs");

    const auto reversed = parse_score_request(
        R"({"query":[4,5],"items":[[1],[2,3]],"item_first":true,"label_token_ids":[5,3]})",
        tokenizer);
    require(reversed.prompts[0].prompt == std::vector<int>{1, 4, 5} &&
            reversed.prompts[1].prompt == std::vector<int>{2, 3, 4, 5} &&
            reversed.prompts[0].options.max_new_tokens == 0 &&
            reversed.prompts[0].options.label_token_ids ==
                reversed.prompts[1].options.label_token_ids, "Token-row request differs");
    const auto omitted = parse_score_request(R"({"items":[[248319]],"label_token_ids":[248043]})",
                                              tokenizer);
    require(omitted.prompts[0].prompt == std::vector<int>{248319}, "Model vocabulary differs");

    reject(tokenizer, R"({"items":"text"})",
           "label_token_ids is required for generation (CausalLM) models.");
    reject(tokenizer, R"({"label_token_ids":[1]})", "items must be provided");
    reject(tokenizer, R"({"items":"x","label_token_ids":[2,2]})",
           "label_token_ids must not contain duplicate tokens");
    reject(tokenizer, R"({"items":"x","label_token_ids":[-1]})",
           "label_token_ids must contain nonnegative integers");
    reject(tokenizer, R"({"items":"x","label_token_ids":[248044]})",
           "Token ID 248044 is out of vocabulary (vocab size: 248044)");
    reject(tokenizer, R"({"items":["x","y"],"label_token_ids":[[2]]})",
           "label_token_ids must have one list per item");
    reject(tokenizer, R"({"items":"x","label_token_ids":[1],"temperature":2})",
           "temperature requires apply_softmax=True");
    reject<sglang::server::ScoreValidationError>(
        tokenizer, R"({"items":"x","label_token_ids":[1],"temperature":0})",
        "temperature must be finite and greater than zero");
}

void check_outputs(const sglang::Tokenizer& tokenizer) {
    const std::vector<float> logits{1000, 1002, 998, 995};
    const std::vector<int> labels{2, 0};
    const auto row = sglang::decoder::label_logprobs(logits, labels);
    const double denominator = 1 + std::exp(-2.) + std::exp(-4.) + std::exp(-7.);
    near(row[0], -4 - std::log(denominator));
    near(row[1], -2 - std::log(denominator));
    std::vector<sglang::DecoderResult> results(1);
    results[0].label_logprobs = std::make_shared<const std::vector<double>>(row);
    auto request = parse_score_request(
        R"({"items":[[5,6]],"label_token_ids":[2,0],"return_token_logprobs":true})", tokenizer);
    const auto verify = [&](double first, double second) {
        const auto response = sglang::json::parse(score_response(request, results));
        const auto& object = *response.object();
        const auto& scores = *get(object, "scores")->array()->front().array();
        const auto& raw = *get(object, "token_logprobs")->array()->front().array();
        const auto& usage = *get(object, "usage")->object();
        require(get(object, "object")->string() == "scoring" &&
                get(object, "model")->string() == "default" &&
                get(object, "pooled_hidden_states")->is_null() &&
                get(usage, "prompt_tokens")->unsigned_integer() == 2 &&
                get(usage, "total_tokens")->unsigned_integer() == 2 &&
                get(usage, "completion_tokens")->unsigned_integer() == 0, "Response shape differs");
        near(*scores[0].number(), first);
        near(*scores[1].number(), second);
        near(*raw[0].number(), row[0]);
        near(*raw[1].number(), row[1]);
    };
    verify(std::exp(row[0]), std::exp(row[1]));
    request.apply_softmax = true;
    request.temperature = 2;
    verify(1 / (1 + std::exp(1.)), 1 / (1 + std::exp(-1.)));
    request = parse_score_request(
        R"({"items":[],"label_token_ids":[-1],"return_token_logprobs":true})", tokenizer);
    const auto empty = sglang::json::parse(score_response(request, {}));
    require(get(*empty.object(), "scores")->array()->empty() &&
            get(*empty.object(), "token_logprobs")->array()->empty(), "Empty response differs");
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Pass the checkpoint tokenizer.json");
        check_budget_arithmetic();
        const sglang::Tokenizer tokenizer(argv[1]);
        check_inputs(tokenizer);
        check_outputs(tokenizer);
        std::cout << "Score CPU contract passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
