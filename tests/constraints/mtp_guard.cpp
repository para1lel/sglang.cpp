#include "sglang/decoder_constraint.h"
#include "sglang/decoder_internal.h"
#include "sglang/json.h"
#include "sglang/tokenizer/tokenizer.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

void check_depth(const std::shared_ptr<const sglang::TokenConstraint>& constraint) {
    const std::vector<int> prompt{32};
    sglang::DecoderOptions options;
    if (options.requires_target_only()) {
        throw std::runtime_error("Ordinary text unexpectedly requires target-only decode");
    }
    options.constraint = constraint;
    if (options.requires_target_only()) {
        throw std::runtime_error("Constrained text unexpectedly requires target-only decode");
    }
    options.max_new_tokens = 16;
    for (int depth = 0; depth <= sglang::decoder_max_mtp_steps; ++depth) {
        options.mtp_steps = depth;
        sglang::decoder::validate_request(prompt, options, true);
    }
    options.stop_token_id = 32;
    try {
        sglang::decoder::validate_request(prompt, options, true);
    } catch (const std::invalid_argument& error) {
        constexpr std::string_view eos_error =
            "Constrained decoding requires the model EOS stop tokens";
        if (error.what() == eos_error) {
            return;
        }
        throw;
    }
    throw std::runtime_error("A constrained decoder request accepted a custom stop token");
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::invalid_argument("Expected the path to checkpoint tokenizer.json");
        }
        const sglang::Tokenizer tokenizer(argv[1]);
        check_depth(sglang::TokenConstraint::compile(tokenizer, R"(root ::= "abc")"));
        check_depth(sglang::TokenConstraint::compile_regex(tokenizer, "^(?:ab|abc)$"));
        std::cout << "EBNF and regex accept K0 through K4 with model EOS\n";
#ifdef SGLANG_CHECK_JSON_SCHEMA
        const auto schema =
            sglang::json::parse(R"({"type":"null"})", sglang::json::Syntax::ordered);
        check_depth(sglang::TokenConstraint::compile_json_schema(tokenizer, schema));
        std::cout << "JSON Schema accepts K0 through K4 with model EOS\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
