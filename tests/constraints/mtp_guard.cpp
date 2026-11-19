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
    options.constraint = constraint;
    options.max_new_tokens = 16;
    sglang::decoder::validate_request(prompt, options, true);
    for (int depth = 1; depth <= sglang::decoder_max_mtp_steps; ++depth) {
        options.mtp_steps = depth;
        try {
            sglang::decoder::validate_request(prompt, options, true);
        } catch (const std::invalid_argument& error) {
            if (std::string_view(error.what()) == "Constrained decoding requires mtp_steps=0") {
                continue;
            }
            throw;
        }
        throw std::runtime_error("A constrained decoder request accepted positive MTP depth");
    }
    options.constraint.reset();
    sglang::decoder::validate_request(prompt, options, true);
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
        std::cout << "EBNF and regex accept K0 and reject every positive depth\n";
#ifdef SGLANG_CHECK_JSON_SCHEMA
        const auto schema =
            sglang::json::parse(R"({"type":"null"})", sglang::json::Syntax::ordered);
        check_depth(sglang::TokenConstraint::compile_json_schema(tokenizer, schema));
        std::cout << "JSON Schema accepts K0 and rejects every positive depth\n";
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
