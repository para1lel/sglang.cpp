#include "sglang/decoder_constraint.h"
#ifdef SGLANG_CHECK_JSON_SCHEMA
#include "sglang/json.h"
#endif

#include <iostream>

namespace {

void print_candidates(const sglang::ConstraintCursor& cursor) {
    const auto& mask = cursor.mask();
    const auto& ids = cursor.ids();
    if (!std::is_sorted(ids.begin(), ids.end()) ||
        std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
        throw std::logic_error("Candidate IDs are not strictly ordered");
    }
    if ((ids.size() > sglang::constraint_max_candidates) != !mask.empty()) {
        throw std::logic_error("Wrong sparse or dense representation");
    }
    if (!mask.empty()) {
        std::vector<std::uint32_t> expected(sglang::constraint_mask_words);
        for (const auto id : ids) {
            expected.at(id / 32) |= std::uint32_t{1} << (id % 32);
        }
        if (mask != expected) {
            throw std::logic_error("Dense bitset differs from candidate IDs");
        }
    }
    std::cout << "allowed";
    for (const auto id : ids) {
        std::cout << ' ' << id;
    }
    std::cout << '\n';
}

void walk(const sglang::TokenConstraint& constraint, int remaining, int argc, char** argv) {
    std::cout << "minimum " << constraint.minimum_tokens() << '\n';
    sglang::ConstraintCursor cursor;
    for (int index = 5; index <= argc; ++index) {
        cursor.candidates(constraint, remaining);
        print_candidates(cursor);
        if (index == argc) {
            break;
        }
        if (remaining <= 0) {
            throw std::logic_error("Probe exceeded its token budget");
        }
        const auto token = std::stoi(argv[index]);
        const auto& ids = cursor.ids();
        if (!std::binary_search(ids.begin(), ids.end(), token)) {
            throw std::logic_error("Scripted token is outside the candidate set");
        }
        if (cursor.commit(constraint, token, --remaining)) {
            if (index != argc - 1) {
                throw std::logic_error("Constraint stopped before the scripted final token");
            }
            std::cout << "stop\n";
            break;
        }
    }
}

#ifdef SGLANG_CHECK_JSON_SCHEMA
sglang::json::Value parse_schema(std::string_view source) {
    try {
        return sglang::json::parse(source, sglang::json::Syntax::ordered);
    } catch (const sglang::json::ParseError& error) {
        throw std::invalid_argument(error.what());
    }
}
#endif

std::shared_ptr<const sglang::TokenConstraint> compile(
    const sglang::Tokenizer& tokenizer, std::string_view kind, std::string_view source) {
    if (kind == "regex") {
        return sglang::TokenConstraint::compile_regex(tokenizer, source);
    }
#ifdef SGLANG_CHECK_JSON_SCHEMA
    if (kind == "schema") {
        return sglang::TokenConstraint::compile_json_schema(tokenizer, parse_schema(source));
    }
#endif
    throw std::invalid_argument("Unknown constraint kind");
}

}

int main(int argc, char** argv) {
    if (argc < 5) {
        return 64;
    }
    try {
        const sglang::Tokenizer tokenizer(argv[1]);
        const auto constraint = compile(tokenizer, argv[2], argv[3]);
        walk(*constraint, std::stoi(argv[4]), argc, argv);
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
