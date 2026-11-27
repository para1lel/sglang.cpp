#include "sglang/decoder_constraint.h"
#include "sglang/server/serving/qwen_serving_profile.h"
#include "sglang/server/http.h"
#include "sglang/server/text_completion.h"

#include <iostream>

namespace {

using namespace sglang;

bool allowed(const ConstraintCursor& cursor, int token) {
    return std::binary_search(cursor.ids().begin(), cursor.ids().end(), token);
}

void check_mask(const ConstraintCursor& cursor) {
    const auto& ids = cursor.ids();
    const auto& mask = cursor.mask();
    if (!std::is_sorted(ids.begin(), ids.end()) ||
        std::adjacent_find(ids.begin(), ids.end()) != ids.end() ||
        (ids.size() > constraint_max_candidates) != !mask.empty()) {
        throw std::logic_error("Invalid candidate representation");
    }
    if (!mask.empty()) {
        std::vector<std::uint32_t> expected(constraint_mask_words);
        for (const auto id : ids) {
            expected.at(id / 32) |= std::uint32_t{1} << (id % 32);
        }
        if (mask != expected) {
            throw std::logic_error("Dense mask differs from the allowed token IDs");
        }
    }
}

void print_state(const ConstraintCursor& cursor, bool terminal) {
    std::cout << "state " << cursor.ids().size() << ' ' << !cursor.mask().empty() << ' '
              << allowed(cursor, end_of_text) << ' ' << allowed(cursor, end_of_message) << ' '
              << terminal << '\n';
}

void walk(const TokenConstraint& constraint, const json::Array& tokens, int remaining) {
    ConstraintCursor live, draft;
    bool terminal = false;
    std::cout << "minimum " << constraint.minimum_tokens() << '\n';
    for (const auto& value : tokens) {
        const auto token = value.unsigned_integer().value_or(model_vocab_size);
        if (token >= model_vocab_size) {
            throw std::invalid_argument("Expected a checkpoint token ID");
        }
        live.candidates(constraint, remaining);
        check_mask(live);
        print_state(live, terminal);
        if (!allowed(live, token) || remaining == 0 || terminal) {
            std::cout << "rejected " << token << '\n';
            return;
        }
        draft.reset_to(live);
        terminal = draft.commit(constraint, token, remaining - 1);
        live.candidates(constraint, remaining);
        if (!allowed(live, token)) {
            throw std::logic_error("Trial commit changed the live cursor");
        }
        --remaining;
        if (live.commit(constraint, token, remaining) != terminal) {
            throw std::logic_error("Live and trial commits returned different stop decisions");
        }
        std::cout << "accepted " << token << '\n';
    }
    live.candidates(constraint, remaining);
    check_mask(live);
    print_state(live, terminal);
}

void probe(const server::CompletionTokenizer& tokenizer, std::string_view input) {
    const auto parsed = json::parse(input);
    const auto* values = parsed.array();
    if (!values || values->size() != 2 || !values->at(0).string()) {
        throw std::invalid_argument("Expected a request body string and output text or token IDs");
    }
    const auto request = server::parse_chat_request(*values->at(0).string(), tokenizer);
    const auto& output = values->at(1);
    json::Array encoded;
    if (const auto text = output.string()) {
        for (const auto token : tokenizer.get().encode(*text)) {
            encoded.emplace_back(static_cast<int>(token));
        }
    } else if (const auto* ids = output.array()) {
        encoded = *ids;
    } else {
        throw std::invalid_argument("Expected output text or token IDs");
    }
    std::string prompt, quoted;
    for (const auto token : request.prompt) {
        prompt += tokenizer.get().piece(token);
    }
    json::append_quoted(quoted, prompt);
    std::cout << "prompt " << quoted << '\n';
    if (const auto& constraint = request.options.constraint) {
        walk(*constraint, encoded, request.options.max_new_tokens);
    } else {
        std::cout << "unconstrained\n";
    }
}

}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: tool_probe CHECKPOINT\n";
        return 64;
    }
    const server::CompletionTokenizer tokenizer(argv[1], server::SamplingDefaults::model,
                                                {true, true}, model::qwen_serving_profile);
    for (std::string line; std::getline(std::cin, line);) {
        try {
            probe(tokenizer, line);
        } catch (const std::invalid_argument& error) {
            std::cout << "invalid " << error.what() << '\n';
        } catch (const server::HttpError& error) {
            std::cout << "invalid " << error.what() << '\n';
        }
        std::cout << "end\n";
    }
}
