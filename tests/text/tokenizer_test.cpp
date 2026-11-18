#include "sglang/json.h"
#include "sglang/tokenizer/tokenizer.h"
#include "sglang/tokenizer/unicode.h"
#include "sglang/utf8.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool passed, std::string_view message) {
    if (!passed) {
        throw std::runtime_error(std::string(message));
    }
}

void check_utf8() {
    for (const std::string_view scalar : {"é", "語", "𠮷"}) {
        for (std::size_t boundary = 1; boundary < scalar.size(); ++boundary) {
            sglang::Utf8Stream complete;
            sglang::Utf8Stream independent;
            sglang::Utf8Stream truncated;
            require(complete.push(scalar.substr(0, boundary)).empty(), "UTF-8 carry");
            require(independent.push("ABC") == "ABC" && independent.finish().empty(),
                    "UTF-8 state must be request-local");
            require(complete.push(scalar.substr(boundary)) == scalar &&
                        complete.finish().empty(), "UTF-8 completion");
            require(truncated.push(scalar.substr(0, boundary)).empty() &&
                        truncated.finish() == "�" && truncated.finish().empty(),
                    "truncated scalar must flush exactly once");
        }
    }
    sglang::Utf8Stream malformed;
    require(malformed.push("\xed\xa0\x80") == "���", "surrogate bytes must be replaced");
    require(malformed.push("\xf0\x90").empty() && malformed.push("A") == "�A",
            "bad continuation must be reconsidered as a lead byte");
    require(!sglang::valid_utf8("\xc0\xaf") && !sglang::valid_utf8("\xf4\x90\x80\x80"),
            "reject overlong and out-of-range UTF-8");
}

void check_unicode() {
    require(sglang::tokenizer::normalize_nfc("Cafe\u0301 한") == "Café 한",
            "canonical composition and Hangul");
    require(sglang::tokenizer::normalize_nfc("A\u0315\u0300") == "À\u0315",
            "canonical combining-class order");
    std::vector<std::string> pieces;
    sglang::tokenizer::split_pieces("don't १२ 3\r\n漢字", [&](std::string_view piece) {
        pieces.emplace_back(piece);
    });
    require(pieces == std::vector<std::string>{"don", "'t", " ", "१", "२", " ", "3", "\r\n",
                                               "漢字"}, "Unicode pretokenizer pieces");
}

void check_checkpoint(const char* tokenizer_path, const char* fixture_path) {
    const sglang::Tokenizer tokenizer(tokenizer_path);
    std::ifstream stream(fixture_path);
    require(stream.is_open(), "open text fixture");
    const std::string bytes((std::istreambuf_iterator<char>(stream)), {});
    const auto fixture = sglang::json::parse(bytes);
    const auto& cases = *fixture.object()->at("cases").object();
    require(!cases.empty(), "no checkpoint cases");
    require(tokenizer.encode("", 0).empty(), "empty input with zero budget");
    for (const auto& [label, entry] : cases) {
        const auto& sample = *entry.object();
        const auto text = sglang::json::get_string(sample, "text").value();
        const auto& ids = *sample.at("ids").array();
        const auto actual = tokenizer.encode(text, ids.size());
        require(!actual.empty() && actual.size() == ids.size(), label);
        for (std::size_t index = 0; index < ids.size(); ++index) {
            require(actual[index] == ids[index].unsigned_integer().value(), label);
        }
        bool rejected = false;
        try {
            tokenizer.encode(text, ids.size() - 1);
        } catch (const std::length_error&) {
            rejected = true;
        }
        require(rejected, label);
    }
    require(tokenizer.piece(248046, true).empty() &&
                tokenizer.piece(248046, false) == "<|im_end|>", "special token output policy");
    require(tokenizer.piece(248319).empty(), "embedding-only token has no tokenizer bytes");
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 1 || argc == 3, "usage: tokenizer-test [TOKENIZER_JSON FIXTURE_JSON]");
        check_utf8();
        check_unicode();
        if (argc == 3) {
            check_checkpoint(argv[1], argv[2]);
        }
        std::cout << "Native UTF-8 and tokenizer checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
