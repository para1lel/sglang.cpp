#include "sglang/decoder_batch.h"
#include "sglang/decoder_constraint.h"
#include "sglang/decoder_constraint_cuda.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <vector>

namespace {

using sglang::Decoder;
using sglang::DecoderBatch;
using sglang::DecoderBatchStepStats;
using sglang::DecoderFinishReason;
using sglang::DecoderOptions;
using sglang::TokenConstraint;
using sglang::Tokenizer;
using Selection = std::array<bool, sglang::decoder_batch_max_lanes>;

const std::vector<int> prompt{8149, 198};
int fail_after = -1;
int injected = 0;
std::vector<int> score_plan;
std::size_t score_step = 0;

void require(bool passed, const char* message) {
    if (!passed) {
        throw std::runtime_error(message);
    }
}

std::string bytes(const Tokenizer& tokenizer, const std::vector<int>& ids) {
    std::string result;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (ids[index] == sglang::end_of_text || ids[index] == sglang::end_of_message) {
            require(index + 1 == ids.size(), "EOS preceded content");
        } else {
            const auto piece = tokenizer.piece(ids[index], true);
            require(!piece.empty(), "Unexpected hidden token in constrained output");
            result += piece;
        }
    }
    return result;
}

DecoderOptions options(const Tokenizer& tokenizer, std::string_view grammar, int budget) {
    DecoderOptions result;
    result.constraint = TokenConstraint::compile(tokenizer, grammar);
    result.max_new_tokens = budget;
    return result;
}

using TokenEdges = std::vector<std::vector<std::pair<int, std::size_t>>>;

TokenEdges reference_edges(const Tokenizer& tokenizer, std::string_view choice) {
    TokenEdges edges(choice.size());
    for (int id = 0; id < sglang::model_vocab_size; ++id) {
        const auto piece = tokenizer.piece(id, true);
        if (piece.empty() || id == sglang::end_of_text || id == sglang::end_of_message) {
            continue;
        }
        for (auto offset = choice.find(piece); offset != std::string_view::npos;
             offset = choice.find(piece, offset + 1)) {
            edges[offset].emplace_back(id, offset + piece.size());
        }
    }
    return edges;
}

std::vector<int> reference_choices(const std::vector<std::string>& choices,
                                   const std::vector<TokenEdges>& spellings,
                                   std::string_view prefix, int budget) {
    std::vector<int> allowed;
    for (std::size_t index = 0; index < choices.size(); ++index) {
        const auto& choice = choices[index];
        if (choice.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        if (prefix.size() == choice.size()) {
            allowed.insert(allowed.end(), {sglang::end_of_text, sglang::end_of_message});
            continue;
        }
        const auto& edges = spellings[index];
        std::vector<int> minimum(choice.size() + 1, static_cast<int>(choice.size()) + 1);
        minimum.back() = 0;
        for (std::size_t offset = choice.size(); offset-- > prefix.size();) {
            for (const auto& [token, end] : edges[offset]) {
                minimum[offset] = std::min(minimum[offset], 1 + minimum[end]);
            }
        }
        for (const auto& [token, end] : edges[prefix.size()]) {
            if (minimum[end] < budget) {
                allowed.push_back(token);
            }
        }
    }
    std::sort(allowed.begin(), allowed.end());
    allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
    return allowed;
}

void check_cursor(const Tokenizer& tokenizer) {
    const auto grammar = TokenConstraint::compile(tokenizer,
        R"( root ::= "hello" | "hello!" | "\uD83D\uDE42" | "e\u0301" | "\"\\\n\u0000" )");
    require(grammar->minimum_tokens() == 1, "Expected a one-token alternative");
    const std::vector<std::string> spellings{"hello", "hello!", "🙂", "é",
                                              std::string("\"\\\n\0", 4)};
    std::vector<TokenEdges> edges;
    for (const auto& spelling : spellings) {
        edges.push_back(reference_edges(tokenizer, spelling));
    }
    std::array<int, 256> singles;
    singles.fill(-1);
    for (int id = 0; id < sglang::model_vocab_size; ++id) {
        const auto piece = tokenizer.piece(id, true);
        if (piece.size() == 1) {
            singles[static_cast<unsigned char>(piece.front())] = id;
        }
    }
    int equivalent_paths = 0;
    for (const auto& spelling : spellings) {
        std::vector<int> raw;
        for (const unsigned char byte : spelling) {
            require(singles[byte] >= 0, "Snapshot is missing a byte fallback token");
            raw.push_back(singles[byte]);
        }
        const auto encoded = tokenizer.encode(spelling);
        const std::vector<int> merged(encoded.begin(), encoded.end());
        std::vector<std::vector<int>> paths{raw};
        if (merged != raw && bytes(tokenizer, merged) == spelling) {
            paths.push_back(merged);
            ++equivalent_paths;
        }
        for (const auto& ids : paths) {
            require(bytes(tokenizer, ids) == spelling, "Fixture tokenization changed bytes");
            sglang::ConstraintCursor cursor;
            std::string prefix;
            for (std::size_t index = 0; index < ids.size(); ++index) {
                const int rest = static_cast<int>(ids.size() - index - 1);
                cursor.candidates(*grammar, rest + 1);
                require(cursor.ids() == reference_choices(spellings, edges, prefix, rest + 1),
                        "Allowed IDs differ from the tokenizer-byte oracle");
                require(std::binary_search(cursor.ids().begin(), cursor.ids().end(), ids[index]),
                        "Missing equivalent token path");
                require(cursor.commit(*grammar, ids[index], rest) == (rest == 0),
                        "Incorrect byte path termination");
                prefix += tokenizer.piece(ids[index], true);
            }
        }
    }
    require(equivalent_paths >= 2, "Fixture did not exercise merged and split token paths");
    for (const int eos : {sglang::end_of_text, sglang::end_of_message}) {
        sglang::ConstraintCursor cursor;
        const auto hello = tokenizer.encode("hello");
        require(hello.size() == 1, "Expected one token for hello");
        require(!cursor.commit(*grammar, hello.front(), 1), "Prefix ended with budget left");
        cursor.candidates(*grammar, 1);
        require(std::binary_search(cursor.ids().begin(), cursor.ids().end(), eos),
                "Accepting prefix omitted checkpoint EOS");
        require(cursor.commit(*grammar, eos, 0), "EOS did not terminate the accepting prefix");
    }
}

void check_controlled_scores(const Decoder& decoder, const Tokenizer& tokenizer) {
    const std::vector<std::string> spellings{"hello", "hello!"};
    const std::vector<TokenEdges> edges{reference_edges(tokenizer, spellings[0]),
                                       reference_edges(tokenizer, spellings[1])};
    const auto root = reference_choices(spellings, edges, "", 2);
    const auto continuation = reference_choices(spellings, edges, "hello", 1);
    const auto hello = tokenizer.encode("hello").at(0);
    const auto hello_at = std::find(root.begin(), root.end(), hello);
    require(hello_at != root.end(), "Missing controlled root token");
    auto constrained = options(tokenizer, R"(root ::= "hello" | "hello!")", 2);
    for (const int eos : {sglang::end_of_text, sglang::end_of_message}) {
        const auto eos_at = std::find(continuation.begin(), continuation.end(), eos);
        require(eos_at != continuation.end(), "Missing controlled EOS");
        score_plan = {static_cast<int>(hello_at - root.begin()),
                      static_cast<int>(eos_at - continuation.begin())};
        score_step = 0;
        constrained.temperature = eos == sglang::end_of_text ? 0 : 1.25;
        constrained.top_k = 1;
        const auto result = decoder.generate(prompt, constrained);
        require(score_step == score_plan.size() &&
                    result.finish_reason == DecoderFinishReason::constraint &&
                    result.output_token_ids == std::vector<int>{static_cast<int>(hello), eos},
                "Controlled greedy or sampled EOS did not stop at the accepting prefix");
    }
    const std::vector<std::string> single_tokens{"h", "z"};
    const std::vector<TokenEdges> single_edges{reference_edges(tokenizer, single_tokens[0]),
                                              reference_edges(tokenizer, single_tokens[1])};
    const auto tied = reference_choices(single_tokens, single_edges, "", 1);
    require(tied.size() >= 2, "Expected at least two tied candidates");
    constrained = options(tokenizer, R"(root ::= "h" | "z")", 1);
    score_plan = {-1};
    score_step = 0;
    const auto result = decoder.generate(prompt, constrained);
    require(score_step == 1 && result.finish_reason == DecoderFinishReason::constraint &&
                result.output_token_ids == std::vector<int>{tied.front()},
            "Greedy allowed-row tie did not select the lowest vocabulary ID");
    score_plan.clear();
    score_step = 0;
}

void check_direct(const Decoder& decoder, const Tokenizer& tokenizer) {
    check_controlled_scores(decoder, tokenizer);
    auto constrained = options(tokenizer, R"(root ::= "hello" | "hello!")", 1);
    const auto boundary = decoder.generate(prompt, constrained);
    require(boundary.finish_reason == DecoderFinishReason::constraint &&
                boundary.output_token_ids.size() == 1 &&
                bytes(tokenizer, boundary.output_token_ids) == "hello",
            "Exact budget did not complete the feasible choice");
    for (const auto invalid : {0, 1, 2}) {
        auto rejected = constrained;
        if (invalid == 0) {
            rejected.max_new_tokens = 0;
        } else if (invalid == 1) {
            rejected.mtp_steps = 1;
        } else {
            rejected.stop_token_id = 42;
        }
        bool threw = false;
        try {
            decoder.generate(prompt, rejected);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        require(threw, "Invalid direct constraint options were accepted");
    }

    constrained = options(tokenizer, R"(root ::= "JQ: 42+17=59" | "JQ: 42+17=591")", 12);
    constrained.temperature = 1.25;
    constrained.top_k = 5;
    constrained.top_p = 0.9;
    constrained.seed = -31;
    const auto stable = decoder.generate(prompt, constrained);
    constrained.eager = true;
    const auto eager = decoder.generate(prompt, constrained);
    require(stable.finish_reason == DecoderFinishReason::constraint &&
                stable.output_token_ids == eager.output_token_ids,
            "Seeded constrained serial eager and graph disagree");
    const auto output = bytes(tokenizer, stable.output_token_ids);
    require(output == "JQ: 42+17=59" || output == "JQ: 42+17=591",
            "Seeded constrained serial output is outside the grammar");
    const auto callback = sglang::DecoderCallbacks{
        [](int) { return false; }, {},
    };
    const auto cancelled = decoder.generate(prompt, constrained, callback);
    require(cancelled.finish_reason == DecoderFinishReason::cancelled &&
                cancelled.output_token_ids == std::vector<int>{stable.output_token_ids.front()},
            "Serial cancellation published an unexpected prefix");
    fail_after = 1;
    int delivered = 0;
    bool threw = false;
    try {
        decoder.generate(prompt, constrained, {[&](int) { ++delivered; return true; }, {}});
    } catch (const sglang::DecoderConstraintError&) {
        threw = true;
    }
    require(threw && delivered == 1 && injected == 1,
            "Serial bad allowed row did not fail after one delivered token");
    require(decoder.generate(prompt, constrained).output_token_ids == stable.output_token_ids,
            "Serial retry changed after a bad allowed row");
}

void check_batch(const Decoder& decoder, const Tokenizer& tokenizer, bool graphs) {
    auto rejected = options(tokenizer, R"(root ::= "RX: 123456789" | "RX: 123456780")", 14);
    auto peer = options(tokenizer, R"(root ::= "XY: 987654321" | "XY: 987654320")", 14);
    peer.temperature = 0.9;
    peer.seed = -17;
    const auto wanted_peer = decoder.generate(prompt, peer).output_token_ids;
    const auto wanted_retry = decoder.generate(prompt, rejected).output_token_ids;
    DecoderBatch batch(decoder, {graphs, 32});
    batch.admit(3, prompt, rejected);
    batch.admit(31, prompt, peer);
    Selection selected{};
    selected[3] = selected[31] = true;
    std::vector<int> right;
    bool failed = false;
    bool observed_graph = false;
    int ticks = 0;
    const int before = injected;
    while (selected[3] || selected[31]) {
        require(++ticks <= 32, "Constrained batch did not retire");
        if (ticks == 3) {
            fail_after = 0;
        }
        DecoderBatchStepStats stats;
        const auto steps = batch.advance_chunked(selected, &stats);
        if (steps[3].constraint_failed) {
            require(ticks == 3 && injected == before + 1 && !steps[3].token &&
                        !steps[3].sampling_failed,
                    "Invalid allowed row produced a token or the wrong error type");
            observed_graph = stats.graph_slots == ((1u << 3) | (1u << 31));
            failed = true;
            selected[3] = false;
        }
        require(!steps[31].constraint_failed && !steps[31].sampling_failed,
                "Peer failed alongside the injected allowed row");
        if (steps[31].token) {
            right.push_back(*steps[31].token);
        }
        if (steps[31].finish_reason) {
            require(steps[31].finish_reason == DecoderFinishReason::constraint,
                    "Unexpected peer finish");
            selected[31] = false;
        }
    }
    require(failed && observed_graph == graphs && right == wanted_peer && batch.available() &&
                batch.request_device_allocation_bytes() == 0,
            "Failed slot retirement, replay, or peer parity");

    batch.admit(3, prompt, peer);
    selected[3] = true;
    batch.advance_chunked(selected);
    batch.cancel(3);
    require(batch.request_device_allocation_bytes() == 0, "Cancelled slot leaked request memory");
    batch.admit(3, prompt, rejected);
    std::vector<int> retry;
    for (int tick = 0; tick < 32; ++tick) {
        const auto step = batch.advance_chunked(selected)[3];
        require(!step.constraint_failed && !step.sampling_failed, "Reused slot failed");
        if (step.token) {
            retry.push_back(*step.token);
        }
        if (step.finish_reason) {
            require(step.finish_reason == DecoderFinishReason::constraint, "Reused slot finish");
            break;
        }
    }
    require(retry == wanted_retry && batch.request_device_allocation_bytes() == 0,
            "Cancel or invalid row changed the reused slot");
}

}

extern "C" cudaError_t real_gather(
    const __nv_bfloat16*, const int*, __nv_bfloat16*, int, cudaStream_t)
    asm("__real__ZN6sglang7decoder25gather_constrained_logits"
        "EPK13__nv_bfloat16PKiPS1_iP11CUstream_st");

extern "C" cudaError_t wrap_gather(
    const __nv_bfloat16*, const int*, __nv_bfloat16*, int, cudaStream_t)
    asm("__wrap__ZN6sglang7decoder25gather_constrained_logits"
        "EPK13__nv_bfloat16PKiPS1_iP11CUstream_st");

extern "C" cudaError_t wrap_gather(
    const __nv_bfloat16* source, const int* ids, __nv_bfloat16* logits, int count,
    cudaStream_t stream) {
    const auto status = real_gather(source, ids, logits, count, stream);
    if (status == cudaSuccess && fail_after >= 0 && fail_after-- == 0) {
        ++injected;
        return cudaMemsetAsync(logits, 0xff, count * sizeof(*logits), stream);
    }
    if (status == cudaSuccess && score_step < score_plan.size()) {
        const auto winner = score_plan[score_step++];
        if (winner >= count) {
            return cudaErrorInvalidValue;
        }
        const auto cleared = cudaMemsetAsync(logits, 0, count * sizeof(*logits), stream);
        if (cleared != cudaSuccess || winner < 0) {
            return cleared;
        }
        return cudaMemsetAsync(logits + winner, 0x3f, sizeof(*logits), stream);
    }
    return status;
}

int main(int argc, char** argv) {
    try {
        require(argc == 2 || (argc == 3 && std::string_view(argv[2]) == "--cursor-only"),
                "Expected a checkpoint path and optional --cursor-only");
        const std::filesystem::path checkpoint = argv[1];
        const Tokenizer tokenizer(checkpoint / "tokenizer.json");
        check_cursor(tokenizer);
        std::cout << "constraint cursor: tokenizer-byte oracle, alternate paths and EOS passed\n";
        if (argc == 2) {
            const Decoder decoder(checkpoint);
            check_direct(decoder, tokenizer);
            check_batch(decoder, tokenizer, false);
            check_batch(decoder, tokenizer, true);
            std::cout << "constraint decoder: score selection, eager, graphs, "
                         "fault retirement and reuse passed\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
