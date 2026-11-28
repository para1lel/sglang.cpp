#include "sglang/decoder_batch.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

void watch_draft_upload(const std::filesystem::path&);
void inspect_serving(const sglang::Decoder&, int);
void check_tp2_prefill_plan();
void check_serving_reductions(const char*, std::size_t);
extern "C" void watch_serving_draft(bool decode);

namespace {
using namespace sglang;
using Tokens = std::vector<int>;
using Prompts = std::array<Tokens, 2>;
constexpr int budget = 8;

void need(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

DecoderOptions options(bool sample, int depth) {
    DecoderOptions config;
    config.eager = true;
    config.ignore_eos = true;
    config.max_new_tokens = budget;
    config.mtp_steps = depth;
    if (sample) {
        config.temperature = 0.8;
        config.top_k = 12;
        config.seed = 7103;
    }
    return config;
}

void record(std::ostream& out, const char* label, int slot, const Tokens& tokens) {
    need(tokens.size() == budget, "Unexpected emitted token count");
    out << label << '\t' << slot;
    for (const int token : tokens)
        out << '\t' << token;
    out << std::endl;
}

Tokens serial(const Decoder& decoder, const Prompts& prompts, int slot, bool sample) {
    const auto config = options(sample, 0);
    const auto result = decoder.generate(prompts[slot], config);
    need(result.finish_reason == DecoderFinishReason::length && !result.mtp_drafted_tokens,
         "Target-only serial request unexpectedly stopped or drafted");
    check_serving_reductions(sample ? "serial_sample" : "serial_greedy",
                             prompts[slot].size() + budget - 1);
    return result.output_token_ids;
}

void batch(const Decoder& decoder, const Prompts& prompts, bool sample, bool graphs,
           std::ostream& output) {
    const char* label = sample ? "batch_mtp4_sample" : "batch_mtp4_greedy";
    DecoderBatchOptions setup;
    setup.cuda_graphs = graphs;
    setup.tp_verification_graphs = graphs;
    setup.max_lanes = 2;
    DecoderBatch runner(decoder, setup);
    auto config = options(sample, 4);
    config.eager = !graphs;
    std::array<bool, decoder_batch_max_lanes> selected{};
    std::array<Tokens, 2> tokens;
    std::array<int, 2> drafted{}, verified{}, accepted{};
    bool simultaneous = false;
    int full_replays = 0;
    for (int slot = 0; slot < 2; ++slot) {
        need(runner.admit(slot, prompts[slot], config) == 0, "Unexpected cached admission");
        selected[slot] = true;
    }
    for (int step = 0; step < 2 * budget && (selected[0] || selected[1]); ++step) {
        DecoderBatchStepStats stats;
        watch_serving_draft(step > 0);
        const auto rows = runner.advance_chunked(selected, &stats);
        watch_serving_draft(false);
        const auto graph_slots = stats.graph_slots | stats.verification_graph_slots;
        const unsigned active =
            static_cast<unsigned>(selected[0]) | (static_cast<unsigned>(selected[1]) << 1);
        need((graphs || graph_slots == 0) && (graph_slots & ~active) == 0,
             "Unexpected target graph in normal serving control");
        full_replays += static_cast<int>(active == 3 && stats.verification_graph_slots == active);
        simultaneous |= rows[0].mtp_verified_rows == 5 && rows[1].mtp_verified_rows == 5;
        for (int slot = 0; slot < 2; ++slot) {
            if (!selected[slot])
                continue;
            const auto& row = rows[slot];
            need(!row.sampling_failed && !row.constraint_failed && !row.prefix_pending,
                 "Unexpected lane failure or pending prefix");
            if (row.token)
                tokens[slot].push_back(*row.token);
            need(row.extra_token_count >= 0 && row.extra_token_count <= 4,
                 "Invalid speculative output count");
            tokens[slot].insert(tokens[slot].end(), row.extra_tokens.begin(),
                                row.extra_tokens.begin() + row.extra_token_count);
            drafted[slot] += row.mtp_drafted_tokens;
            accepted[slot] += row.mtp_accepted_tokens;
            verified[slot] += row.mtp_verified_rows;
            if (row.finish_reason) {
                need(*row.finish_reason == DecoderFinishReason::length, "Unexpected batch finish");
                selected[slot] = false;
            }
        }
    }
    for (int slot = 0; slot < 2; ++slot) {
        std::cout << "case=" << label << " lane=" << slot << " drafted=" << drafted[slot]
                  << " accepted=" << accepted[slot]
                  << " rejected=" << drafted[slot] - accepted[slot]
                  << " verified=" << verified[slot] << '\n';
        need(!selected[slot] && drafted[slot] >= 4 && verified[slot] >= 5 && simultaneous,
             "Two-lane MTP4 never verified both lanes");
        need(accepted[slot] <= drafted[slot], "Draft accepted more than it proposed");
        record(output, label, slot, tokens[slot]);
    }
    need(drafted[0] + drafted[1] > accepted[0] + accepted[1],
         "Draft rejection control did not reject a proposal");
    need(!graphs || full_replays >= 2, "Normal fused verifier did not replay both lanes twice");
    std::cout << "case=" << label << " full_fused_replays=" << full_replays << '\n';
    const auto minimum_rows = prompts[0].size() + prompts[1].size() + 2 * (budget - 1);
    check_serving_reductions(label, minimum_rows);
}
}

int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc != 4 && argc != 5) {
        throw std::invalid_argument(
            "usage: serving-check CHECKPOINT TP OUTPUT [mixed|mixed-graph]");
    }
    const int world = std::stoi(argv[2]);
    need(world == 1 || world == 2, "Invalid world size");
    const Prompts short_prompts{{{760, 6511, 314, 9338, 369}, {760, 6511, 314, 6124, 369}}};
    auto prompts = short_prompts;
    bool graphs = false;
    if (argc == 5) {
        const std::string_view shape = argv[4];
        graphs = shape == "mixed-graph";
        need(shape == "mixed" || (world == 2 && graphs), "Invalid mixed prompt control");
        for (int row = 0; row < 124; ++row) {
            prompts[1].push_back(short_prompts[1][row % short_prompts[1].size()]);
        }
    }
    std::ofstream output(argv[3]);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    if (world == 2) {
        watch_draft_upload(argv[1]);
    }
    sglang::Decoder decoder(argv[1], 0, true, world);
    inspect_serving(decoder, world);
    if (world == 2)
        check_tp2_prefill_plan();
    for (int slot = 0; slot < 2; ++slot) {
        record(output, "serial_greedy", slot, serial(decoder, prompts, slot, false));
        record(output, "serial_sample", slot, serial(decoder, prompts, slot, true));
    }
    for (const bool sample : {false, true})
        batch(decoder, prompts, sample, graphs, output);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "serving-check: %s\n", e.what());
    return 1;
}
