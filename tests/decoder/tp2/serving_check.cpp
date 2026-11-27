#include "sglang/decoder_batch.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <stdexcept>

void watch_draft_upload(const std::filesystem::path&);
void inspect_serving(const sglang::Decoder&, int);
void check_tp2_prefill_plan();
void check_serving_reductions(const char*, std::size_t);

namespace {
using namespace sglang;
using Tokens = std::vector<int>;
const std::array<Tokens, 2> prompts{{{760, 6511, 314, 9338, 369}, {760, 6511, 314, 6124, 369}}};
constexpr int budget = 8;

void need(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
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
    for (int token : tokens) out << '\t' << token;
    out << std::endl;
}

Tokens serial(const Decoder& decoder, int slot, bool sample) {
    const auto config = options(sample, 0);
    const auto result = decoder.generate(prompts[slot], config);
    need(result.finish_reason == DecoderFinishReason::length && !result.mtp_drafted_tokens,
         "Target-only serial request unexpectedly stopped or drafted");
    check_serving_reductions(sample ? "serial_sample" : "serial_greedy",
                             prompts[slot].size() + budget - 1);
    return result.output_token_ids;
}

void batch(const Decoder& decoder, bool sample, std::ostream& output) {
    const char* label = sample ? "batch_mtp4_sample" : "batch_mtp4_greedy";
    DecoderBatchOptions setup;
    setup.max_lanes = 2;
    DecoderBatch runner(decoder, setup);
    const auto config = options(sample, 4);
    std::array<bool, decoder_batch_max_lanes> selected{};
    std::array<Tokens, 2> tokens;
    std::array<int, 2> drafted{}, verified{}, accepted{};
    bool simultaneous = false;
    for (int slot = 0; slot < 2; ++slot) {
        need(runner.admit(slot, prompts[slot], config) == 0, "Unexpected cached admission");
        selected[slot] = true;
    }
    for (int step = 0; step < 2 * budget && (selected[0] || selected[1]); ++step) {
        DecoderBatchStepStats stats;
        const auto rows = runner.advance_chunked(selected, &stats);
        simultaneous |= rows[0].mtp_verified_rows == 5 && rows[1].mtp_verified_rows == 5;
        for (int slot = 0; slot < 2; ++slot) {
            if (!selected[slot]) continue;
            const auto& row = rows[slot];
            need(!row.sampling_failed && !row.constraint_failed && !row.prefix_pending,
                 "Unexpected lane failure or pending prefix");
            if (row.token) tokens[slot].push_back(*row.token);
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
        std::printf("case=%s lane=%d drafted=%d accepted=%d verified=%d\n", label, slot,
                    drafted[slot], accepted[slot], verified[slot]);
        need(!selected[slot] && drafted[slot] >= 4 && verified[slot] >= 5 && simultaneous,
             "Two-lane MTP4 never verified both lanes");
        record(output, label, slot, tokens[slot]);
    }
    const auto minimum_rows = prompts[0].size() + prompts[1].size() + 2 * (budget - 1);
    check_serving_reductions(label, minimum_rows);
}
}

int main(int argc, char** argv) try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc != 4) throw std::invalid_argument("usage: serving-check CHECKPOINT TP OUTPUT");
    const int world = std::stoi(argv[2]);
    need(world == 1 || world == 2, "Invalid world size");
    std::ofstream output(argv[3]);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    if (world == 2) {
        check_tp2_prefill_plan();
        watch_draft_upload(argv[1]);
    }
    sglang::Decoder decoder(argv[1], 0, true, world);
    inspect_serving(decoder, world);
    for (int slot = 0; slot < 2; ++slot) {
        record(output, "serial_greedy", slot, serial(decoder, slot, false));
        record(output, "serial_sample", slot, serial(decoder, slot, true));
    }
    for (bool sample : {false, true}) batch(decoder, sample, output);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "serving-check: %s\n", e.what());
    return 1;
}
