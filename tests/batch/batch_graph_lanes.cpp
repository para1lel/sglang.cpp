#include "sglang/decoder_batch.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using namespace sglang;
using Mask = std::array<bool, decoder_batch_max_lanes>;

void require(bool condition, const char* failure) {
    if (!condition) {
        throw std::runtime_error(failure);
    }
}

Mask mask(unsigned value) {
    return {bool(value & 1), bool(value & 2), bool(value & 4), bool(value & 8)};
}

struct Instance {
    DecoderBatch batch;
    std::array<int, decoder_batch_max_lanes> jobs{};
    int next = 0;
    int steps = 0;
    unsigned graph_slots = 0;

    Instance(const Decoder& decoder, bool graphs) : batch(decoder, {graphs}) {}

    void admit(int slot, int length, int budget, std::optional<int> stop = std::nullopt) {
        constexpr std::array pool{13, 198, 279, 826, 760, 6511, 314, 9338, 369, 1570,
                                  315, 4950, 31784, 303, 806, 352, 952, 969};
        ++next;
        std::vector<int> prompt;
        for (int index = 0; index < length; ++index) {
            prompt.push_back(pool[(index + next * 3) % pool.size()]);
        }
        DecoderOptions options;
        options.max_new_tokens = budget;
        options.stop_token_id = stop;
        batch.admit(slot, std::move(prompt), options);
        jobs[slot] = next;
        std::cout << "admit " << slot << ' ' << next << ' ' << length << ' ' << budget << ' '
                  << batch.request_device_allocation_bytes() << '\n';
    }

    void cancel(int slot) {
        const auto before = batch.request_device_allocation_bytes();
        batch.cancel(slot);
        const auto after = batch.request_device_allocation_bytes();
        require(after < before, "Cancelled lane retained its request buffers");
        std::cout << "cancel " << slot << ' ' << jobs[slot] << ' ' << before
                  << ' ' << after << '\n';
        jobs[slot] = 0;
    }

    std::array<DecoderBatchStep, decoder_batch_max_lanes> step(unsigned bits) {
        DecoderBatchStepStats stats;
        const auto selected = mask(bits);
        const auto result = batch.advance_chunked(selected, &stats);
        std::cout << "step " << ++steps << ' ' << bits << ' ' << stats.graph_slots << ' '
                  << batch.request_device_allocation_bytes() << ' '
                  << stats.prefill_workspace_bytes << '\n';
        unsigned decode_slots = 0;
        for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
            const auto& lane = result[slot];
            if (!selected[slot] || !jobs[slot]) {
                require(!lane.decode && !lane.token && !lane.finish_reason, "Inactive lane wrote");
                continue;
            }
            if (lane.decode) {
                require(lane.token.has_value(), "Decode lane did not emit a token");
                decode_slots |= 1u << slot;
            }
            std::cout << "lane " << slot << ' ' << jobs[slot] << ' ' << stats.prefill_rows[slot]
                      << ' ' << lane.decode << ' ' << lane.token.value_or(-1) << ' '
                      << (lane.finish_reason ? static_cast<int>(*lane.finish_reason) : -1) << '\n';
            if (lane.finish_reason) {
                jobs[slot] = 0;
            }
        }
        require(stats.graph_slots == 0 || stats.graph_slots == decode_slots,
                "Graph mask differs from the slots that decoded");
        graph_slots = stats.graph_slots;
        std::cout << std::flush;
        return result;
    }

    void repeat(unsigned bits, int count) {
        for (int index = 0; index < count; ++index) {
            step(bits);
        }
    }

    void prefill(unsigned bits, int slot) {
        for (int index = 0; index < 100; ++index) {
            if (step(bits)[slot].token) {
                return;
            }
        }
        throw std::runtime_error("Boundary prefill did not complete");
    }

    void check_step(unsigned bits, unsigned graph, std::size_t increase = 0) {
        const auto before = batch.request_device_allocation_bytes();
        step(bits);
        require(graph_slots == graph, "Unexpected boundary replay mask");
        require(batch.request_device_allocation_bytes() == before + increase,
                "Unexpected boundary request bytes");
    }

    void drain() {
        for (int index = 0; index < 40; ++index) {
            unsigned bits = 0;
            for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
                if (jobs[slot]) {
                    bits |= 1u << slot;
                }
            }
            if (!bits) {
                require(batch.request_device_allocation_bytes() == 0, "Final live request buffers");
                return;
            }
            step(bits);
        }
        throw std::runtime_error("Drain did not finish all slots");
    }
};

void setup(Instance& run) {
    run.admit(0, 127, 15);
    run.admit(1, 8, 14);
    run.admit(2, 21, 19);
    run.admit(3, 6, 26);
    run.repeat(15, 2);
}

void scenario(const Decoder& decoder, bool graphs, bool hard) {
    Instance run(decoder, graphs);
    setup(run);
    if (hard) {
        try {
            run.repeat(15, 6);
            throw std::logic_error("Expected a post-launch failure");
        } catch (const std::runtime_error& error) {
            require(std::string_view(error.what()).find("Complete decoder batch step") !=
                    std::string_view::npos, "Unexpected post-launch failure");
        }
        try {
            run.step(15);
            throw std::logic_error("Expected an unavailable batch after a post-launch failure");
        } catch (const std::runtime_error& error) {
            require(std::string_view(error.what()) == "Decoder batch is unavailable",
                    "Batch accepted work after a post-launch failure");
        }
        std::cout << "failed-closed\n";
        return;
    }
    run.repeat(15, 5);
    run.repeat(10, 3);
    run.repeat(5, 2);
    run.cancel(1);
    run.repeat(13, 3);
    run.admit(1, 13, 0);
    const auto zero = run.step(11);
    require(!zero[1].token && zero[1].finish_reason == DecoderFinishReason::length,
            "Zero-budget reused slot emitted a token");
    run.admit(1, 7, 12);
    run.repeat(15, 5);
    run.repeat(11, 3);
    run.drain();
    require(run.steps > 20, "Insufficient replay steps");
}

void score_boundary(const Decoder& decoder, std::string_view role) {
    const bool graph = role != "eager";
    const bool oom = role == "score-oom";
    constexpr std::size_t score_bytes = 24 * sizeof(float);
    constexpr auto first_growth = 5248 * score_bytes;
    constexpr auto replacement_growth = 5376 * score_bytes;
    Instance run(decoder, graph);
    run.admit(0, 5117, 16);
    run.admit(2, 5117, 16);
    const auto before_prefill = run.batch.request_device_allocation_bytes();
    run.prefill(5, 0);
    require(run.batch.request_device_allocation_bytes() == before_prefill,
            "Prefill allocated long decode scores");
    for (int position = 5117; position <= 5122; ++position) {
        const bool replay = graph && position > 5117 && (position < 5120 || !oom);
        run.check_step(5, replay ? 5u : 0u, replay && position == 5120 ? 2 * first_growth : 0);
    }
    run.cancel(0);
    run.admit(1, 2, 5);
    run.check_step(6, 0);
    run.check_step(6, graph ? 6u : 0u, oom ? first_growth : 0);
    run.cancel(1);
    run.admit(0, 5121, 200);
    run.admit(3, 9, 6);
    run.prefill(1, 0);
    run.prefill(8, 3);
    run.check_step(13, 0);
    run.check_step(13, graph ? 13u : 0u, graph ? replacement_growth : 0);
    run.check_step(1, 0);
    run.admit(1, 5, 0);
    const auto zero = run.step(7);
    require(run.graph_slots == (graph ? 5u : 0u) && !zero[1].token &&
            zero[1].finish_reason == DecoderFinishReason::length,
            "Zero-budget lane changed the replay mask");
    while (run.jobs[2]) {
        run.step(5);
        require(run.graph_slots == (graph ? 5u : 0u), "Rebound width stopped replaying");
    }
    run.cancel(0);
    run.cancel(3);
    run.drain();
    run.check_step(15, 0);
}

void verify_wide(const Decoder& decoder) {
    constexpr int width = decoder_batch_max_lanes;
    constexpr int budget = decoder_batch_max_step_tokens;
    std::array<std::array<int, 3>, width> target_ids{};
    Mask ready;
    ready.fill(true);
    for (const int depth : {0, 1}) {
        DecoderBatch batch(decoder, {false, width, 0, budget});
        DecoderOptions options;
        options.max_new_tokens = 4;
        options.mtp_steps = depth;
        for (int slot = 0; slot < width; ++slot) {
            batch.admit(slot, {13, 198, 279, 826 + slot}, options);
        }
        DecoderBatchStepStats prefill;
        const auto seeds = batch.advance_chunked(ready, &prefill);
        require(prefill.target_rows == width * 4 && prefill.verification_lanes == 0,
                "Unexpected target charge while seeding wide verification");
        for (int slot = 0; slot < width; ++slot) {
            require(prefill.prefill_rows[slot] == 4 && seeds[slot].token && !seeds[slot].decode,
                    "Wide verification lane was not ready after prefill");
            if (depth == 0) {
                target_ids[slot][0] = *seeds[slot].token;
            } else {
                require(seeds[slot].token == target_ids[slot][0], "K1 changed a prefill seed");
            }
        }
        const int rounds = depth == 0 ? 2 : 1;
        for (int round = 0; round < rounds; ++round) {
            DecoderBatchStepStats stats;
            const auto decoded = batch.advance_chunked(ready, &stats);
            require(stats.target_rows == width * (depth + 1) && stats.graph_slots == 0,
                    "Unexpected target charge while decoding wide verification");
            require(stats.verification_lanes == width * depth &&
                    stats.verification_rows == width * depth * (depth + 1),
                    "Ready wide cohort did not verify together");
            for (int slot = 0; slot < width; ++slot) {
                const auto& step = decoded[slot];
                require(step.decode && step.token && !step.finish_reason && !step.mtp_boundary,
                        "Ready wide lane did not decode");
                if (depth == 0) {
                    target_ids[slot][round + 1] = *step.token;
                    continue;
                }
                require(step.mtp_drafted_tokens == 1 && step.mtp_verified_rows == 2 &&
                        step.extra_token_count >= 0 && step.extra_token_count <= 1,
                        "Wide K1 lane did not verify its draft and target rows");
                const bool bonus_matches = step.extra_token_count == 0 ||
                    step.extra_tokens[0] == target_ids[slot][2];
                require(step.token == target_ids[slot][1] && bonus_matches,
                        "Wide K1 lane changed its target-only tokens");
            }
            std::cout << "K" << depth << " ready " << width << " verified "
                      << stats.verification_lanes << " rows " << stats.verification_rows << '\n';
        }
        for (int slot = 0; slot < width; ++slot) {
            batch.cancel(slot);
        }
        require(batch.request_device_allocation_bytes() == 0, "Wide requests retained buffers");
    }
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 4 || (argc == 5 && std::string_view(argv[4]) == "boundary"),
                "Expected CHECKPOINT ROLE FP8_PREFILL [boundary]");
        const bool boundary = argc == 5;
        const std::string_view role = argv[2];
        const std::string_view policy = argv[3];
        const bool wide = !boundary && role == "verify-wide" && policy == "exact";
        require(wide || role == "eager" || role == "graph" ||
                role == (boundary ? "score-oom" : "hard"),
                "Unexpected graph test role");
        require(policy == "exact" || policy == "tensor-core", "Expected exact or tensor-core");
        const Decoder decoder(argv[1], 0, wide, policy == "exact" ?
            Fp8PrefillPolicy::exact : Fp8PrefillPolicy::tensor_core);
        if (wide) {
            verify_wide(decoder);
        } else if (boundary) {
            score_boundary(decoder, role);
        } else {
            scenario(decoder, role != "eager", role == "hard");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
