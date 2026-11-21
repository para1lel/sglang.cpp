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
    Mask selected{};
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        selected[slot] = (value & (1u << slot)) != 0;
    }
    return selected;
}

struct Instance {
    DecoderBatch batch;
    std::array<int, decoder_batch_max_lanes> jobs{};
    int next = 0;
    int steps = 0;
    unsigned graph_slots = 0;

    Instance(const Decoder& decoder, bool graphs, int width = 4) :
        batch(decoder, {graphs, width}) {}

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

void singleton(const Decoder& decoder, std::string_view role) {
    constexpr int far_slot = decoder_batch_max_lanes - 1;
    constexpr unsigned far_mask = 1u << far_slot;
    constexpr unsigned pair = far_mask | 1u;
    const bool graphs = role != "eager";
    const bool rejected = role == "launch";
    const auto singleton_mask = graphs && !rejected ? far_mask : 0u;
    Instance run(decoder, graphs, decoder_batch_max_lanes);

    run.admit(far_slot, 1534, 6);
    run.prefill(far_mask, far_slot);
    run.step(far_mask);
    require(run.graph_slots == 0, "First singleton decode did not warm eagerly");
    run.step(far_mask);
    require(run.graph_slots == (graphs ? far_mask : 0u),
            "Expected singleton replay at 1536 visible tokens");
    run.admit(0, 1, 6);
    const auto mixed = run.step(pair);
    require(run.graph_slots == 0 && mixed[far_slot].decode &&
            mixed[0].token && !mixed[0].decode, "Mixed prefill and singleton decode used a graph");
    run.admit(2, 1, 0);
    const auto zero = run.step(far_mask | 4u);
    require(run.graph_slots == singleton_mask && zero[far_slot].decode &&
            !zero[2].token && zero[2].finish_reason == DecoderFinishReason::length,
            "Unexpected singleton mask or zero-budget retirement");
    run.cancel(far_slot);

    run.admit(far_slot, 2558, 6);
    run.prefill(far_mask, far_slot);
    for (int index = 0; index < 3; ++index) {
        run.step(far_mask);
        require(run.graph_slots == singleton_mask, "Unexpected replay mask on recycled slot");
    }
    run.step(pair);
    require(run.graph_slots == 0, "First pair decode did not warm eagerly");
    run.step(pair);
    require(run.graph_slots == (graphs ? pair : 0u),
            "Singleton graph state prevented pair replay");
    require(!run.jobs[far_slot], "Recycled slot did not finish");
    run.cancel(0);

    run.admit(far_slot, 5119, 3);
    run.prefill(far_mask, far_slot);
    run.step(far_mask);
    require(run.graph_slots == singleton_mask, "Last inline score position did not replay");
    run.step(far_mask);
    require(run.graph_slots == 0, "Singleton beyond inline score capacity used a graph");
    run.drain();
    run.check_step(pair, 0);
}

void expect_verification_error(DecoderBatch& batch, unsigned selected, std::string_view message) {
    bool failed = false;
    try {
        batch.advance_chunked(mask(selected));
    } catch (const std::exception& error) {
        require(std::string_view(error.what()).find(message) != std::string_view::npos,
                error.what());
        failed = true;
    }
    require(failed, "K1 operation did not propagate its failure");
}

void expect_unavailable_admission(DecoderBatch& batch) {
    try {
        batch.admit(0, {13}, DecoderOptions{});
    } catch (const std::exception& error) {
        require(std::string_view(error.what()) == "Decoder batch is unavailable", error.what());
        return;
    }
    require(false, "Failed K1 batch accepted another request");
}

void verify_failure(DecoderBatch& batch, unsigned selected, std::string_view message,
                    bool drained) {
    expect_verification_error(batch, selected, message);
    require(!batch.available(), "Failed K1 operation left the batch available");
    if (drained) {
        require(batch.request_device_allocation_bytes() == 0,
                "Drained K1 failure retained request buffers");
    }
    expect_verification_error(batch, selected, "Decoder batch is unavailable");
    expect_unavailable_admission(batch);
    std::cout << "verify failed-closed\n";
}

void verify_recovery(const Decoder& decoder, DecoderBatch& batch, int slot,
                     std::string_view message) {
    expect_verification_error(batch, 1u << slot, message);
    require(batch.available() && batch.request_device_allocation_bytes() == 0,
            "Drained K1 operation did not retire its requests");
    DecoderBatch eager(decoder, {false, 2, 0, 4});
    DecoderOptions options;
    options.mtp_steps = 1;
    options.max_new_tokens = 8;
    batch.admit(slot, {13, 198, 279, 826}, options);
    eager.admit(1, {13, 198, 279, 826}, options);
    for (int step = 0; step < 2; ++step) {
        DecoderBatchStepStats stats;
        const auto actual = batch.advance_chunked(mask(1u << slot), &stats)[slot];
        const auto expected = eager.advance_chunked(mask(2))[1];
        require(actual.token && actual.token == expected.token &&
                actual.extra_token_count == expected.extra_token_count &&
                actual.extra_tokens == expected.extra_tokens &&
                actual.mtp_verified_rows == expected.mtp_verified_rows &&
                stats.verification_graph_slots == (step ? 1u << slot : 0),
                "Replacement after K1 failure differed from eager");
    }
    batch.cancel(slot);
    require(batch.request_device_allocation_bytes() == 0, "K1 replacement retained buffers");
    std::cout << "verify recovered\n";
}

void admit_verifier(DecoderBatch& batch, int slot, int depth, int budget, int length, int offset) {
    constexpr std::array pool{13, 198, 279, 826, 760, 6511, 314, 9338, 369, 1570,
                              315, 4950, 31784, 303, 806, 352, 952, 969};
    std::vector<int> prompt;
    for (int index = 0; index < length; ++index) {
        prompt.push_back(pool[(index + offset) % pool.size()]);
    }
    DecoderOptions options;
    options.mtp_steps = depth;
    options.max_new_tokens = budget;
    options.incremental_kv = true;
    batch.admit(slot, std::move(prompt), options);
    std::cout << "verify admit " << slot << ' ' << depth << ' ' << budget << ' '
              << length << ' ' << offset << '\n';
}

void verify_exclusions(const Decoder& decoder, bool graphs) {
    DecoderBatch limited(decoder, {graphs, 2, 0, 1});
    DecoderOptions options;
    options.mtp_steps = 1;
    options.max_new_tokens = 4;
    limited.admit(1, {13}, options);
    require(limited.advance_chunked(mask(2))[1].token.has_value(), "No budget-case seed");
    DecoderBatchStepStats stats;
    const auto downgraded = limited.advance_chunked(mask(2), &stats)[1];
    require(downgraded.token && downgraded.mtp_verified_rows == 1 &&
            !downgraded.mtp_drafted_tokens && stats.target_rows == 1 &&
            stats.graph_slots == 0 && stats.verification_graph_slots == 0,
            "One-row budget did not downgrade K1");
    std::cout << "verify downgrade " << *downgraded.token << '\n';
    limited.cancel(1);

    DecoderBatch wider(decoder, {graphs, 2, 0, 8});
    for (int depth = 2; depth <= 4; ++depth) {
        admit_verifier(wider, 1, depth, 16, 4, depth);
        require(wider.advance_chunked(mask(2))[1].token.has_value(), "No higher-depth seed");
        for (int round = 0; round < 2; ++round) {
            const auto result = wider.advance_chunked(mask(2), &stats)[1];
            require(result.token && result.mtp_drafted_tokens == depth &&
                    result.mtp_verified_rows == depth + 1 &&
                    stats.graph_slots == 0 && stats.verification_graph_slots == 0,
                    "Higher-depth verification changed shape or used the K1 graph");
            std::cout << "verify depth " << depth << ' ' << *result.token;
            for (int extra = 0; extra < result.extra_token_count; ++extra) {
                std::cout << ' ' << result.extra_tokens[extra];
            }
            std::cout << '\n';
        }
        wider.cancel(1);
    }
}

void verify_graph(const Decoder& decoder, std::string_view role) {
    const bool graphs = role != "eager";
    constexpr int first = 3;
    constexpr int second = 29;
    constexpr unsigned first_mask = 1u << first;
    constexpr unsigned second_mask = 1u << second;
    constexpr unsigned both = first_mask | second_mask;
    DecoderBatch batch(decoder, {graphs, decoder_batch_max_lanes, 0, 4});
    int steps = 0;
    const auto step = [&](unsigned selected, int lanes, int rows, unsigned k0 = 0,
                          unsigned k1 = 0) {
        DecoderBatchStepStats stats;
        const auto result = batch.advance_chunked(mask(selected), &stats);
        require(stats.verification_lanes == lanes && stats.verification_rows == rows,
                "Unexpected K1 verification shape");
        require(stats.graph_slots == k0, "K1 step changed the K0 graph mask");
        require(stats.verification_graph_slots == k1, "Unexpected K1 target graph mask");
        std::cout << "verify step " << ++steps << ' ' << selected << ' '
                  << stats.target_rows << ' ' << lanes << ' ' << rows << '\n'
                  << "verify graphs " << steps << ' ' << k0 << ' ' << k1 << '\n';
        for (const int slot : {first, second}) {
            const auto& lane = result[slot];
            std::cout << "verify lane " << slot << ' ' << lane.decode << ' '
                      << lane.token.value_or(-1) << ' ' << lane.extra_token_count << ' '
                      << (lane.extra_token_count ? lane.extra_tokens[0] : -1) << ' '
                      << lane.mtp_drafted_tokens << ' ' << lane.mtp_accepted_tokens << ' '
                      << lane.mtp_verified_rows << ' ' << lane.mtp_replayed_rows << ' '
                      << lane.mtp_boundary << ' '
                      << (lane.finish_reason ? static_cast<int>(*lane.finish_reason) : -1)
                      << '\n';
        }
        return result;
    };
    const auto prefill = [&](int slot, int length) {
        for (int done = 0; done < length; done += 4) {
            require(step(1u << slot, 0, 0)[slot].token.has_value() == (done + 4 >= length),
                    "K1 prefill seed arrived at the wrong position");
        }
    };
    const auto cancel = [&](int slot) {
        const auto before = batch.request_device_allocation_bytes();
        batch.cancel(slot);
        require(batch.request_device_allocation_bytes() < before,
                "Cancelled K1 lane retained request buffers");
        std::cout << "verify cancel " << slot << '\n';
    };
    admit_verifier(batch, second, 1, 24, 127, 1);
    prefill(second, 127);
    step(second_mask, 1, 2);
    if (role == "hard") {
        verify_failure(batch, second_mask, "illegal memory access", false);
        return;
    }
    if (role == "hard-drain") {
        verify_failure(batch, second_mask, "Drain failed decoder batch step", false);
        return;
    }
    if (role == "cleanup") {
        verify_failure(batch, second_mask, "Destroy decode capture graph", false);
        return;
    }
    const bool launch_fault = role == "launch" || role == "submitted" || role == "exec-cleanup";
    const bool replays = role == "graph" || role == "upload" || launch_fault;
    step(second_mask, 1, 2, 0, replays ? second_mask : 0);
    if (role == "upload") {
        verify_recovery(decoder, batch, second, "Upload K1 verifier graph bindings");
        return;
    }
    admit_verifier(batch, first, 0, 16, 4, 2);
    prefill(first, 4);
    if (launch_fault) {
        verify_failure(batch, both, "Launch batch graph", role != "exec-cleanup");
        return;
    }
    step(both, 1, 2, 0, role == "graph" ? second_mask : 0);
    cancel(second);
    step(first_mask, 0, 0);
    step(first_mask, 0, 0, graphs ? first_mask : 0);
    cancel(first);

    admit_verifier(batch, first, 1, 16, 4, 3);
    admit_verifier(batch, second, 1, 16, 4, 4);
    prefill(first, 4);
    prefill(second, 4);
    step(both, 2, 4);
    cancel(second);
    step(first_mask, 1, 2, 0, role == "graph" ? first_mask : 0);
    cancel(first);

    admit_verifier(batch, second, 1, 2, 4, 5);
    prefill(second, 4);
    const auto boundary = step(second_mask, 0, 0)[second];
    require(boundary.mtp_boundary && boundary.finish_reason.has_value(),
            "Single output remaining did not take the K1 boundary path");
    require(batch.request_device_allocation_bytes() == 0, "K1 requests retained buffers");

    verify_exclusions(decoder, graphs);
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
        require(argc == 4 || argc == 5,
                "Expected CHECKPOINT ROLE FP8_PREFILL [boundary|singleton|verify]");
        const std::string_view test = argc == 5 ? argv[4] : "";
        require(test.empty() || test == "boundary" || test == "singleton" || test == "verify",
                "Unexpected graph test scenario");
        const std::string_view role = argv[2];
        const std::string_view policy = argv[3];
        const bool wide = test.empty() && role == "verify-wide" && policy == "exact";
        const bool verify_fault = test == "verify" &&
            (role == "prepare" || role == "begin" || role == "instantiate" || role == "upload" ||
             role == "launch" || role == "submitted" || role == "cleanup" ||
             role == "exec-cleanup" || role == "hard" || role == "hard-drain");
        require(wide || verify_fault || role == "eager" || role == "graph" ||
                role == (test == "boundary" ? "score-oom" :
                         test == "singleton" ? "launch" : "hard"),
                "Unexpected graph test role");
        require(policy == "exact" || policy == "tensor-core", "Expected exact or tensor-core");
        const Decoder decoder(argv[1], 0, wide || test == "verify", policy == "exact" ?
            Fp8PrefillPolicy::exact : Fp8PrefillPolicy::tensor_core);
        if (wide) {
            verify_wide(decoder);
        } else if (test == "boundary") {
            score_boundary(decoder, role);
        } else if (test == "singleton") {
            singleton(decoder, role);
        } else if (test == "verify") {
            verify_graph(decoder, role);
        } else {
            scenario(decoder, role != "eager", role == "hard");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
