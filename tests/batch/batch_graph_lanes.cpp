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
        std::cout << std::flush;
        return result;
    }

    void repeat(unsigned bits, int count) {
        for (int index = 0; index < count; ++index) {
            step(bits);
        }
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

}

int main(int argc, char** argv) {
    try {
        require(argc == 4, "Expected CHECKPOINT ROLE FP8_PREFILL");
        const std::string_view role = argv[2];
        const std::string_view policy = argv[3];
        require(role == "eager" || role == "graph" || role == "hard",
                "Expected eager, graph or hard");
        require(policy == "exact" || policy == "tensor-core", "Expected exact or tensor-core");
        const Decoder decoder(argv[1], 0, false, policy == "exact" ?
            Fp8PrefillPolicy::exact : Fp8PrefillPolicy::tensor_core);
        scenario(decoder, role != "eager", role == "hard");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
