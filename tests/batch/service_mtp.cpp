#include "sglang/decoder_batch.h"
#include "sglang/server/batch_service.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

using Finish = sglang::DecoderFinishReason;
using Service = sglang::server::BatchService;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

enum class Interrupt { none, callback, poll };

struct Scenario {
    const char* name;
    Interrupt interrupt;
    std::vector<int> expected_ids;
    std::size_t expected_accepted;
    Finish finish;
};

bool check(const Scenario& scenario) {
    const bool seed_only = scenario.expected_ids.size() == 1;
    sglang::DecoderOptions options;
    options.max_new_tokens = seed_only ? 1 : 6;
    std::vector<int> emitted;
    sglang::DecoderCallbacks callbacks;
    callbacks.on_token = [&](int token) {
        emitted.push_back(token);
        return scenario.interrupt != Interrupt::callback ||
               emitted.size() < scenario.expected_ids.size();
    };
    callbacks.is_cancelled = [&] {
        return scenario.interrupt == Interrupt::poll &&
               emitted.size() >= scenario.expected_ids.size();
    };
    std::ostringstream trace;
    auto* original = std::cerr.rdbuf(trace.rdbuf());
    sglang::DecoderResult result;
    try {
        const sglang::Decoder decoder("unused");
        Service service(decoder, true, false, 4);
        result = service.generate(3, {7}, options, callbacks);
    } catch (...) {
        std::cerr.rdbuf(original);
        throw;
    }
    std::cerr.rdbuf(original);

    require(emitted == scenario.expected_ids && result.output_token_ids == scenario.expected_ids,
            "Unexpected callback or result token IDs");
    require(result.finish_reason == scenario.finish, "Unexpected finish reason");
    require(result.mtp_drafted_tokens == (seed_only ? 0 : 4) &&
                result.mtp_rounds == (seed_only ? 0 : 1),
            "Unexpected proposal or round counters");
    const std::string observed = trace.str();
    const std::string round = "batch mtp slot=3 drafted=4 accepted=4 verified=5 "
                              "replayed=0 boundary=0\n";
    if (seed_only) {
        require(observed.find("batch mtp slot=") == std::string::npos &&
                    observed.find("batch verify lanes=0 rows=0 ") != std::string::npos,
                "Seed-only request performed verification");
    } else {
        require(observed.find("batch verify lanes=1 rows=5 ") != std::string::npos &&
                    observed.find(round) != std::string::npos &&
                    observed.find("batch mtp slot=") == observed.rfind("batch mtp slot="),
                "Missing committed-round scheduler trace");
    }
    const bool exact = result.mtp_accepted_tokens == scenario.expected_accepted;
    std::cout << scenario.name << " emitted=" << emitted.size()
              << " accepted=" << result.mtp_accepted_tokens
              << " expected=" << scenario.expected_accepted
              << ' ' << (exact ? "passed" : "failed") << '\n';
    return exact;
}

}

namespace sglang {

struct Decoder::Model {};
Decoder::Decoder(const std::filesystem::path&, int) {}
Decoder::~Decoder() = default;

struct DecoderBatch::State {
    int budget = 0;
    bool seeded = false;
};

DecoderBatch::DecoderBatch(const Decoder&, DecoderBatchOptions)
    : state_(std::make_unique<State>()) {}
DecoderBatch::~DecoderBatch() = default;

void DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    require(slot == 3 && prompt == std::vector<int>{7} && options.mtp_steps == 4 &&
                (options.max_new_tokens == 1 || options.max_new_tokens == 6),
            "Unexpected admission");
    require(state_->budget == 0, "Overlapping admission");
    state_->budget = options.max_new_tokens;
    state_->seeded = false;
}

void DecoderBatch::cancel(int slot) {
    require(slot == 3 && std::exchange(state_->budget, 0) != 0, "Cancelled a retired slot");
}

std::array<DecoderBatchStep, decoder_batch_max_lanes>
DecoderBatch::advance_chunked(const std::array<bool, decoder_batch_max_lanes>& selected,
                              DecoderBatchStepStats* stats) {
    constexpr std::array<bool, decoder_batch_max_lanes> expected{false, false, false, true};
    require(selected == expected && state_->budget != 0 && stats,
            "Unexpected scheduler selection");
    std::array<DecoderBatchStep, decoder_batch_max_lanes> steps{};
    auto& step = steps[3];
    if (!std::exchange(state_->seeded, true)) {
        step.token = 100;
        stats->prefill_rows[3] = 1;
        if (state_->budget == 1) {
            step.finish_reason = Finish::length;
            state_->budget = 0;
        }
        return steps;
    }
    state_->budget = 0;
    step.token = 101;
    step.finish_reason = Finish::length;
    step.decode = true;
    step.extra_token = 102;
    step.extra_tokens = {102, 103, 104, 105};
    step.extra_token_count = 4;
    step.mtp_accepted_tokens = 4;
    step.mtp_drafted_tokens = 4;
    stats->verification_lanes = 1;
    stats->verification_rows = 5;
    return steps;
}

std::size_t DecoderBatch::request_device_allocation_bytes() const { return 0; }
bool DecoderBatch::available() const { return true; }

}

int main() {
    try {
        const bool seed = check({"seed only", Interrupt::none, {100}, 0, Finish::length});
        const bool callback = check({"callback false", Interrupt::callback, {100, 101}, 1,
                                     Finish::cancelled});
        const bool poll = check({"mid-round cancel", Interrupt::poll, {100, 101, 102}, 2,
                                 Finish::cancelled});
        const bool full = check({"full round", Interrupt::none, {100, 101, 102, 103, 104, 105}, 4,
                                 Finish::length});
        return seed && callback && poll && full ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
