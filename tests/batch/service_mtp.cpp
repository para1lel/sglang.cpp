#include "sglang/decoder_batch.h"
#include "sglang/runtime/server/single_device_batch_executor.h"
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
    bool scored = false;
};

bool check(const Scenario& scenario) {
    const bool seed_only = scenario.expected_ids.size() == 1;
    sglang::DecoderOptions options;
    options.max_new_tokens = seed_only ? 1 : 6;
    if (scenario.scored) options.generated_top_logprobs = 1;
    std::vector<int> emitted;
    std::vector<std::shared_ptr<const sglang::DecoderTokenLogprobs>> scores;
    bool admitted = false;
    sglang::DecoderCallbacks callbacks;
    callbacks.on_admitted = [&] {
        require(!admitted, "Duplicate admission notification");
        admitted = true;
        return true;
    };
    const auto emit = [&](int token) {
        require(admitted, "Token callback before admission callback");
        emitted.push_back(token);
        return scenario.interrupt != Interrupt::callback ||
               emitted.size() < scenario.expected_ids.size();
    };
    callbacks.on_token = [&](int token) {
        require(!scenario.scored, "Scored token used the unscored callback");
        return emit(token);
    };
    callbacks.on_scored_token = [&](int token, const auto& score) {
        require(scenario.scored && score && score->selected.token_id == token &&
                    score->selected.value == -0.25f * token && score->alternatives.size() == 1 &&
                    score->alternatives[0].token_id == token &&
                    score->alternatives[0].value == score->selected.value,
                "Missing or misordered score at token callback");
        scores.push_back(score);
        return emit(token);
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
        auto executor = std::make_unique<sglang::server::SingleDeviceBatchExecutor>(decoder,
            sglang::server::BatchExecutor::Options{
                false, 4, 0, sglang::decoder_default_max_num_batched_tokens});
        Service service(std::move(executor), true, 4);
        result = service.generate(3, {7}, options, callbacks);
    } catch (...) {
        std::cerr.rdbuf(original);
        throw;
    }
    std::cerr.rdbuf(original);

    require(emitted == scenario.expected_ids && result.output_token_ids == scenario.expected_ids,
            "Unexpected callback or result token IDs");
    require(scores == result.generated_logprobs &&
                scores.size() == (scenario.scored ? emitted.size() : 0),
            "Result score rows differ from committed callback score rows");
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

void check_media_callbacks() {
    const sglang::Decoder decoder("unused");
    auto executor = std::make_unique<sglang::server::SingleDeviceBatchExecutor>(decoder,
        sglang::server::BatchExecutor::Options{
            false, 4, 0, sglang::decoder_default_max_num_batched_tokens});
    Service service(std::move(executor), true, 4);
    sglang::DecoderOptions options;
    options.max_new_tokens = 1;
    const std::vector<sglang::DecoderMedia> media(1);
    constexpr std::size_t expanded_rows = 9;
    sglang::DecoderCallbacks single;
    std::optional<std::size_t> admitted;
    single.on_prompt_admitted = [&](std::size_t cached) {
        require(!admitted, "Duplicate media admission notification");
        admitted = cached;
        return true;
    };
    const auto result = service.generate(3, {7}, options, single, media, expanded_rows);
    require(admitted == 0 && result.cached_prompt_tokens == 0 &&
                result.output_token_ids == std::vector{100},
            "Uncached media admission or scalar output was lost");

    bool emitted = false;
    sglang::server::BatchCallbacks indexed;
    indexed.on_output = [&](std::size_t index, const sglang::DecoderResult& output, bool final) {
        require(!emitted && index == 0 && final && output.cached_prompt_tokens == 0 &&
                    output.output_token_ids == std::vector{100}, "Unexpected indexed media output");
        emitted = true;
        return true;
    };
    sglang::server::BatchPrompt prompt{{7}, options};
    prompt.media = media;
    prompt.expanded_tokens = expanded_rows;
    const auto results = service.generate_batch(3, {prompt}, indexed);
    require(emitted && results && results->size() == 1 &&
                results->front().cached_prompt_tokens == 0 &&
                results->front().output_token_ids == std::vector{100},
            "Uncached media admission or indexed output was lost");
    std::cout << "Media admission and indexed output passed with LMCache disabled\n";
}

}

namespace sglang {

struct Decoder::Model {};
#define SGLANG_TEST_TP_HEADER "sglang/runtime/execution/decoder_tp.h"
#if __has_include(SGLANG_TEST_TP_HEADER)
struct Decoder::TpRuntime {};
const decoder::TpRuntimeInfo* Decoder::tensor_parallel_info() const { return nullptr; }
#endif
Decoder::Decoder(const std::filesystem::path&, int) {}
Decoder::~Decoder() = default;

struct DecoderBatch::State {
    int budget = 0;
    bool seeded = false;
    bool scored = false;
};

DecoderBatch::DecoderBatch(const Decoder&, DecoderBatchOptions options)
    : state_(std::make_unique<State>()) {
    require(!options.cuda_graphs && options.max_lanes == 4 &&
                options.prefix_cache_entries == 0 && options.max_num_batched_tokens == 32768,
            "Unexpected service scheduler configuration");
}
DecoderBatch::~DecoderBatch() = default;

int DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    require(slot == 3 && prompt == std::vector<int>{7} && options.mtp_steps == 4 &&
                (options.max_new_tokens == 1 || options.max_new_tokens == 6),
            "Unexpected admission");
    require(state_->budget == 0, "Overlapping admission");
    state_->budget = options.max_new_tokens;
    state_->seeded = false;
    state_->scored = options.generated_top_logprobs == 1;
    return 0;
}

int DecoderBatch::admit_multimodal(int slot, std::vector<int> prompt,
                                  const DecoderOptions& options,
                                  const std::vector<DecoderMedia>& media) {
    require(media.size() == 1, "Unexpected media count");
    return admit(slot, std::move(prompt), options);
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
    const auto score = [](int token) {
        const DecoderLogprob value{token, -0.25f * token};
        return std::make_shared<const DecoderTokenLogprobs>(DecoderTokenLogprobs{value, {value}});
    };
    if (!std::exchange(state_->seeded, true)) {
        step.token = 100;
        if (state_->scored) step.generated_logprobs = score(*step.token);
        stats->prefill_rows[3] = 1;
        stats->target_rows = 1;
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
    if (state_->scored) {
        step.generated_logprobs = score(*step.token);
        for (int index = 0; index < step.extra_token_count; ++index) {
            step.extra_generated_logprobs[index] = score(step.extra_tokens[index]);
        }
    }
    step.mtp_accepted_tokens = 4;
    step.mtp_drafted_tokens = 4;
    step.mtp_verified_rows = 5;
    stats->verification_lanes = 1;
    stats->verification_rows = 5;
    stats->target_rows = 5;
    return steps;
}

std::size_t DecoderBatch::request_device_allocation_bytes() const { return 0; }
DecoderPrefixCacheStats DecoderBatch::prefix_cache_stats() const { return {}; }
DecoderBatchPrefixUpdate DecoderBatch::poll_prefix(int) { std::abort(); }
void DecoderBatch::reset_prefix_cache() {
    throw std::logic_error("Unexpected prefix cache reset");
}
void DecoderBatch::release_retired_allocations() {}
bool DecoderBatch::available() const { return true; }

}

int main() {
    try {
        require(sglang::DecoderOptions{}.max_num_batched_tokens == 32768,
                "Unexpected standalone decoder budget default");
        require(sglang::DecoderBatchOptions{}.max_num_batched_tokens == 32768,
                "Unexpected batch decoder budget default");
        const bool seed = check({"seed only", Interrupt::none, {100}, 0, Finish::length});
        const bool callback = check({"callback false", Interrupt::callback, {100, 101}, 1,
                                     Finish::cancelled});
        const bool poll = check({"mid-round cancel", Interrupt::poll, {100, 101, 102}, 2,
                                 Finish::cancelled});
        const bool full = check({"full round", Interrupt::none, {100, 101, 102, 103, 104, 105}, 4,
                                 Finish::length});
        const bool scored_full = check({"scored full round", Interrupt::none,
            {100, 101, 102, 103, 104, 105}, 4, Finish::length, true});
        const bool scored_callback = check({"scored callback false", Interrupt::callback,
            {100, 101, 102}, 2, Finish::cancelled, true});
        const bool scored_poll = check({"scored mid-round cancel", Interrupt::poll,
            {100, 101, 102, 103}, 3, Finish::cancelled, true});
        check_media_callbacks();
        return seed && callback && poll && full && scored_full && scored_callback && scored_poll
            ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
