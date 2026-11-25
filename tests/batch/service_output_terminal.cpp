#include "sglang/runtime/server/single_device_batch_executor.h"
#include "sglang/server/batch_service.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>

namespace {

using namespace sglang;
using namespace sglang::server;
constexpr int width = decoder_batch_max_lanes;
constexpr int held_token = 3;
constexpr int queued_token = held_token + 1;
using Selection = std::array<bool, width>;
using Steps = std::array<DecoderBatchStep, width>;

enum class Outcome { abort, stop, length };

void require(bool condition, const char* description) {
    if (!condition) {
        std::cerr << description << '\n';
        std::abort();
    }
}

struct Scenario {
    std::mutex mutex;
    std::condition_variable changed;
    int first_callbacks = 0;
    int target_produced = 0;
    int target_cancels = 0;
    bool target_blocked = false;
    bool publication_observed = false;
    bool release = false;

    template <class Predicate>
    void await(std::unique_lock<std::mutex>& lock, Predicate predicate, const char* description) {
        require(changed.wait_for(lock, std::chrono::seconds(5), predicate), description);
    }

    bool deliver(int slot, int ordinal, int token) {
        require(token == 1000 * slot + ordinal, "Callback changed token identity or order");
        std::unique_lock lock(mutex);
        if (ordinal == 1) {
            ++first_callbacks;
            changed.notify_all();
            await(lock, [&] { return first_callbacks == width; }, "Missing first callback");
        }
        if (slot == 0 && ordinal == held_token) {
            target_blocked = true;
            changed.notify_all();
            await(lock, [&] { return release; }, "Third callback was never released");
        }
        return true;
    }
};

Scenario* scenario;

void check_scores(const DecoderResult& result) {
    require(result.generated_logprobs.size() == result.output_token_ids.size(),
            "Score count differs from retained token count");
    for (std::size_t index = 0; index < result.output_token_ids.size(); ++index) {
        const int token = result.output_token_ids[index];
        require(static_cast<bool>(result.generated_logprobs[index]), "Retained score was empty");
        const auto& score = *result.generated_logprobs[index];
        require(score.selected.token_id == token && score.selected.value == -0.25f * token &&
                    score.alternatives.size() == 1 &&
                    score.alternatives.front().token_id == token,
                "A queued score no longer belongs to its retained token");
    }
}

void check_terminal(Outcome outcome, bool scored) {
    Scenario events;
    scenario = &events;
    const Decoder decoder({}, 0, false);
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{true, width, 0, decoder_batch_max_step_tokens}), false);
    BatchRequestId id("terminal-race");
    std::array<DecoderResult, width> results;
    std::array<int, width> received{};
    std::array<std::thread, width> streams;
    for (int slot = 0; slot < width; ++slot) {
        streams[slot] = std::thread([&, slot] {
            DecoderOptions options;
            options.max_new_tokens = slot == 0 && outcome == Outcome::length ? queued_token : 16;
            if (slot == 0 && outcome == Outcome::stop) {
                options.stop_token_id = queued_token;
            }
            if (scored && slot == 0) {
                options.generated_top_logprobs = 1;
                BatchPrompt prompt;
                prompt.prompt.assign(decoder_batch_max_step_tokens, slot);
                prompt.options = options;
                prompt.request_id = &id;
                const BatchCallbacks callbacks{{}, {}, {},
                    [&](std::size_t index, const DecoderResult& result, bool terminal) {
                        require(index == 0, "Scored scalar got an unexpected batch index");
                        check_scores(result);
                        const auto delivered = static_cast<std::size_t>(received[slot]);
                        if (result.output_token_ids.size() > delivered) {
                            events.deliver(slot, ++received[slot], result.output_token_ids.back());
                        }
                        if (terminal) {
                            results[slot] = result;
                        }
                        return true;
                    }, true};
                require(service.generate_batch(slot, {prompt}, callbacks).has_value(),
                        "Scored scalar batch disconnected");
                return;
            }
            DecoderCallbacks callbacks{[&, slot](int token) {
                return events.deliver(slot, ++received[slot], token);
            }, {}, {}, {}, true};
            results[slot] = service.generate(
                slot, std::vector<int>(decoder_batch_max_step_tokens, slot),
                options, callbacks, {}, 0, slot == 0 ? &id : nullptr);
        });
    }
    {
        std::unique_lock lock(events.mutex);
        events.await(lock, [&] { return events.target_blocked && events.publication_observed; },
                     "The fourth token was not published during the third callback");
        require(events.target_produced == queued_token, "Exceeded one queued token");
        const bool abort_wins = outcome == Outcome::abort;
        require(id.abort(true) == abort_wins, "Wrong terminal transition won");
        require(!id.abort(true), "Repeated abort won a second terminal transition");
        service.notify_abort();
        if (abort_wins) {
            events.await(lock, [&] { return events.target_cancels == 1; },
                         "Abort did not retire the decoder lane");
        }
        events.release = true;
        events.changed.notify_all();
    }
    for (auto& stream : streams) {
        stream.join();
    }
    const auto& target = results[0];
    const bool aborted = outcome == Outcome::abort;
    const auto reason = aborted ? DecoderFinishReason::aborted :
        outcome == Outcome::stop ? DecoderFinishReason::stop : DecoderFinishReason::length;
    const auto tokens = aborted ? std::vector{1, 2, 3} : std::vector{1, 2, 3, 4};
    require(id.finished() && id.aborted() == aborted, "Wrong final request-ID ownership");
    require(target.finish_reason == reason && target.output_token_ids == tokens &&
                received[0] == static_cast<int>(tokens.size()),
            "Wrong terminal result, queued ID or delivery count");
    require(events.target_cancels == int(aborted), "Unexpected decoder retirement count");
    if (scored) {
        check_scores(target);
    }
    for (int slot = 1; slot < width; ++slot) {
        require(results[slot].finish_reason == DecoderFinishReason::length &&
                    received[slot] == 16 &&
                    results[slot].output_token_ids.size() == 16 &&
                    results[slot].generated_logprobs.empty(),
                "A peer lost output or its terminal result");
    }
    require(service.reset_prefix_cache(), "The worker did not release all slots");
    std::cout << "Buffered terminal " << static_cast<int>(outcome)
              << (scored ? " scored passed\n" : " unscored passed\n");
}

}

namespace sglang {

struct Decoder::Model {};
Decoder::Decoder(const std::filesystem::path&, int, bool) {}
Decoder::~Decoder() = default;

struct DecoderBatch::State {
    std::array<DecoderOptions, width> options{};
    std::array<int, width> produced{};
    Selection live{};
};

DecoderBatch::DecoderBatch(const Decoder&, DecoderBatchOptions options)
    : state_(std::make_unique<State>()) {
    require(options.cuda_graphs && options.max_lanes == width, "Wrong worker options");
}

DecoderBatch::~DecoderBatch() = default;

int DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    require(!state_->live[slot] && prompt.size() == decoder_batch_max_step_tokens &&
                prompt.front() == slot, "Unexpected decoder admission");
    state_->live[slot] = true;
    state_->options[slot] = options;
    return 0;
}

int DecoderBatch::admit_multimodal(int, std::vector<int>, const DecoderOptions&,
                                  const std::vector<DecoderMedia>&) {
    std::abort();
}

void DecoderBatch::cancel(int slot) {
    require(slot == 0 && state_->live[slot], "Unexpected decoder cancellation");
    state_->live[slot] = false;
    std::lock_guard lock(scenario->mutex);
    ++scenario->target_cancels;
    scenario->changed.notify_all();
}

bool DecoderBatch::available() const { return true; }
std::size_t DecoderBatch::request_device_allocation_bytes() const { return 0; }
DecoderPrefixCacheStats DecoderBatch::prefix_cache_stats() const { return {}; }
DecoderBatchPrefixUpdate DecoderBatch::poll_prefix(int) { std::abort(); }
void DecoderBatch::reset_prefix_cache() {}
void DecoderBatch::release_retired_allocations() {}

Steps DecoderBatch::advance_chunked(const Selection& selected, DecoderBatchStepStats*) {
    Steps steps{};
    std::lock_guard lock(scenario->mutex);
    if (scenario->target_produced == queued_token && !selected[0]) {
        scenario->publication_observed = true;
    }
    for (int slot = 0; slot < width; ++slot) {
        if (!selected[slot]) {
            continue;
        }
        require(state_->live[slot], "Selected a retired decoder slot");
        const int ordinal = ++state_->produced[slot];
        auto& step = steps[slot];
        step.decode = ordinal > 1;
        step.token = 1000 * slot + ordinal;
        const auto& options = state_->options[slot];
        if (options.generated_top_logprobs == 1) {
            const DecoderLogprob score{*step.token, -0.25f * *step.token};
            step.generated_logprobs =
                std::make_shared<const DecoderTokenLogprobs>(DecoderTokenLogprobs{score, {score}});
        }
        if (options.stops_at(*step.token)) {
            step.finish_reason = DecoderFinishReason::stop;
        } else if (ordinal == options.max_new_tokens) {
            step.finish_reason = DecoderFinishReason::length;
        }
        if (step.finish_reason) {
            state_->live[slot] = false;
        }
        if (slot == 0) {
            scenario->target_produced = ordinal;
        }
    }
    scenario->changed.notify_all();
    return steps;
}

}

int main() {
    for (const auto outcome : {Outcome::abort, Outcome::stop, Outcome::length}) {
        check_terminal(outcome, false);
        check_terminal(outcome, true);
    }
}
