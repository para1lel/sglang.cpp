#include "sglang/runtime/server/single_device_batch_executor.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/indexed_admission.h"
#include "sglang/tokenizer/tokenizer.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <iostream>
#include <mutex>
#include <utility>

namespace {

using namespace std::chrono_literals;
using namespace sglang;
using namespace sglang::server;

void require(bool passed, const char* message) {
    if (!passed) {
        std::cerr << message << '\n';
        std::abort();
    }
}

struct Gate {
    bool hold() {
        entered.set_value();
        return release.get_future().wait_for(5s) == std::future_status::ready;
    }

    void wait() {
        require(entered.get_future().wait_for(5s) == std::future_status::ready,
                "Timed out waiting for the admission callback");
    }

    std::promise<void> entered;
    std::promise<void> release;
};

struct Admissions {
    void add(int token, int slot) {
        std::lock_guard lock(mutex);
        events.emplace_back(token, slot);
        changed.notify_all();
    }

    std::vector<std::pair<int, int>> wait(std::size_t count) {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, 5s, [&] { return events.size() >= count; }),
                "Timed out waiting for decoder admissions");
        return events;
    }

    void clear() {
        std::lock_guard lock(mutex);
        events.clear();
    }

    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::pair<int, int>> events;
};

Admissions admissions;
std::atomic<bool> hold_first_lane = false;

std::vector<BatchPrompt> prompts(int first, int count) {
    std::vector<BatchPrompt> items;
    for (int index = 0; index < count; ++index) {
        BatchPrompt item;
        item.prompt = {first + index};
        item.options.max_new_tokens = 2;
        items.push_back(std::move(item));
    }
    return items;
}

void check_results(const std::optional<std::vector<DecoderResult>>& results, int first, int count) {
    require(results && results->size() == static_cast<std::size_t>(count),
            "Wrong indexed result count");
    for (int index = 0; index < count; ++index) {
        const auto token = first + index;
        const auto& result = results->at(index);
        require(result.output_token_ids == std::vector{token, token + 1000} &&
                    result.finish_reason == DecoderFinishReason::length,
                "Wrong indexed result or finish reason");
    }
}

void check_tickets() {
    IndexedAdmission admission;
    const auto now = IndexedAdmission::Clock::time_point{};
    admission.wait(0, 0b0011);
    admission.wait(1, 0b0111);
    require(!admission.can_claim(1, 0, now) && admission.can_claim(1, 2, now),
            "An older request must only claim lanes it can use");
    admission.claimed(0, 0);
    require(admission.can_claim(1, 1, now) && !admission.can_claim(0, 1, now),
            "A successful refill must rotate priority to the other waiter");
    admission.poll(1, now + 10ms);
    require(!admission.can_claim(0, 1, now + 9ms) && admission.can_claim(0, 1, now + 10ms),
            "A polling request must keep its turn only until the poll deadline");
    admission.wait(1, 0b0010);
    require(!admission.can_claim(0, 1, now + 11ms),
            "A returned polling request must regain eligibility with the same ticket");
    admission.pause(1);
    require(admission.can_claim(0, 1, now), "A blocked writer retained its admission claim");
    admission.wait(1, 0b0010);
    require(!admission.can_claim(0, 1, now), "A paused waiter lost its earlier ticket");
    admission.finish(1);
    require(admission.can_claim(0, 1, now), "A retired request retained its admission claim");
}

void check_two_owners(const Decoder& decoder) {
    admissions.clear();
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{false, 4, 0, decoder_default_max_num_batched_tokens}), false);
    Gate gate;
    auto left = std::async(std::launch::async, [&] {
        BatchCallbacks callbacks;
        callbacks.on_admitted = [&] { return gate.hold(); };
        return service.generate_batch(2, prompts(10, 2), callbacks);
    });
    gate.wait();
    require(!service.reset_prefix_cache(), "An indexed request must prevent prefix reset");
    require(admissions.wait(2) == (std::vector<std::pair<int, int>>{{10, 0}, {11, 1}}),
            "Indexed physical placement still depends on the HTTP owner");
    auto right = std::async(std::launch::async, [&] {
        return service.generate_batch(3, prompts(20, 2), {});
    });
    require(right.wait_for(3s) == std::future_status::ready,
            "A waiting indexed group cannot use all free lanes");
    check_results(right.get(), 20, 2);
    require(admissions.wait(4) ==
                (std::vector<std::pair<int, int>>{{10, 0}, {11, 1}, {20, 2}, {21, 3}}),
            "Concurrent owners stranded a decoder lane");
    gate.release.set_value();
    check_results(left.get(), 10, 2);
    check_results(service.generate_batch(3, prompts(30, 2), {}), 30, 2);
    check_results(service.generate_batch(3, prompts(40, 1), {}), 40, 1);
    const auto events = admissions.wait(7);
    require(events[4] == std::pair{30, 0} && events[5] == std::pair{31, 1} &&
                events[6] == std::pair{40, 3},
            "Isolated indexed or singleton placement changed");
}

void check_refill(const Decoder& decoder) {
    admissions.clear();
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{false, 2, 0, decoder_default_max_num_batched_tokens}), false);
    Gate gate;
    auto left = std::async(std::launch::async, [&] {
        BatchCallbacks callbacks;
        callbacks.on_admitted = [&] { return gate.hold(); };
        return service.generate_batch(1, prompts(10, 8), callbacks);
    });
    gate.wait();
    admissions.wait(2);
    std::promise<void> waiting;
    auto right = std::async(std::launch::async, [&] {
        int polls = 0;
        BatchCallbacks callbacks;
        callbacks.is_cancelled = [&] {
            if (++polls == 2) {
                waiting.set_value();
            }
            return false;
        };
        return service.generate_batch(0, prompts(30, 8), callbacks);
    });
    require(waiting.get_future().wait_for(5s) == std::future_status::ready,
            "Competing group did not reach the saturated queue");
    gate.release.set_value();
    check_results(left.get(), 10, 8);
    check_results(right.get(), 30, 8);
    const auto events = admissions.wait(16);
    require(events[2].first == 30 || events[3].first == 30,
            "An oversubscribed group refilled twice ahead of a waiting peer");
}

void check_paused_cancellation(const Decoder& decoder) {
    admissions.clear();
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{false, 2, 0, decoder_default_max_num_batched_tokens}), false);
    Gate incumbent_gate;
    auto incumbent = std::async(std::launch::async, [&] {
        BatchCallbacks callbacks;
        callbacks.on_admitted = [&] { return incumbent_gate.hold(); };
        return service.generate_batch(0, prompts(10, 2), callbacks);
    });
    incumbent_gate.wait();
    admissions.wait(2);

    Gate poll_gate;
    auto paused = std::async(std::launch::async, [&] {
        int polls = 0;
        BatchCallbacks callbacks;
        callbacks.is_cancelled = [&] {
            if (++polls == 2) {
                require(poll_gate.hold(), "Cancellation callback was never released");
                return true;
            }
            return false;
        };
        return service.generate_batch(1, prompts(30, 2), callbacks);
    });
    poll_gate.wait();
    incumbent_gate.release.set_value();
    check_results(incumbent.get(), 10, 2);

    auto replacement = std::async(std::launch::async, [&] {
        return service.generate_batch(0, prompts(70, 2), {});
    });
    const bool progressed = replacement.wait_for(500ms) == std::future_status::ready;
    poll_gate.release.set_value();
    require(!paused.get(), "Cancelled group returned a result");
    check_results(replacement.get(), 70, 2);
    const auto events = admissions.wait(4);
    require(progressed && events ==
                (std::vector<std::pair<int, int>>{{10, 0}, {11, 1}, {70, 0}, {71, 1}}),
            "A paused cancellation callback stranded available decoder lanes");
    check_results(service.generate_batch(1, prompts(90, 2), {}), 90, 2);
}

void check_scalar_reservation(const Decoder& decoder) {
    admissions.clear();
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{false, 2, 0, decoder_default_max_num_batched_tokens}), false);
    Gate gate;
    auto group = std::async(std::launch::async, [&] {
        BatchCallbacks callbacks;
        callbacks.on_admitted = [&] { return gate.hold(); };
        return service.generate_batch(1, prompts(10, 4), callbacks);
    });
    gate.wait();
    admissions.wait(2);
    std::promise<void> reserved;
    auto scalar = std::async(std::launch::async, [&] {
        bool reported = false;
        DecoderCallbacks callbacks;
        callbacks.is_cancelled = [&] {
            if (!std::exchange(reported, true)) {
                reserved.set_value();
            }
            return false;
        };
        const auto item = prompts(99, 1).front();
        return service.generate(0, item.prompt, item.options, callbacks);
    });
    require(reserved.get_future().wait_for(5s) == std::future_status::ready,
            "Scalar did not reserve its busy physical lane");
    gate.release.set_value();
    check_results(group.get(), 10, 4);
    require(scalar.get().output_token_ids == std::vector{99, 1099}, "Wrong scalar output");
    const auto events = admissions.wait(5);
    const auto refill = std::find_if(events.begin() + 2, events.end(), [](const auto& event) {
        return event.second == 0;
    });
    require(refill != events.end() && refill->first == 99,
            "An indexed refill took a scalar's reserved lane");
}

void check_singleton_owner(const Decoder& decoder) {
    admissions.clear();
    hold_first_lane = true;
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
        BatchExecutor::Options{false, 2, 0, decoder_default_max_num_batched_tokens}), false);
    std::promise<void> peer_finished;
    auto peer_finished_signal = peer_finished.get_future();
    auto group = std::async(std::launch::async, [&] {
        BatchCallbacks callbacks;
        callbacks.on_output = [&](std::size_t index, const DecoderResult&, bool terminal) {
            if (terminal && index == 1) {
                peer_finished.set_value();
            }
            return true;
        };
        return service.generate_batch(1, prompts(10, 2), callbacks);
    });
    require(admissions.wait(2) == (std::vector<std::pair<int, int>>{{10, 0}, {11, 1}}),
            "Indexed incumbent did not occupy both lanes");
    require(peer_finished_signal.wait_for(5s) == std::future_status::ready,
            "The non-owner lane did not finish");

    std::promise<void> polled;
    auto singleton = std::async(std::launch::async, [&] {
        int polls = 0;
        BatchCallbacks callbacks;
        callbacks.is_cancelled = [&] {
            if (++polls == 1) {
                polled.set_value();
            }
            return false;
        };
        return service.generate_batch(0, prompts(90, 1), callbacks);
    });
    require(polled.get_future().wait_for(5s) == std::future_status::ready,
            "One-item request did not start polling");
    const bool waited_for_owner = singleton.wait_for(150ms) == std::future_status::timeout;
    hold_first_lane = false;
    check_results(group.get(), 10, 2);
    check_results(singleton.get(), 90, 1);
    require(waited_for_owner && admissions.wait(3).back() == std::pair{90, 0},
            "One-item batch used another physical lane while its owner's lane was occupied");
}

}

namespace sglang {

struct Decoder::Model {};
#define SGLANG_TEST_TP_HEADER "sglang/runtime/execution/decoder_tp.h"
#if __has_include(SGLANG_TEST_TP_HEADER)
struct Decoder::TpRuntime {};
const decoder::TpRuntimeInfo* Decoder::tensor_parallel_info() const { return nullptr; }
#endif
Decoder::Decoder(const std::filesystem::path&, int, bool) {}
Decoder::~Decoder() = default;

std::string_view Tokenizer::piece(TokenId, bool) const { std::abort(); }

struct DecoderBatch::State {
    std::array<int, decoder_batch_max_lanes> tokens{};
    std::array<int, decoder_batch_max_lanes> generated{};
};

DecoderBatch::DecoderBatch(const Decoder&, DecoderBatchOptions)
    : state_(std::make_unique<State>()) {}
DecoderBatch::~DecoderBatch() = default;

int DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    require(prompt.size() == 1 && options.max_new_tokens == 2, "Unexpected decoder admission");
    state_->tokens[slot] = prompt.front();
    state_->generated[slot] = 0;
    admissions.add(prompt.front(), slot);
    return 0;
}

int DecoderBatch::admit_multimodal(int, std::vector<int>, const DecoderOptions&,
                                  const std::vector<DecoderMedia>&) {
    std::abort();
}

void DecoderBatch::cancel(int) {}
bool DecoderBatch::available() const { return true; }
std::size_t DecoderBatch::request_device_allocation_bytes() const { return 0; }
DecoderPrefixCacheStats DecoderBatch::prefix_cache_stats() const { return {}; }
DecoderBatchPrefixUpdate DecoderBatch::poll_prefix(int) { std::abort(); }
void DecoderBatch::reset_prefix_cache() {}
void DecoderBatch::release_retired_allocations() {}

std::array<DecoderBatchStep, decoder_batch_max_lanes>
DecoderBatch::advance_chunked(const std::array<bool, decoder_batch_max_lanes>& selected,
                              DecoderBatchStepStats*) {
    std::array<DecoderBatchStep, decoder_batch_max_lanes> steps{};
    for (int slot = 0; slot < decoder_batch_max_lanes; ++slot) {
        if (!selected[slot]) {
            continue;
        }
        if (slot == 0 && state_->tokens[slot] == 10 && state_->generated[slot] == 1 &&
            hold_first_lane) {
            continue;
        }
        const int count = state_->generated[slot]++;
        require(count < 2, "Retired lane was advanced");
        steps[slot].token = state_->tokens[slot] + 1000 * count;
        steps[slot].decode = count != 0;
        if (count == 1) {
            steps[slot].finish_reason = DecoderFinishReason::length;
        }
    }
    return steps;
}

}

int main(int argc, char** argv) {
    const Decoder decoder({}, 0, false);
    if (argc == 2) {
        std::vector<std::array<std::uint8_t, 16>> devices(2);
        devices[0][0] = 1;
        devices[1][0] = 2;
        BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder,
            BatchExecutor::Options{false, 2, 0, decoder_default_max_num_batched_tokens}),
            false, 0, decoder::LmCacheConfig{argv[1], "service", 2000}, devices, "service");
        check_results(service.generate_batch(0, prompts(90, 1), {}), 90, 1);
        return 0;
    }
    check_tickets();
    check_two_owners(decoder);
    check_refill(decoder);
    check_paused_cancellation(decoder);
    check_scalar_reservation(decoder);
    check_singleton_owner(decoder);
    std::cout << "Indexed capacity, refill fairness, cancellation progress "
                 "and reservations passed\n";
}
