#include "service_mtp_mailbox_fixture.h"

#include "sglang/runtime/server/single_device_batch_executor.h"
#include "sglang/server/batch_service.h"

#include <chrono>
#include <future>
#include <iostream>

namespace sglang::server::mailbox_fixture {
namespace {
using namespace std::chrono_literals;
using Finish = DecoderFinishReason;

enum class CapacityOutcome { admitted, aborted, exhausted };

void check_mixed_mailboxes(int lanes, bool sampled_writer) {
    Fixture observed;
    observed.max_lanes = lanes;
    const Decoder decoder({}, 0, true);
    BatchService service(
        std::make_unique<SingleDeviceBatchExecutor>(
            decoder,
            BatchExecutor::Options{true, lanes, 0, decoder_default_max_num_batched_tokens}),
        false, 4);
    const auto start = [&](int slot, bool sampled) {
        return std::async(std::launch::async, [&, slot, sampled] {
            const int base = 100 + (100 * slot);
            DecoderOptions options;
            options.max_new_tokens = 16;
            if (sampled) {
                options.temperature = 0.7F;
                options.top_k = 40;
                options.seed = 123 + slot;
            }
            DecoderCallbacks callbacks;
            callbacks.buffer_decode_output = true;
            std::vector<int> delivered;
            callbacks.on_token = [&](int token) {
                require(token == base + static_cast<int>(delivered.size()),
                        "Mixed mailbox reordered or duplicated output");
                delivered.push_back(token);
                if (slot == 0 && token == base + 1) {
                    std::unique_lock lock(observed.mutex);
                    observed.hold(lock, observed.writer);
                } else if (slot != 0 && token == base) {
                    std::unique_lock lock(observed.mutex);
                    observed.hold(lock, observed.consumers[slot]);
                }
                return true;
            };
            const auto result = service.generate(slot, {base}, options, callbacks);
            require(result.output_token_ids == delivered && delivered.size() == 16 &&
                        result.finish_reason == Finish::length && result.mtp_rounds == 3 &&
                        result.mtp_accepted_tokens == 12,
                    "Mixed mailbox changed result or speculation counters");
        });
    };
    auto slow = start(0, sampled_writer);
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.writer.entered; });
    }
    std::vector<std::future<void>> peers;
    for (int slot = 1; slot < lanes; ++slot)
        peers.push_back(start(slot, slot % 2 != 0));
    {
        std::unique_lock lock(observed.mutex);
        for (int slot = 1; slot < lanes; ++slot) {
            observed.await(lock, [&] { return observed.consumers[slot].entered; });
        }
        require(observed.primary_steps == 3, "Slow writer did not fill exactly two steps");
    }
    const auto grace_before = service.metrics().mailbox_grace_timeouts;
    for (int slot = 1; slot < lanes; ++slot) {
        {
            const std::scoped_lock lock(observed.mutex);
            observed.release(observed.consumers[slot]);
        }
        require(peers[slot - 1].wait_for(5s) == std::future_status::ready,
                "Slow writer blocked a ready peer");
        peers[slot - 1].get();
    }
    const auto grace_after = service.metrics().mailbox_grace_timeouts;
    require(grace_after - grace_before <= 1,
            "A previously blocked writer repeatedly delayed independent peers");
    {
        const std::scoped_lock lock(observed.mutex);
        require(observed.primary_steps == 3 && slow.wait_for(0s) == std::future_status::timeout,
                "Full mailbox advanced or finished before its writer returned");
        observed.release(observed.writer);
    }
    slow.get();
    require(service.runtime_state().running_requests == 0 && service.reset_prefix_cache(),
            "Drained mailboxes retained decoder ownership");
}

void check_capacity_settlement(CapacityOutcome outcome, std::future<DecoderResult>& oldest,
                               const int& oldest_acks) {
    if (outcome == CapacityOutcome::exhausted) {
        try {
            static_cast<void>(oldest.get());
            require(false, "Exhausted resident request did not report capacity failure");
        } catch (const DecoderCapacityError& error) {
            require(error.retryable && oldest_acks == 0,
                    "Terminal resident rejection acknowledged admission or changed its error");
        }
        return;
    }
    const auto result = oldest.get();
    if (outcome == CapacityOutcome::aborted) {
        require(result.finish_reason == Finish::aborted && result.output_token_ids.empty() &&
                    oldest_acks == 0,
                "Aborted resident wait acknowledged admission or emitted output");
    } else {
        require(result.finish_reason == Finish::length &&
                    result.output_token_ids == std::vector{200, 201} && oldest_acks == 1,
                "Oldest retry did not acknowledge exactly one successful admission");
    }
}

void check_capacity_order(CapacityOutcome outcome) {
    Fixture observed;
    observed.limit_capacity = true;
    observed.exhaust_capacity = outcome == CapacityOutcome::exhausted;
    const Decoder decoder({}, 0, true);
    BatchExecutor::Options executor_options{true, 4, 0, decoder_default_max_num_batched_tokens};
    executor_options.resident_prefix_cache = true;
    BatchService service(std::make_unique<SingleDeviceBatchExecutor>(decoder, executor_options),
                         false, 0);
    DecoderOptions options;
    options.max_new_tokens = 2;
    DecoderCallbacks holder_callbacks;
    holder_callbacks.on_token = [&](int token) {
        if (token < 102) {
            std::unique_lock lock(observed.mutex);
            observed.hold(lock, token == 100 ? observed.writer : observed.peer);
        }
        return true;
    };
    auto holder = std::async(std::launch::async, [&] {
        auto holder_options = options;
        holder_options.max_new_tokens = 3;
        return service.generate(0, {100}, holder_options, holder_callbacks);
    });
    observed.wait_for(observed.writer);
    BatchRequestId oldest_id("capacity-oldest");
    int oldest_acks = 0;
    DecoderCallbacks oldest_callbacks;
    oldest_callbacks.on_admitted = [&] {
        ++oldest_acks;
        return true;
    };
    auto oldest = std::async(std::launch::async, [&] {
        return service.generate(1, {200}, options, oldest_callbacks, {}, 0, &oldest_id);
    });
    observed.wait_for(observed.capacity_rejected);
    int younger_acks = 0;
    DecoderCallbacks younger_callbacks;
    younger_callbacks.is_cancelled = [&] {
        std::unique_lock lock(observed.mutex);
        observed.hold(lock, observed.consumers[2]);
        return false;
    };
    younger_callbacks.on_admitted = [&] {
        ++younger_acks;
        return true;
    };
    auto younger = std::async(
        std::launch::async, [&] { return service.generate(2, {300}, options, younger_callbacks); });
    {
        std::unique_lock lock(observed.mutex);
        observed.await(lock, [&] { return observed.consumers[2].entered; });
        observed.release(observed.capacity_rejected);
        observed.await(lock, [&] { return observed.capacity_rejections >= 2; });
        observed.release(observed.writer);
        observed.await(lock, [&] { return observed.peer.entered; });
        require(observed.capacity_admissions == std::vector{0},
                "Younger allocation bypassed the oldest resident retry");
    }
    const auto blocked = service.metrics();
    require(blocked.resident_admission_retries > 0 && blocked.admission_waiting_for_capacity == 1,
            "Resident admission wait was not counted");
    require(!service.reset_prefix_cache(), "Cache reset took ownership during a resident retry");
    if (outcome == CapacityOutcome::aborted) {
        require(oldest_id.abort(true), "Resident wait could not be aborted");
        service.notify_abort();
        check_capacity_settlement(outcome, oldest, oldest_acks);
    }
    {
        const std::scoped_lock lock(observed.mutex);
        observed.release(observed.consumers[2]);
        if (outcome != CapacityOutcome::aborted)
            observed.release(observed.peer);
    }
    if (outcome != CapacityOutcome::aborted)
        check_capacity_settlement(outcome, oldest, oldest_acks);
    const auto result = younger.get();
    require(result.finish_reason == Finish::length &&
                result.output_token_ids == std::vector{300, 301} && younger_acks == 1,
            "Younger request did not progress after the oldest retry settled");
    {
        const std::scoped_lock lock(observed.mutex);
        require(
            observed.capacity_admissions ==
                (outcome == CapacityOutcome::admitted ? std::vector{0, 1, 2} : std::vector{0, 2}),
            "Unexpected resident admission order");
        if (outcome == CapacityOutcome::aborted)
            observed.release(observed.peer);
    }
    require(holder.get().output_token_ids == std::vector{100, 101, 102} &&
                service.metrics().admission_waiting_for_capacity == 0 &&
                service.reset_prefix_cache(),
            "Resident retry retained cache ownership or changed an admitted peer's output");
}

}

void check_progress() {
    for (const int lanes : {7, 8, 9}) {
        check_mixed_mailboxes(lanes, false);
        check_mixed_mailboxes(lanes, true);
    }
    for (const auto outcome : {
             CapacityOutcome::admitted,
             CapacityOutcome::aborted,
             CapacityOutcome::exhausted,
         }) {
        check_capacity_order(outcome);
    }
    std::cout << "Mixed mailboxes and resident admission progress passed\n";
}

}
