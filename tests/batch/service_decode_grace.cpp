#include "sglang/server/batch_service.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace sglang;
using Mask = std::uint32_t;
using Steps = std::array<DecoderBatchStep, decoder_batch_max_lanes>;
using Selection = std::array<bool, decoder_batch_max_lanes>;
using Steady = std::chrono::steady_clock;
using System = std::chrono::system_clock;

constexpr int width = decoder_batch_max_lanes;
constexpr Mask all = ~Mask{0};
constexpr Mask first25 = (Mask{1} << 25) - 1;
static_assert(width == 32);

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::abort();
    }
}

struct Fixture {
    std::mutex mutex;
    std::condition_variable changed;
    pthread_cond_t* worker_condition = nullptr;
    pthread_mutex_t* worker_mutex = nullptr;
    int blocking_waits = 0;
    int grace_waits = 0;
    int seed_limit = 0;
    int blocked_item = -1;
    int resets = 0;
    std::vector<Mask> advances;

    template<class Predicate>
    void await(Predicate predicate) {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, std::chrono::seconds(5), predicate),
                "Timed out waiting for worker or stream progress");
    }

    void record_wait(std::condition_variable* condition, std::mutex* lock) {
        std::lock_guard guard(mutex);
        worker_condition = condition->native_handle();
        worker_mutex = lock->native_handle();
        ++blocking_waits;
        changed.notify_all();
    }

    void release_seeds(int limit) {
        std::lock_guard lock(mutex);
        seed_limit = limit;
        changed.notify_all();
    }

    std::pair<int, int> wake_worker() {
        pthread_cond_t* condition;
        pthread_mutex_t* lock;
        {
            std::lock_guard guard(mutex);
            condition = worker_condition;
            lock = worker_mutex;
        }
        pthread_mutex_lock(lock);
        const auto waits = [&] {
            std::lock_guard guard(mutex);
            return std::pair{blocking_waits, grace_waits};
        }();
        pthread_cond_broadcast(condition);
        pthread_mutex_unlock(lock);
        return waits;
    }
};

Fixture fixture;
thread_local bool decoder_worker = false;

bool check_stream(sglang::server::BatchService& service) {
    std::vector<sglang::server::BatchPrompt> prompts(width);
    for (int index = 0; index < width; ++index) {
        prompts[index].prompt = {index};
        prompts[index].options.max_new_tokens = 2;
    }
    std::array<int, width> deliveries{};
    std::optional<std::vector<DecoderResult>> results;
    const int initial_waits = [&] {
        std::lock_guard lock(fixture.mutex);
        return fixture.blocking_waits;
    }();
    std::thread stream([&] {
        sglang::server::BatchCallbacks callbacks;
        callbacks.on_output = [&](std::size_t index, const DecoderResult& result, bool done) {
            const int received = ++deliveries.at(index);
            require(received <= 2 && done == (received == 2) &&
                        result.output_token_ids.size() == static_cast<std::size_t>(received) &&
                        result.output_token_ids.back() == 100 + 2 * static_cast<int>(index) +
                                                             received - 1,
                    "Native stream delivered the wrong item, token, or finish flag");
            if (!done) {
                const int item = static_cast<int>(index);
                std::unique_lock lock(fixture.mutex);
                if (item >= fixture.seed_limit) {
                    fixture.blocked_item = item;
                    fixture.changed.notify_all();
                    fixture.changed.wait(lock, [&] { return item < fixture.seed_limit; });
                }
            }
            return true;
        };
        results = service.generate_batch(0, prompts, callbacks);
    });

    fixture.await([&] {
        return fixture.blocked_item == 0 && fixture.blocking_waits > initial_waits;
    });
    const auto pending = fixture.wake_worker();
    fixture.await([&] {
        return fixture.blocking_waits > pending.first || fixture.grace_waits > pending.second ||
               fixture.advances.size() > 1;
    });
    {
        std::lock_guard lock(fixture.mutex);
        require(fixture.grace_waits == pending.second && fixture.advances == std::vector{all},
                "A wake with every mailbox pending entered grace or dispatched work");
    }
    require(!service.reset_prefix_cache(), "Reset succeeded while the stream was active");

    fixture.release_seeds(24);
    fixture.await([&] { return fixture.blocked_item == 24; });
    const auto wait = fixture.wake_worker();
    fixture.await([&] { return fixture.grace_waits > wait.second || fixture.advances.size() > 1; });
    {
        std::lock_guard lock(fixture.mutex);
        require(fixture.advances == std::vector{all},
                "Twenty-four acknowledged mailboxes bypassed grace");
    }

    fixture.release_seeds(25);
    fixture.await([&] { return fixture.blocked_item == 25; });
    const auto threshold = fixture.wake_worker();
    fixture.await([&] {
        return fixture.grace_waits > threshold.second || fixture.advances.size() > 1;
    });
    const bool early = [&] {
        std::lock_guard lock(fixture.mutex);
        return fixture.advances.size() == 2 && fixture.advances.back() == first25;
    }();

    fixture.release_seeds(width);
    stream.join();
    require(results && results->size() == width, "Native stream returned the wrong result count");
    for (int index = 0; index < width; ++index) {
        const auto& result = results->at(index);
        require(result.finish_reason == DecoderFinishReason::length && deliveries[index] == 2 &&
                    result.output_token_ids == std::vector{100 + 2 * index, 101 + 2 * index},
                "Mailbox drain lost, duplicated, or reordered output");
    }
    std::lock_guard lock(fixture.mutex);
    if (!early) {
        std::cerr << "Twenty-five acknowledged mailboxes remained in grace\n";
        return false;
    }
    require(fixture.advances == std::vector{all, first25, all ^ first25},
            "Late acknowledgements failed to drain as the remaining seven lanes");
    return true;
}

}

extern "C" Steady::time_point __real__ZNSt6chrono3_V212steady_clock3nowEv() noexcept;
extern "C" Steady::time_point __wrap__ZNSt6chrono3_V212steady_clock3nowEv() noexcept {
    return decoder_worker ? Steady::time_point{} : __real__ZNSt6chrono3_V212steady_clock3nowEv();
}

extern "C" System::time_point __real__ZNSt6chrono3_V212system_clock3nowEv() noexcept;
extern "C" System::time_point __wrap__ZNSt6chrono3_V212system_clock3nowEv() noexcept {
    return decoder_worker ? System::time_point{} : __real__ZNSt6chrono3_V212system_clock3nowEv();
}

extern "C" void __real__ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE(
    std::condition_variable*, std::unique_lock<std::mutex>&);
extern "C" void __wrap__ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE(
    std::condition_variable* condition, std::unique_lock<std::mutex>& lock) {
    if (decoder_worker) {
        fixture.record_wait(condition, lock.mutex());
    }
    __real__ZNSt18condition_variable4waitERSt11unique_lockISt5mutexE(condition, lock);
}

extern "C" int __real_pthread_cond_timedwait(pthread_cond_t*, pthread_mutex_t*, const timespec*);
extern "C" int __wrap_pthread_cond_timedwait(pthread_cond_t* condition, pthread_mutex_t* lock,
                                           const timespec* deadline) {
    if (!decoder_worker) {
        return __real_pthread_cond_timedwait(condition, lock, deadline);
    }
    {
        std::lock_guard guard(fixture.mutex);
        ++fixture.grace_waits;
        fixture.changed.notify_all();
    }
    return pthread_cond_wait(condition, lock);
}

namespace sglang {

struct Decoder::Model {};
Decoder::Decoder(const std::filesystem::path&, int, bool, Fp8PrefillPolicy) {}
Decoder::~Decoder() = default;

struct DecoderBatch::State {
    std::array<int, width> emitted{};
};

DecoderBatch::DecoderBatch(const Decoder&, DecoderBatchOptions options)
    : state_(std::make_unique<State>()) {
    require(options.cuda_graphs && options.max_lanes == width, "Wrong fixture configuration");
    decoder_worker = true;
}

DecoderBatch::~DecoderBatch() = default;

int DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    require(prompt == std::vector{slot} && options.max_new_tokens == 2 && options.mtp_steps == 0,
            "Unexpected native admission");
    return 0;
}

int DecoderBatch::admit_multimodal(int, std::vector<int>, const DecoderOptions&,
                                  const std::vector<DecoderMedia>&) {
    std::abort();
}

void DecoderBatch::cancel(int) { std::abort(); }
bool DecoderBatch::available() const { return true; }
std::size_t DecoderBatch::request_device_allocation_bytes() const { return 0; }
DecoderPrefixCacheStats DecoderBatch::prefix_cache_stats() const { return {}; }

void DecoderBatch::reset_prefix_cache() {
    std::lock_guard lock(fixture.mutex);
    ++fixture.resets;
}

Steps DecoderBatch::advance_chunked(const Selection& selected, DecoderBatchStepStats*) {
    Mask selection = 0;
    Steps steps{};
    for (int slot = 0; slot < width; ++slot) {
        if (selected[slot]) {
            selection |= Mask{1} << slot;
            auto& emitted = state_->emitted[slot];
            require(emitted < 2, "Finished lane was selected again");
            auto& step = steps[slot];
            step.decode = emitted != 0;
            step.token = 100 + 2 * slot + emitted++;
            if (emitted == 2) {
                step.finish_reason = DecoderFinishReason::length;
            }
        }
    }
    std::lock_guard lock(fixture.mutex);
    fixture.advances.push_back(selection);
    fixture.changed.notify_all();
    return steps;
}

}

int main() {
    bool passed;
    {
        const Decoder decoder({}, 0, false, Fp8PrefillPolicy::exact);
        sglang::server::BatchService service(decoder, false, true, 0, width);
        require(service.ready(), "Batch worker did not start");
        fixture.await([] { return fixture.blocking_waits > 0; });
        require(service.reset_prefix_cache(), "Idle reset before streaming failed");
        passed = check_stream(service);
        require(service.reset_prefix_cache(), "Idle reset after draining failed");
    }
    require(fixture.resets == 2, "Idle reset did not reach the worker");
    if (passed) {
        std::cout << "Native mailbox grace and drain passed\n";
    }
    return passed ? 0 : 1;
}
