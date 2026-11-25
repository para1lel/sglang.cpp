#include "sglang/runtime/server/batch_cache.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace sglang::server {
namespace {

using namespace decoder;
using Clock = std::chrono::steady_clock;

std::uint64_t elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

class Tasks {
public:
    explicit Tasks(int count) {
        try {
            for (int i = 0; i < count; ++i) workers_.emplace_back([this] { run(); });
        } catch (...) {
            close();
            throw;
        }
    }
    ~Tasks() { close(); }

    void push(std::function<void()> job, BatchCache::Cancel cancel = nullptr) {
        const std::lock_guard lock(mutex_);
        queue_.push_back({std::move(job), std::move(cancel)});
        changed_.notify_one();
    }

    void discard_cancelled() {
        const std::lock_guard lock(mutex_);
        queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [](const auto& task) {
            return task.cancel && *task.cancel;
        }), queue_.end());
        if (queue_.empty() && active_ == 0) idle_.notify_all();
    }

    void drain() {
        std::unique_lock lock(mutex_);
        idle_.wait(lock, [&] { return queue_.empty() && active_ == 0; });
    }

private:
    void close() {
        {
            const std::lock_guard lock(mutex_);
            closing_ = true;
        }
        changed_.notify_all();
        for (auto& worker : workers_) worker.join();
    }

    void run() {
        for (;;) {
            Task task;
            {
                std::unique_lock lock(mutex_);
                changed_.wait(lock, [&] { return closing_ || !queue_.empty(); });
                if (queue_.empty()) return;
                task = std::move(queue_.front());
                queue_.pop_front();
                ++active_;
            }
            if (!task.cancel || !*task.cancel) task.run();
            {
                const std::lock_guard lock(mutex_);
                if (--active_ == 0 && queue_.empty()) idle_.notify_all();
            }
        }
    }

    struct Task {
        std::function<void()> run;
        BatchCache::Cancel cancel;
    };
    std::mutex mutex_;
    std::condition_variable changed_, idle_;
    std::deque<Task> queue_;
    std::vector<std::thread> workers_;
    int active_ = 0;
    bool closing_ = false;
};

}

struct BatchCache::State {
    State(std::unique_ptr<LmCacheClient> connection, std::string identity, int lanes)
        : client(std::move(connection)), identity(std::move(identity)),
          readers(std::min(lanes, 4)), transfers(std::min(lanes, 4)) {
        try { available = client->ready(); } catch (const std::exception&) {}
    }

    bool apply(const PrefixCacheKeys& keys, const PrefixDeviceTransfer& command, bool restore) {
        LmCacheReceipt receipt;
        const auto status = restore ? client->retrieve(keys, command, &receipt)
                                    : client->store(keys, command, &receipt);
        available = true;
        if (status != LmCacheStatus::complete) {
            misses += restore && status == LmCacheStatus::miss;
            declined += !restore;
            return false;
        }
        auto& bytes = restore ? read_bytes : written_bytes;
        for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] += receipt.bytes[i];
        if (!restore) {
            gqa_written += receipt.gqa_written_tokens;
            ++stores;
        }
        return true;
    }

    std::unique_ptr<LmCacheClient> client;
    std::string identity;
    std::array<std::atomic_uint64_t, 3> read_bytes{}, written_bytes{};
    std::atomic_bool available{false};
    std::atomic_uint64_t misses{0}, errors{0}, stores{0}, declined{0}, gqa_written{0};
    std::atomic_uint64_t lookup_ns{0}, write_ns{0}, restore_ns{0};
    Tasks readers, transfers;
};

BatchCache::BatchCache(std::unique_ptr<LmCacheClient> client, std::string identity, int lanes)
    : state_(std::make_unique<State>(std::move(client), std::move(identity), lanes)) {}
BatchCache::~BatchCache() = default;

void BatchCache::lookup(Keys keys, int score_start, bool draft, Cancel cancelled, ReadDone done) {
    auto* state = state_.get();
    state->readers.push([state, keys = std::move(keys), score_start, draft,
                         cancelled, done = std::move(done)] {
        BatchCacheRead result;
        const auto start = Clock::now();
        try {
            result.keys = keys(state->identity);
            if (!result.keys || *cancelled) return;
            const auto match = state->client->lookup_prefix(*result.keys, score_start, draft);
            result.match = {match.gqa, match.matched_gdn, match.gqa, match.gdn, match.draft};
            state->available = true;
            state->misses += !*cancelled && result.keys->cacheable_tokens() && !match.gdn;
        } catch (const std::exception&) {
            ++state->errors;
            state->available = false;
        }
        state->lookup_ns += elapsed(start);
        done(std::move(result));
    }, cancelled);
}

std::shared_future<bool> BatchCache::transfer(std::shared_ptr<const PrefixCacheKeys> keys,
                                              const PrefixDeviceTransfer& command, bool restore,
                                              std::function<void()> done) {
    auto* state = state_.get();
    auto complete = std::make_shared<std::promise<bool>>();
    auto result = complete->get_future().share();
    // The decoder retains the command and its live device allocations until this future is ready.
    state->transfers.push([state, keys = std::move(keys), &command, restore, complete,
                           done = std::move(done)] {
        const auto start = Clock::now();
        bool success = false;
        try {
            success = state->apply(*keys, command, restore);
        } catch (const std::exception&) {
            ++state->errors;
            state->available = false;
        }
        (restore ? state->restore_ns : state->write_ns) += elapsed(start);
        complete->set_value(success);
        done();
    });
    return result;
}

void BatchCache::wake_readers() { state_->readers.discard_cancelled(); }

void BatchCache::reset() {
    state_->readers.drain();
    state_->transfers.drain();
    try {
        state_->client->reset();
        state_->available = true;
    } catch (...) {
        ++state_->errors;
        state_->available = false;
        throw CacheResetUnavailable("LMCache namespace reset was not acknowledged");
    }
}

LmCacheMetrics BatchCache::metrics() const {
    const auto& state = *state_;
    return {true, state.available, state.misses, state.errors, state.stores, state.declined,
            state.gqa_written, state.read_bytes[0], state.read_bytes[1], state.read_bytes[2],
            state.written_bytes[0], state.written_bytes[1], state.written_bytes[2],
            state.lookup_ns, state.write_ns, state.restore_ns};
}

}
