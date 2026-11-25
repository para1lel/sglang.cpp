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

struct StoreReservations {
    explicit StoreReservations(int limit) : limit(limit) {}

    const int limit;
    std::atomic_int reserved{0};
    std::atomic_int pending{0};
};

struct StoreReservation {
    explicit StoreReservation(const std::shared_ptr<StoreReservations>& pool) : counts(pool) {
        ++counts->pending;
    }
    ~StoreReservation() {
        if (reserved) --counts->reserved;
        --counts->pending;
    }

    std::shared_ptr<StoreReservations> counts;
    bool reserved = false;
};

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
          reservations(std::make_shared<StoreReservations>(2 * lanes)),
          readers(std::min(lanes, 4)), restores(std::min(lanes, 4)),
          stores_queue(std::min(lanes, 4)) {
        try { available = client->ready(); } catch (const std::exception&) {}
    }

    bool record(LmCacheStatus status, const LmCacheReceipt& receipt, bool restore) {
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
    std::shared_ptr<StoreReservations> reservations;
    Tasks readers, restores, stores_queue;
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

std::shared_future<bool> BatchCache::restore(std::shared_ptr<const PrefixCacheKeys> keys,
                                             const PrefixDeviceTransfer& view,
                                             std::function<void()> done) {
    auto* state = state_.get();
    auto complete = std::make_shared<std::promise<bool>>();
    auto result = complete->get_future().share();
    // The importing request owns its destination until the connector is quiescent.
    state->restores.push([state, keys = std::move(keys), &view, complete,
                           done = std::move(done)] {
        const auto start = Clock::now();
        bool success = false;
        try {
            LmCacheReceipt receipt;
            const auto status = state->client->retrieve(*keys, view, &receipt);
            success = state->record(status, receipt, true);
        } catch (const std::exception&) {
            ++state->errors;
            state->available = false;
        }
        state->restore_ns += elapsed(start);
        complete->set_value(success);
        done();
    });
    return result;
}

bool BatchCache::reserve_store(std::shared_ptr<void>& reservation) {
    if (!reservation) reservation = std::make_shared<StoreReservation>(state_->reservations);
    auto& ticket = *static_cast<StoreReservation*>(reservation.get());
    // Only the decoder worker reserves a view. Connector workers return reservations.
    if (ticket.reserved) return true;
    if (!store_capacity()) return false;
    ticket.reserved = true;
    ++ticket.counts->reserved;
    return true;
}

bool BatchCache::store_capacity() const {
    const auto& counts = *state_->reservations;
    return counts.reserved < counts.limit;
}

bool BatchCache::stores_pending() const { return state_->reservations->pending != 0; }

void BatchCache::store(const PrefixCacheKeys& keys, PrefixDeviceTransfer&& view,
                       std::shared_ptr<void> reservation,
                       std::shared_ptr<std::atomic_uint64_t> stores, std::function<void()> done) {
    auto* state = state_.get();
    auto metadata = state->client->prepare_store(keys, view);
    state->stores_queue.push([state, metadata = std::move(metadata), view = std::move(view),
                             reservation = std::move(reservation), stores = std::move(stores),
                             done = std::move(done)]() mutable {
        const auto start = Clock::now();
        try {
            LmCacheReceipt receipt;
            const auto status = state->client->store_prepared(metadata, &receipt);
            if (state->record(status, receipt, false)) ++*stores;
        } catch (const std::exception&) {
            ++state->errors;
            state->available = false;
        }
        state->write_ns += elapsed(start);
        view = {};
        reservation.reset();
        done();
    });
}

void BatchCache::wake_readers() { state_->readers.discard_cancelled(); }

void BatchCache::reset() {
    state_->readers.drain();
    state_->restores.drain();
    state_->stores_queue.drain();
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
    const auto pending = static_cast<std::uint64_t>(state.reservations->pending.load());
    return {true, state.available, state.misses, state.errors, state.stores, state.declined,
            pending, state.gqa_written, state.read_bytes[0], state.read_bytes[1],
            state.read_bytes[2], state.written_bytes[0], state.written_bytes[1],
            state.written_bytes[2], state.lookup_ns, state.write_ns, state.restore_ns};
}

}
