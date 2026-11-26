#include "sglang/runtime/server/batch_cache.h"

#include <cstring>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <optional>
#include <mutex>
#include <sys/eventfd.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace sglang::server {
namespace {

using namespace decoder;
using Clock = std::chrono::steady_clock;

std::uint64_t elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

class RestoreStop {
public:
    RestoreStop() : fd(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "restore eventfd");
    }
    ~RestoreStop() { close(fd); }
    RestoreStop(const RestoreStop&) = delete;
    RestoreStop& operator=(const RestoreStop&) = delete;

    void signal() const noexcept {
        while (eventfd_write(fd, 1) < 0 && errno == EINTR) {}
    }

    const int fd;
};

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

struct BatchRestoreJoin::State {
    explicit State(int world) : views(static_cast<std::size_t>(world), nullptr) {}

    std::mutex mutex;
    std::vector<const PrefixDeviceTransfer*> views;
    std::vector<std::shared_ptr<const PrefixCacheKeys>> keys;
    std::promise<bool> completion;
    std::shared_future<bool> result;
    std::function<void()> done;
    bool submitted = false;
};

BatchRestoreJoin::~BatchRestoreJoin() = default;

void BatchRestoreJoin::fail() noexcept {
    auto& join = *state_;
    const std::lock_guard lock(join.mutex);
    if (join.submitted) return;
    join.submitted = true;
    try {
        join.completion.set_exception(std::make_exception_ptr(
            std::runtime_error("LMCache joint prefix restore was abandoned")));
    } catch (const std::future_error&) {
    }
}

void BatchRestoreJoin::abandon() noexcept {
    auto& join = *state_;
    const std::lock_guard lock(join.mutex);
    try {
        join.completion.set_exception(std::make_exception_ptr(
            std::runtime_error("LMCache joint prefix restore was abandoned")));
    } catch (const std::future_error&) {
    }
}
struct StoreCheckpoint {
    explicit StoreCheckpoint(int world)
        : attached(static_cast<std::size_t>(world), false),
          metadata(static_cast<std::size_t>(world)),
          views(static_cast<std::size_t>(world)),
          reservations(static_cast<std::size_t>(world)),
          dones(static_cast<std::size_t>(world)) {}

    std::vector<bool> attached;
    std::vector<std::string> metadata;
    std::vector<PrefixDeviceTransfer> views;
    std::vector<std::shared_ptr<void>> reservations;
    std::vector<std::function<void()>> dones;
    std::shared_ptr<std::atomic_uint64_t> stores;
};

struct BatchStorePairing::State {
    std::mutex mutex;
    std::deque<std::shared_ptr<StoreCheckpoint>> pending;
};

BatchStorePairing::BatchStorePairing() : state_(std::make_shared<State>()) {}

BatchStorePairing::~BatchStorePairing() {
    if (!state_) return;
    const std::lock_guard lock(state_->mutex);
    for (const auto& checkpoint : state_->pending) {
        for (auto& reservation : checkpoint->reservations) reservation.reset();
    }
}

namespace {
template <typename CacheState>
void run_joint_store(CacheState* state,
                     const std::shared_ptr<StoreCheckpoint>& checkpoint) {
    const auto start = Clock::now();
    const auto ranks = checkpoint->metadata.size();
    std::vector<LmCacheStatus> statuses(ranks, LmCacheStatus::declined);
    std::vector<LmCacheReceipt> receipts(ranks);
    std::vector<std::exception_ptr> errors(ranks);
    const auto run_rank = [state, &checkpoint, &statuses, &receipts,
                           &errors](std::size_t lane) noexcept {
        try {
            statuses[lane] = state->group->store_prepared(
                static_cast<int>(lane), checkpoint->metadata[lane], &receipts[lane]);
            if (statuses[lane] != LmCacheStatus::complete) return;
            if (checkpoint->stores) ++*checkpoint->stores;
            if (lane < state->rank_stores.size()) {
                ++state->rank_stores[lane];
                for (int family = 0; family < 3; ++family) {
                    state->rank_written_bytes[lane][family] += receipts[lane].bytes[family];
                }
            }
        } catch (...) {
            errors[lane] = std::current_exception();
        }
    };
    if (ranks == 1) {
        run_rank(0);
    } else {
        std::vector<std::optional<std::thread>> workers(ranks);
        for (std::size_t lane = 1; lane < ranks; ++lane) {
            try {
                workers[lane].emplace(run_rank, lane);
            } catch (const std::system_error&) {
                workers[lane].reset();
            }
        }
        run_rank(0);
        for (std::size_t lane = 1; lane < ranks; ++lane) {
            if (workers[lane]) {
                workers[lane]->join();
            } else {
                run_rank(lane);
            }
        }
    }
    bool logical = true, failure = false;
    LmCacheReceipt joint;
    auto first_incomplete = LmCacheStatus::declined;
    for (std::size_t lane = 0; lane < ranks; ++lane) {
        if (errors[lane]) {
            failure = true;
            logical = false;
            continue;
        }
        if (statuses[lane] != LmCacheStatus::complete) {
            if (logical) first_incomplete = statuses[lane];
            logical = false;
            continue;
        }
        for (int family = 0; family < 3; ++family) {
            joint.bytes[family] += receipts[lane].bytes[family];
        }
        joint.gqa_written_tokens =
            std::max(joint.gqa_written_tokens, receipts[lane].gqa_written_tokens);
    }
    if (failure) {
        ++state->errors;
        state->available = false;
    } else if (logical) {
        state->record(LmCacheStatus::complete, joint, false);
    } else {
        state->record(first_incomplete, joint, false);
    }
    state->write_ns += elapsed(start);
    for (auto& lane_view : checkpoint->views) lane_view = {};
    for (auto& lane_reservation : checkpoint->reservations) lane_reservation.reset();
    for (auto& lane_done : checkpoint->dones) {
        if (!lane_done) continue;
        auto notify = std::move(lane_done);
        notify();
    }
}
}

struct BatchCache::State {
    State(std::unique_ptr<LmCacheClient> connection, std::string identity, int lanes)
        : identity(std::move(identity)),
          reservations(std::make_shared<StoreReservations>(2 * lanes)),
          readers(std::min(lanes, 4)), restores(std::min(lanes, 4)),
          stores_queue(std::min(lanes, 4)) {
        if (connection) {
            std::vector<std::unique_ptr<LmCacheClient>> clients;
            clients.push_back(std::move(connection));
            group = std::make_unique<LmCacheGroup>(std::move(clients));
            try { available = group->ready(); } catch (const std::exception&) {}
        }
    }

    State(const LmCacheConfig& base, std::vector<std::array<std::uint8_t, 16>> rank_devices,
          std::string identity, int lanes)
        : identity(std::move(identity)),
          reservations(std::make_shared<StoreReservations>(2 * lanes)),
          readers(std::min(lanes, 4)), restores(std::min(lanes, 4)),
          stores_queue(std::min(lanes, 4)) {
        if (rank_devices.empty()) throw std::invalid_argument("Empty LMCache rank device list");
        group = std::make_unique<LmCacheGroup>(ranked_lmcache_configs(base,
                                                                     rank_devices.size()));
        devices = std::move(rank_devices);
        rank_stores = std::vector<std::atomic_uint64_t>(devices.size());
        rank_restores = std::vector<std::atomic_uint64_t>(devices.size());
        rank_read_bytes = std::vector<std::array<std::atomic_uint64_t, 3>>(devices.size());
        rank_written_bytes = std::vector<std::array<std::atomic_uint64_t, 3>>(devices.size());
        try { available = group->ready(); } catch (const std::exception&) {}
    }

    int rank_of(const PrefixDeviceTransfer& view) const {
        for (std::size_t rank = 0; rank < devices.size(); ++rank) {
            if (std::memcmp(devices[rank].data(), view.transfer.device.data(),
                            view.transfer.device.size()) == 0) {
                return static_cast<int>(rank);
            }
        }
        throw std::invalid_argument("Prefix transfer originates from an unknown device");
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

    std::unique_ptr<LmCacheGroup> group;
    std::vector<std::array<std::uint8_t, 16>> devices;
    std::vector<std::atomic_uint64_t> rank_stores;
    std::vector<std::atomic_uint64_t> rank_restores;
    std::vector<std::array<std::atomic_uint64_t, 3>> rank_read_bytes;
    std::vector<std::array<std::atomic_uint64_t, 3>> rank_written_bytes;
    std::string identity;
    std::array<std::atomic_uint64_t, 3> read_bytes{}, written_bytes{};
    std::atomic_bool available{false};
    std::atomic_uint64_t misses{0}, errors{0}, stores{0}, declined{0}, gqa_written{0};
    std::atomic_uint64_t lookup_ns{0}, write_ns{0}, restore_ns{0};
    std::shared_ptr<StoreReservations> reservations;
    RestoreStop restore_stop;
    Tasks readers, restores, stores_queue;
};

BatchCache::BatchCache(std::unique_ptr<LmCacheClient> client, std::string identity, int lanes)
    : state_(std::make_unique<State>(std::move(client), std::move(identity), lanes)) {}

BatchCache::BatchCache(const LmCacheConfig& base,
                       std::vector<std::array<std::uint8_t, 16>> devices, std::string identity,
                       int lanes)
    : state_(std::make_unique<State>(base, std::move(devices), std::move(identity), lanes)) {}

BatchCache::~BatchCache() = default;

int BatchCache::world_size() const {
    return state_->devices.empty() ? 1 : static_cast<int>(state_->devices.size());
}

void BatchCache::lookup(Keys keys, int score_start, bool draft, Cancel cancelled, ReadDone done) {
    auto* state = state_.get();
    state->readers.push([state, keys = std::move(keys), score_start, draft,
                         cancelled, done = std::move(done)] {
        BatchCacheRead result;
        const auto start = Clock::now();
        try {
            result.keys = keys(state->identity);
            if (*cancelled || result.keys.empty()) return;
            if (result.keys.size() != static_cast<std::size_t>(state->group->world_size())) {
                result.keys.clear();
            } else {
                std::vector<const PrefixCacheKeys*> rank_keys;
                rank_keys.reserve(result.keys.size());
                for (const auto& rank : result.keys) {
                    if (!rank) {
                        result.keys.clear();
                        break;
                    }
                    rank_keys.push_back(rank.get());
                }
                if (!result.keys.empty()) {
                    result.match = state->group->lookup_prefix(rank_keys, score_start, draft);
                    state->available = true;
                    state->misses += !*cancelled && result.keys.front()->cacheable_tokens() &&
                                     !result.match.usable_gdn;
                }
            }
        } catch (const std::exception&) {
            ++state->errors;
            state->available = false;
        }
        state->lookup_ns += elapsed(start);
        done(std::move(result));
    }, cancelled);
}

std::shared_ptr<BatchRestoreJoin> BatchCache::begin_restore() {
    auto join = std::make_shared<BatchRestoreJoin::State>(state_->group->world_size());
    join->result = join->completion.get_future().share();
    return std::shared_ptr<BatchRestoreJoin>(new BatchRestoreJoin(std::move(join)));
}

std::shared_future<bool> BatchCache::restore(
    const std::shared_ptr<BatchRestoreJoin>& join,
    const std::vector<std::shared_ptr<PrefixCacheKeys>>& keys,
    const PrefixDeviceTransfer& view, std::function<void()> done) {
    auto* state = state_.get();
    if (!join) throw std::invalid_argument("Missing prefix restore join");
    auto& join_state = *join->state_;
    const auto world = static_cast<std::size_t>(state->group->world_size());
    bool submit = false;
    try {
        if (keys.size() != world) {
            throw std::invalid_argument("LMCache rank key count mismatch");
        }
        for (const auto& key : keys) {
            if (!key) throw std::invalid_argument("Missing LMCache rank restore keys");
        }
        const int rank = world == 1 ? 0 : state->rank_of(view);
        const std::lock_guard lock(join_state.mutex);
        if (join_state.keys.empty()) join_state.keys.assign(keys.begin(), keys.end());
        if (join_state.views[rank]) {
            throw std::logic_error("Duplicate rank prefix restore");
        }
        join_state.views[rank] = &view;
        if (!join_state.done) join_state.done = std::move(done);
        if (!join_state.submitted &&
            std::find(join_state.views.begin(), join_state.views.end(), nullptr) ==
                join_state.views.end()) {
            join_state.submitted = true;
            submit = true;
        }
    } catch (...) {
        join->fail();
        throw;
    }
    if (submit) {
        try {
            state->restores.push([state, join = join->state_] {
                const auto start = Clock::now();
                bool success = false;
                try {
                    LmCacheReceipt receipt;
                    std::vector<LmCacheReceipt> rank_receipts;
                    std::vector<const PrefixCacheKeys*> key_ptrs;
                    key_ptrs.reserve(join->keys.size());
                    for (const auto& key : join->keys) key_ptrs.push_back(key.get());
                    const auto status = state->group->retrieve(key_ptrs, join->views, &receipt,
                                                               &rank_receipts,
                                                               state->restore_stop.fd);
                    success = state->record(status, receipt, true);
                    if (success && state->rank_restores.size() == join->views.size() &&
                        state->rank_read_bytes.size() == join->views.size()) {
                        for (std::size_t lane = 0; lane < join->views.size(); ++lane) {
                            if (!join->views[lane]) continue;
                            ++state->rank_restores[lane];
                            for (int family = 0; family < 3; ++family) {
                                state->rank_read_bytes[lane][family] +=
                                    rank_receipts[lane].bytes[family];
                            }
                        }
                    }
                } catch (const std::exception&) {
                    ++state->errors;
                    state->available = false;
                }
                state->restore_ns += elapsed(start);
                join->completion.set_value(success);
                std::function<void()> notify;
                {
                    const std::lock_guard lock(join->mutex);
                    notify = std::move(join->done);
                }
                if (notify) notify();
            });
        } catch (...) {
            join->abandon();
            throw;
        }
    }
    return join_state.result;
}

bool BatchCache::reserve_store(std::shared_ptr<void>& reservation) {
    if (!reservation) reservation = std::make_shared<StoreReservation>(state_->reservations);
    auto& ticket = *static_cast<StoreReservation*>(reservation.get());
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

void BatchCache::store(const std::shared_ptr<BatchStorePairing>& pairing,
                       const std::vector<std::shared_ptr<PrefixCacheKeys>>& keys,
                       const int* generated, int count, PrefixDeviceTransfer&& view,
                       std::shared_ptr<void> reservation,
                       std::shared_ptr<std::atomic_uint64_t> stores, std::function<void()> done) {
    auto* state = state_.get();
    const auto world = static_cast<std::size_t>(state->group->world_size());
    if (!pairing) throw std::invalid_argument("Missing prefix store pairing");
    if (keys.size() != world || count < 0) {
        throw std::invalid_argument("Invalid LMCache rank store");
    }
    for (const auto& key : keys) {
        if (!key) throw std::invalid_argument("Invalid LMCache rank store");
    }
    const int rank = world == 1 ? 0 : state->rank_of(view);
    const auto lane = static_cast<std::size_t>(rank);
    auto& rank_keys = *keys[rank];
    for (int i = 0; i < count; ++i) rank_keys.append(generated[i]);
    auto metadata = state->group->prepare_store(rank, rank_keys, view);

    std::shared_ptr<StoreCheckpoint> checkpoint;
    bool submit = false;
    {
        const std::lock_guard lock(pairing->state_->mutex);
        for (const auto& candidate : pairing->state_->pending) {
            if (!candidate->attached[lane]) {
                checkpoint = candidate;
                break;
            }
        }
        if (!checkpoint) {
            checkpoint = std::make_shared<StoreCheckpoint>(static_cast<int>(world));
            pairing->state_->pending.push_back(checkpoint);
        }
        checkpoint->attached[lane] = true;
        checkpoint->metadata[lane] = std::move(metadata);
        checkpoint->views[lane] = std::move(view);
        checkpoint->reservations[lane] = std::move(reservation);
        if (stores && !checkpoint->stores) checkpoint->stores = std::move(stores);
        checkpoint->dones[lane] = std::move(done);
        if (std::find(checkpoint->attached.begin(), checkpoint->attached.end(), false) ==
            checkpoint->attached.end()) {
            pairing->state_->pending.erase(
                std::find(pairing->state_->pending.begin(), pairing->state_->pending.end(),
                          checkpoint));
            submit = true;
        }
    }
    if (!submit) return;
    state->stores_queue.push(
        [state, checkpoint] { run_joint_store(state, checkpoint); });
}

void BatchCache::wake_readers() { state_->readers.discard_cancelled(); }

void BatchCache::stop_restores() noexcept { state_->restore_stop.signal(); }

void BatchCache::reset() {
    state_->readers.drain();
    state_->restores.drain();
    state_->stores_queue.drain();
    try {
        state_->group->reset();
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
    LmCacheMetrics metrics;
    metrics.enabled = true;
    metrics.available = state.available;
    metrics.lookup_misses = state.misses;
    metrics.io_errors = state.errors;
    metrics.stores = state.stores;
    metrics.declined_stores = state.declined;
    metrics.pending_stores = pending;
    metrics.gqa_written_tokens = state.gqa_written;
    metrics.gqa_read_bytes = state.read_bytes[0];
    metrics.gdn_read_bytes = state.read_bytes[1];
    metrics.draft_read_bytes = state.read_bytes[2];
    metrics.gqa_written_bytes = state.written_bytes[0];
    metrics.gdn_written_bytes = state.written_bytes[1];
    metrics.draft_written_bytes = state.written_bytes[2];
    metrics.lookup_ns = state.lookup_ns;
    metrics.write_ns = state.write_ns;
    metrics.restore_ns = state.restore_ns;
    metrics.world_size = state.group ? state.group->world_size() : 1;
    for (const auto& rank : state.rank_stores) metrics.rank_stores.push_back(rank);
    for (std::size_t lane = 0; lane < state.rank_restores.size(); ++lane) {
        metrics.rank_restores.push_back(state.rank_restores[lane]);
        metrics.rank_read_bytes.push_back({state.rank_read_bytes[lane][0],
                                           state.rank_read_bytes[lane][1],
                                           state.rank_read_bytes[lane][2]});
        metrics.rank_written_bytes.push_back({state.rank_written_bytes[lane][0],
                                              state.rank_written_bytes[lane][1],
                                              state.rank_written_bytes[lane][2]});
    }
    return metrics;
}

}
