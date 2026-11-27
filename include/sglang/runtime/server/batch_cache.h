#pragma once

#include "sglang/runtime/cache/lmcache_client.h"
#include "sglang/runtime/cache/prefix_payload.h"

#include <array>
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sglang::server {

struct LmCacheMetrics {
    bool enabled = false;
    bool available = false;
    std::uint64_t lookup_misses = 0;
    std::uint64_t io_errors = 0;
    std::uint64_t stores = 0;
    std::uint64_t declined_stores = 0;
    std::uint64_t pending_stores = 0;
    std::uint64_t gqa_written_tokens = 0;
    std::uint64_t gqa_read_bytes = 0, gdn_read_bytes = 0, draft_read_bytes = 0;
    std::uint64_t gqa_written_bytes = 0, gdn_written_bytes = 0, draft_written_bytes = 0;
    std::uint64_t lookup_ns = 0;
    std::uint64_t write_ns = 0;
    std::uint64_t restore_ns = 0;
    int world_size = 1;
    std::vector<std::uint64_t> rank_stores;
    std::vector<std::uint64_t> rank_restores;
    std::vector<std::array<std::uint64_t, 3>> rank_read_bytes;
    std::vector<std::array<std::uint64_t, 3>> rank_written_bytes;
};

struct BatchCacheRead {
    std::vector<std::shared_ptr<decoder::PrefixCacheKeys>> keys;
    decoder::PrefixCacheRead match;
};

class CacheResetUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class BatchRestoreJoin {
public:
    ~BatchRestoreJoin();

    void fail() noexcept;

private:
    friend class BatchCache;
    struct State;
    std::shared_ptr<State> state_;
    explicit BatchRestoreJoin(std::shared_ptr<State> state);
    void abandon() noexcept;
};

class BatchStorePairing {
public:
    BatchStorePairing();
    ~BatchStorePairing();

private:
    friend class BatchCache;
    struct State;
    std::shared_ptr<State> state_;
};

class BatchCache {
public:
    using Cancel = std::shared_ptr<std::atomic_bool>;
    using Keys = std::function<std::vector<std::shared_ptr<decoder::PrefixCacheKeys>>(
        std::string_view)>;
    using ReadDone = std::function<void(BatchCacheRead)>;

    BatchCache(std::unique_ptr<decoder::LmCacheClient> client, std::string identity, int lanes);
    BatchCache(const decoder::LmCacheConfig& base,
               std::vector<std::array<std::uint8_t, 16>> devices, std::string identity,
               int lanes);
    ~BatchCache();

    int world_size() const;

    void lookup(Keys keys, int score_start, bool draft, Cancel cancelled, ReadDone done);

    std::shared_future<bool> restore(
        const std::shared_ptr<BatchRestoreJoin>& join,
        const std::vector<std::shared_ptr<decoder::PrefixCacheKeys>>& keys,
        const decoder::PrefixDeviceTransfer& view, std::function<void()> done);
    std::shared_ptr<BatchRestoreJoin> begin_restore();

    bool reserve_store(std::shared_ptr<void>& reservation);
    bool store_capacity() const;
    bool stores_pending() const;

    void store(const std::shared_ptr<BatchStorePairing>& pairing,
               const std::vector<std::shared_ptr<decoder::PrefixCacheKeys>>& keys,
               const int* generated, int count, decoder::PrefixDeviceTransfer&& view,
               std::shared_ptr<void> reservation, std::shared_ptr<std::atomic_uint64_t> stores,
               std::function<void()> done);
    void wake_readers();
    void stop_restores() noexcept;
    void reset();
    LmCacheMetrics metrics() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}
