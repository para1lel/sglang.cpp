#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/runtime/cache/lmcache_client.h"
#include "sglang/runtime/cache/prefix_payload.h"

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>

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
};

struct BatchCacheRead {
    std::shared_ptr<decoder::PrefixCacheKeys> keys;
    decoder::PrefixCacheRead match;
};

class CacheResetUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class BatchCache {
public:
    using Cancel = std::shared_ptr<std::atomic_bool>;
    using Keys = std::function<std::shared_ptr<decoder::PrefixCacheKeys>(std::string_view)>;
    using ReadDone = std::function<void(BatchCacheRead)>;

    BatchCache(std::unique_ptr<decoder::LmCacheClient> client, std::string identity, int lanes);
    ~BatchCache();

    void lookup(Keys keys, int score_start, bool draft, Cancel cancelled, ReadDone done);
    std::shared_future<bool> restore(std::shared_ptr<const decoder::PrefixCacheKeys> keys,
                                     const decoder::PrefixDeviceTransfer& view,
                                     std::function<void()> done);
    bool reserve_store(std::shared_ptr<void>& reservation);
    bool store_capacity() const;
    bool stores_pending() const;
    void store(const decoder::PrefixCacheKeys& keys, decoder::PrefixDeviceTransfer&& view,
               std::shared_ptr<void> reservation, std::shared_ptr<std::atomic_uint64_t> stores,
               std::function<void()> done);
    void wake_readers();
    void reset();
    LmCacheMetrics metrics() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

}
