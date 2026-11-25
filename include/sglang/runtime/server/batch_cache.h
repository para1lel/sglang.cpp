#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/runtime/cache/lmcache_client.h"
#include "sglang/runtime/cache/lmcache_group.h"
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
    // Ranked worlds: number of per-rank daemons and physical stores per rank
    // (one per completed rank write). `stores` above counts one LOGICAL
    // checkpoint store, credited only when every rank's write completed.
    int world_size = 1;
    std::vector<std::uint64_t> rank_stores;
    // Per-rank completions: successful joint restores credited to every
    // participating rank lane, and per-family transfer bytes from each
    // rank's actual daemon receipts. Empty (absent) for TP1.
    std::vector<std::uint64_t> rank_restores;
    std::vector<std::array<std::uint64_t, 3>> rank_read_bytes;
    std::vector<std::array<std::uint64_t, 3>> rank_written_bytes;
};

struct BatchCacheRead {
    // One key set per rank, each built with that rank's PrefixCacheLayout.
    std::vector<std::shared_ptr<decoder::PrefixCacheKeys>> keys;
    decoder::PrefixCacheRead match;
};

class CacheResetUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// One in-flight joint restore across the rank lanes of a single request. The
/// decoder invokes restore() once per lane with the lane's own exported view;
/// the cache dispatches by device UUID, and the shared future resolves only
/// after every rank's transfer settled (jointly complete or jointly cold).
class BatchRestoreJoin {
public:
    ~BatchRestoreJoin();

    // Breaks a join that will never complete (a lane failed to register its
    // view): the shared future settles with an exception so no decoder
    // waiter can hang. No-op once the joint transfer was submitted — an
    // in-flight GET owns its settlement and the borrowed rank buffers.
    void fail() noexcept;

private:
    friend class BatchCache;
    struct State;
    std::shared_ptr<State> state_;
    explicit BatchRestoreJoin(std::shared_ptr<State> state) : state_(std::move(state)) {}
    // Settles a join whose worker was submitted but never admitted to the
    // queue (admission threw): the worker provably cannot run, so this
    // cannot race an in-flight transfer. Cache-internal: only BatchCache's
    // admission failure path may call it.
    void abandon() noexcept;
};

/// Per-request store pairing across rank lanes. Each lane enqueues its
/// checkpoints in order, so a lane's k-th store joins the other lanes' k-th
/// store: the pairing holds the checkpoint's shared reservation, exported
/// views, and acknowledgment until every rank's store settled, and its
/// logical accounting (stores / gqa_written_tokens) completes only then.
/// Destroyed with the request; incomplete pairs release their reservations.
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

    // TP1: one daemon, historical wire bytes.
    BatchCache(std::unique_ptr<decoder::LmCacheClient> client, std::string identity, int lanes);
    // Ranked (TP2): one daemon per rank, derived from the base configuration;
    // devices[rank] is that rank's CUDA device UUID, used to dispatch each
    // lane's exported view to its rank.
    BatchCache(const decoder::LmCacheConfig& base,
               std::vector<std::array<std::uint8_t, 16>> devices, std::string identity,
               int lanes);
    ~BatchCache();

    int world_size() const;

    void lookup(Keys keys, int score_start, bool draft, Cancel cancelled, ReadDone done);

    // Per-lane restore registration. keys is the per-rank key set of one
    // request; view must have been exported on the calling rank's device.
    std::shared_future<bool> restore(
        const std::shared_ptr<BatchRestoreJoin>& join,
        const std::vector<std::shared_ptr<decoder::PrefixCacheKeys>>& keys,
        const decoder::PrefixDeviceTransfer& view, std::function<void()> done);
    // Creates the join for a request's first lane; harmless to call per lane.
    std::shared_ptr<BatchRestoreJoin> begin_restore();

    bool reserve_store(std::shared_ptr<void>& reservation);
    bool store_capacity() const;
    bool stores_pending() const;

    // Per-lane store: dispatches by view device UUID, appends the generated
    // tokens to that rank's keys, and pairs the lane's checkpoint with the
    // other lanes' through `pairing`. The shared reservation is held until
    // every rank's store settled; `stores` advances once per completed
    // physical rank store (a world of W advances it W times per checkpoint),
    // while the logical stores metric and gqa_written_tokens are credited
    // once per checkpoint only when all ranks completed. Terminal and
    // periodic checkpoints take the same path.
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

}  // namespace sglang::server
