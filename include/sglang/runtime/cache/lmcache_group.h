#pragma once

#include "sglang/runtime/cache/lmcache_client.h"
#include "sglang/runtime/cache/prefix_payload.h"

#include <memory>
#include <string_view>
#include <vector>

namespace sglang::decoder {

/// Default per-rank daemon configuration: socket <default>-r<rank>.sock and
/// namespace sglang-r<rank>, matching tools/lmcache/serve.sh --rank. A world of
/// one keeps the historical TP1 endpoint and namespace.
std::vector<LmCacheConfig> ranked_lmcache_configs(int world_size, int timeout_ms = 1000);
// Same derivation from an explicitly configured endpoint and namespace (the
// --lmcache-socket / --lmcache-namespace path): the per-rank suffixes are
// appended to what was configured instead of the defaults.
std::vector<LmCacheConfig> ranked_lmcache_configs(const LmCacheConfig& base, int world_size);

/// One prefix-cache client per rank daemon, presented as a single composite
/// cache for a sharded decoder world. Keys and device transfers stay
/// rank-local (each rank stores its own KV shard and GDN state under its own
/// layout identity); the group only coordinates coverage and completion:
///
///   - lookup_prefix intersects per-rank GDN endpoint inventories and picks
///     the largest endpoint present on every rank, so a record missing on one
///     rank never silently truncates the other (the min-of-maxima trap);
///   - retrieve reports complete only when every rank completed, giving the
///     batch layer one atomic signal for its joint cold fallback;
///   - rank 0 is the MTP draft lane; draft coverage is required there only.
class LmCacheGroup {
public:
    explicit LmCacheGroup(std::vector<LmCacheConfig> configs);
    // Takes ownership of already-connected rank clients; a single client is
    // the TP1 passthrough (identical operations on the wire).
    explicit LmCacheGroup(std::vector<std::unique_ptr<LmCacheClient>> clients);

    int world_size() const { return static_cast<int>(clients_.size()); }
    const LmCacheClient& client(int rank) const { return *clients_.at(rank); }

    bool ready() const;

    // keys[rank] was built with rank's PrefixCacheLayout. The returned read is
    // the single composite extent every rank restores to.
    PrefixCacheRead lookup_prefix(const std::vector<const PrefixCacheKeys*>& keys,
                                  int max_restore, bool draft) const;

    std::string prepare_store(int rank, const PrefixCacheKeys& keys,
                              const PrefixDeviceTransfer& transfer) const;
    LmCacheStatus store(int rank, const PrefixCacheKeys& keys,
                        const PrefixDeviceTransfer& transfer,
                        LmCacheReceipt* receipt = nullptr) const;
    LmCacheStatus store_prepared(int rank, std::string_view metadata,
                                 LmCacheReceipt* receipt = nullptr) const;

    // Joint GPU transfer. Returns complete only if every rank completed, miss
    // if any rank missed, declined if any rank declined. When receipt is set,
    // per-family bytes sum the rank shards while gqa_written_tokens counts the
    // restored tokens once (not once per rank). A transport error on one rank
    // lets the other ranks settle first, then rethrows: the caller treats it
    // as a joint failure and resets every rank cold.
    LmCacheStatus retrieve(const std::vector<const PrefixCacheKeys*>& keys,
                           const std::vector<const PrefixDeviceTransfer*>& transfers,
                           LmCacheReceipt* receipt = nullptr, int stop_fd = -1) const;

    // Reset every rank's namespace, in rank order.
    void reset() const;

private:
    std::vector<std::unique_ptr<LmCacheClient>> clients_;
};

}  // namespace sglang::decoder
