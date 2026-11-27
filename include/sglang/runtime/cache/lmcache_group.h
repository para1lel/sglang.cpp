#pragma once

#include "sglang/runtime/cache/lmcache_client.h"
#include "sglang/runtime/cache/prefix_payload.h"

#include <memory>
#include <string_view>
#include <vector>

namespace sglang::decoder {

std::vector<LmCacheConfig> ranked_lmcache_configs(int world_size, int timeout_ms = 1000);
std::vector<LmCacheConfig> ranked_lmcache_configs(const LmCacheConfig& base, int world_size);

class LmCacheGroup {
public:
    explicit LmCacheGroup(std::vector<LmCacheConfig> configs);
    explicit LmCacheGroup(std::vector<std::unique_ptr<LmCacheClient>> clients);

    int world_size() const;
    const LmCacheClient& client(int rank) const;

    bool ready() const;

    PrefixCacheRead lookup_prefix(const std::vector<const PrefixCacheKeys*>& keys,
                                  int max_restore, bool draft) const;

    std::string prepare_store(int rank, const PrefixCacheKeys& keys,
                              const PrefixDeviceTransfer& transfer) const;
    LmCacheStatus store(int rank, const PrefixCacheKeys& keys,
                        const PrefixDeviceTransfer& transfer,
                        LmCacheReceipt* receipt = nullptr) const;
    LmCacheStatus store_prepared(int rank, std::string_view metadata,
                                 LmCacheReceipt* receipt = nullptr) const;

    LmCacheStatus retrieve(const std::vector<const PrefixCacheKeys*>& keys,
                           const std::vector<const PrefixDeviceTransfer*>& transfers,
                           LmCacheReceipt* receipt,
                           std::vector<LmCacheReceipt>* per_rank = nullptr,
                           int stop_fd = -1) const;

    void reset() const;

private:
    std::vector<std::unique_ptr<LmCacheClient>> clients_;
};

}
