#pragma once

#include "sglang/runtime/cache/lmcache_client.h"
#include "sglang/runtime/cache/prefix_payload.h"

#include <string>
#include <string_view>
#include <vector>

namespace sglang::decoder {

class LmCacheGroup {
public:
  explicit LmCacheGroup(LmCacheClient client);
  LmCacheGroup(LmCacheConfig config, int world_size);

  int world_size() const;

  bool ready() const;

  PrefixCacheRead lookup_prefix(const std::vector<const PrefixCacheKeys*>& keys, int max_restore,
                                bool draft) const;

  std::string prepare_store(int rank, const PrefixCacheKeys& keys,
                            const PrefixDeviceTransfer& transfer) const;
  LmCacheStatus store(int rank, const PrefixCacheKeys& keys, const PrefixDeviceTransfer& transfer,
                      LmCacheReceipt* receipt = nullptr) const;
  LmCacheStatus store_prepared(std::string_view metadata, LmCacheReceipt* receipt = nullptr) const;

  LmCacheStatus retrieve(const std::vector<const PrefixCacheKeys*>& keys,
                         const std::vector<const PrefixDeviceTransfer*>& transfers,
                         LmCacheReceipt* receipt, std::vector<LmCacheReceipt>* per_rank = nullptr,
                         int stop_fd = -1) const;

  void reset() const;

private:
  LmCacheClient client_;
  int world_size_ = 1;
};

}
