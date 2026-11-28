#pragma once

#include "sglang/runtime/cache/lmcache_client.h"
#include "sglang/runtime/cache/prefix_payload.h"

#include <atomic>
#include <string>
#include <string_view>
#include <vector>

namespace sglang::decoder {

struct LmCacheGroupStoreMetrics {
    std::uint64_t pending = 0, stores = 0, declined = 0, errors = 0, elapsed_ns = 0;
    std::uint64_t gqa_written_tokens = 0;
    std::array<std::uint64_t, 3> bytes{};
    std::vector<std::uint64_t> rank_stores;
    std::vector<std::array<std::uint64_t, 3>> rank_bytes;
};

class LmCacheGroup {
public:
  explicit LmCacheGroup(LmCacheClient client);
  LmCacheGroup(LmCacheConfig config, int world_size);

  int world_size() const;

  bool ready() const;
  bool ready(const std::vector<std::array<std::uint8_t, 16>>& devices) const;

  PrefixCacheRead lookup_prefix(const std::vector<const PrefixCacheKeys*>& keys, int max_restore,
                                bool draft) const;

  std::string prepare_store(int rank, const PrefixCacheKeys& keys,
                            const PrefixDeviceTransfer& transfer) const;
  LmCacheStatus store(int rank, const PrefixCacheKeys& keys, const PrefixDeviceTransfer& transfer,
                      LmCacheReceipt* receipt = nullptr) const;
  LmCacheStatus store_prepared(std::string_view metadata, LmCacheReceipt* receipt = nullptr) const;
  LmCacheStatus store(const std::vector<const PrefixCacheKeys*>& keys,
                      const std::vector<const PrefixDeviceTransfer*>& transfers,
                      LmCacheReceipt* receipt = nullptr,
                      std::vector<LmCacheReceipt>* per_rank = nullptr, int stop_fd = -1) const;
  LmCacheGroupStoreMetrics store_metrics() const;

  LmCacheStatus retrieve(const std::vector<const PrefixCacheKeys*>& keys,
                         const std::vector<const PrefixDeviceTransfer*>& transfers,
                         LmCacheReceipt* receipt, std::vector<LmCacheReceipt>* per_rank = nullptr,
                         int stop_fd = -1) const;

  void reset() const;

private:
  LmCacheStatus transfer_ranked(const std::vector<const PrefixCacheKeys*>& keys,
                                const std::vector<const PrefixDeviceTransfer*>& transfers,
                                LmCacheReceipt* receipt, std::vector<LmCacheReceipt>* per_rank,
                                int stop_fd, bool store) const;
  LmCacheStatus transfer_one(std::size_t rank, const PrefixCacheKeys& keys,
                             const PrefixDeviceTransfer& transfer, LmCacheReceipt& receipt,
                             int stop_fd, bool store) const;

  LmCacheClient client_;
  int world_size_ = 1;
  struct RankStoreCounters {
      std::atomic_uint64_t stores{0};
      std::array<std::atomic_uint64_t, 3> bytes{};
  };
  mutable std::vector<RankStoreCounters> rank_stores_;
  mutable std::atomic_uint64_t pending_{0}, stores_{0}, declined_{0}, errors_{0}, write_ns_{0};
  mutable std::atomic_uint64_t gqa_written_{0};
  mutable std::array<std::atomic_uint64_t, 3> written_bytes_{};
};

}
