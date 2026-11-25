#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sglang::decoder {

enum class LmCacheFamily { gqa = 1, gdn = 2, draft = 3 };
enum class LmCacheStatus { complete, miss, declined };

struct LmCacheKey {
    LmCacheFamily family;
    std::string identity;
};

struct LmCacheGpuAllocation {
    std::array<std::uint8_t, 64> handle;
    std::uint64_t bytes;
};

struct LmCacheGpuTransfer {
    std::array<std::uint8_t, 16> device;
    std::array<std::uint8_t, 64> ready_event;
    std::vector<LmCacheGpuAllocation> allocations;
};

class PrefixCacheKeys;
struct PrefixDeviceTransfer;

struct LmCachePrefix {
    int gqa = 0;
    int gdn = 0;
    int draft = 0;
    int matched_gdn = 0;
};

struct LmCacheReceipt {
    std::array<std::uint64_t, 3> bytes{};
    std::uint32_t gqa_written_tokens = 0;
};

std::string default_lmcache_endpoint();

struct LmCacheConfig {
    std::string endpoint = default_lmcache_endpoint();
    std::string name_space = "sglang";
    int timeout_ms = 1000;
};

class LmCacheClient {
public:
    explicit LmCacheClient(LmCacheConfig config);

    bool ready() const;
    std::vector<bool> lookup(const std::vector<LmCacheKey>& keys) const;
    LmCachePrefix lookup_prefix(const PrefixCacheKeys& keys, int max_restore, bool draft) const;

    // Encode store metadata while the prefix keys and transfer view are still stable.
    std::string prepare_store(const PrefixCacheKeys& keys,
                              const PrefixDeviceTransfer& transfer) const;

    // Keep exported allocations and the ready event alive until the operation returns.
    LmCacheStatus store(const PrefixCacheKeys& keys, const PrefixDeviceTransfer& transfer,
                        LmCacheReceipt* receipt = nullptr) const;
    LmCacheStatus store_prepared(std::string_view metadata,
                                 LmCacheReceipt* receipt = nullptr) const;
    LmCacheStatus retrieve(const PrefixCacheKeys& keys, const PrefixDeviceTransfer& transfer,
                           LmCacheReceipt* receipt = nullptr) const;
    void reset() const;

private:
    LmCachePrefix lookup_prefix(std::string_view digests, int max_restore, bool draft) const;
    LmCacheConfig config_;
};

}  // namespace sglang::decoder
