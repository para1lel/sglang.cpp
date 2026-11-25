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

struct LmCacheEndpointCoverage {
    int end = 0;
    bool draft = false;
};

// Full endpoint inventory behind lookup_prefix: every present GDN state record
// at or below min(gqa, max_restore), in descending endpoint order, each flagged
// for usable draft coverage at endpoint - 1.
struct LmCachePrefixDetail {
    int gqa = 0;
    int matched_gdn = 0;
    std::vector<LmCacheEndpointCoverage> endpoints;
};

struct LmCacheReceipt {
    std::array<std::uint64_t, 3> bytes{};
    std::uint32_t gqa_written_tokens = 0;
};

std::string default_lmcache_endpoint();
// Per-rank daemon socket of the default endpoint: sglang-lmcache.sock becomes
// sglang-lmcache-r<rank>.sock.
std::string rank_lmcache_endpoint(int rank);
// Same derivation applied to an explicit endpoint (the --lmcache-socket path).
std::string rank_lmcache_endpoint(const std::string& endpoint, int rank);

struct LmCacheConfig {
    std::string endpoint = default_lmcache_endpoint();
    std::string name_space = "sglang";
    int timeout_ms = 1000;
    // Ranked (TP2) dialect: detail lookups (op 9) and retrieve/store (ops 10
    // and 11) whose MXFP8 sections carry their KV head count. The default
    // false keeps the historical TP1 wire bytes and operations exactly, so a
    // TP1 client works against old and new daemons alike; a ranked client
    // requires a daemon that answers the ranked operations.
    bool native_v4 = false;
};

class LmCacheClient {
public:
    explicit LmCacheClient(LmCacheConfig config);

    const LmCacheConfig& config() const { return config_; }
    bool ready() const;
    std::vector<bool> lookup(const std::vector<LmCacheKey>& keys) const;
    LmCachePrefix lookup_prefix(const PrefixCacheKeys& keys, int max_restore, bool draft) const;
    LmCachePrefixDetail lookup_prefix_detail(const PrefixCacheKeys& keys, int max_restore,
                                             bool draft) const;

    // Encode store metadata while the prefix keys and transfer view are still stable.
    std::string prepare_store(const PrefixCacheKeys& keys,
                              const PrefixDeviceTransfer& transfer) const;

    // Keep exported allocations and the ready event alive until the operation returns.
    LmCacheStatus store(const PrefixCacheKeys& keys, const PrefixDeviceTransfer& transfer,
                        LmCacheReceipt* receipt = nullptr) const;
    LmCacheStatus store_prepared(std::string_view metadata,
                                 LmCacheReceipt* receipt = nullptr) const;
    // A readable stop_fd interrupts the GET. Keep it open through lease reconciliation.
    LmCacheStatus retrieve(const PrefixCacheKeys& keys, const PrefixDeviceTransfer& transfer,
                           LmCacheReceipt* receipt = nullptr, int stop_fd = -1) const;
    void reset() const;

private:
    LmCachePrefix lookup_prefix(std::string_view digests, int max_restore, bool draft) const;
    LmCachePrefixDetail lookup_prefix_detail(std::string_view digests, int max_restore,
                                             bool draft) const;
    LmCacheConfig config_;
};

}  // namespace sglang::decoder
