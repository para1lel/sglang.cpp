#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace sglang::decoder {

enum class LmCacheFamily { gqa = 1, gdn = 2, draft = 3 };

struct LmCacheKey {
    LmCacheFamily family;
    std::string identity;
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
    std::unique_ptr<std::byte[]> get(const LmCacheKey& key, std::size_t expected_bytes) const;
    bool put(const LmCacheKey& key, const std::byte* data, std::size_t size) const;
    void reset() const;

private:
    LmCacheConfig config_;
};

}  // namespace sglang::decoder
