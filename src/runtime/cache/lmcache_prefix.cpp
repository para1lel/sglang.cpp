#include "sglang/runtime/cache/lmcache_client.h"

#include "sglang/runtime/cache/prefix_payload.h"

#include <algorithm>

namespace sglang::decoder {

LmCachePrefix LmCacheClient::lookup_prefix(const PrefixCacheKeys& keys, int max_restore,
                                           bool draft) const {
    const int limit = keys.cacheable_tokens();
    if (limit <= 0) return {};
    std::string digests;
    digests.reserve(std::size_t(limit) * 64);
    for (int end = 1; end <= limit; ++end) digests += keys.key(end);
    return lookup_prefix(digests, std::clamp(max_restore, 0, limit), draft);
}

}  // namespace sglang::decoder
