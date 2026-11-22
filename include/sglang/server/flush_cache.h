#pragma once

#include <optional>

namespace sglang::server {

struct HttpRequest;
class Socket;

struct FlushCacheRequest {
    bool matched = false;
    std::optional<double> timeout;
};

enum class FlushCacheResult { flushed, busy, timed_out };

FlushCacheRequest read_flush_cache(const Socket& connection, const HttpRequest& request);
void write_flush_cache(const Socket& connection, FlushCacheResult result);

}
