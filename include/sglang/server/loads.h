#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace sglang::server {

class Socket;
struct BatchMetrics;
struct HttpRequest;

inline constexpr std::string_view loads_route = "/v1/loads";

struct LoadSnapshot {
    int running;
    std::size_t waiting;
    int capacity;
    std::string_view accelerator;
};

void write_loads(const Socket& connection, const HttpRequest& request, LoadSnapshot snapshot);
std::string format_metrics(const BatchMetrics& metrics, int running, std::size_t waiting);

}
