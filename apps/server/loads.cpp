#include "sglang/server/loads.h"

#include "sglang/json.h"
#include "sglang/mxfp8_gqa.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/http.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>

namespace sglang::server {
namespace {

std::string_view trim(std::string_view text) {
    constexpr std::string_view whitespace = " \t\n\r\f\v";
    const auto first = text.find_first_not_of(whitespace);
    if (first == std::string_view::npos)
        return {};
    return text.substr(first, text.find_last_not_of(whitespace) - first + 1);
}

struct LoadFilters {
    std::optional<std::string> rank;
    std::string include;
    bool prometheus = false;
};

LoadFilters read_filters(std::string_view query) {
    LoadFilters filters;
    while (!query.empty()) {
        const auto end = query.find('&');
        const auto field = query.substr(0, end);
        const auto equals = field.find('=');
        const auto key = decode_url(field.substr(0, equals), true);
        const auto value = equals == std::string_view::npos
            ? std::string_view{} : field.substr(equals + 1);
        if (key == "dp_rank") {
            filters.rank = decode_url(value, true);
        } else if (key == "include") {
            filters.include = decode_url(value, true);
        } else if (key == "format") {
            filters.prometheus = decode_url(value, true) == "prometheus";
        }
        query.remove_prefix(end == std::string_view::npos ? query.size() : end + 1);
    }
    return filters;
}

std::optional<bool> selects_rank_zero(std::string_view input) {
    input = trim(input);
    if (!input.empty() && (input.front() == '+' || input.front() == '-'))
        input.remove_prefix(1);
    if (input.empty() || input.front() < '0' || input.front() > '9')
        return std::nullopt;

    bool zero = true;
    bool fractional = false;
    bool needs_digit = false;
    for (const char symbol : input) {
        if (symbol >= '0' && symbol <= '9') {
            if (fractional && symbol != '0')
                return std::nullopt;
            zero &= symbol == '0';
            needs_digit = false;
        } else if (symbol == '_' && !needs_digit && !fractional) {
            needs_digit = true;
        } else if (symbol == '.' && !needs_digit && !fractional) {
            fractional = true;
            needs_digit = true;
        } else {
            return std::nullopt;
        }
    }
    return needs_digit ? std::nullopt : std::optional<bool>{zero};
}

std::optional<bool> includes_queues(std::string_view include) {
    if (include.empty())
        return true;
    bool queues = false;
    bool valid = true;
    constexpr std::array<std::string_view, 5> other_sections{
        "core", "memory", "spec", "lora", "disagg"};
    for (;;) {
        const auto end = include.find(',');
        const auto section = trim(include.substr(0, end));
        if (section == "all")
            return true;
        if (section == "queues") {
            queues = true;
        } else if (std::find(other_sections.begin(), other_sections.end(), section) ==
                   other_sections.end()) {
            valid = false;
        }
        if (end == std::string_view::npos)
            return valid ? std::optional<bool>{queues} : std::nullopt;
        include.remove_prefix(end + 1);
    }
}

struct LoadMetric {
    std::string_view name;
    std::string value;
};

void append_metric(std::string& body, const LoadMetric& metric) {
    body += ",\"";
    body += metric.name;
    body += "\":";
    body += metric.value;
}

void append_gauge(std::string& body, const LoadMetric& metric,
                  std::string_view section = std::string_view{}) {
    std::string name = "sglang_";
    name += section;
    name += metric.name;
    body += "# TYPE " + name + " gauge\n";
    body += name + "{dp_rank=\"0\"} " + metric.value + '\n';
}

void write_snapshot(const Socket& connection, LoadSnapshot snapshot, bool present, bool queues,
                    bool prometheus) {
    const auto now = std::chrono::system_clock::now();
    const std::chrono::duration<double> epoch = now.time_since_epoch();
    const std::array<LoadMetric, 4> core{{
        {"timestamp", std::to_string(epoch.count())},
        {"num_running_reqs", std::to_string(snapshot.running)},
        {"num_waiting_reqs", std::to_string(snapshot.waiting)},
        {"max_running_requests", std::to_string(snapshot.capacity)},
    }};
    const LoadMetric waiting{"waiting", std::to_string(snapshot.waiting)};
    if (prometheus) {
        std::string body;
        if (present) {
            for (const auto& metric : core)
                append_gauge(body, metric);
            if (queues)
                append_gauge(body, waiting, "queues_");
        } else {
            body = '\n';
        }
        write_raw_response(connection, "200 OK", body, "text/plain; version=0.0.4; charset=utf-8");
        return;
    }

    const auto seconds = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    std::array<char, 26> timestamp{};
    gmtime_r(&seconds, &utc);
    std::strftime(timestamp.data(), timestamp.size(), "%Y-%m-%dT%H:%M:%S+00:00", &utc);
    std::string body = R"({"timestamp":)";
    json::append_quoted(body, timestamp.data());
    body += R"(,"version":null,"accelerator":)";
    if (snapshot.accelerator.empty()) {
        body += "null";
    } else {
        json::append_quoted(body, snapshot.accelerator);
    }
    body += R"(,"num_accelerators":1,"loads":[)";
    if (present) {
        body += R"({"dp_rank":0)";
        for (const auto& metric : core)
            append_metric(body, metric);
        if (queues)
            body += R"(,"queues":{"waiting":)" + waiting.value + '}';
        body += '}';
    }
    body += "]}";
    write_json(connection, "200 OK", body);
}

}

void write_loads(const Socket& connection, const HttpRequest& request, LoadSnapshot snapshot) {
    if (request.method != "GET") {
        write_get_only(connection, request.method == "HEAD");
        return;
    }
    const auto filters = read_filters(request.query);
    const auto present =
        filters.rank ? selects_rank_zero(*filters.rank) : std::optional<bool>{true};
    if (!present) {
        std::string message =
            "query.dp_rank: Input should be a valid integer, unable to parse string as an integer"
            " (input: ";
        json::append_quoted(message, *filters.rank);
        message += ')';
        write_tokenization_error(connection, "400 Bad Request", "Bad Request", message);
        return;
    }
    const auto queues = *present ? includes_queues(filters.include) : std::optional<bool>{false};
    if (!queues) {
        write_raw_response(connection, "500 Internal Server Error", "Internal Server Error",
                           "text/plain; charset=utf-8");
        return;
    }
    write_snapshot(connection, snapshot, *present, *queues, filters.prometheus);
}

std::string format_metrics(const BatchMetrics& metrics, int running, std::size_t waiting) {
    std::string text = "# Native sglang.cpp metrics with frozen W3-prefix compatibility names\n";
    const auto emit = [&text](std::string_view name, auto value) {
        text += name;
        text += "{engine=\"0\"} ";
        text += std::to_string(value);
        text += '\n';
    };
    const int capacity = metrics.cache_capacity * (decoder_batch_prefix_max_tokens /
                                                   mxfp8_page_size);
    emit("vllm:num_requests_running", running);
    emit("vllm:num_requests_waiting", waiting);
    emit("vllm:kv_cache_usage_perc", capacity ? double(metrics.cache.pages) / capacity : 0.0);
    emit("vllm:prompt_tokens_total", metrics.prompt_tokens);
    emit("vllm:generation_tokens_total", metrics.generation_tokens);
    emit("vllm:external_prefix_cache_hits_total", 0);
    emit("vllm:prefix_cache_hits_total", metrics.cache.cached_tokens);
    emit("vllm:request_prefill_kv_computed_tokens_count", metrics.prefill_requests);
    emit("vllm:request_prefill_kv_computed_tokens_sum", metrics.cache.prefill_tokens);
    emit("sglang_cpp:prefix_cache_entries", metrics.cache.entries);
    emit("sglang_cpp:prefix_cache_pages", metrics.cache.pages);
    emit("sglang_cpp:prefix_cache_device_bytes", metrics.cache.device_bytes);
    emit("sglang_cpp:prefix_cache_hits_total", metrics.cache.hits);
    emit("sglang_cpp:prefix_cache_stores_total", metrics.cache.stores);
    emit("sglang_cpp:prefix_cache_evictions_total", metrics.cache.evictions);
    emit("sglang_cpp:prefix_cache_draft_device_bytes", metrics.cache.draft_device_bytes);
    emit("sglang_cpp:prefix_cache_draft_cached_rows_total", metrics.cache.draft_cached_rows);
    emit("sglang_cpp:prefix_cache_draft_prefill_rows_total", metrics.cache.draft_prefill_rows);
    emit("sglang_cpp:prefix_cache_draft_misses_total", metrics.cache.draft_misses);
    return text;
}

}
