#include "sglang/server/flush_cache.h"

#include "sglang/json.h"
#include "sglang/server/http.h"
#include "sglang/utf8.h"

#include <algorithm>
#include <cstdlib>

namespace sglang::server {
namespace {

constexpr std::string_view flush_path = "/flush_cache";

std::optional<std::string> timeout_input(std::string_view query) {
    std::optional<std::string> input;
    while (!query.empty()) {
        const auto end = query.find('&');
        const auto field = query.substr(0, end);
        const auto equals = field.find('=');
        if (decode_url(field.substr(0, equals), true) == "timeout") {
            input = equals == std::string_view::npos
                ? std::string{} : decode_url(field.substr(equals + 1), true);
        }
        query.remove_prefix(end == std::string_view::npos ? query.size() : end + 1);
    }
    return input;
}

bool unicode_space(char32_t scalar) {
    return (scalar >= U'\t' && scalar <= U'\r') || scalar == U' ' ||
           scalar == U'\u0085' || scalar == U'\u00a0' || scalar == U'\u1680' ||
           (scalar >= U'\u2000' && scalar <= U'\u200a') || scalar == U'\u2028' ||
           scalar == U'\u2029' || scalar == U'\u202f' || scalar == U'\u205f' ||
           scalar == U'\u3000';
}

std::optional<double> parse_timeout(std::string_view input) {
    const auto scalars = read_utf8(input);
    const auto first = std::find_if_not(scalars.begin(), scalars.end(), unicode_space);
    const auto last = std::find_if_not(scalars.rbegin(), scalars.rend(), unicode_space).base();
    if (first >= last || *first == U'_' || last[-1] == U'_') {
        return std::nullopt;
    }
    std::string number;
    number.reserve(last - first);
    for (auto scalar = first; scalar != last; ++scalar) {
        if (*scalar == U'_' && scalar[-1] != U'_') {
            continue;
        }
        if (*scalar >= U'A' && *scalar <= U'Z') {
            number += static_cast<char>(*scalar + (U'a' - U'A'));
        } else if (*scalar < 128) {
            number += static_cast<char>(*scalar);
        } else {
            return std::nullopt;
        }
    }
    std::string_view magnitude(number);
    if (magnitude.front() == '-' || magnitude.front() == '+') {
        magnitude.remove_prefix(1);
    }
    if (magnitude != "inf" && magnitude != "infinity" && magnitude != "nan" &&
        magnitude.find_first_not_of("0123456789.e+-") != std::string_view::npos) {
        return std::nullopt;
    }
    char* end = nullptr;
    const double timeout = std::strtod(number.c_str(), &end);
    if (end == number.c_str() || end != number.c_str() + number.size()) {
        return std::nullopt;
    }
    return timeout;
}

void write_invalid_timeout(const Socket& connection, std::string_view input, bool parsed) {
    std::string message = "query.timeout: ";
    message += parsed ? "Input should be greater than or equal to 0"
                      : "Input should be a valid number, unable to parse string as a number";
    message += " (input: ";
    json::append_quoted(message, input);
    message += ')';
    std::string body = R"({"object":"error","message":)";
    json::append_quoted(body, message);
    body += R"(,"type":"Bad Request","param":null,"code":400})";
    write_json(connection, "400 Bad Request", body);
}

}

FlushCacheRequest read_flush_cache(const Socket& connection, const HttpRequest& request) {
    if (request.route_path() != flush_path) {
        return {};
    }
    if (request.method != "GET" && request.method != "POST") {
        write_method_not_allowed(connection, "GET, POST", request.method == "HEAD");
        return {true, std::nullopt};
    }
    const auto input = timeout_input(request.query);
    const auto timeout = input ? parse_timeout(*input) : 0.0;
    if (!timeout || !(*timeout >= 0.0)) {
        write_invalid_timeout(connection, *input, timeout.has_value());
        return {true, std::nullopt};
    }
    return {true, timeout};
}

void write_flush_cache(const Socket& connection, FlushCacheResult result) {
    switch (result) {
        case FlushCacheResult::flushed:
            write_raw_response(connection, "200 OK",
                "Cache flushed.\nPlease check backend logs for more details. "
                "(When there are running or waiting requests, the operation will not be "
                "performed.)\n");
            break;
        case FlushCacheResult::busy:
            write_raw_response(connection, "400 Bad Request", "Flush cache failed.\n");
            break;
        case FlushCacheResult::timed_out:
            write_raw_response(connection, "400 Bad Request", "Timed out waiting for idle state.");
            break;
    }
}

}
