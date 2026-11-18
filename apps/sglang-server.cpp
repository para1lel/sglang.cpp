#include "sglang/decoder.h"
#include "sglang/server/completions.h"
#include "sglang/server/http.h"

#include <charconv>
#include <exception>
#include <iostream>
#include <string_view>

namespace {

using namespace sglang::server;

struct Arguments {
    std::filesystem::path checkpoint = "/data0/shared/Qwen3.8-27B-FP8";
    int device = 0;
    int port = 17384;
};

int parse_integer(std::string_view value, const char* error) {
    int number = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || number < 0) {
        throw std::invalid_argument(error);
    }
    return number;
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (++index == argc) {
            throw std::invalid_argument("Missing value for " + std::string(option));
        }
        const std::string_view value = argv[index];
        if (option == "--checkpoint" && !value.empty()) {
            arguments.checkpoint = value;
        } else if (option == "--device") {
            arguments.device = parse_integer(value, "Expected a nonnegative CUDA device ordinal");
        } else if (option == "--port") {
            arguments.port = parse_integer(value, "Expected a TCP port in [1, 65535]");
            if (arguments.port == 0 || arguments.port > 65535) {
                throw std::invalid_argument("Expected a TCP port in [1, 65535]");
            }
        } else {
            throw std::invalid_argument("Invalid argument " + std::string(option));
        }
    }
    return arguments;
}

void serve_connection(const Socket& connection, const sglang::Decoder& decoder) {
    try {
        const auto request = read_request(connection);
        if (!request) {
            return;
        }
        if (request->path == "/health" && request->method == "GET" && request->body.empty()) {
            write_json(connection, "200 OK", "{}");
        } else if (request->path == "/v1/completions" && request->method == "POST") {
            complete(connection, decoder, request->body);
        } else {
            write_error(connection, "404 Not Found",
                        "Expected GET /health or POST /v1/completions");
        }
    } catch (const HttpError& error) {
        write_error(connection, error.status, error.what());
    } catch (const std::invalid_argument& error) {
        write_error(connection, "400 Bad Request", error.what());
    } catch (const std::exception& error) {
        std::cerr << "Request failed: " << error.what() << '\n';
        write_error(connection, "500 Internal Server Error", "Native request failed");
    }
}

}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: sglang-server [--checkpoint PATH] [--device ORDINAL] [--port PORT]\n"
                     "Serves on 127.0.0.1 only. Default port: 17384.\n";
        return 0;
    }
    try {
        const auto arguments = parse_arguments(argc, argv);
        install_shutdown_handlers();
        const sglang::Decoder decoder(arguments.checkpoint, arguments.device);
        if (stopping()) {
            return 0;
        }
        const auto listener = listen_loopback(arguments.port);
        std::cerr << "Listening on http://127.0.0.1:" << arguments.port << '\n';
        while (!stopping()) {
            const auto connection = accept_connection(listener);
            if (connection.get() >= 0) {
                serve_connection(connection, decoder);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "sglang-server: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
