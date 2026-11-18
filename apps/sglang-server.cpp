#include "fp8_prefill_option.h"

#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/completions.h"
#include "sglang/server/http.h"

#include <charconv>
#include <exception>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>

namespace {

using namespace sglang::server;

struct Arguments {
    std::filesystem::path checkpoint = "/data0/shared/Qwen3.8-27B-FP8";
    sglang::Fp8PrefillPolicy fp8_prefill = sglang::Fp8PrefillPolicy::exact;
    int device = 0;
    int port = 17384;
    int max_active = 1;
    int mtp_steps = 0;
    bool eager = false;
    bool trace_batches = false;
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
        if (option == "--trace-batches") {
            arguments.trace_batches = true;
            continue;
        }
        if (option == "--eager") {
            arguments.eager = true;
            continue;
        }
        if (++index == argc) {
            throw std::invalid_argument("Missing value for " + std::string(option));
        }
        const std::string_view value = argv[index];
        if (option == "--checkpoint" && !value.empty()) {
            arguments.checkpoint = value;
        } else if (option == "--fp8-prefill") {
            arguments.fp8_prefill = sglang::cli::parse_fp8_prefill(value);
        } else if (option == "--device") {
            arguments.device = parse_integer(value, "Expected a nonnegative CUDA device ordinal");
        } else if (option == "--port") {
            arguments.port = parse_integer(value, "Expected a TCP port in [1, 65535]");
            if (arguments.port == 0 || arguments.port > 65535) {
                throw std::invalid_argument("Expected a TCP port in [1, 65535]");
            }
        } else if (option == "--max-active") {
            const auto error = "Expected --max-active in [1, 4]";
            arguments.max_active = parse_integer(value, error);
            if (arguments.max_active == 0 ||
                arguments.max_active > sglang::decoder_batch_max_lanes) {
                throw std::invalid_argument(error);
            }
        } else if (option == "--mtp-steps") {
            const auto error = "Expected --mtp-steps in [0, 4]";
            arguments.mtp_steps = parse_integer(value, error);
            if (arguments.mtp_steps > sglang::decoder_max_mtp_steps) {
                throw std::invalid_argument(error);
            }
        } else {
            throw std::invalid_argument("Invalid argument " + std::string(option));
        }
    }
    if (arguments.mtp_steps != 0 && arguments.max_active != 1) {
        throw std::invalid_argument("--mtp-steps requires --max-active 1");
    }
    if (arguments.eager && (arguments.max_active != 1 || arguments.mtp_steps != 0)) {
        throw std::invalid_argument("--eager requires --max-active 1 and --mtp-steps 0");
    }
    return arguments;
}

template <typename Complete>
void serve_connection(const Socket& connection, const Complete& complete_request) {
    try {
        const auto request = read_request(connection);
        if (!request) {
            return;
        }
        if (request->path == "/health" && request->method == "GET" && request->body.empty()) {
            write_json(connection, "200 OK", "{}");
        } else if (request->path == "/v1/completions" && request->method == "POST") {
            complete_request(connection, request->body);
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

template <typename Complete>
void serve_connections(const Socket& listener, const Complete& complete_request) {
    while (!stopping()) {
        const auto connection = accept_connection(listener);
        if (connection.get() >= 0) {
            serve_connection(connection, complete_request);
        }
    }
}

void serve_batch(const Socket& listener, BatchService& service,
                 const CompletionTokenizer& tokenizer, int max_active) {
    std::vector<std::thread> workers;
    workers.reserve(max_active);
    std::mutex failure_mutex;
    std::exception_ptr failure;
    const auto stop_on_error = [&] {
        const std::lock_guard lock(failure_mutex);
        if (!failure) {
            failure = std::current_exception();
        }
        stop_serving();
    };
    try {
        for (int slot = 0; slot < max_active; ++slot) {
            workers.emplace_back([&, slot] {
                try {
                    serve_connections(listener, [&](const Socket& connection,
                                                    std::string_view body) {
                        complete(connection, service, slot, tokenizer, body);
                    });
                } catch (...) {
                    stop_on_error();
                }
            });
        }
    } catch (...) {
        stop_on_error();
    }
    for (auto& worker : workers) {
        worker.join();
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
}

}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: sglang-server [--checkpoint PATH] [--device ORDINAL] [--port PORT]"
                     " [--max-active N] [--mtp-steps N] [--eager]\n"
                     "                     [--fp8-prefill exact|tensor-core] [--trace-batches]\n"
                     "Active requests: 1 by default, up to 4.\n"
                     "MTP steps: 0 by default, up to 4 with one active request.\n"
                     "Eager decode: one active request with MTP off.\n"
                     "Serves on 127.0.0.1 only. Default port: 17384.\n"
                  << sglang::cli::fp8_prefill_usage;
        return 0;
    }
    try {
        const auto arguments = parse_arguments(argc, argv);
        install_shutdown_handlers();
        const sglang::Decoder decoder(arguments.checkpoint, arguments.device,
                                      arguments.mtp_steps != 0, arguments.fp8_prefill);
        if (stopping()) {
            return 0;
        }
        const auto listener = listen_loopback(arguments.port);
        std::cerr << "Listening on http://127.0.0.1:" << arguments.port << '\n';
        const CompletionTokenizer tokenizer(arguments.checkpoint);
        if (arguments.max_active == 1) {
            serve_connections(listener, [&](const Socket& connection, std::string_view body) {
                complete(connection, decoder, tokenizer, body, arguments.trace_batches,
                         arguments.mtp_steps, arguments.eager);
            });
        } else {
            BatchService service(decoder, arguments.trace_batches);
            serve_batch(listener, service, tokenizer, arguments.max_active);
        }
    } catch (const std::exception& error) {
        std::cerr << "sglang-server: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
