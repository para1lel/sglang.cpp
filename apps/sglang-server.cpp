#include "sglang/fp8_prefill_option.h"

#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/completions.h"
#include "sglang/server/http.h"
#include "sglang/server/prefix_controls.h"

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
    int prefix_cache_entries = 0;
    bool eager = false;
    bool trace_batches = false;
    bool batch_cuda_graphs = false;
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
        if (option == "--batch-cuda-graphs") {
            arguments.batch_cuda_graphs = true;
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
            const auto error = "Expected --max-active in [1, 32]";
            arguments.max_active = parse_integer(value, error);
            if (arguments.max_active == 0 ||
                arguments.max_active > sglang::decoder_batch_max_lanes) {
                throw std::invalid_argument(error);
            }
        } else if (option == "--prefix-cache-entries") {
            const auto error = "Expected --prefix-cache-entries in [0, 4]";
            arguments.prefix_cache_entries = parse_integer(value, error);
            if (arguments.prefix_cache_entries > sglang::decoder_batch_prefix_max_entries) {
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
    if (arguments.eager && (arguments.max_active != 1 || arguments.mtp_steps != 0)) {
        throw std::invalid_argument("--eager requires --max-active 1 and --mtp-steps 0");
    }
    if (arguments.batch_cuda_graphs && arguments.max_active == 1) {
        throw std::invalid_argument("--batch-cuda-graphs requires --max-active at least 2");
    }
    if (arguments.mtp_steps > 0 && arguments.max_active > sglang::decoder_batch_max_mtp_lanes) {
        throw std::invalid_argument("--max-active above 4 requires --mtp-steps 0");
    }
    if (arguments.prefix_cache_entries != 0 &&
        (arguments.eager || arguments.mtp_steps != 0 ||
         arguments.fp8_prefill != sglang::Fp8PrefillPolicy::exact)) {
        throw std::invalid_argument(
            "--prefix-cache-entries requires exact prefill and --mtp-steps 0. Omit --eager.");
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
            complete_request(connection, request->body, false);
        } else if (request->path == "/v1/chat/completions" && request->method == "POST") {
            complete_request(connection, request->body, true);
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
                                                    std::string_view body, bool chat) {
                        if (chat) {
                            complete_chat(connection, service, slot, tokenizer, body);
                        } else {
                            complete(connection, service, slot, tokenizer, body);
                        }
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
                     " [--max-active N] [--mtp-steps N] [--eager] [--batch-cuda-graphs]\n"
                     "                     [--fp8-prefill exact|tensor-core] [--trace-batches]\n"
                     "                     [--prefix-cache-entries 0..4]\n"
                     "Active requests: 1 by default, up to 32.\n"
                     "MTP steps: 0 by default, up to 4 with at most 4 active requests.\n"
                     "Batch graphs: ordinary decode steps at widths 2-4. Other steps use eager.\n"
                     "Eager decode: one active request with MTP off.\n"
                     "Prefix cache: disabled by default. Requires exact prefill and MTP off.\n"
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
        if (arguments.max_active == 1 && arguments.prefix_cache_entries == 0) {
            serve_connections(listener, [&](const Socket& connection,
                                            std::string_view body, bool chat) {
                if (chat) {
                    complete_chat(connection, decoder, tokenizer, body, arguments.trace_batches,
                                  arguments.mtp_steps, arguments.eager);
                } else {
                    complete(connection, decoder, tokenizer, body, arguments.trace_batches,
                             arguments.mtp_steps, arguments.eager);
                }
            });
        } else {
            BatchService service(decoder, arguments.trace_batches, arguments.batch_cuda_graphs,
                                 arguments.mtp_steps, arguments.max_active,
                                 arguments.prefix_cache_entries);
            if (arguments.prefix_cache_entries != 0) {
                serve_prefix_connections(listener, service, arguments.max_active,
                    [&](const Socket& connection, int slot, std::string_view body, bool chat) {
                        if (chat) {
                            complete_chat(connection, service, slot, tokenizer, body);
                        } else {
                            complete(connection, service, slot, tokenizer, body);
                        }
                    });
            } else {
                serve_batch(listener, service, tokenizer, arguments.max_active);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "sglang-server: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
