#include "sglang/fp8_prefill_option.h"

#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/completions.h"
#include "sglang/server/http.h"
#include "sglang/server/prefix_controls.h"
#include "sglang/server/vision_completion.h"

#include <charconv>
#include <exception>
#include <iostream>
#include <string_view>

namespace {

using namespace sglang::server;

struct Arguments {
    std::filesystem::path checkpoint = "/data0/shared/Qwen3.8-27B-FP8";
    sglang::Fp8PrefillPolicy fp8_prefill = sglang::Fp8PrefillPolicy::exact;
    SamplingDefaults sampling = SamplingDefaults::model;
    ChatParsers chat_parsers;
    int device = 0;
    int port = 17384;
    int max_active = sglang::decoder_batch_max_lanes;
    int max_num_batched_tokens = sglang::decoder_batch_max_step_tokens;
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

int parse_target_rows(std::string_view value) {
    const auto error = "Expected --max-num-batched-tokens in [1, 32768]";
    const int rows = parse_integer(value, error);
    if (rows == 0 || rows > sglang::decoder_batch_max_step_tokens) {
        throw std::invalid_argument(error);
    }
    return rows;
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
        } else if (option == "--sampling-defaults" && value == "model") {
            arguments.sampling = SamplingDefaults::model;
        } else if (option == "--sampling-defaults" && value == "openai") {
            arguments.sampling = SamplingDefaults::openai;
        } else if (option == "--reasoning-parser") {
            if (value != "qwen3") {
                throw std::invalid_argument("Expected --reasoning-parser qwen3");
            }
            arguments.chat_parsers.qwen3_reasoning = true;
        } else if (option == "--tool-call-parser") {
            if (value != "qwen3_coder") {
                throw std::invalid_argument("Expected --tool-call-parser qwen3_coder");
            }
            arguments.chat_parsers.qwen3_coder_tools = true;
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
        } else if (option == "--max-num-batched-tokens") {
            arguments.max_num_batched_tokens = parse_target_rows(value);
        } else if (option == "--prefix-cache-entries") {
            const auto error = "Expected --prefix-cache-entries in [0, 4]";
            arguments.prefix_cache_entries = parse_integer(value, error);
            if (arguments.prefix_cache_entries > sglang::decoder_batch_prefix_max_entries) {
                throw std::invalid_argument(error);
            }
        } else if (option == "--mtp-steps" || option == "--mtp-k") {
            const auto error = "Expected --mtp-steps in [0, 4]";
            arguments.mtp_steps = parse_integer(value, error);
            if (arguments.mtp_steps > sglang::decoder_max_mtp_steps) {
                throw std::invalid_argument(error);
            }
        } else {
            throw std::invalid_argument("Invalid argument " + std::string(option));
        }
    }
    if (arguments.prefix_cache_entries != 0) {
        if (arguments.eager) {
            throw std::invalid_argument("--prefix-cache-entries requires omitting --eager");
        }
        if (arguments.mtp_steps != 0 && arguments.fp8_prefill ==
            sglang::Fp8PrefillPolicy::tensor_core) {
            throw std::invalid_argument(
                "MTP with --prefix-cache-entries requires --fp8-prefill exact");
        }
    }
    return arguments;
}

std::string health_json(const Arguments& arguments) {
    return "{\"max_active\":" + std::to_string(arguments.max_active) +
        ",\"max_num_batched_tokens\":" + std::to_string(arguments.max_num_batched_tokens) + "}";
}

Socket listen(const Arguments& arguments) {
    auto listener = listen_loopback(arguments.port);
    std::cerr << "Listening on http://127.0.0.1:" << arguments.port << '\n';
    return listener;
}

}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: sglang-server [--checkpoint PATH] [--device ORDINAL] [--port PORT]"
                     " [--max-active N] [--mtp-steps N] [--eager] [--batch-cuda-graphs]\n"
                     "                     [--max-num-batched-tokens N] [--trace-batches]\n"
                     "                     [--fp8-prefill exact|tensor-core]\n"
                     "                     [--prefix-cache-entries 0..4]\n"
                     "                     [--reasoning-parser qwen3]\n"
                     "                     [--tool-call-parser qwen3_coder]\n"
                     "Active requests: 32 by default, range 1-32.\n"
                     "Waiting completions: up to --max-active. Full queue returns HTTP 503.\n"
                     "Target rows per step: 32768 by default, range 1-32768.\n"
                     "MTP steps: 0 by default, up to 4. --mtp-k is an alias.\n"
                     "Batch graphs: enable graph replay for eligible decode steps.\n"
                     "Eager decode: disable graph replay.\n"
                     "Prefix cache: disabled by default. Omit --eager. MTP requires exact.\n"
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
        const CompletionTokenizer tokenizer(arguments.checkpoint, arguments.sampling,
                                            arguments.chat_parsers);
        const auto health = health_json(arguments);
        VisionCompletion vision(arguments.checkpoint, arguments.device);
        BatchService service(decoder, arguments.trace_batches,
                             arguments.batch_cuda_graphs && !arguments.eager,
                             arguments.mtp_steps, arguments.max_active,
                             arguments.prefix_cache_entries, arguments.max_num_batched_tokens);
        if (stopping()) {
            return 0;
        }
        const auto listener = listen(arguments);
        serve_connections(listener, service, arguments.max_active,
            [&](const Socket& connection, int slot, std::string_view body, bool chat) {
                if (chat) {
                    complete_chat(connection, service, slot, tokenizer, body, &vision);
                } else {
                    complete(connection, service, slot, tokenizer, body);
                }
            }, health, arguments.prefix_cache_entries != 0);
    } catch (const std::exception& error) {
        std::cerr << "sglang-server: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
