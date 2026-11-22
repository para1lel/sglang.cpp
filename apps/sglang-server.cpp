#include "sglang/fp8_prefill_option.h"

#include "sglang/checkpoint.h"
#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/json.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/completions.h"
#include "sglang/server/http.h"
#include "sglang/server/native_generate.h"
#include "sglang/server/ollama_discovery.h"
#include "sglang/server/prefix_controls.h"
#include "sglang/server/score.h"
#include "sglang/server/tokenization.h"
#include "sglang/server/vision_completion.h"

#include <charconv>
#include <chrono>
#include <exception>
#include <iostream>
#include <string_view>

namespace {

using namespace sglang::server;

struct Arguments {
    std::filesystem::path checkpoint = "/data0/shared/Qwen3.8-27B-FP8";
    std::string launch_command = "sglang-server";
    sglang::Fp8PrefillPolicy fp8_prefill = sglang::Fp8PrefillPolicy::exact;
    SamplingDefaults sampling = SamplingDefaults::model;
    ChatParsers chat_parsers;
    int device = 0;
    int port = 17384;
    int max_active = sglang::decoder_batch_max_lanes;
    int max_num_batched_tokens = sglang::decoder_default_max_num_batched_tokens;
    int mtp_steps = 4;
    int prefix_cache_entries = 0;
    bool mtp_steps_explicit = false;
    bool eager = false;
    bool trace_batches = false;
    bool batch_cuda_graphs = true;

    bool enable_batch_cuda_graphs() const { return batch_cuda_graphs && !eager; }
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

void append_launch_option(std::string& command, std::string_view option,
                          std::string_view value = std::string_view{}) {
    command += ' ';
    command += option;
    if (value.empty()) {
        return;
    }
    command += ' ';
    if (option == "--sampling-defaults" || option == "--reasoning-parser" ||
        option == "--tool-call-parser" || option == "--fp8-prefill" ||
        option == "--device" || option == "--port" || option == "--max-active" ||
        option == "--max-num-batched-tokens" || option == "--prefix-cache-entries" ||
        option == "--mtp-steps" || option == "--mtp-k") {
        command += value;
    } else {
        command += "'<redacted>'";
    }
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (option == "--trace-batches") {
            arguments.trace_batches = true;
            append_launch_option(arguments.launch_command, option);
            continue;
        }
        if (option == "--batch-cuda-graphs") {
            arguments.batch_cuda_graphs = true;
            append_launch_option(arguments.launch_command, option);
            continue;
        }
        if (option == "--eager") {
            arguments.eager = true;
            append_launch_option(arguments.launch_command, option);
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
            arguments.mtp_steps_explicit = true;
            if (arguments.mtp_steps > sglang::decoder_max_mtp_steps) {
                throw std::invalid_argument(error);
            }
        } else {
            throw std::invalid_argument("Invalid argument " + std::string(option));
        }
        append_launch_option(arguments.launch_command, option, value);
    }
    if (arguments.prefix_cache_entries != 0) {
        if (sglang::token_ordered_decode(arguments.fp8_prefill)) {
            throw std::invalid_argument("Selected FP8 policy requires prefix cache off");
        }
        if (arguments.eager) {
            throw std::invalid_argument("--prefix-cache-entries requires omitting --eager");
        }
        if (arguments.fp8_prefill == sglang::Fp8PrefillPolicy::tensor_core) {
            if (arguments.mtp_steps_explicit && arguments.mtp_steps != 0) {
                throw std::invalid_argument(
                    "MTP with --prefix-cache-entries requires --fp8-prefill exact");
            }
            arguments.mtp_steps = 0;
        }
    }
    return arguments;
}

std::string health_json(const Arguments& arguments) {
    return "{\"max_active\":" + std::to_string(arguments.max_active) +
        ",\"max_num_batched_tokens\":" + std::to_string(arguments.max_num_batched_tokens) + "}";
}

std::string model_info_json(const Arguments& arguments) {
    std::string body = R"({"model_path":)";
    const auto checkpoint = arguments.checkpoint.string();
    sglang::json::append_quoted(body, checkpoint);
    body += R"(,"served_model_name":)";
    sglang::json::append_quoted(body, sglang::model_name);
    body += R"(,"tokenizer_path":)";
    sglang::json::append_quoted(body, checkpoint);
    body += R"(,"is_generation":true,"preferred_sampling_params":null,)";
    body += R"("weight_version":"default","load_format":"safetensors","reasoning_parser":)";
    body += arguments.chat_parsers.qwen3_reasoning ? R"("qwen3")" : "null";
    body += R"(,"tool_call_parser":)";
    body += arguments.chat_parsers.qwen3_coder_tools ? R"("qwen3_coder")" : "null";
    body += R"(,"disaggregation_mode":"null","has_image_understanding":true,)";
    body += R"("has_audio_understanding":false,"model_type":)";
    sglang::json::append_quoted(body, sglang::checkpoint_model_type);
    body += R"(,"architectures":[)";
    sglang::json::append_quoted(body, sglang::checkpoint_architecture);
    body += "]}";
    return body;
}

std::string server_info_json(const Arguments& arguments, double startup_seconds,
                             const BatchRuntimeState& state) {
    std::string body = R"({"frontend":"cpp","version":null,"supports_inline_system":false)";
    body += R"(,"host":"127.0.0.1","dp_size":1,"tp_size":1,"pp_size":1)";
    body += R"(,"launch_command":)";
    sglang::json::append_quoted(body, arguments.launch_command);
    body += R"(,"model_path":)";
    sglang::json::append_quoted(body, arguments.checkpoint.string());
    body += R"(,"tokenizer_path":)";
    sglang::json::append_quoted(body, arguments.checkpoint.string());
    body += R"(,"served_model_name":)";
    sglang::json::append_quoted(body, sglang::model_name);
    body += R"(,"port":)" + std::to_string(arguments.port);
    body += R"(,"device":)" + std::to_string(arguments.device);
    body += R"(,"max_running_requests":)" + std::to_string(arguments.max_active);
    body += R"(,"max_num_batched_tokens":)" + std::to_string(arguments.max_num_batched_tokens);
    body += R"(,"mtp_steps":)" + std::to_string(arguments.mtp_steps);
    body += R"(,"prefix_cache_entries":)" + std::to_string(arguments.prefix_cache_entries);
    body += R"(,"fp8_prefill":)";
    sglang::json::append_quoted(body, sglang::cli::fp8_prefill_name(arguments.fp8_prefill));
    body += R"(,"sampling_defaults":)";
    body += arguments.sampling == SamplingDefaults::model ? "\"model\"" : "\"openai\"";
    body += R"(,"reasoning_parser":)";
    body += arguments.chat_parsers.qwen3_reasoning ? "\"qwen3\"" : "null";
    body += R"(,"tool_call_parser":)";
    body += arguments.chat_parsers.qwen3_coder_tools ? "\"qwen3_coder\"" : "null";
    body += R"(,"eager":)";
    body += arguments.eager ? "true" : "false";
    body += R"(,"trace_batches":)";
    body += arguments.trace_batches ? "true" : "false";
    body += R"(,"batch_cuda_graphs":)";
    body += arguments.enable_batch_cuda_graphs() ? "true" : "false";
    body += R"(,"startup_time":{"server_e2e":)" + std::to_string(startup_seconds) + '}';
    body += R"(,"internal_states":[{"dp_rank":0,"world_size":1)";
    body += R"(,"effective_max_running_requests_per_dp":)" + std::to_string(arguments.max_active);
    body += R"(,"ready":)";
    body += state.ready ? "true" : "false";
    body += R"(,"num_running_reqs":)" + std::to_string(state.running_requests);
    body += R"(,"generation_tokens_total":)" + std::to_string(state.generation_tokens);
    return body + "}]}";
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
                     "                     [--fp8-prefill exact|tensor-core|tensor-core-mtp|\n"
                     "                      tensor-core-target|tensor-core-mtp-target]\n"
                     "                     [--prefix-cache-entries 0..4]\n"
                     "                     [--reasoning-parser qwen3]\n"
                     "                     [--tool-call-parser qwen3_coder]\n"
                     "Active requests: 32 by default, range 1-32.\n"
                     "Waiting completions: up to --max-active. Full queue returns HTTP 503.\n"
                     "Target rows per step: " << sglang::decoder_default_max_num_batched_tokens
                  << " by default, range 1-32768.\n"
                     "MTP steps: 4 by default, range 0-4. --mtp-k is an alias.\n"
                     "Default MTP uses 0 steps for logprobs and with a Tensor Core prefix cache.\n"
                     "Batch graphs: enabled by default for eligible decode steps.\n"
                     "Eager decode: disable graph replay.\n"
                     "Prefix cache: disabled by default. Omit --eager. MTP requires exact.\n"
                     "Serves on 127.0.0.1 only. Default port: 17384.\n"
                  << sglang::cli::fp8_prefill_usage;
        return 0;
    }
    try {
        const auto started = std::chrono::steady_clock::now();
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
                             arguments.enable_batch_cuda_graphs(),
                             {arguments.mtp_steps, !arguments.mtp_steps_explicit},
                             arguments.max_active,
                             arguments.prefix_cache_entries, arguments.max_num_batched_tokens);
        if (stopping()) {
            return 0;
        }
        const auto model_info = model_info_json(arguments);
        const auto listener = listen(arguments);
        const std::chrono::duration<double> startup = std::chrono::steady_clock::now() - started;
        NativeGenerate native_generate(service, tokenizer);
        const OllamaDiscovery discovery(native_generate);
        serve_connections(listener, service, arguments.max_active,
            [&](const Socket& connection, int slot, std::string_view body, bool chat) {
                if (chat) {
                    complete_chat(connection, service, slot, tokenizer, body, &vision);
                } else {
                    complete(connection, service, slot, tokenizer, body);
                }
            }, health, arguments.prefix_cache_entries != 0,
            [&](std::string_view body) {
                return native_generate.prepare(body);
            }, [&](const Socket& connection, TokenizationRoute route, std::string_view body) {
                serve_tokenization(connection, route, tokenizer, body);
            }, model_info, [&](const Socket& connection, std::string_view body) {
                native_generate.abort(connection, body);
            }, [&] {
                return server_info_json(arguments, startup.count(), service.runtime_state());
            }, [&](const Socket& connection, int slot, std::string_view body) {
                serve_score(connection, service, slot, tokenizer.get(), body);
            }, &discovery, [&](const Socket& connection, int slot, std::string_view body) {
                complete_messages(connection, service, slot, tokenizer, body, &vision);
            });
    } catch (const std::exception& error) {
        std::cerr << "sglang-server: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
