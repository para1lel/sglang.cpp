#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/json.h"
#include "sglang/runtime/backend/cuda_backend_policy.h"
#include "sglang/runtime/model/qwen_serving_profile.h"
#include "sglang/runtime/server/single_device_batch_executor.h"
#include "sglang/server/batch_service.h"
#include "sglang/server/completions.h"
#include "sglang/server/decisions.h"
#include "sglang/server/http.h"
#include "sglang/server/native_generate.h"
#include "sglang/server/ollama_discovery.h"
#include "sglang/server/prefix_controls.h"
#include "sglang/server/responses.h"
#include "sglang/server/score.h"
#include "sglang/server/tokenization.h"
#include "sglang/server/vision_completion.h"
#include "sglang/tokenizer/digest.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <string_view>
#include <sys/stat.h>
#include <utility>

namespace {

using namespace sglang::server;

constexpr int default_max_active = sglang::decoder_default_max_running_requests *
    single_device_batch_topology.device_count();

struct Arguments {
    std::filesystem::path checkpoint;
    std::string launch_command = "sglang-server";
    SamplingDefaults sampling = SamplingDefaults::model;
    ChatParsers chat_parsers;
    int device = 0;
    int port = 17384;
    int max_active = default_max_active;
    int max_num_batched_tokens = sglang::decoder_default_max_num_batched_tokens;
    int mtp_steps = 4;
    sglang::decoder::LmCacheConfig lmcache;
    bool lmcache_enabled = true;
    bool eager = false;
    bool trace_batches = false;
    bool enable_response_store = false;

    bool enable_batch_cuda_graphs() const { return !eager; }
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
        option == "--tool-call-parser" ||
        option == "--device" || option == "--port" || option == "--max-active" ||
        option == "--max-num-batched-tokens" || option == "--prefix-cache-entries" ||
        option == "--mtp-steps" || option == "--mtp-k") {
        command += value;
    } else {
        command += "'<redacted>'";
    }
}

Arguments parse_arguments(int argc, char** argv, const sglang::model::ServingProfile& profile) {
    Arguments arguments;
    arguments.checkpoint = profile.checkpoint_path;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (option == "--trace-batches") {
            arguments.trace_batches = true;
            append_launch_option(arguments.launch_command, option);
            continue;
        }
        if (option == "--batch-cuda-graphs") {
            append_launch_option(arguments.launch_command, option);
            continue;
        }
        if (option == "--enable-response-store") {
            arguments.enable_response_store = true;
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
            if (value != profile.reasoning_parser) {
                throw std::invalid_argument("Expected --reasoning-parser qwen3");
            }
            arguments.chat_parsers.qwen3_reasoning = true;
        } else if (option == "--tool-call-parser") {
            if (value != profile.tool_call_parser) {
                throw std::invalid_argument("Expected --tool-call-parser qwen3_coder");
            }
            arguments.chat_parsers.qwen3_coder_tools = true;
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
        } else if (option == "--lmcache-socket" && !value.empty()) {
            arguments.lmcache.endpoint = value;
        } else if (option == "--prefix-cache-entries") {
            if (value != "0") {
                throw std::invalid_argument("Only --prefix-cache-entries 0 is supported");
            }
            arguments.lmcache_enabled = false;
        } else if (option == "--mtp-steps" || option == "--mtp-k") {
            const auto error = "Expected --mtp-steps in [0, 4]";
            arguments.mtp_steps = parse_integer(value, error);
            if (arguments.mtp_steps > sglang::decoder_max_mtp_steps) {
                throw std::invalid_argument(error);
            }
        } else {
            throw std::invalid_argument("Invalid argument " + std::string(option));
        }
        append_launch_option(arguments.launch_command, option, value);
    }
    return arguments;
}

std::string cache_identity(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> files{directory / "config.json",
                                           directory / "model.safetensors.index.json"};
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".safetensors") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    std::string identity = "sglang-qwen3.8-prefix-v1\n";
    for (const auto& path : files) {
        struct stat info{};
        if (stat(path.c_str(), &info) != 0) {
            throw std::runtime_error("Cannot identify checkpoint file " + path.string());
        }
        const auto name = path.filename().string();
        identity += std::to_string(name.size()) + ':' + name;
        for (const auto field : {static_cast<std::int64_t>(info.st_dev),
                                 static_cast<std::int64_t>(info.st_ino), info.st_size,
                                 info.st_mtim.tv_sec, info.st_mtim.tv_nsec,
                                 info.st_ctim.tv_sec, info.st_ctim.tv_nsec}) {
            identity += ':' + std::to_string(field);
        }
        identity += '\n';
    }
    return sglang::tokenizer::sha256(identity);
}

std::string health_json(const Arguments& arguments) {
    return "{\"max_active\":" + std::to_string(arguments.max_active) +
        ",\"max_num_batched_tokens\":" + std::to_string(arguments.max_num_batched_tokens) + "}";
}

void append_parser(std::string& body, bool enabled, std::string_view parser) {
    if (enabled) sglang::json::append_quoted(body, parser);
    else body += "null";
}

std::string model_info_json(const Arguments& arguments,
                            const sglang::model::ServingProfile& profile) {
    std::string body = R"({"model_path":)";
    const auto checkpoint = arguments.checkpoint.string();
    sglang::json::append_quoted(body, checkpoint);
    body += R"(,"served_model_name":)";
    sglang::json::append_quoted(body, profile.served_name);
    body += R"(,"tokenizer_path":)";
    sglang::json::append_quoted(body, checkpoint);
    body += R"(,"is_generation":true,"preferred_sampling_params":null,)";
    body += R"("weight_version":)";
    sglang::json::append_quoted(body, profile.weight_version);
    body += R"(,"load_format":)";
    sglang::json::append_quoted(body, profile.load_format);
    body += R"(,"reasoning_parser":)";
    append_parser(body, arguments.chat_parsers.qwen3_reasoning, profile.reasoning_parser);
    body += R"(,"tool_call_parser":)";
    append_parser(body, arguments.chat_parsers.qwen3_coder_tools, profile.tool_call_parser);
    body += R"(,"disaggregation_mode":"null","has_image_understanding":)";
    body += profile.has_image_understanding ? "true" : "false";
    body += R"(,"has_audio_understanding":)";
    body += profile.has_audio_understanding ? "true" : "false";
    body += R"(,"model_type":)";
    sglang::json::append_quoted(body, profile.model_type);
    body += R"(,"architectures":[)";
    sglang::json::append_quoted(body, profile.architecture);
    body += "]}";
    return body;
}
std::string draft_seed_fc_json(const Arguments& arguments, bool wmma);
std::string server_info_json(const Arguments& arguments, double startup_seconds,
                             const BatchService& service, const sglang::Decoder& decoder,
                             const sglang::model::ServingProfile& profile) {
    const auto state = service.runtime_state();
    const auto topology = service.topology();
    const auto& backend = decoder.backend_policy();
    const bool draft_enabled = arguments.mtp_steps != 0;
    std::string body = R"({"frontend":"cpp","version":null,"supports_inline_system":)";
    body += profile.supports_inline_system ? "true" : "false";
    body += R"(,"enable_multimodal":)";
    body += profile.has_image_understanding ? "true" : "false";
    body += R"(,"host":"127.0.0.1","dp_size":)" + std::to_string(topology.dp_size);
    body += R"(,"tp_size":)" + std::to_string(topology.tp_size);
    body += R"(,"pp_size":)" + std::to_string(topology.pp_size);
    body += R"(,"launch_command":)";
    sglang::json::append_quoted(body, arguments.launch_command);
    body += R"(,"model_path":)";
    sglang::json::append_quoted(body, arguments.checkpoint.string());
    body += R"(,"tokenizer_path":)";
    sglang::json::append_quoted(body, arguments.checkpoint.string());
    body += R"(,"served_model_name":)";
    sglang::json::append_quoted(body, profile.served_name);
    body += R"(,"port":)" + std::to_string(arguments.port);
    body += R"(,"device":)" + std::to_string(arguments.device);
    body += R"(,"max_running_requests":)" + std::to_string(arguments.max_active);
    body += R"(,"max_num_batched_tokens":)" + std::to_string(arguments.max_num_batched_tokens);
    body += R"(,"mtp_steps":)" + std::to_string(arguments.mtp_steps);
    body += R"(,"mtp_int8_head":)";
    body += decoder.has_int8_draft_head() ? "true" : "false";
    body += R"(,"mtp_online_gqa":)";
    body += backend.draft.online_attention && draft_enabled && arguments.max_active >= 2
        ? "true" : "false";
    body += R"(,"certified_int8_verify_head":)";
    body += decoder.has_certified_vocab() ? "true" : "false";
    const auto lmcache = service.metrics().lmcache;
    body += R"(,"prefix_cache_entries":0,"lmcache_enabled":)";
    body += lmcache.enabled ? "true" : "false";
    body += R"(,"lmcache_available":)";
    body += lmcache.available ? "true" : "false";
    body += R"(,"fp8_prefill":)";
    const bool tc_prefill = backend.tensor_core_prefill;
    sglang::json::append_quoted(body, tc_prefill ? "tensor-core" : "exact");
    body += R"(,"target_gqa":)";
    sglang::json::append_quoted(body, backend.grouped_target_attention ? "grouped-split" : "exact");
    body += R"(,"paired_prefill":)";
    const bool paired_prefill = backend.paired_prefill && arguments.max_active >= 2 &&
        arguments.max_num_batched_tokens >= decoder.paired_prefill_threshold();
    body += paired_prefill ? "true" : "false";
    body += R"(,"sampling_defaults":)";
    body += arguments.sampling == SamplingDefaults::model ? "\"model\"" : "\"openai\"";
    body += R"(,"reasoning_parser":)";
    append_parser(body, arguments.chat_parsers.qwen3_reasoning, profile.reasoning_parser);
    body += R"(,"tool_call_parser":)";
    append_parser(body, arguments.chat_parsers.qwen3_coder_tools, profile.tool_call_parser);
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
    return body + "}]" + draft_seed_fc_json(arguments, tc_prefill && backend.draft.wmma_seed_fc) +
        "}";
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
                     " [--max-active N] [--mtp-steps N] [--eager]\n"
                     "                     [--max-num-batched-tokens N] [--trace-batches]\n"
                     "                     [--lmcache-socket PATH] [--prefix-cache-entries 0]\n"
                     "                     [--reasoning-parser qwen3]\n"
                     "                     [--tool-call-parser qwen3_coder]\n"
                     "                     [--enable-response-store]\n"
                     "Active requests: " << default_max_active << " by default, range 1-32.\n"
                     "Waiting completions: up to --max-active. Full queue returns HTTP 503.\n"
                     "Target rows per step: " << sglang::decoder_default_max_num_batched_tokens
                  << " by default, range 1-32768.\n"
                     "MTP steps: 4 by default, range 0-4. --mtp-k is an alias.\n"
                     "Default MTP uses 0 steps for output logprobs.\n"
                     "Batch graphs: enabled by default for eligible decode steps.\n"
                     "Eager decode: disable graph replay.\n"
                     "LMCache enabled by default. Set --prefix-cache-entries 0 to disable.\n"
                     "Serves on 127.0.0.1 only. Default port: 17384.\n";
        return 0;
    }
    try {
        const auto started = std::chrono::steady_clock::now();
        const auto& profile = sglang::model::qwen_serving_profile;
        const auto arguments = parse_arguments(argc, argv, profile);
        install_shutdown_handlers();
        const sglang::Decoder decoder(arguments.checkpoint, arguments.device,
                                      arguments.mtp_steps != 0);
        if (stopping()) {
            return 0;
        }
        const CompletionTokenizer tokenizer(arguments.checkpoint, arguments.sampling,
                                            arguments.chat_parsers, profile);
        const auto health = health_json(arguments);
        VisionCompletion vision(arguments.checkpoint, arguments.device);
        const BatchExecutor::Options batch_options{
            arguments.enable_batch_cuda_graphs(), arguments.max_active, 0,
            arguments.max_num_batched_tokens};
        auto executor = std::make_unique<SingleDeviceBatchExecutor>(decoder, batch_options);
        auto cache = arguments.lmcache_enabled
            ? std::make_unique<sglang::decoder::LmCacheClient>(arguments.lmcache) : nullptr;
        const auto identity = cache ? cache_identity(arguments.checkpoint) : std::string{};
        BatchService service(std::move(executor), arguments.trace_batches, arguments.mtp_steps,
                             std::move(cache), identity);
        if (stopping()) {
            return 0;
        }
        const auto model_info = model_info_json(arguments, profile);
        const auto listener = listen(arguments);
        const std::chrono::duration<double> startup = std::chrono::steady_clock::now() - started;
        NativeGenerate native_generate(service, tokenizer);
        const OllamaDiscovery discovery(native_generate);
        ResponseStore response_store(arguments.enable_response_store);
        serve_connections(listener, service, arguments.max_active,
            [&](const Socket& connection, int slot, std::string_view body, bool chat) {
                if (chat) {
                    complete_chat(connection, service, slot, tokenizer, body, &vision);
                } else {
                    complete(connection, service, slot, tokenizer, body);
                }
            }, health, arguments.lmcache_enabled,
            [&](std::string_view body) {
                return native_generate.prepare(body);
            }, [&](const Socket& connection, TokenizationRoute route, std::string_view body) {
                serve_tokenization(connection, route, tokenizer, body);
            }, model_info, [&](const Socket& connection, std::string_view body) {
                native_generate.abort(connection, body);
            }, [&] {
                return server_info_json(arguments, startup.count(), service, decoder, profile);
            }, [&](const Socket& connection, int slot, std::string_view body) {
                serve_score(connection, service, slot, tokenizer.get(), body);
            }, &discovery, [&](const Socket& connection, int slot, std::string_view body) {
                complete_messages(connection, service, slot, tokenizer, body, &vision);
            }, [&](const Socket& connection, int slot, std::string_view body) {
                complete_responses(connection, service, slot, tokenizer, body, response_store,
                                   &vision);
            }, &response_store,
            [&](const Socket& connection, int slot, std::string_view body, bool systemone) {
                if (systemone) {
                    serve_systemone(connection, service, slot, tokenizer, body,
                                    arguments.checkpoint.string(), vision);
                } else {
                    serve_decisions(connection, service, slot, tokenizer, body, vision);
                }
            }, decoder.accelerator_name(), &native_generate);
    } catch (const std::exception& error) {
        std::cerr << "sglang-server: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

namespace {

std::string draft_seed_fc_json(const Arguments& arguments, bool wmma) {
    using namespace sglang::decoder;
    const std::string field = R"(,"mtp_seed_fc":)";
    if (!arguments.mtp_steps) return field + "null";
    if (!wmma) return field + R"({"preferred":"simt_bf16_fp32"})";
    return field + R"({"preferred":"tma_tcgen05_bf16_fp32","fallback":"wmma_bf16_fp32",)"
        R"("outside_shape_limits":"simt_bf16_fp32",)"
        R"("min_prompt_tokens":)" + std::to_string(wmma_seed_min_prompt_tokens) +
        R"(,"min_rows":)" + std::to_string(wmma_seed_min_rows) +
        R"(,"max_rows":)" + std::to_string(wmma_seed_max_rows) + "}";
}

}
