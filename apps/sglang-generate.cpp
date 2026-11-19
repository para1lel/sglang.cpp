#include "sglang/fp8_prefill_option.h"

#include "sglang/decoder.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <locale>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using Clock = std::chrono::steady_clock;

struct Arguments {
    std::filesystem::path checkpoint = "/data0/shared/Qwen3.8-27B-FP8";
    std::vector<int> prompt;
    sglang::DecoderOptions options;
    sglang::Fp8PrefillPolicy fp8_prefill = sglang::Fp8PrefillPolicy::exact;
    int device = 0;
    int warmup = 0;
    int repeats = 1;
    bool help = false;
};

void usage(std::ostream& output) {
    output << "Usage: sglang-generate (--prompt-ids ID,ID,... | --prompt-file FILE)\n"
           << "                       [--max-new-tokens N] [--stop-token ID] [--top-logits N]\n"
           << "                       [--mtp-steps N] (0–4, default 0)\n"
           << "                       [--temperature T] [--top-p P] [--top-k K]\n"
           << "                       [--min-p P] [--seed INTEGER]\n"
           << "                       [--fp8-prefill exact|tensor-core]\n"
           << "                       [--warmup N] [--repeats N] [--checkpoint DIR] [--device N]\n"
           << "The checkpoint defaults to /data0/shared/Qwen3.8-27B-FP8. "
              "Device is a CUDA ordinal.\n"
           << "Prompt IDs may be separated by commas or whitespace. Results are written as JSON.\n"
           << sglang::cli::fp8_prefill_usage;
}

int positive_or_zero(std::string_view input, std::string_view option) {
    int value = 0;
    const auto result = std::from_chars(input.data(), input.data() + input.size(), value);
    if (input.empty() || result.ec != std::errc{} || result.ptr != input.end() || value < 0) {
        throw std::invalid_argument("Expected a nonnegative integer for " + std::string(option));
    }
    return value;
}

template <typename Number>
Number number(std::string_view input, std::string_view option) {
    Number value{};
    const auto* end = input.data() + input.size();
    const auto result = std::from_chars(input.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument("Invalid number for " + std::string(option));
    }
    return value;
}

bool parse_sampling(std::string_view option, std::string_view value,
                    sglang::DecoderOptions& options) {
    if (option == "--temperature") {
        options.temperature = number<double>(value, option);
        if (!std::isfinite(options.temperature) || options.temperature < 0) {
            throw std::invalid_argument("Expected finite --temperature >= 0");
        }
    } else if (option == "--top-p") {
        options.top_p = number<double>(value, option);
        if (!(options.top_p > 0 && options.top_p <= 1)) {
            throw std::invalid_argument("Expected --top-p in (0, 1]");
        }
    } else if (option == "--top-k") {
        options.top_k = number<std::int64_t>(value, option);
        if (options.top_k != -1 && options.top_k < 1) {
            throw std::invalid_argument("Expected --top-k of -1 or at least 1");
        }
    } else if (option == "--min-p") {
        options.min_p = number<double>(value, option);
        if (!(options.min_p >= 0 && options.min_p <= 1)) {
            throw std::invalid_argument("Expected --min-p in [0, 1]");
        }
    } else if (option == "--seed") {
        options.seed = number<std::int64_t>(value, option);
    } else {
        return false;
    }
    return true;
}

std::vector<int> parse_ids(std::string_view input) {
    constexpr std::string_view separators = ", \t\n\r\f\v";
    constexpr auto whitespace = separators.substr(1);
    std::vector<int> ids;
    auto start = input.find_first_not_of(whitespace);
    while (start != std::string_view::npos) {
        const auto end = input.find_first_of(separators, start);
        ids.push_back(positive_or_zero(input.substr(start, end - start), "prompt token"));
        start = input.find_first_not_of(whitespace, end);
        if (start != std::string_view::npos && input[start] == ',') {
            start = input.find_first_not_of(whitespace, start + 1);
            if (start == std::string_view::npos) {
                throw std::invalid_argument("Missing prompt token after comma");
            }
        }
    }
    return ids;
}

std::vector<int> read_prompt(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open prompt file " + path.string());
    }
    const std::string text(std::istreambuf_iterator<char>{input}, {});
    return parse_ids(text);
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    bool have_prompt = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option = argv[index];
        if (option == "--help" || option == "-h") {
            arguments.help = true;
            return arguments;
        }
        if (++index == argc) {
            throw std::invalid_argument("Missing argument for " + std::string(option));
        }
        const std::string_view value = argv[index];
        if (option == "--prompt-ids" || option == "--prompt-file") {
            if (have_prompt) {
                throw std::invalid_argument("Specify one prompt");
            }
            arguments.prompt = option == "--prompt-ids" ? parse_ids(value) : read_prompt(value);
            have_prompt = true;
        } else if (option == "--checkpoint") {
            arguments.checkpoint = value;
        } else if (option == "--fp8-prefill") {
            arguments.fp8_prefill = sglang::cli::parse_fp8_prefill(value);
        } else if (option == "--max-new-tokens") {
            arguments.options.max_new_tokens = positive_or_zero(value, option);
        } else if (option == "--stop-token") {
            arguments.options.stop_token_id = positive_or_zero(value, option);
        } else if (option == "--top-logits") {
            arguments.options.top_logits = positive_or_zero(value, option);
        } else if (option == "--mtp-steps") {
            arguments.options.mtp_steps = positive_or_zero(value, option);
            if (arguments.options.mtp_steps > sglang::decoder_max_mtp_steps) {
                throw std::invalid_argument("Expected --mtp-steps in [0, 4]");
            }
        } else if (option == "--warmup") {
            arguments.warmup = positive_or_zero(value, option);
        } else if (option == "--repeats") {
            arguments.repeats = positive_or_zero(value, option);
        } else if (option == "--device") {
            arguments.device = positive_or_zero(value, option);
        } else if (!parse_sampling(option, value, arguments.options)) {
            throw std::invalid_argument("Unknown option " + std::string(option));
        }
    }
    if (arguments.prompt.empty() || arguments.repeats == 0) {
        throw std::invalid_argument("A nonempty prompt and at least one repetition are required");
    }
    return arguments;
}

template <typename Values, typename WriteValue>
void write_array(std::ostream& output, const Values& values, WriteValue write_value) {
    output << '[';
    bool first = true;
    for (const auto& value : values) {
        if (!first) {
            output << ',';
        }
        first = false;
        write_value(output, value);
    }
    output << ']';
}

template <typename Value>
void write_number(std::ostream& output, Value value) {
    output << value;
}

void write_logit(std::ostream& output, const sglang::DecoderLogit& entry) {
    output << "{\"token_id\":" << entry.token_id << ",\"logit\":";
    if (std::isfinite(entry.logit)) {
        output << entry.logit;
    } else {
        output << "null";
    }
    output << '}';
}

void write_step(std::ostream& output, const std::vector<sglang::DecoderLogit>& step) {
    write_array(output, step, write_logit);
}

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

struct Measurements {
    sglang::DecoderResult result;
    std::vector<double> samples;
    std::size_t peak_bytes = 0;
    std::size_t output_tokens = 0;
    double elapsed_ms = 0;
};

Measurements measure(const sglang::Decoder& decoder, const Arguments& arguments) {
    Measurements measured;
    const bool repeatable = !arguments.options.samples() || arguments.options.seed.has_value();
    for (int index = 0; index < arguments.warmup; ++index) {
        const auto result = decoder.generate(arguments.prompt, arguments.options);
        measured.peak_bytes = std::max(measured.peak_bytes, result.peak_device_allocation_bytes);
    }
    for (int index = 0; index < arguments.repeats; ++index) {
        const auto start = Clock::now();
        auto result = decoder.generate(arguments.prompt, arguments.options);
        const double elapsed = milliseconds(Clock::now() - start);
        measured.samples.push_back(elapsed);
        measured.elapsed_ms += elapsed;
        measured.output_tokens += result.output_token_ids.size();
        measured.peak_bytes = std::max(measured.peak_bytes, result.peak_device_allocation_bytes);
        if (index == 0) {
            measured.result = std::move(result);
        } else if (repeatable &&
                   (result.output_token_ids != measured.result.output_token_ids ||
                    result.finish_reason != measured.result.finish_reason)) {
            throw std::runtime_error("Output changed between identical seeded or greedy requests");
        }
    }
    return measured;
}

void write_result(const Arguments& arguments, const Measurements& measured,
                  std::size_t model_bytes, double load_ms) {
    const auto& result = measured.result;
    const bool stopped = result.finish_reason == sglang::DecoderFinishReason::stop;
    const double rate = measured.elapsed_ms == 0 ? 0 :
        measured.output_tokens * (1000.0 / measured.elapsed_ms);
    auto& output = std::cout;
    output.imbue(std::locale::classic());
    output << std::setprecision(12);
    output << "{\"prompt_token_ids\":";
    write_array(output, arguments.prompt, write_number<int>);
    output << ",\"output_token_ids\":";
    write_array(output, result.output_token_ids, write_number<int>);
    output << ",\"finish_reason\":\"" << (stopped ? "stop" : "length") << '"';
    if (arguments.options.top_logits > 0) {
        output << ",\"step_top_logits\":";
        write_array(output, result.step_top_logits, write_step);
    }
    if (arguments.options.mtp_steps != 0) {
        output << ",\"mtp_drafted_tokens\":" << result.mtp_drafted_tokens
               << ",\"mtp_accepted_tokens\":" << result.mtp_accepted_tokens
               << ",\"mtp_rounds\":" << result.mtp_rounds;
    }
    output << ",\"device\":" << arguments.device << ",\"warmup\":" << arguments.warmup
           << ",\"repeats\":" << arguments.repeats << ",\"load_ms\":" << load_ms
           << ",\"model_device_allocation_bytes\":" << model_bytes
           << ",\"peak_device_allocation_bytes\":" << measured.peak_bytes
           << ",\"output_tokens_per_second\":" << rate << ",\"request_ms\":";
    write_array(output, measured.samples, write_number<double>);
    output << "}\n";
}

}

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        if (arguments.help) {
            usage(std::cout);
            return 0;
        }
        const auto start = Clock::now();
        const sglang::Decoder decoder(arguments.checkpoint, arguments.device,
                                      arguments.options.mtp_steps != 0, arguments.fp8_prefill);
        const double load_ms = milliseconds(Clock::now() - start);
        const auto measured = measure(decoder, arguments);
        write_result(arguments, measured, decoder.model_device_allocation_bytes(), load_ms);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sglang-generate: " << error.what() << '\n';
        return 1;
    }
}
