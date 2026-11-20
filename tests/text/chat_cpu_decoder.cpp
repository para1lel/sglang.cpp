#include "sglang/decoder_batch.h"
#include "sglang/server/vision_completion.h"
#include "sglang/tokenizer/tokenizer.h"
#include "sglang/vision.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace sglang {
namespace {

constexpr std::array<int, 7> sequence{172, 253, 239, 102, 32, 248046, 33};
constexpr std::array<int, 7> sampling_sequence{32, 248046, 33, 248044, 34, 35, 36};
constexpr std::size_t han_partial_prompt_tokens = 41;

void validate_token_ids(const std::vector<int>& prompt, const DecoderOptions& options) {
    const auto valid = [](int token) {
        return token >= 0 && static_cast<TokenId>(token) < model_vocab_size;
    };
    if (prompt.empty() || !std::all_of(prompt.begin(), prompt.end(), valid)) {
        throw std::invalid_argument("Prompt must contain token IDs in [0, 248320)");
    }
    if (options.stop_token_id && !valid(*options.stop_token_id)) {
        throw std::invalid_argument("Stop token ID falls outside the vocabulary");
    }
}

bool sampling_probe() {
    return std::getenv("SGLANG_TEST_SAMPLING_RECEIPTS") != nullptr;
}

void record_sampling(const DecoderOptions& options, std::size_t prompt_size) {
    if (!sampling_probe()) {
        return;
    }
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "cpu sampling {\"temperature\":" << options.temperature
           << ",\"top_k\":" << options.top_k << ",\"top_p\":" << options.top_p
           << ",\"min_p\":" << options.min_p
           << ",\"budget\":" << options.max_new_tokens << ",\"seed\":";
    if (options.seed) {
        output << *options.seed;
    } else {
        output << "null";
    }
    output << ",\"sampled\":" << (options.samples() ? "true" : "false")
           << ",\"prompt_tokens\":" << prompt_size << ",\"stops\":[";
    std::vector<int> stops(options.additional_stop_token_ids.begin(),
                           options.additional_stop_token_ids.end());
    if (options.stop_token_id) {
        stops.push_back(*options.stop_token_id);
    }
    std::sort(stops.begin(), stops.end());
    stops.erase(std::unique(stops.begin(), stops.end()), stops.end());
    stops.erase(std::remove_if(stops.begin(), stops.end(),
                               [&](const auto token) { return !options.stops_at(token); }),
                stops.end());
    for (std::size_t index = 0; index < stops.size(); ++index) {
        output << (index == 0 ? "" : ",") << stops[index];
    }
    output << "]}\n";
    std::cerr << output.str();
}

std::map<std::int64_t, std::vector<int>> read_output_scripts() {
    std::map<std::int64_t, std::vector<int>> scripts;
    const auto* path = std::getenv("SGLANG_TEST_CHAT_OUTPUT_IDS");
    if (!path) {
        return scripts;
    }
    std::ifstream input(path);
    for (std::string line; std::getline(input, line);) {
        std::istringstream values(line);
        std::int64_t seed;
        std::vector<int> ids;
        if (!(values >> seed)) {
            throw std::runtime_error("Missing scripted output seed");
        }
        for (int id; values >> id;) {
            ids.push_back(id);
        }
        if (!values.eof() || ids.empty() || !scripts.emplace(seed, std::move(ids)).second) {
            throw std::runtime_error("Invalid scripted output token IDs");
        }
    }
    if (!input.eof() || scripts.empty()) {
        throw std::runtime_error("Cannot read scripted chat outputs");
    }
    return scripts;
}

const std::vector<int>* scripted_output(const DecoderOptions& options) {
    static const auto scripts = read_output_scripts();
    const auto found = scripts.find(options.seed.value_or(0));
    return found == scripts.end() ? nullptr : &found->second;
}

struct Lane {
    DecoderOptions options;
    std::size_t prompt_size;
    const std::vector<int>* script = scripted_output(options);
    int tokens = 0;

    DecoderBatchStep step() {
        if (tokens == options.max_new_tokens) {
            return {std::nullopt, DecoderFinishReason::length, false};
        }
        const auto offset = prompt_size == han_partial_prompt_tokens
            ? 0 : (prompt_size % sequence.size() + sequence.size() - 17 % sequence.size());
        const auto index = (static_cast<std::size_t>(tokens) + offset) % sequence.size();
        auto token = sequence[index];
        if (script) {
            token = script->at(static_cast<std::size_t>(tokens) % script->size());
        } else if (sampling_probe()) {
            token = sampling_sequence[static_cast<std::size_t>(tokens) % sampling_sequence.size()];
        }
        ++tokens;
        if (options.stops_at(token)) {
            return {token, DecoderFinishReason::stop, true};
        }
        return {token, tokens == options.max_new_tokens
            ? std::optional{DecoderFinishReason::length} : std::nullopt, true};
    }
};

}

struct Decoder::Model {};

Decoder::Decoder(const std::filesystem::path&, int, bool, Fp8PrefillPolicy) {}
Decoder::~Decoder() = default;

DecoderResult Decoder::generate(const std::vector<int>& prompt, const DecoderOptions& options,
                                const DecoderCallbacks& callbacks) const {
    validate_token_ids(prompt, options);
    record_sampling(options, prompt.size());
    Lane lane{options, prompt.size()};
    DecoderResult result;
    while (true) {
        if (callbacks.is_cancelled && callbacks.is_cancelled()) {
            result.finish_reason = DecoderFinishReason::cancelled;
            return result;
        }
        const auto step = lane.step();
        if (step.token) {
            result.output_token_ids.push_back(*step.token);
            if (callbacks.on_token && !callbacks.on_token(*step.token)) {
                result.finish_reason = DecoderFinishReason::cancelled;
                return result;
            }
        }
        if (step.finish_reason) {
            result.finish_reason = *step.finish_reason;
            return result;
        }
    }
}

struct DecoderBatch::State {
    std::array<std::optional<Lane>, decoder_batch_max_lanes> lanes;
};

DecoderBatch::DecoderBatch(const Decoder&, DecoderBatchOptions options)
    : state_(std::make_unique<State>()) {
    if (options.cuda_graphs) {
        throw std::invalid_argument("The CPU decoder has no CUDA graphs");
    }
}

DecoderBatch::~DecoderBatch() = default;

int DecoderBatch::admit(int slot, std::vector<int> prompt, const DecoderOptions& options) {
    validate_token_ids(prompt, options);
    auto& lane = state_->lanes.at(slot);
    if (lane) {
        throw std::logic_error("Invalid CPU lane admission");
    }
    lane.emplace(Lane{options, prompt.size()});
    record_sampling(options, prompt.size());
    std::cerr << ("cpu admit slot=" + std::to_string(slot) +
                  " prompt=" + std::to_string(prompt.size()) +
                  " budget=" + std::to_string(options.max_new_tokens) + '\n');
    return 0;
}

#ifdef SGLANG_TEST_BATCH_MEDIA
int DecoderBatch::admit_multimodal(int, std::vector<int>, const DecoderOptions&,
                                  const std::vector<DecoderMedia>&) {
    throw std::logic_error("The CPU decoder does not accept media");
}
#endif

void DecoderBatch::cancel(int slot) {
    auto& lane = state_->lanes.at(slot);
    if (!lane) {
        throw std::logic_error("Cancelling an empty CPU lane");
    }
    std::cerr << ("cpu cancel slot=" + std::to_string(slot) +
                  " prompt=" + std::to_string(lane->prompt_size) +
                  " budget=" + std::to_string(lane->options.max_new_tokens) + '\n');
    lane.reset();
}

std::array<DecoderBatchStep, decoder_batch_max_lanes>
DecoderBatch::advance_chunked(const std::array<bool, decoder_batch_max_lanes>& selected,
                              DecoderBatchStepStats*) {
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    std::array<DecoderBatchStep, decoder_batch_max_lanes> result{};
    bool has_greedy = false;
    bool has_sampled = false;
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
        if (!selected[slot]) {
            continue;
        }
        auto& lane = state_->lanes[slot];
        if (!lane) {
            throw std::logic_error("Selected an empty CPU lane");
        }
        has_greedy |= !lane->options.samples();
        has_sampled |= lane->options.samples();
        result[slot] = lane->step();
        if (result[slot].finish_reason) {
            lane.reset();
        }
    }
    if (sampling_probe() && has_greedy && has_sampled) {
        std::cerr << "cpu sampling mixed step\n";
    }
    return result;
}

std::size_t DecoderBatch::request_device_allocation_bytes() const {
    return std::count_if(state_->lanes.begin(), state_->lanes.end(),
                         [](const auto& lane) { return lane.has_value(); });
}

bool DecoderBatch::available() const {
    return true;
}

DecoderPrefixCacheStats DecoderBatch::prefix_cache_stats() const {
    return {};
}

void DecoderBatch::reset_prefix_cache() {}

struct VisionEncoder::Impl {};

VisionEncoder::~VisionEncoder() = default;

namespace server {

VisionCompletion::VisionCompletion(const std::filesystem::path& checkpoint, int device)
    : checkpoint_(checkpoint), device_(device) {}

VisionCompletion::~VisionCompletion() = default;

#ifdef SGLANG_TEST_SERIAL_MEDIA
DecoderResult VisionCompletion::generate(const Decoder&, const MediaPrompt&,
                                         const DecoderOptions&, const DecoderCallbacks&) {
    throw std::logic_error("The CPU decoder does not accept media");
}
#endif

#ifdef SGLANG_TEST_BATCH_MEDIA
DecoderResult VisionCompletion::generate(BatchService&, int, const MediaPrompt&,
                                        const DecoderOptions&, const DecoderCallbacks&) {
    throw std::logic_error("The CPU decoder does not accept media");
}
#endif

}

}
