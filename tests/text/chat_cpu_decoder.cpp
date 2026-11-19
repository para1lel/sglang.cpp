#include "sglang/decoder_batch.h"
#include "sglang/server/vision_completion.h"
#include "sglang/tokenizer/tokenizer.h"
#include "sglang/vision.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace sglang {
namespace {

constexpr std::array<int, 7> sequence{172, 253, 239, 102, 32, 248046, 33};

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

struct Lane {
    DecoderOptions options;
    std::size_t prompt_size;
    int tokens = 0;

    DecoderBatchStep step() {
        if (tokens == options.max_new_tokens) {
            return {std::nullopt, DecoderFinishReason::length, false};
        }
        const auto index = (static_cast<std::size_t>(tokens) + prompt_size % sequence.size() +
                            sequence.size() - 17 % sequence.size()) % sequence.size();
        const auto token = sequence[index];
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
    Lane lane{options, prompt.size()};
    DecoderResult result;
    while (true) {
        const auto step = lane.step();
        if ((callbacks.is_cancelled && callbacks.is_cancelled()) ||
            (step.token && callbacks.on_token && !callbacks.on_token(*step.token))) {
            result.finish_reason = DecoderFinishReason::cancelled;
            return result;
        }
        if (step.token) {
            result.output_token_ids.push_back(*step.token);
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
    std::cerr << ("cpu admit slot=" + std::to_string(slot) +
                  " prompt=" + std::to_string(prompt.size()) +
                  " budget=" + std::to_string(options.max_new_tokens) + '\n');
    return 0;
}

int DecoderBatch::admit_multimodal(int, std::vector<int>, const DecoderOptions&,
                                  const std::vector<DecoderMedia>&) {
    throw std::logic_error("The CPU decoder does not accept media");
}

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
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
        if (!selected[slot]) {
            continue;
        }
        auto& lane = state_->lanes[slot];
        if (!lane) {
            throw std::logic_error("Selected an empty CPU lane");
        }
        result[slot] = lane->step();
        if (result[slot].finish_reason) {
            lane.reset();
        }
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

DecoderResult VisionCompletion::generate(const Decoder&, const MediaPrompt&,
                                         const DecoderOptions&, const DecoderCallbacks&) {
    throw std::logic_error("The CPU decoder does not accept media");
}

DecoderResult VisionCompletion::generate(BatchService&, int, const MediaPrompt&,
                                         const DecoderOptions&, const DecoderCallbacks&) {
    throw std::logic_error("The CPU decoder does not accept media");
}

}

}
