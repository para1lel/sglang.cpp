#include "sglang/decoder_internal.h"

#include <algorithm>
#include <cmath>

namespace sglang {
namespace {

using namespace decoder;

bool valid_token(int id) {
    return id >= 0 && id < vocab_size;
}

void validate_request(const std::vector<int>& prompt, const DecoderOptions& options) {
    if (prompt.empty() || !std::all_of(prompt.begin(), prompt.end(), valid_token)) {
        throw std::invalid_argument("Prompt must contain token IDs in [0, 248320)");
    }
    if (options.stop_token_id && !valid_token(*options.stop_token_id)) {
        throw std::invalid_argument("Stop token ID falls outside the vocabulary");
    }
    if (options.top_logits < 0 || options.top_logits > vocab_size) {
        throw std::invalid_argument("Top-logit count falls outside the vocabulary");
    }
    if (options.max_new_tokens < 0 || options.max_new_tokens > text_attention_max_tokens ||
        prompt.size() > std::size_t(text_attention_max_tokens - options.max_new_tokens)) {
        throw std::invalid_argument("Prompt and output budget must fit the 262144-token context");
    }
}

std::vector<DecoderLogit> top_logits(const std::vector<Bf16>& scores, std::size_t count,
                                    std::vector<DecoderLogit>& ranking) {
    ranking.clear();
    for (std::size_t index = 0; index < scores.size(); ++index) {
        const float logit = __bfloat162float(scores[index]);
        if (!std::isnan(logit)) {
            ranking.push_back({static_cast<int>(index), logit});
        }
    }
    const auto end = ranking.begin() + std::min(count, ranking.size());
    std::partial_sort(ranking.begin(), end, ranking.end(),
                      [](const DecoderLogit& left, const DecoderLogit& right) {
                          if (left.logit == right.logit) {
                              return left.token_id < right.token_id;
                          }
                          return left.logit > right.logit;
                      });
    return {ranking.begin(), end};
}

void read_step(Request& request) {
    auto& transfers = request.memory.transfers();
    const auto stream = request.memory.stream();
    auto& winner = transfers.winner;
    auto& host_logits = transfers.logits;
    check(cudaMemcpyAsync(&winner, request.work.argmax, sizeof(winner),
                          cudaMemcpyDeviceToHost, stream), "Read greedy token");
    if (!host_logits.empty()) {
        check(cudaMemcpyAsync(host_logits.data(), request.work.logits,
                              host_logits.size() * sizeof(Bf16), cudaMemcpyDeviceToHost, stream),
              "Read step logits");
    }
    check(cudaStreamSynchronize(stream), "Complete decoder step");
    if (!valid_token(winner.token_id)) {
        throw std::runtime_error("Greedy selection did not produce a valid token ID");
    }
}

}

Decoder::Decoder(const std::filesystem::path& checkpoint, int cuda_device)
    : model_(std::make_unique<Model>(checkpoint, cuda_device)) {}

Decoder::~Decoder() = default;

std::size_t Decoder::model_device_allocation_bytes() const {
    return model_->memory->bytes();
}

DecoderResult Decoder::generate(const std::vector<int>& prompt_token_ids,
                                const DecoderOptions& options,
                                const DecoderCallbacks& callbacks) const {
    validate_request(prompt_token_ids, options);
    DecoderResult result;
    result.peak_device_allocation_bytes = model_device_allocation_bytes();
    const auto cancelled = [&] { return callbacks.is_cancelled && callbacks.is_cancelled(); };
    if (cancelled()) {
        result.finish_reason = DecoderFinishReason::cancelled;
        return result;
    }
    if (options.max_new_tokens == 0) {
        return result;
    }
    DeviceScope selected_device(model_->device);
    const int prompt_count = static_cast<int>(prompt_token_ids.size());
    const int capacity = prompt_count + options.max_new_tokens - 1;
    Request request(model_->device, capacity, model_->memory);
    auto& transfers = request.memory.transfers();
    std::vector<DecoderLogit> ranking;
    if (options.top_logits > 0) {
        transfers.logits.resize(vocab_size);
        ranking.reserve(vocab_size);
    }
    auto& winner = transfers.winner;
    result.peak_device_allocation_bytes += request.memory.bytes();
    result.output_token_ids.reserve(options.max_new_tokens);
    const auto run_request = [&] {
        for (int position = 0; position < prompt_count; ++position) {
            if (cancelled()) {
                return DecoderFinishReason::cancelled;
            }
            model_->forward(request, prompt_token_ids[position], position, TextGdnMode::prefill);
        }
        for (int step = 0; step < options.max_new_tokens; ++step) {
            if (cancelled()) {
                return DecoderFinishReason::cancelled;
            }
            if (step != 0) {
                const int position = prompt_count + step - 1;
                model_->forward(request, winner.token_id, position, TextGdnMode::decode);
            }
            model_->compute_logits(request);
            read_step(request);
            result.output_token_ids.push_back(winner.token_id);
            const bool keep_running = !callbacks.on_token || callbacks.on_token(winner.token_id);
            if (options.top_logits > 0) {
                result.step_top_logits.push_back(
                    top_logits(transfers.logits, options.top_logits, ranking));
            }
            if (!keep_running) {
                return DecoderFinishReason::cancelled;
            }
            if (options.stop_token_id == winner.token_id) {
                return DecoderFinishReason::stop;
            }
        }
        return DecoderFinishReason::length;
    };
    result.finish_reason = run_request();
    request.memory.close();
    selected_device.restore();
    return result;
}

}
