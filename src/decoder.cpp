#include "sglang/decoder_internal.h"

#include "sglang/decoder_cuda.cuh"

#include <algorithm>
#include <cmath>

namespace sglang {
namespace decoder {
namespace {

bool valid_token(int id) {
    return id >= 0 && id < vocab_size;
}

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

void copy_greedy_token(Request& request, cudaStream_t stream) {
    auto& winner = request.memory.transfers().winner;
    check(cudaMemcpyAsync(&winner, &request.work.argmax->token_id, sizeof(winner),
                          cudaMemcpyDeviceToHost, stream), "Read greedy token");
}

int read_greedy_token(Request& request) {
    const int winner = request.memory.transfers().winner;
    if (!valid_token(winner)) {
        throw std::runtime_error("Greedy selection did not produce a valid token ID");
    }
    return winner;
}

}

namespace {

using namespace decoder;

class RequestLease {
public:
    explicit RequestLease(std::atomic_flag& claimed)
        : claimed_(claimed), owns_slot_(!claimed_.test_and_set(std::memory_order_acquire)) {}

    ~RequestLease() {
        if (owns_slot_) {
            claimed_.clear(std::memory_order_release);
        }
    }

    RequestLease(const RequestLease&) = delete;
    RequestLease& operator=(const RequestLease&) = delete;

    bool owns_slot() const { return owns_slot_; }

private:
    std::atomic_flag& claimed_;
    bool owns_slot_;
};

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

int read_step(Request& request) {
    const auto stream = request.memory.stream();
    auto& host_logits = request.memory.transfers().logits;
    copy_greedy_token(request, stream);
    if (!host_logits.empty()) {
        check(cudaMemcpyAsync(host_logits.data(), request.work.logits,
                              host_logits.size() * sizeof(Bf16), cudaMemcpyDeviceToHost, stream),
              "Read step logits");
    }
    check(cudaStreamSynchronize(stream), "Complete decoder step");
    return read_greedy_token(request);
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
    int capacity = prompt_count + options.max_new_tokens - 1;
    const int pages = (capacity + mxfp8_page_size - 1) / mxfp8_page_size;
    result.output_token_ids.reserve(options.max_new_tokens);
    const RequestLease lease(model_->request_in_use);
    std::unique_ptr<Request> eager_request;
    auto& storage = lease.owns_slot() ? model_->decode_request : eager_request;
    try {
        if (storage && storage->pages.page_count < pages) {
            const int previous_capacity = storage->pages.page_count * mxfp8_page_size;
            capacity = std::max(capacity, std::min(2 * previous_capacity,
                                                  text_attention_max_tokens));
            storage->memory.close();
            storage.reset();
        }
        if (storage) {
            storage->memory.reclaim_deferred();
        } else {
            storage = std::make_unique<Request>(model_->device, capacity, model_->memory,
                                                lease.owns_slot());
        }
        auto& request = *storage;
        request.reset();
        auto& transfers = request.memory.transfers();
        transfers.logits.resize(options.top_logits > 0 ? vocab_size : 0);
        std::vector<DecoderLogit> ranking;
        if (options.top_logits > 0) {
            ranking.reserve(vocab_size);
        }
        result.peak_device_allocation_bytes += request.memory.bytes();
        const auto run_request = [&] {
            for (int position = 0; position < prompt_count; ++position) {
                if (cancelled()) {
                    return DecoderFinishReason::cancelled;
                }
                model_->forward(request, prompt_token_ids[position], position,
                                TextGdnMode::prefill);
            }
            int winner = -1;
            for (int step = 0; step < options.max_new_tokens; ++step) {
                if (cancelled()) {
                    return DecoderFinishReason::cancelled;
                }
                if (step != 0 && request.decode_position) {
                    if (step == 1) {
                        check(set_decode_position(request.decode_position, prompt_count,
                                                  request.memory.stream()), "Set decode position");
                    }
                    model_->decode_graph(request);
                } else {
                    if (step != 0) {
                        model_->forward(request, winner, prompt_count + step - 1,
                                        TextGdnMode::decode);
                    }
                    model_->compute_logits(request);
                }
                winner = read_step(request);
                result.output_token_ids.push_back(winner);
                const bool keep_running = !callbacks.on_token || callbacks.on_token(winner);
                if (options.top_logits > 0) {
                    result.step_top_logits.push_back(
                        top_logits(transfers.logits, options.top_logits, ranking));
                }
                if (!keep_running) {
                    return DecoderFinishReason::cancelled;
                }
                if (options.stop_token_id == winner) {
                    return DecoderFinishReason::stop;
                }
            }
            return DecoderFinishReason::length;
        };
        result.finish_reason = run_request();
        if (eager_request) {
            request.memory.close();
        } else if (result.output_token_ids.empty()) {
            check(cudaStreamSynchronize(request.memory.stream()), "Complete interrupted prefill");
        }
    } catch (...) {
        storage.reset();
        throw;
    }
    selected_device.restore();
    return result;
}

}
