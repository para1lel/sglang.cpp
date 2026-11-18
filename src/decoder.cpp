#include "sglang/decoder_prefill.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_verify.h"

#include <algorithm>
#include <cmath>

namespace sglang {
namespace decoder {
namespace {

bool valid_token(int id) {
    return id >= 0 && id < vocab_size;
}

}

void validate_request(const std::vector<int>& prompt, const DecoderOptions& options,
                      bool allow_mtp) {
    if (prompt.empty() || !std::all_of(prompt.begin(), prompt.end(), valid_token)) {
        throw std::invalid_argument("Prompt must contain token IDs in [0, 248320)");
    }
    if ((options.stop_token_id && !valid_token(*options.stop_token_id)) ||
        !std::all_of(options.additional_stop_token_ids.begin(),
                     options.additional_stop_token_ids.end(), valid_token)) {
        throw std::invalid_argument("Stop token ID falls outside the vocabulary");
    }
    if (options.top_logits < 0 || options.top_logits > vocab_size) {
        throw std::invalid_argument("Top-logit count falls outside the vocabulary");
    }
    if (options.mtp_steps < 0 || options.mtp_steps > decoder_max_mtp_steps) {
        throw std::invalid_argument("MTP proposal count must be in [0, 4]");
    }
    if (!allow_mtp && options.mtp_steps != 0) {
        throw std::invalid_argument("MTP requires serial generation");
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
    RequestLease(std::atomic_flag& claimed, bool reusable)
        : claimed_(claimed),
          owns_slot_(reusable && !claimed_.test_and_set(std::memory_order_acquire)) {}

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

std::vector<DecoderLogit> top_logits(const Bf16* scores, std::size_t count,
                                    std::vector<DecoderLogit>& ranking) {
    ranking.clear();
    for (int index = 0; index < vocab_size; ++index) {
        const float logit = __bfloat162float(scores[index]);
        if (!std::isnan(logit)) {
            ranking.push_back({index, logit});
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

struct Decoder::Model::MtpGeneration {
    const Model& model;
    Request& target;
    MtpDraftRequest& draft;
    const DecoderOptions& options;
    const DecoderCallbacks& callbacks;
    DecoderResult& result;
    std::vector<DecoderLogit> ranking{};
    int next_target_position = 0;
    int seed = -1;

    bool cancelled() const { return callbacks.is_cancelled && callbacks.is_cancelled(); }

    std::optional<DecoderFinishReason> emit(int token, const Bf16* logits, bool accepted = false) {
        if (cancelled()) {
            return DecoderFinishReason::cancelled;
        }
        seed = token;
        result.output_token_ids.push_back(token);
        result.mtp_accepted_tokens += accepted;
        const bool keep_running = !callbacks.on_token || callbacks.on_token(token);
        if (options.top_logits > 0) {
            result.step_top_logits.push_back(
                top_logits(logits, options.top_logits, ranking));
        }
        if (!keep_running) {
            return DecoderFinishReason::cancelled;
        }
        if (options.stops_at(token)) {
            return DecoderFinishReason::stop;
        }
        if (result.output_token_ids.size() == static_cast<std::size_t>(options.max_new_tokens)) {
            return DecoderFinishReason::length;
        }
        return std::nullopt;
    }

    using VerificationTokens = std::array<int, target_verify_max_rows>;

    bool propose(VerificationTokens& tokens, int count) {
        const auto stream = target.memory.stream();
        const int first_draft_position = next_target_position - 1;
        const Bf16* hidden = target.work.normalized;
        int shifted_token = seed;
        tokens.front() = seed;
        ++result.mtp_rounds;
        for (int index = 0; index < count; ++index) {
            if (cancelled()) {
                return false;
            }
            hidden = draft.enqueue(first_draft_position + index, shifted_token, hidden, true,
                                   stream);
            check(cudaStreamSynchronize(stream), "Complete MTP draft step");
            shifted_token = draft.read_token();
            tokens[index + 1] = shifted_token;
            ++result.mtp_drafted_tokens;
        }
        return true;
    }

    std::optional<DecoderFinishReason> consume(TargetVerification& verification,
                                               const VerificationTokens& tokens, int rows,
                                               int count, int& processed) {
        for (int index = 0; index < rows; ++index) {
            if (index > 0 && cancelled()) {
                return DecoderFinishReason::cancelled;
            }
            ++processed;
            const int token = verification.winner(index);
            const bool accepted = index < count && token == tokens[index + 1];
            if (const auto finish = emit(token, verification.logits(index), accepted)) {
                return finish;
            }
            if (!accepted) {
                return std::nullopt;
            }
            const auto row = verification.row(index);
            draft.enqueue(row.position, seed, row.work->normalized, false, target.memory.stream());
        }
        return std::nullopt;
    }

    std::optional<DecoderFinishReason> verify(TargetVerification& verification,
                                              const VerificationTokens& tokens, int count,
                                              int remaining) {
        draft.rewind(next_target_position);
        if (cancelled()) {
            return DecoderFinishReason::cancelled;
        }
        const auto stream = target.memory.stream();
        const int rows = std::min(remaining, count + 1);
        model.forward_verify(verification, next_target_position, tokens.data(), rows, stream);
        verification.read();
        int processed = 0;
        const auto finish = consume(verification, tokens, rows, count, processed);
        if (!finish) {
            const auto* hidden = verification.row(processed - 1).work->normalized;
            check(cudaMemcpyAsync(target.work.normalized, hidden, hidden_size * sizeof(Bf16),
                                  cudaMemcpyDeviceToDevice, stream),
                  "Carry committed target hidden state into next draft");
        }
        verification.commit(processed);
        next_target_position += processed;
        return finish;
    }

    DecoderFinishReason run(const std::vector<int>& prompt) {
        const auto stream = target.memory.stream();
        const int prompt_count = static_cast<int>(prompt.size());
        for (; next_target_position < prompt_count; ++next_target_position) {
            if (cancelled()) {
                return DecoderFinishReason::cancelled;
            }
            const int position = next_target_position;
            model.forward(target, prompt[position], position, TextGdnMode::prefill);
            if (position + 1 < prompt_count) {
                model.normalize_final(target.work, stream);
                draft.enqueue(position, prompt[position + 1], target.work.normalized, false,
                              stream);
            }
        }
        model.compute_logits(target);
        const int token = read_step(target);
        if (const auto finish = emit(token, target.memory.transfers().logits.data())) {
            return *finish;
        }
        const auto before = target.memory.bytes();
        const int rows = std::min(options.max_new_tokens - 1, options.mtp_steps + 1);
        TargetVerification verification(target, prompt_count + options.max_new_tokens - 1,
                                        next_target_position, rows, options.top_logits > 0);
        result.peak_device_allocation_bytes += target.memory.bytes() - before;
        while (!cancelled()) {
            const int remaining = options.max_new_tokens -
                                  static_cast<int>(result.output_token_ids.size());
            const int count = std::min(remaining, options.mtp_steps);
            VerificationTokens tokens{};
            if (!propose(tokens, count)) {
                break;
            }
            if (const auto finish = verify(verification, tokens, count, remaining)) {
                return *finish;
            }
        }
        return DecoderFinishReason::cancelled;
    }
};

Decoder::Model::Model(const std::filesystem::path& checkpoint, int cuda_device,
                      Fp8PrefillPolicy fp8_prefill)
    : Model(checkpoint, cuda_device) {
    fp8_prefill_ = fp8_prefill;
}

Decoder::Decoder(const std::filesystem::path& checkpoint, int cuda_device)
    : Decoder(checkpoint, cuda_device, false) {}

Decoder::Decoder(const std::filesystem::path& checkpoint, int cuda_device, bool enable_mtp)
    : Decoder(checkpoint, cuda_device, enable_mtp, Fp8PrefillPolicy::exact) {}

Decoder::Decoder(const std::filesystem::path& checkpoint, int cuda_device, bool enable_mtp,
                 Fp8PrefillPolicy fp8_prefill) {
    if (fp8_prefill != Fp8PrefillPolicy::exact && fp8_prefill != Fp8PrefillPolicy::tensor_core) {
        throw std::invalid_argument("Invalid FP8 prefill policy");
    }
    model_ = std::make_unique<Model>(checkpoint, cuda_device, fp8_prefill);
    if (enable_mtp) {
        model_->mtp_weights = std::make_shared<MtpDraftWeights>(
            checkpoint, cuda_device, model_->memory, model_->embedding, model_->lm_head);
    }
}

Decoder::~Decoder() = default;

std::size_t Decoder::model_device_allocation_bytes() const {
    return model_->memory->bytes() + (model_->mtp_weights ?
                                     model_->mtp_weights->device_allocation_bytes() : 0);
}

DecoderFinishReason Decoder::Model::generate_mtp(const std::vector<int>& prompt,
                                                const DecoderOptions& options,
                                                const DecoderCallbacks& callbacks,
                                                DecoderResult& result) const {
    const int capacity = static_cast<int>(prompt.size()) + options.max_new_tokens - 1;
    const auto draft = std::make_shared<MtpDraftRequest>(mtp_weights, capacity);
    Request target(device, capacity, draft);
    target.reset();
    target.memory.transfers().logits.resize(options.top_logits > 0 ? vocab_size : 0);
    result.peak_device_allocation_bytes +=
        target.memory.bytes() + draft->device_allocation_bytes();
    MtpGeneration generation{*this, target, *draft, options, callbacks, result};
    if (options.top_logits > 0) {
        generation.ranking.reserve(vocab_size);
    }
    const auto finish = generation.run(prompt);
    target.memory.close();
    return finish;
}

void Decoder::Model::generate_serial(const std::vector<int>& prompt, const DecoderOptions& options,
                                     const DecoderCallbacks& callbacks,
                                     DecoderResult& result) const {
    const auto cancelled = [&] { return callbacks.is_cancelled && callbacks.is_cancelled(); };
    const int prompt_count = static_cast<int>(prompt.size());
    int capacity = prompt_count + options.max_new_tokens - 1;
    const int pages = (capacity + mxfp8_page_size - 1) / mxfp8_page_size;
    const RequestLease lease(request_in_use, !options.eager);
    std::unique_ptr<Request> eager_request;
    auto& storage = lease.owns_slot() ? decode_request : eager_request;
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
            storage = std::make_unique<Request>(device, capacity, memory, lease.owns_slot());
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
            if (cancelled()) {
                return DecoderFinishReason::cancelled;
            }
            {
                PrefillWorkspace workspace(request.memory.stream(),
                                           std::min(prompt_count, prefill_chunk_rows));
                result.peak_device_allocation_bytes += workspace.bytes();
                const int tile_rows = workspace.bytes() ? workspace.capacity() : 1;
                for (int position = 0; position < prompt_count; position += tile_rows) {
                    const int rows = std::min(tile_rows, prompt_count - position);
                    prefill_chunk(request, prompt.data() + position, rows, position, workspace);
                    if (cancelled()) {
                        workspace.close();
                        return DecoderFinishReason::cancelled;
                    }
                }
                workspace.close();
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
                    decode_graph(request);
                } else {
                    if (step != 0) {
                        forward(request, winner, prompt_count + step - 1, TextGdnMode::decode);
                    }
                    compute_logits(request);
                }
                winner = read_step(request);
                result.output_token_ids.push_back(winner);
                const bool keep_running = !callbacks.on_token || callbacks.on_token(winner);
                if (options.top_logits > 0) {
                    result.step_top_logits.push_back(
                        top_logits(transfers.logits.data(), options.top_logits, ranking));
                }
                if (!keep_running) {
                    return DecoderFinishReason::cancelled;
                }
                if (options.stops_at(winner)) {
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
}

DecoderResult Decoder::generate(const std::vector<int>& prompt_token_ids,
                                const DecoderOptions& options,
                                const DecoderCallbacks& callbacks) const {
    validate_request(prompt_token_ids, options, true);
    if (options.mtp_steps != 0 && !model_->mtp_weights) {
        throw std::invalid_argument("MTP weights must be enabled when constructing the decoder");
    }
    DecoderResult result;
    result.peak_device_allocation_bytes = model_device_allocation_bytes();
    if (callbacks.is_cancelled && callbacks.is_cancelled()) {
        result.finish_reason = DecoderFinishReason::cancelled;
        return result;
    }
    if (options.max_new_tokens == 0) {
        return result;
    }
    DeviceScope selected_device(model_->device);
    reclaim_prefill_workspaces(model_->device);
    result.output_token_ids.reserve(options.max_new_tokens);
    if (options.mtp_steps == 0) {
        model_->generate_serial(prompt_token_ids, options, callbacks, result);
    } else {
        result.finish_reason = model_->generate_mtp(prompt_token_ids, options, callbacks, result);
    }
    selected_device.restore();
    return result;
}

}
