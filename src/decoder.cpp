#include "sglang/decoder_prefill.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_sampling.h"
#include "sglang/decoder_verify.h"
#include "sglang/prefill_fp8.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string_view>

namespace sglang {
namespace decoder {
namespace {

bool valid_token(int id) {
    return id >= 0 && id < vocab_size;
}

int serial_prefill_rows(bool b200, int prompt_rows) {
    const auto* setting = std::getenv("SGLANG_SERIAL_PREFILL_ROWS");
    if (!setting) {
        return b200 && prompt_rows >= 512 ? 1024 : prefill_chunk_rows;
    }
    const std::string_view value(setting);
    if (value == "256") {
        return 256;
    }
    if (value == "512") {
        return 512;
    }
    if (value == "1024") {
        return 1024;
    }
    if (value == "2048") {
        return serial_prefill_max_rows;
    }
    return prefill_chunk_rows;
}

void validate_constraint(const DecoderOptions& options) {
    if (!options.constraint) {
        return;
    }
    if ((options.stop_token_id && !constraint_eos(*options.stop_token_id)) ||
        !std::all_of(options.additional_stop_token_ids.begin(),
                     options.additional_stop_token_ids.end(), constraint_eos)) {
        throw std::invalid_argument("Constrained decoding requires the model EOS stop tokens");
    }
    if (options.max_new_tokens < options.constraint->minimum_tokens()) {
        throw std::invalid_argument(options.constraint->budget_error());
    }
}

void validate_logit_bias(const DecoderOptions& options) {
    if (!options.requires_target_only()) {
        return;
    }
    int previous = -1;
    for (const auto [token, bias] : *options.logit_bias) {
        if (token <= previous || !valid_token(token) || !std::isfinite(bias) ||
            bias == 0.0f || std::abs(bias) > 1e30f) {
            throw std::invalid_argument("Invalid, duplicate or unsorted decoder logit bias");
        }
        previous = token;
    }
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
    if (!std::isfinite(options.temperature) || options.temperature < 0) {
        throw std::invalid_argument("temperature must be finite and nonnegative");
    }
    if (!(options.top_p > 0 && options.top_p <= 1)) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }
    if (!(options.min_p >= 0 && options.min_p <= 1)) {
        throw std::invalid_argument("min_p must be in [0, 1]");
    }
    if (options.top_k != -1 && options.top_k < 1) {
        throw std::invalid_argument("top_k must be -1 or at least 1");
    }
    if (options.mtp_steps < 0 || options.mtp_steps > decoder_max_mtp_steps) {
        throw std::invalid_argument("MTP proposal count must be in [0, 4]");
    }
    if (options.max_num_batched_tokens < 0) {
        throw std::invalid_argument("Target step budget must be nonnegative");
    }
    validate_constraint(options);
    validate_logit_bias(options);
    if (!allow_mtp && options.mtp_steps != 0) {
        throw std::invalid_argument("MTP requires serial generation");
    }
    if (options.max_new_tokens < 0 || options.max_new_tokens > text_attention_max_tokens ||
        prompt.size() > std::size_t(text_attention_max_tokens - options.max_new_tokens)) {
        throw std::invalid_argument("Prompt and output budget must fit the 262144-token context");
    }
}

SamplingParameters request_sampling(const DecoderOptions& options) {
    std::uint64_t seed;
    if (options.seed) {
        seed = static_cast<std::uint64_t>(*options.seed);
    } else {
        std::random_device entropy;
        const auto high = static_cast<std::uint64_t>(entropy());
        seed = (high << 32) | entropy();
    }
    const int top_k = options.top_k == -1 ? vocab_size
        : static_cast<int>(std::min<std::int64_t>(options.top_k, vocab_size));
    return {options.temperature, options.top_p, top_k, seed, options.min_p};
}

void Request::upload_logit_bias(const DecoderOptions& options, cudaStream_t stream) {
    const auto& entries = *options.logit_bias;
    auto* buffer = memory.allocate<DecoderLogitBias>(entries.size());
    check(cudaMemcpyAsync(buffer, entries.data(), entries.size() * sizeof(*buffer),
                          cudaMemcpyHostToDevice, stream), "Upload logit bias");
    logit_bias = buffer;
}

void BiasedWorkspace::prepare(Allocations& memory, Request& request, const DecoderOptions& options,
                              cudaStream_t stream) {
    request.upload_logit_bias(options, stream);
    if (!logits) {
        logits = memory.allocate<float>(vocab_size);
    }
    if (options.samples() && !sampling) {
        sampling_bytes = biased_sampling_workspace_bytes(vocab_size);
        sampling = memory.allocate<std::byte>(sampling_bytes);
    }
    if (options.constraint && !mask) {
        mask = memory.allocate<std::uint32_t>(constraint_mask_words);
    }
}

void BiasedWorkspace::select(const Request& request, Workspace& work,
                             const DecoderOptions& options, const ConstraintCursor* allowed,
                             const SamplingParameters* parameters, int position,
                             cudaStream_t stream) const {
    if (allowed) {
        if (allowed->ids().empty()) {
            throw std::logic_error("Output constraint has no remaining continuation");
        }
        check(cudaMemcpyAsync(mask, allowed->mask().data(), constraint_mask_words *
                              sizeof(*mask), cudaMemcpyHostToDevice, stream),
              "Upload allowed token mask");
    }
    check(apply_logit_bias(work.logits, logits, vocab_size, request.logit_bias,
                          options.logit_bias->size(), allowed ? mask : nullptr, stream),
          "Apply logit bias");
    if (parameters) {
        check(sample_logits(logits, vocab_size, &work.argmax->token_id, *parameters,
                            position, sampling, sampling_bytes, stream), "Sample biased logits");
    } else {
        check(text_argmax(logits, vocab_size, work.argmax_scratch, work.argmax, stream),
              "Choose highest biased logit");
        if (allowed) {
            check(check_allowed_logit(logits, &work.argmax->token_id, stream),
                  "Validate biased constrained token");
        }
    }
}

void RequestProgress::select(Request& request, Workspace& work, const DecoderOptions& options,
                             const ConstraintWorkspace& scratch, const SamplingParameters* sampling,
                             void* sampling_workspace, cudaStream_t stream,
                             const BiasedWorkspace* biased) {
    if (options.constraint) {
        constraint.candidates(*options.constraint, remaining_output(options), biased != nullptr);
    }
    if (biased) {
        biased->select(request, work, options, options.constraint ? &constraint : nullptr,
                       sampling, generated, stream);
    } else if (options.constraint) {
        select_constrained(work, scratch, constraint, sampling, generated, sampling_workspace,
                            stream);
    } else if (sampling) {
        check(sample_logits(work.logits, vocab_size, &work.argmax->token_id, *sampling,
                            generated, sampling_workspace, stream), "Sample decoder logits");
    }
    auto& transfers = request.memory.transfers();
    auto& winner = transfers.winner;
    check(cudaMemcpyAsync(&winner, &work.argmax->token_id, sizeof(winner),
                          cudaMemcpyDeviceToHost, stream), "Read selected token");
    auto& logits = transfers.logits;
    if (!logits.empty()) {
        check(cudaMemcpyAsync(logits.data(), work.logits, logits.size() * sizeof(Bf16),
                              cudaMemcpyDeviceToHost, stream), "Read decoder step logits");
    }
}

std::optional<DecoderFinishReason> RequestProgress::commit(const DecoderOptions& options, int token,
                                                           const Bf16* logits,
                                                           SerialOutput* output) {
    last_token = token;
    ++generated;
    if (output && !output->emit(token, logits)) {
        return DecoderFinishReason::cancelled;
    }
    const int remaining = remaining_output(options);
    if (options.constraint) {
        if (constraint.commit(*options.constraint, token, remaining)) {
            return DecoderFinishReason::constraint;
        }
    } else if (options.stops_at(token)) {
        return DecoderFinishReason::stop;
    }
    if (remaining == 0) {
        return DecoderFinishReason::length;
    }
    return std::nullopt;
}

int validate_winner(int winner, bool sampled, bool constrained) {
    if (!valid_token(winner)) {
        if (constrained && winner == -1) {
            throw DecoderConstraintError();
        }
        if (sampled && winner == -1) {
            throw DecoderSamplingError();
        }
        throw std::runtime_error("Greedy selection did not produce a valid token ID");
    }
    return winner;
}

int read_token(Request& request, bool sampled, bool constrained) {
    return validate_winner(request.memory.transfers().winner, sampled, constrained);
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
                                    std::vector<DecoderLogit>& ranking,
                                    const std::vector<DecoderLogitBias>* bias) {
    ranking.clear();
    std::size_t entry = 0;
    for (int index = 0; index < vocab_size; ++index) {
        float logit = __bfloat162float(scores[index]);
        if (bias && entry < bias->size() && (*bias)[entry].token_id == index) {
            logit += (*bias)[entry++].bias;
        }
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

void prepare_serial_request(Request& request, const DecoderOptions& options) {
    request.reset();
    if (options.samples() && !request.sampling_workspace) {
        request.sampling_workspace =
            request.memory.allocate<std::byte>(sampling_workspace_bytes());
    }
    if (options.constraint) {
        request.constraint_workspace.allocate(request.memory, options.constraint->uses_wide_mask());
    }
    request.memory.transfers().logits.resize(options.top_logits > 0 ? vocab_size : 0);
}

}

SerialOutput::SerialOutput(const DecoderCallbacks& callbacks, int top_count, DecoderResult& result,
                           const std::vector<DecoderLogitBias>* bias)
    : callbacks_(callbacks), top_count_(top_count), result_(result), bias_(bias) {
    if (top_count_ > 0) {
        ranking_.reserve(vocab_size);
    }
}

bool SerialOutput::emit(int token, const Bf16* logits) {
    result_.output_token_ids.push_back(token);
    const bool keep_running = !callbacks_.on_token || callbacks_.on_token(token);
    if (top_count_ > 0) {
        result_.step_top_logits.push_back(top_logits(logits, top_count_, ranking_, bias_));
    }
    return keep_running;
}

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

bool Decoder::Model::prefill_serial(Request& request, RequestProgress& progress,
                                    const std::vector<int>& prompt, const SerialOutput& output,
                                    DecoderResult& result, const MediaPrefill* media,
                                    int step_budget) const {
    const int count = static_cast<int>(prompt.size());
    const auto stream = request.memory.stream();
    const int aligned = count / prefill_chunk_rows * prefill_chunk_rows;
    std::optional<PrefillWorkspace> tiled;
    if (!media && fp8_prefill() == Fp8PrefillPolicy::tensor_core) {
        const int configured = serial_prefill_rows(b200_, count);
        const int limit = step_budget ? step_budget / prefill_chunk_rows * prefill_chunk_rows
                                      : configured;
        const int larger = std::min({aligned, configured, limit});
        if (larger > prefill_chunk_rows) {
            tiled.emplace(stream, larger);
        }
    }
    if (!tiled || !tiled->bytes()) {
        const int limit = step_budget ? std::min(step_budget, prefill_chunk_rows)
                                      : prefill_chunk_rows;
        tiled.emplace(stream, std::min(count, limit));
    }
    auto& workspace = *tiled;
    result.peak_device_allocation_bytes += workspace.bytes();
    const int tile_rows = workspace.bytes() ? workspace.capacity() : 1;
    while (progress.remaining_prompt(count)) {
        const int position = progress.position;
        const int end = position < aligned ? aligned : count;
        const int rows = std::min(tile_rows, end - position);
        prefill_chunk(request, prompt.data() + position, rows, position, workspace,
                      nullptr, 0, media);
        progress.advance(rows);
        if (output.cancelled()) {
            workspace.close();
            return false;
        }
    }
    workspace.close();
    return true;
}

void Decoder::Model::advance_serial(Request& request, RequestProgress& progress,
                                    int input, bool media, int rotary_delta) const {
    if (progress.generated == 0) {
        compute_logits(request);
        return;
    }
    const auto stream = request.memory.stream();
    const int position = progress.position;
    if (request.kv_limit) {
        request.grow_kv(position + 1, position, stream);
    }
    if (request.decode_position) {
        if (progress.generated == 1) {
            check(set_decode_position(request.decode_position, position, stream),
                  "Set decode position");
        }
        decode_graph(request);
    } else {
        if (media) {
            const ForwardLane lane{&request, input, position, TextGdnMode::decode, false,
                                   nullptr, nullptr, position + rotary_delta};
            forward(&lane, 1, stream);
        } else {
            forward(request, input, position, TextGdnMode::decode);
        }
        compute_logits(request, true);
    }
    progress.advance(1);
}

void Decoder::Model::generate_serial(const std::vector<int>& prompt, const DecoderOptions& options,
                                     const DecoderCallbacks& callbacks,
                                     DecoderResult& result,
                                     std::shared_ptr<const MultimodalInput> media,
                                     const std::vector<int>* teacher) const {
    const int prompt_count = static_cast<int>(prompt.size());
    const int limit = RequestProgress::capacity(prompt_count, options);
    const int kv_limit = options.incremental_kv ? limit : 0;
    int capacity = kv_limit ? prompt_count : limit;
    const int pages = (capacity + mxfp8_page_size - 1) / mxfp8_page_size;
    const auto sampling = options.samples() ? std::optional{request_sampling(options)}
                                            : std::nullopt;
    const bool biased = options.requires_target_only();
    const RequestLease lease(request_in_use, !options.eager && !media && !biased);
    std::unique_ptr<Request> eager_request;
    auto& storage = lease.owns_slot() ? decode_request : eager_request;
    try {
        if (storage && storage->kv_limit != kv_limit) {
            storage->memory.close();
            storage.reset();
        }
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
            const std::shared_ptr<const void> owners = media
                ? std::shared_ptr<const void>(media) : memory;
            storage = std::make_unique<Request>(device, capacity, owners, lease.owns_slot(),
                                                kv_limit);
        }
        auto& request = *storage;
        BiasedWorkspace biased_workspace;
        if (biased) {
            request.reset();
            request.memory.transfers().logits.resize(options.top_logits > 0 ? vocab_size : 0);
            biased_workspace.prepare(request.memory, request, options, request.memory.stream());
        } else {
            prepare_serial_request(request, options);
        }
        const auto prefill = media
            ? upload_media_positions(request, *media, request.memory.stream()) : MediaPrefill{};
        const int rotary_delta = media ? media->prompt.next_position - prompt_count : 0;
        SerialOutput output(callbacks, options.top_logits, result, options.logit_bias.get());
        result.peak_device_allocation_bytes += request.memory.bytes();
        const auto run_request = [&] {
            RequestProgress progress;
            if (output.cancelled() ||
                !prefill_serial(request, progress, prompt, output, result,
                                media ? &prefill : nullptr, options.max_num_batched_tokens)) {
                return DecoderFinishReason::cancelled;
            }
            const auto stream = request.memory.stream();
            while (progress.remaining_output(options) > 0) {
                if (output.cancelled()) {
                    return DecoderFinishReason::cancelled;
                }
                const int input = progress.generated != 0 && teacher
                    ? (*teacher)[progress.generated - 1] : progress.last_token;
                advance_serial(request, progress, input, bool(media), rotary_delta);
                progress.select(request, request.work, options, request.constraint_workspace,
                                sampling ? &*sampling : nullptr, request.sampling_workspace,
                                stream, biased ? &biased_workspace : nullptr);
                check(cudaStreamSynchronize(stream), "Complete decoder step");
                const int token = read_token(request, sampling.has_value(),
                                             options.constraint != nullptr);
                const auto* logits = request.memory.transfers().logits.data();
                if (const auto finish = progress.commit(options, token, logits, &output)) {
                    return *finish;
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
    if (options.generated_top_logprobs != -1) {
        throw std::invalid_argument("Generated logprobs require the batch decoder");
    }
    const int proposals = options.requires_target_only() ? 0 : options.mtp_steps;
    if (proposals != 0 && !model_->mtp_weights) {
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
    if (proposals == 0) {
        model_->generate_serial(prompt_token_ids, options, callbacks, result);
    } else {
        const int limit = options.max_num_batched_tokens
            ? std::min(options.max_num_batched_tokens, prefill_chunk_rows) : prefill_chunk_rows;
        DecoderBatch batch(*this, DecoderBatchOptions{false, 1, 0, limit});
        SerialOutput output(callbacks, options.top_logits, result);
        batch.generate_mtp(prompt_token_ids, options, output, result);
    }
    selected_device.restore();
    return result;
}

}
