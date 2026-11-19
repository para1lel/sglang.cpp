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
    if (options.mtp_steps != 0) {
        throw std::invalid_argument("Constrained decoding requires mtp_steps=0");
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

void copy_token(Request& request, const Workspace& work, cudaStream_t stream) {
    auto& winner = request.memory.transfers().winner;
    check(cudaMemcpyAsync(&winner, &work.argmax->token_id, sizeof(winner),
                          cudaMemcpyDeviceToHost, stream), "Read selected token");
}

int read_token(Request& request, bool sampled, bool constrained) {
    const int winner = request.memory.transfers().winner;
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

int read_step(Request& request, const std::optional<SamplingParameters>& sampling = std::nullopt,
              const ConstraintCursor* constraint = nullptr, int position = 0) {
    const auto stream = request.memory.stream();
    auto& host_logits = request.memory.transfers().logits;
    if (constraint) {
        select_constrained(request.work, request.constraint_workspace, *constraint,
                            sampling ? &*sampling : nullptr, position,
                            request.sampling_workspace, stream);
    } else if (sampling) {
        check(sample_logits(request.work.logits, vocab_size, &request.work.argmax->token_id,
                            *sampling, position, request.sampling_workspace, stream),
              "Sample decoder logits");
    }
    copy_token(request, request.work, stream);
    if (!host_logits.empty()) {
        check(cudaMemcpyAsync(host_logits.data(), request.work.logits,
                              host_logits.size() * sizeof(Bf16), cudaMemcpyDeviceToHost, stream),
              "Read step logits");
    }
    check(cudaStreamSynchronize(stream), "Complete decoder step");
    return read_token(request, sampling.has_value(), constraint != nullptr);
}

}

struct Decoder::Model::MtpGeneration {
    const Model& model;
    Request& target;
    MtpDraftRequest& draft;
    const DecoderOptions& options;
    const DecoderCallbacks& callbacks;
    DecoderResult& result;
    std::optional<SamplingParameters> sampling;
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
        result.mtp_rounds += count > 0;
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
            const int token = verification.winner(index, sampling.has_value());
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
        if (count > 0) {
            draft.rewind(next_target_position);
        }
        if (cancelled()) {
            return DecoderFinishReason::cancelled;
        }
        const auto stream = target.memory.stream();
        const int rows = std::min(remaining, count + 1);
        if (options.incremental_kv) {
            target.grow_kv(next_target_position + rows, next_target_position, stream);
        }
        model.forward_verify(verification, next_target_position, tokens.data(), rows, stream);
        if (sampling) {
            verification.sample(*sampling, static_cast<int>(result.output_token_ids.size()),
                                target.sampling_workspace);
        }
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

    bool prefill(const std::vector<int>& prompt, std::size_t& prefill_bytes) {
        if (cancelled()) {
            return false;
        }
        const auto stream = target.memory.stream();
        const int prompt_count = static_cast<int>(prompt.size());
        const int limit = options.max_num_batched_tokens
            ? std::min(prefill_chunk_rows, options.max_num_batched_tokens) : prefill_chunk_rows;
        PrefillWorkspace workspace(stream, std::min(prompt_count, limit));
        prefill_bytes = workspace.bytes();
        result.peak_device_allocation_bytes += prefill_bytes;
        const int tile_rows = prefill_bytes ? workspace.capacity() : 1;
        while (next_target_position < prompt_count) {
            if (cancelled()) {
                workspace.close();
                return false;
            }
            const int position = next_target_position;
            const int remaining = prompt_count - position;
            const int rows = std::min(tile_rows, remaining);
            model.prefill_chunk(target, prompt.data() + position, rows, position, workspace,
                                &draft, remaining);
            if (callbacks.is_cancelled) {
                check(cudaStreamSynchronize(stream), "Complete cancellable MTP prefill tile");
            }
            next_target_position += rows;
        }
        workspace.close();
        return !cancelled();
    }

    DecoderFinishReason run(const std::vector<int>& prompt) {
        std::size_t prefill_bytes = 0;
        if (!prefill(prompt, prefill_bytes)) {
            return DecoderFinishReason::cancelled;
        }
        model.compute_logits(target);
        const int token = read_step(target, sampling);
        if (const auto finish = emit(token, target.memory.transfers().logits.data())) {
            return *finish;
        }
        const auto before = target.memory.bytes();
        const int proposals = options.max_num_batched_tokens
            ? std::min(options.mtp_steps, options.max_num_batched_tokens - 1) : options.mtp_steps;
        const int rows = std::min(options.max_new_tokens - 1, proposals + 1);
        const int prompt_count = static_cast<int>(prompt.size());
        TargetVerification verification(target, prompt_count + options.max_new_tokens - 1,
                                        next_target_position, rows, options.top_logits > 0);
        const auto verify_bytes = target.memory.bytes() - before;
        result.peak_device_allocation_bytes +=
            std::max(prefill_bytes, verify_bytes) - prefill_bytes;
        while (!cancelled()) {
            const int remaining = options.max_new_tokens -
                                  static_cast<int>(result.output_token_ids.size());
            const int count = std::min(remaining - (sampling ? 1 : 0), proposals);
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
    const int initial_kv = options.incremental_kv ? static_cast<int>(prompt.size()) : capacity;
    Request target(device, initial_kv, draft, false, options.incremental_kv ? capacity : 0);
    prepare_serial_request(target, options);
    result.peak_device_allocation_bytes +=
        target.memory.bytes() + draft->device_allocation_bytes();
    const auto sampling = options.samples() ? std::optional{request_sampling(options)}
                                            : std::nullopt;
    MtpGeneration generation{*this, target, *draft, options, callbacks, result, sampling};
    if (options.top_logits > 0) {
        generation.ranking.reserve(vocab_size);
    }
    const auto finish = generation.run(prompt);
    target.memory.close();
    return finish;
}

bool Decoder::Model::prefill_serial(Request& request, const std::vector<int>& prompt,
                                   const DecoderCallbacks& callbacks, DecoderResult& result,
                                   const MediaPrefill* media, int step_budget) const {
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
    for (int position = 0; position < count;) {
        const int end = position < aligned ? aligned : count;
        const int rows = std::min(tile_rows, end - position);
        prefill_chunk(request, prompt.data() + position, rows, position, workspace,
                      nullptr, 0, media);
        position += rows;
        if (callbacks.is_cancelled && callbacks.is_cancelled()) {
            workspace.close();
            return false;
        }
    }
    workspace.close();
    return true;
}

void Decoder::Model::advance_serial(Request& request, int step, int prompt_count, int input,
                                    bool media, int rotary_delta) const {
    const int position = prompt_count + step - 1;
    if (step != 0 && request.kv_limit) {
        request.grow_kv(position + 1, position, request.memory.stream());
    }
    if (step != 0 && request.decode_position) {
        if (step == 1) {
            check(set_decode_position(request.decode_position, prompt_count,
                                      request.memory.stream()), "Set decode position");
        }
        decode_graph(request);
        return;
    }
    if (step != 0) {
        if (media) {
            const ForwardLane lane{&request, input, position, TextGdnMode::decode, false,
                                   nullptr, nullptr, position + rotary_delta};
            forward(&lane, 1, request.memory.stream());
        } else {
            forward(request, input, position, TextGdnMode::decode);
        }
    }
    compute_logits(request, step != 0);
}

void Decoder::Model::generate_serial(const std::vector<int>& prompt, const DecoderOptions& options,
                                     const DecoderCallbacks& callbacks,
                                     DecoderResult& result,
                                     std::shared_ptr<const MultimodalInput> media,
                                     const std::vector<int>* teacher) const {
    const auto cancelled = [&] { return callbacks.is_cancelled && callbacks.is_cancelled(); };
    const int prompt_count = static_cast<int>(prompt.size());
    const int limit = prompt_count + options.max_new_tokens - 1;
    const int kv_limit = options.incremental_kv ? limit : 0;
    int capacity = kv_limit ? prompt_count : limit;
    const int pages = (capacity + mxfp8_page_size - 1) / mxfp8_page_size;
    const auto sampling = options.samples() ? std::optional{request_sampling(options)}
                                            : std::nullopt;
    const RequestLease lease(request_in_use, !options.eager && !media);
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
        prepare_serial_request(request, options);
        const auto prefill = media
            ? upload_media_positions(request, *media, request.memory.stream()) : MediaPrefill{};
        const int rotary_delta = media ? media->prompt.next_position - prompt_count : 0;
        auto& transfers = request.memory.transfers();
        std::vector<DecoderLogit> ranking;
        if (options.top_logits > 0) {
            ranking.reserve(vocab_size);
        }
        result.peak_device_allocation_bytes += request.memory.bytes();
        const auto run_request = [&] {
            if (cancelled() ||
                !prefill_serial(request, prompt, callbacks, result, media ? &prefill : nullptr,
                                options.max_num_batched_tokens)) {
                return DecoderFinishReason::cancelled;
            }
            int winner = -1;
            ConstraintCursor constraint;
            for (int step = 0; step < options.max_new_tokens; ++step) {
                if (cancelled()) {
                    return DecoderFinishReason::cancelled;
                }
                const int input = step != 0 && teacher ? (*teacher)[step - 1] : winner;
                advance_serial(request, step, prompt_count, input, bool(media), rotary_delta);
                const int remaining = options.max_new_tokens - step;
                if (options.constraint) {
                    constraint.candidates(*options.constraint, remaining);
                }
                winner = read_step(request, sampling,
                                   options.constraint ? &constraint : nullptr, step);
                result.output_token_ids.push_back(winner);
                const bool keep_running = !callbacks.on_token || callbacks.on_token(winner);
                if (options.top_logits > 0) {
                    result.step_top_logits.push_back(
                        top_logits(transfers.logits.data(), options.top_logits, ranking));
                }
                if (!keep_running) {
                    return DecoderFinishReason::cancelled;
                }
                if (options.constraint &&
                    constraint.commit(*options.constraint, winner, remaining - 1)) {
                    return DecoderFinishReason::constraint;
                }
                if (!options.constraint && options.stops_at(winner)) {
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
