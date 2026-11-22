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
    if (!options.logit_bias) {
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
    if (options.score_prompt && (options.prompt_logprob_start < 0 ||
                                 options.prompt_logprob_start >= int(prompt.size()))) {
        throw std::invalid_argument("Prompt logprob start falls outside the prompt");
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
    if (!(options.frequency_penalty >= -2 && options.frequency_penalty <= 2)) {
        throw std::invalid_argument("frequency_penalty must be finite and in [-2, 2]");
    }
    if (!(options.presence_penalty >= -2 && options.presence_penalty <= 2)) {
        throw std::invalid_argument("presence_penalty must be finite and in [-2, 2]");
    }
    if (!(options.repetition_penalty > 0 && options.repetition_penalty <= 2)) {
        throw std::invalid_argument("repetition_penalty must be finite and in (0, 2]");
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
    if (options.min_new_tokens < 0 || options.min_new_tokens > options.max_new_tokens) {
        throw std::invalid_argument("min_new_tokens must be in [0, max_new_tokens]");
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

bool needs_seen_history(const DecoderOptions& options) {
    return options.presence_penalty != 0 || options.repetition_penalty != 1;
}

std::size_t logit_adjustment_bytes(const DecoderOptions& options) {
    const auto stop_bytes = options.min_new_tokens > 0
        ? constraint_mask_words * sizeof(std::uint32_t) : 0;
    return stop_bytes + vocab_size * ((options.frequency_penalty != 0 ? sizeof(float) : 0) +
                                      (needs_seen_history(options) ? sizeof(std::uint8_t) : 0));
}

void Request::prepare_minimum_stops(const DecoderOptions& options, cudaStream_t stream) {
    minimum_stop_upload.resize(constraint_mask_words);
    const auto mark = [this](int token) {
        minimum_stop_upload[token / 32] |= std::uint32_t{1} << (token % 32);
    };
    mark(end_of_text);
    mark(end_of_message);
    if (options.stop_token_id) mark(*options.stop_token_id);
    for (const int token : options.additional_stop_token_ids) mark(token);
    auto* buffer = memory.allocate<std::uint32_t>(constraint_mask_words);
    check(cudaMemcpyAsync(buffer, minimum_stop_upload.data(),
                          constraint_mask_words * sizeof(*buffer), cudaMemcpyHostToDevice,
                          stream), "Upload minimum-output stop mask");
    minimum_stop_mask = buffer;
}

void Request::prepare_logit_adjustments(const DecoderOptions& options, cudaStream_t stream) {
    if (options.min_new_tokens > 0) prepare_minimum_stops(options, stream);
    if (options.logit_bias && !options.logit_bias->empty()) {
        const auto& entries = *options.logit_bias;
        auto* buffer = memory.allocate<DecoderLogitBias>(entries.size());
        check(cudaMemcpyAsync(buffer, entries.data(), entries.size() * sizeof(*buffer),
                              cudaMemcpyHostToDevice, stream), "Upload logit bias");
        logit_bias = buffer;
    }
    if (options.frequency_penalty != 0) {
        output_penalties.frequency = static_cast<float>(options.frequency_penalty);
        auto*& totals = output_penalties.frequency_totals;
        totals = memory.allocate<float>(vocab_size);
        check(cudaMemsetAsync(totals, 0, vocab_size * sizeof(*totals), stream),
              "Initialize frequency penalties");
    }
    if (needs_seen_history(options)) {
        output_penalties.presence = static_cast<float>(options.presence_penalty);
        output_penalties.repetition = options.repetition_penalty;
        auto*& seen = output_penalties.seen;
        seen = memory.allocate<std::uint8_t>(vocab_size);
        check(cudaMemsetAsync(seen, 0, vocab_size * sizeof(*seen), stream),
              "Initialize seen-token history");
    }
}

void AdjustedLogitsWorkspace::prepare(Allocations& memory, Request& request,
                                      const DecoderOptions& options, cudaStream_t stream) {
    request.prepare_logit_adjustments(options, stream);
    if (options.constraint && !mask) {
        mask = memory.allocate<std::uint32_t>(constraint_mask_words);
    }
    if (options.repetition_penalty != 1) {
        const bool ranked = options.samples() || options.generated_top_logprobs >= 0 ||
            options.top_logits > 0;
        if (!repetition || (ranked && !repetition_ranked)) {
            const auto bytes = repetition_workspace_bytes(vocab_size, ranked);
            repetition = memory.allocate<std::byte>(bytes);
            repetition_bytes = bytes;
            repetition_ranked = ranked;
        }
    } else {
        if (!logits) logits = memory.allocate<float>(vocab_size);
        if (options.samples() && !sampling) {
            sampling_bytes = biased_sampling_workspace_bytes(vocab_size);
            sampling = memory.allocate<std::byte>(sampling_bytes);
        }
    }
}

void AdjustedLogitsWorkspace::apply(const Request& request, const Workspace& work,
                                    const DecoderOptions& options, const std::uint32_t* allowed,
                                    int generated, int previous_token, cudaStream_t stream) const {
    const int count = options.logit_bias ? options.logit_bias->size() : 0;
    const auto* denied = generated < options.min_new_tokens ? request.minimum_stop_mask : nullptr;
    check(adjust_logits(work.logits, logits, vocab_size, request.logit_bias, count,
                        {allowed, denied}, request.output_penalties, previous_token, stream),
          "Adjust decoder logits");
}

void AdjustedLogitsWorkspace::select(const Request& request, Workspace& work,
                                      const DecoderOptions& options,
                                      const ConstraintCursor* allowed,
                                      const SamplingParameters* parameters, int position,
                                      int previous_token, cudaStream_t stream,
                                      GeneratedLogprobReadback* logprobs) const {
    if (allowed) {
        if (allowed->ids().empty()) {
            throw std::logic_error("Output constraint has no remaining continuation");
        }
        check(cudaMemcpyAsync(mask, allowed->mask().data(), constraint_mask_words *
                              sizeof(*mask), cudaMemcpyHostToDevice, stream),
              "Upload allowed token mask");
    }
    if (options.repetition_penalty != 1) {
        select_repetition(request, work, options, allowed ? mask : nullptr, previous_token,
                          {parameters, position, &work.argmax->token_id,
                           logprobs ? options.generated_top_logprobs : -1, logprobs}, stream);
        return;
    }
    apply(request, work, options, allowed ? mask : nullptr, position, previous_token, stream);
    if (parameters) {
        check(sample_logits(logits, vocab_size, &work.argmax->token_id, *parameters,
                            position, sampling, sampling_bytes, stream), "Sample adjusted logits");
    } else {
        check(text_argmax(logits, vocab_size, work.argmax_scratch, work.argmax, stream),
              "Choose highest adjusted logit");
        if (allowed || position < options.min_new_tokens) {
            check(check_allowed_logit(logits, &work.argmax->token_id, stream),
                  "Validate unmasked adjusted token");
        }
    }
}

void AdjustedLogitsWorkspace::select_repetition(
    const Request& request, const Workspace& work, const DecoderOptions& options,
    const std::uint32_t* allowed, int previous_token, RepetitionSelection selection,
    cudaStream_t stream) const {
    const int count = options.logit_bias ? options.logit_bias->size() : 0;
    const auto* denied =
        selection.position < options.min_new_tokens ? request.minimum_stop_mask : nullptr;
    check(select_repetition_logits(work.logits, vocab_size, request.logit_bias, count,
                                  {allowed, denied},
                                  request.output_penalties, previous_token, selection,
                                  repetition, repetition_bytes, stream),
          "Select from repetition-adjusted logits");
}

void RequestProgress::select(Request& request, Workspace& work, const DecoderOptions& options,
                             const ConstraintWorkspace& scratch, const SamplingParameters* sampling,
                             void* sampling_workspace, cudaStream_t stream,
                             const AdjustedLogitsWorkspace* adjusted) {
    if (options.constraint) {
        constraint.candidates(*options.constraint, remaining_output(options), adjusted != nullptr);
    }
    if (adjusted) {
        adjusted->select(request, work, options, options.constraint ? &constraint : nullptr,
                         sampling, generated, last_token, stream);
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
    auto& adjusted_logits = transfers.adjusted_logits;
    if (adjusted && !adjusted_logits.empty()) {
        if (options.constraint) {
            adjusted->apply(request, work, options, nullptr, generated, -1, stream);
        }
        check(cudaMemcpyAsync(adjusted_logits.data(), adjusted->logits,
                              adjusted_logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                              stream), "Read adjusted decoder step logits");
    }
    auto& repetition_logits = transfers.repetition_logits;
    if (adjusted && !repetition_logits.empty()) {
        RepetitionSelection selection;
        selection.position = generated;
        selection.debug = repetition_logits.data();
        selection.debug_count = repetition_logits.size();
        adjusted->select_repetition(request, work, options, nullptr, -1, selection, stream);
    }
}

std::optional<DecoderFinishReason> RequestProgress::commit(
    const DecoderOptions& options, int token, const Bf16* logits, SerialOutput* output,
    const float* adjusted_logits, const std::vector<DecoderLogit>* ranked) {
    last_token = token;
    ++generated;
    if (output && !output->emit(token, logits, adjusted_logits, ranked)) {
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

std::vector<DecoderLogit> top_logits(const Bf16* scores, const float* adjusted, std::size_t count,
                                    std::vector<DecoderLogit>& ranking) {
    ranking.clear();
    for (int index = 0; index < vocab_size; ++index) {
        const float logit = adjusted ? adjusted[index] : __bfloat162float(scores[index]);
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

SerialOutput::SerialOutput(const DecoderCallbacks& callbacks, int top_count, DecoderResult& result)
    : callbacks_(callbacks), top_count_(top_count), result_(result) {
    if (top_count_ > 0) {
        ranking_.reserve(vocab_size);
    }
}

bool SerialOutput::emit(int token, const Bf16* logits, const float* adjusted_logits,
                        const std::vector<DecoderLogit>* repetition_logits) {
    result_.output_token_ids.push_back(token);
    const bool keep_running = !callbacks_.on_token || callbacks_.on_token(token);
    if (repetition_logits && !repetition_logits->empty()) {
        const auto end = std::find_if(repetition_logits->begin(), repetition_logits->end(),
                                      [](const auto& score) { return score.token_id < 0; });
        result_.step_top_logits.emplace_back(repetition_logits->begin(), end);
    } else if (top_count_ > 0) {
        result_.step_top_logits.push_back(
            top_logits(logits, adjusted_logits, top_count_, ranking_));
    }
    return keep_running;
}

Decoder::Model::Model(const std::filesystem::path& checkpoint, int cuda_device,
                      Fp8PrefillPolicy fp8_prefill)
    : Model(checkpoint, cuda_device) {
    fp8_prefill_ = fp8_prefill;
    if (is_b200() && fp8_prefill == Fp8PrefillPolicy::tensor_core_mtp_target) {
        const auto* setting = std::getenv("SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA");
        const std::string_view mode = setting ? setting : "";
        target_split_end_ = mode == "long" ? text_attention_max_tokens
                         : mode == "1" ? target_split_end_position : 0;
    }
}

Decoder::Decoder(const std::filesystem::path& checkpoint, int cuda_device)
    : Decoder(checkpoint, cuda_device, false) {}

Decoder::Decoder(const std::filesystem::path& checkpoint, int cuda_device, bool enable_mtp)
    : Decoder(checkpoint, cuda_device, enable_mtp, Fp8PrefillPolicy::exact) {}

Decoder::Decoder(const std::filesystem::path& checkpoint, int cuda_device, bool enable_mtp,
                 Fp8PrefillPolicy fp8_prefill)
    : Decoder(checkpoint, cuda_device, enable_mtp, fp8_prefill, false) {}

Decoder::Decoder(const std::filesystem::path& checkpoint, int cuda_device, bool enable_mtp,
                 Fp8PrefillPolicy fp8_prefill, bool mtp_int8_head) {
    if (fp8_prefill != Fp8PrefillPolicy::exact && fp8_prefill != Fp8PrefillPolicy::tensor_core &&
        !token_ordered_decode(fp8_prefill)) {
        throw std::invalid_argument("Invalid FP8 prefill policy");
    }
    model_ = std::make_unique<Model>(checkpoint, cuda_device, fp8_prefill);
    if (enable_mtp) {
        model_->mtp_weights = std::make_shared<MtpDraftWeights>(
            checkpoint, cuda_device, model_->memory, model_->embedding, model_->lm_head,
            mtp_int8_head, model_->is_b200());
        model_->tc_vocab = model_->mtp_weights->shared_vocab();
    }
}

Decoder::~Decoder() = default;

bool Decoder::experimental_target_split_gqa() const {
    return model_->target_split_end() == target_split_end_position;
}

bool Decoder::experimental_long_target_split_gqa() const {
    return model_->target_split_end() == text_attention_max_tokens;
}

std::size_t Decoder::model_device_allocation_bytes() const {
    return model_->memory->bytes() + (model_->mtp_weights ?
                                     model_->mtp_weights->device_allocation_bytes() : 0);
}

std::string_view Decoder::accelerator_name() const {
    return model_->accelerator_name;
}

bool Decoder::Model::prefill_serial(Request& request, RequestProgress& progress,
                                    const std::vector<int>& prompt, const SerialOutput& output,
                                    DecoderResult& result, const MediaPrefill* media,
                                    int step_budget) const {
    const int count = static_cast<int>(prompt.size());
    const auto stream = request.memory.stream();
    const bool shared_tiles = !media && mtp_tensor_core_prefill(fp8_prefill());
    const int aligned = shared_tiles ? count : count / prefill_chunk_rows * prefill_chunk_rows;
    std::optional<PrefillWorkspace> tiled;
    if (shared_tiles) {
        const int limit = step_budget ? step_budget : batch_prefill_tile_rows;
        tiled.emplace(stream, std::min({count, limit, batch_prefill_tile_rows}));
    } else if (!media && fp8_prefill() == Fp8PrefillPolicy::tensor_core) {
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

bool Decoder::Model::serial_graph(const DecoderOptions& options, bool media, int prompt_count,
                                  int limit) const {
    const bool split_decode = target_split_end_ && limit > target_split_first_position &&
                              prompt_count < target_split_end_;
    return !options.eager && !media && !options.requires_target_only() && !split_decode;
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
    const bool adjusted = options.requires_target_only();
    const RequestLease lease(request_in_use, serial_graph(options, !!media, prompt_count, limit));
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
        AdjustedLogitsWorkspace adjusted_workspace;
        if (adjusted) {
            request.reset();
            auto& transfers = request.memory.transfers();
            if (options.repetition_penalty != 1) {
                transfers.repetition_logits.resize(options.top_logits);
            } else if (options.top_logits > 0) {
                transfers.adjusted_logits.resize(vocab_size);
            }
            adjusted_workspace.prepare(request.memory, request, options, request.memory.stream());
        } else {
            prepare_serial_request(request, options);
        }
        const auto prefill = media
            ? upload_media_positions(request, *media, request.memory.stream()) : MediaPrefill{};
        const int rotary_delta = media ? media->prompt.next_position - prompt_count : 0;
        SerialOutput output(callbacks, options.top_logits, result);
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
                                stream, adjusted ? &adjusted_workspace : nullptr);
                check(cudaStreamSynchronize(stream), "Complete decoder step");
                const int token = read_token(request, sampling || progress.generated <
                                             options.min_new_tokens,
                                             options.constraint != nullptr);
                const auto& transfers = request.memory.transfers();
                if (const auto finish = progress.commit(options, token, transfers.logits.data(),
                                                        &output,
                                                        transfers.adjusted_logits.data(),
                                                        &transfers.repetition_logits)) {
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
    if (options.generated_top_logprobs != -1 || options.score_prompt || options.label_token_ids) {
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
        const int tile = mtp_tensor_core_prefill(model_->fp8_prefill())
            ? batch_prefill_tile_rows : prefill_chunk_rows;
        const int limit = options.max_num_batched_tokens
            ? std::min(options.max_num_batched_tokens, tile) : tile;
        DecoderBatch batch(*this, DecoderBatchOptions{false, 1, 0, limit});
        SerialOutput output(callbacks, options.top_logits, result);
        batch.generate_mtp(prompt_token_ids, options, output, result);
    }
    selected_device.restore();
    return result;
}

}
