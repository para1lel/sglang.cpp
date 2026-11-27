#include "sglang/decoder_prefill.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_sampling.h"
#include "sglang/decoder_verify.h"
#include "sglang/runtime/backend/tensor_parallel.h"
#include "sglang/prefill_fp8.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include <unistd.h>

namespace sglang {
namespace decoder {
namespace {

int check_tensor_parallel(int tensor_parallel) {
    if (tensor_parallel != 1 && tensor_parallel != 2) {
        throw std::invalid_argument("Tensor parallel size must be 1 or 2");
    }
    return tensor_parallel;
}
bool valid_token(int id) {
    return id >= 0 && id < vocab_size;
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
    copy_cuda_async(buffer, minimum_stop_upload.data(), constraint_mask_words * sizeof(*buffer),
                    cudaMemcpyHostToDevice, stream, "Upload minimum-output stop mask");
    minimum_stop_mask = buffer;
}

void Request::prepare_logit_adjustments(const DecoderOptions& options, cudaStream_t stream) {
    if (options.min_new_tokens > 0) prepare_minimum_stops(options, stream);
    if (options.logit_bias && !options.logit_bias->empty()) {
        const auto& entries = *options.logit_bias;
        auto* buffer = memory.allocate<DecoderLogitBias>(entries.size());
        copy_cuda_async(buffer, entries.data(), entries.size() * sizeof(*buffer),
                        cudaMemcpyHostToDevice, stream, "Upload logit bias");
        logit_bias = buffer;
    }
    if (options.frequency_penalty != 0) {
        output_penalties.frequency = static_cast<float>(options.frequency_penalty);
        auto*& totals = output_penalties.frequency_totals;
        totals = memory.allocate<float>(vocab_size);
        zero_cuda_async(totals, vocab_size * sizeof(*totals), stream,
                        "Initialize frequency penalties");
    }
    if (needs_seen_history(options)) {
        output_penalties.presence = static_cast<float>(options.presence_penalty);
        output_penalties.repetition = options.repetition_penalty;
        auto*& seen = output_penalties.seen;
        seen = memory.allocate<std::uint8_t>(vocab_size);
        zero_cuda_async(seen, vocab_size * sizeof(*seen), stream,
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
        copy_cuda_async(mask, allowed->mask().data(), constraint_mask_words * sizeof(*mask),
                        cudaMemcpyHostToDevice, stream, "Upload allowed token mask");
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
    copy_cuda_async(&winner, &work.argmax->token_id, sizeof(winner), cudaMemcpyDeviceToHost,
                    stream, "Read selected token");
    auto& logits = transfers.logits;
    if (!logits.empty()) {
        copy_cuda_async(logits.data(), work.logits, logits.size() * sizeof(Bf16),
                        cudaMemcpyDeviceToHost, stream, "Read decoder step logits");
    }
    auto& adjusted_logits = transfers.adjusted_logits;
    if (adjusted && !adjusted_logits.empty()) {
        if (options.constraint) {
            adjusted->apply(request, work, options, nullptr, generated, -1, stream);
        }
        copy_cuda_async(adjusted_logits.data(), adjusted->logits,
                        adjusted_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, stream,
                        "Read adjusted decoder step logits");
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
    return finish_token(options, constraint, token, remaining_output(options));
}

std::optional<DecoderFinishReason> RequestProgress::finish_token(
    const DecoderOptions& options, ConstraintCursor& constraint, int token, int remaining) {
    if (options.constraint) {
        if (constraint.commit(*options.constraint, token, remaining)) {
            return DecoderFinishReason::constraint;
        }
    } else if (options.stops_at(token)) {
        return DecoderFinishReason::stop;
    }
    return remaining == 0 ? std::optional{DecoderFinishReason::length} : std::nullopt;
}

std::optional<DecoderFinishReason> RequestProgress::commit_single(
    const DecoderOptions& options, int token, const Bf16* logits, SerialOutput* output,
    const float* adjusted_logits, const std::vector<DecoderLogit>* ranked) {
    const auto finish = commit(options, token, logits, output, adjusted_logits, ranked);
    if (finish != DecoderFinishReason::cancelled && options.text_stop &&
        options.text_stop(&token, 1, finish.has_value()) == 1) {
        return DecoderFinishReason::stop;
    }
    return finish;
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

void prepare_serial_request(Request& request, const DecoderOptions& options,
                            AdjustedLogitsWorkspace* adjusted) {
    request.reset();
    if (adjusted) {
        auto& transfers = request.memory.transfers();
        if (options.repetition_penalty != 1) {
            transfers.repetition_logits.resize(options.top_logits);
        } else if (options.top_logits > 0) {
            transfers.adjusted_logits.resize(vocab_size);
        }
        adjusted->prepare(request.memory, request, options, request.memory.stream());
        return;
    }
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

Decoder::Decoder(const std::filesystem::path& checkpoint, int cuda_device, bool enable_mtp)
    : Decoder(checkpoint, cuda_device, enable_mtp, 1) {}

Decoder::Decoder(const std::filesystem::path& checkpoint, int cuda_device, bool enable_mtp,
                 int tensor_parallel)
    : model_(std::make_unique<Model>(
          checkpoint, cuda_device,
          decoder::WeightSharding{0, check_tensor_parallel(tensor_parallel)})) {
    if (tensor_parallel == 2) {
        int device_count = 0;
        check(cudaGetDeviceCount(&device_count), "Count CUDA devices for tensor parallel");
        if (cuda_device < 0 || cuda_device + 1 >= device_count) {
            throw std::invalid_argument("Tensor parallel size 2 requires two CUDA devices");
        }
        tp_ = std::make_unique<TpRuntime>();
        tp_->device = cuda_device;
        tp_->peer_device = cuda_device + 1;
        tp_->group = decoder::TensorParallel::create({cuda_device, tp_->peer_device});
        const std::size_t slab = std::size_t(decoder::tp_max_forward_lanes) * hidden_size *
                                 sizeof(Bf16);
        const std::size_t packed_bytes =
            std::size_t(decoder::max_wide_mtp_rows) * intermediate_size;
        {
            decoder::DeviceScope peer_scope(tp_->peer_device);
            tp_->peer = std::make_unique<Model>(checkpoint, tp_->peer_device,
                                                decoder::WeightSharding{1, 2});
            check(cudaStreamCreate(&tp_->peer_stream), "Create rank-1 stream");
            check(cudaMalloc(reinterpret_cast<void**>(&tp_->partials[1].attention), slab),
                  "Allocate rank-1 attention slab");
            check(cudaMalloc(reinterpret_cast<void**>(&tp_->partials[1].hidden), slab),
                  "Allocate rank-1 hidden slab");
            check(cudaMalloc(reinterpret_cast<void**>(&tp_->packed[1]), packed_bytes),
                  "Allocate rank-1 packed activation staging");
        }
        {
            decoder::DeviceScope scope(cuda_device);
            check(cudaMalloc(reinterpret_cast<void**>(&tp_->partials[0].attention), slab),
                  "Allocate rank-0 attention slab");
            check(cudaMalloc(reinterpret_cast<void**>(&tp_->partials[0].hidden), slab),
                  "Allocate rank-0 hidden slab");
            check(cudaMalloc(reinterpret_cast<void**>(&tp_->packed[0]), packed_bytes),
                  "Allocate rank-0 packed activation staging");
        }
        tp_->stream = model_->memory->stream();
        model_->tp = tp_.get();
        tp_->peer->tp = tp_.get();
    }
    if (enable_mtp) {
        model_->mtp_weights = std::make_shared<decoder::MtpDraftWeights>(
            checkpoint, cuda_device, model_->memory, model_->embedding, model_->lm_head,
            model_->backend.draft);
        model_->tc_vocab = model_->mtp_weights->shared_vocab();
    }
    if (tensor_parallel == 2) {
        
        
        
        auto fill = [this](int rank, int device) {
            const Model& model = rank == 0 ? *model_ : *tp_->peer;
            auto& entry = tp_info_.ranks[rank];
            entry.local_rank = rank;
            entry.cuda_device = device;
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, device), "Read CUDA device UUID");
            std::memcpy(entry.uuid, properties.uuid.bytes, sizeof(entry.uuid));
            entry.pid = static_cast<long>(getpid());
            entry.nccl_established = true;
            entry.weight_bytes = model.memory->bytes();
            if (model.mtp_weights) {
                entry.weight_bytes += model.mtp_weights->device_allocation_bytes();
            }
            entry.lm_head = model.lm_head.values != nullptr;
            entry.mtp = model.mtp_weights != nullptr;
        };
        fill(0, cuda_device);
        fill(1, tp_->peer_device);
        tp_info_.world_size = 2;
    }
}

Decoder::TpRuntime::~TpRuntime() {
    if (stream) {
        decoder::DeviceScope scope(device);
        synchronize_cuda(stream, "Fence rank-0 stream for tensor-parallel teardown");
    }
    if (peer_stream) {
        decoder::DeviceScope peer_scope(peer_device);
        synchronize_cuda(peer_stream, "Fence rank-1 stream for tensor-parallel teardown");
    }
    verification_capture.reset();
    group.reset();
    peer.reset();
    if (partials[1].attention || partials[1].hidden || peer_stream || packed[1]) {
        decoder::DeviceScope peer_scope(peer_device);
        if (partials[1].attention) {
            check(cudaFree(partials[1].attention), "Free rank-1 attention slab");
        }
        if (partials[1].hidden) {
            check(cudaFree(partials[1].hidden), "Free rank-1 hidden slab");
        }
        if (packed[1]) {
            check(cudaFree(packed[1]), "Free rank-1 packed activation staging");
        }
        if (peer_stream) {
            check(cudaStreamDestroy(peer_stream), "Destroy rank-1 stream");
        }
    }
    if (partials[0].attention || partials[0].hidden || packed[0]) {
        decoder::DeviceScope scope(device);
        if (partials[0].attention) {
            check(cudaFree(partials[0].attention), "Free rank-0 attention slab");
        }
        if (partials[0].hidden) {
            check(cudaFree(partials[0].hidden), "Free rank-0 hidden slab");
        }
        if (packed[0]) {
            check(cudaFree(packed[0]), "Free rank-0 packed activation staging");
        }
    }
}

const decoder::TpRuntimeInfo* Decoder::tensor_parallel_info() const {
    return tp_info_.world_size > 1 ? &tp_info_ : nullptr;
}

decoder::TpArCounters Decoder::tp_ar_counters() const {
    decoder::TpArCounters counters;
    if (tp_ != nullptr && tp_->group != nullptr) {
        tp_->group->snapshot_transport(counters);
    }
    return counters;
}

Decoder::~Decoder() = default;

const CudaBackendPolicy& Decoder::backend_policy() const {
    return model_->backend;
}

bool Decoder::uses_b200_operators() const {
    return model_->is_b200();
}

bool Decoder::has_int8_draft_head() const {
    return model_->mtp_weights && model_->backend.draft.int8_head;
}

bool Decoder::has_certified_vocab() const {
    return model_->tc_vocab != nullptr;
}

int Decoder::paired_prefill_threshold() const {
    return 2 * model_->backend.prefill_tile_rows;
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
    const int limit = step_budget ? step_budget : batch_prefill_tile_rows;
    const int capacity = std::min({count, limit, execution.prefill_limit()});
    std::optional<PrefillWorkspace> tiled(std::in_place, stream, capacity);
    if (!tiled->bytes() && capacity > prefill_chunk_rows) {
        tiled.emplace(stream, prefill_chunk_rows);
    }
    auto& workspace = *tiled;
    result.peak_device_allocation_bytes += workspace.bytes();
    const int tile_rows = workspace.bytes() ? workspace.capacity() : 1;
    while (progress.remaining_prompt(count)) {
        const int position = progress.position;
        const int rows = std::min(tile_rows, count - position);
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
                                    int input, int rotary_delta) const {
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
            check(set_decode_position(request.decode_position, position, stream,
                                      request.decode_position + 1, position + rotary_delta),
                  "Set decode position");
        }
        decode_graph(request);
    } else {
        const ForwardLane lane{&request, input, position, TextGdnMode::decode, true,
                               nullptr, nullptr, position + rotary_delta};
        forward(&lane, 1, stream);
        compute_logits(request, true);
    }
    progress.advance(1);
}

bool Decoder::Model::serial_graph(const DecoderOptions& options, int prompt_count,
                                  int limit) const {
    const bool split_decode = execution.split_decode(prompt_count, limit);
    return !options.eager && !options.requires_target_only() && !split_decode;
}

void Decoder::Model::generate_serial(const std::vector<int>& prompt, const DecoderOptions& options,
                                     const DecoderCallbacks& callbacks,
                                     DecoderResult& result,
                                     std::shared_ptr<const MultimodalInput> media,
                                     const std::vector<int>* teacher) const {
    if (tp) {
        throw std::runtime_error("Serial generation requires tensor parallel size 1");
    }
    const int prompt_count = static_cast<int>(prompt.size());
    const int limit = RequestProgress::capacity(prompt_count, options);
    const int kv_limit = options.incremental_kv ? limit : 0;
    int capacity = kv_limit ? prompt_count : limit;
    const int pages = (capacity + mxfp8_page_size - 1) / mxfp8_page_size;
    const auto sampling = options.samples() ? std::optional{request_sampling(options)}
                                            : std::nullopt;
    const bool adjusted = options.requires_target_only();
    const bool reuse_graph = !teacher && serial_graph(options, prompt_count, limit);
    const RequestLease lease(request_in_use, reuse_graph);
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
            if (media) storage->memory.retain_stream_owner(media);
        } else {
            const std::shared_ptr<const void> owners = media
                ? std::shared_ptr<const void>(media) : memory;
            storage = std::make_unique<Request>(device, capacity, owners, lease.owns_slot(),
                                                kv_limit);
        }
        auto& request = *storage;
        AdjustedLogitsWorkspace adjusted_workspace;
        prepare_serial_request(request, options, adjusted ? &adjusted_workspace : nullptr);
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
                advance_serial(request, progress, input, rotary_delta);
                progress.select(request, request.work, options, request.constraint_workspace,
                                sampling ? &*sampling : nullptr, request.sampling_workspace,
                                stream, adjusted ? &adjusted_workspace : nullptr);
                synchronize_cuda(stream, "Complete decoder step");
                const int token = read_token(request, sampling || progress.generated <
                                             options.min_new_tokens,
                                             options.constraint != nullptr);
                const auto& transfers = request.memory.transfers();
                if (const auto finish = progress.commit_single(
                        options, token, transfers.logits.data(), &output,
                        transfers.adjusted_logits.data(), &transfers.repetition_logits)) {
                    return *finish;
                }
            }
            return DecoderFinishReason::length;
        };
        result.finish_reason = run_request();
        if (eager_request) {
            request.memory.close();
        } else {
            if (result.output_token_ids.empty()) {
                synchronize_cuda(request.memory.stream(), "Complete interrupted prefill");
            }
            if (media) {
                request.memory.free_block(const_cast<MropePosition*>(prefill.positions),
                                          media->prompt.positions.size() * sizeof(MropePosition));
                request.memory.retain_stream_owner(memory);
            }
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
    if (proposals == 0 && !tp_) {
        result.output_token_ids.reserve(options.max_new_tokens);
        model_->generate_serial(prompt_token_ids, options, callbacks, result);
    } else {
        
        
        
        const int tile = model_->execution.prefill_limit();
        const int limit = options.max_num_batched_tokens
            ? std::min(options.max_num_batched_tokens, tile) : tile;
        DecoderBatch batch(*this, DecoderBatchOptions{false, 1, 0, limit});
        SerialOutput output(callbacks, options.top_logits, result);
        batch.generate_mtp(prompt_token_ids, options, output, result);
    }
    selected_device.restore();
    return result;
}

void Decoder::Model::compute_logits(Request& request, bool decode) const {
    compute_logits(request.work, request.memory.stream(), decode);
}

void Decoder::Model::normalize_final(Workspace& work, cudaStream_t stream,
                                     const TextModelStateRow* state) const {
    if (state) {
        check(text_final_rms_bound(work.hidden, work.residual, final_norm, work.normalized,
                                   state, stream), "Normalize bound final residual");
        return;
    }
    check(text_rms_norm(work.hidden, work.residual, final_norm, hidden_size,
                        work.normalized, nullptr, stream), "Normalize final residual");
}

bool Decoder::Model::compute_eager_decode_logits(const ForwardLane* lanes, int count,
                                                cudaStream_t stream) const {
    DecodeHeadRows rows;
    if (count != static_cast<int>(rows.size())) {
        return false;
    }
    for (int index = 0; index < count; ++index) {
        rows[index] = &lanes[index].request->work;
    }
    compute_decode_logits(rows, stream);
    return true;
}

void Decoder::Model::compute_decode_logits(const DecodeHeadRows& rows, cudaStream_t stream,
                                           const DecodeStateRows* states) const {
    DecodeHeadProjections projections;
    static_assert(std::tuple_size_v<DecodeHeadRows> == std::tuple_size_v<DecodeHeadProjections>);
    for (std::size_t index = 0; index < rows.size(); ++index) {
        auto& work = *rows[index];
        normalize_final(work, stream, states ? (*states)[index] : nullptr);
        projections[index] = {work.normalized, work.logits};
    }
    check(decode_bf16_lm_head_batch(lm_head.values, lm_head.rows, lm_head.columns, projections,
                                   stream), "Run grouped decode LM head");
    for (auto* work : rows) {
        check(text_argmax(work->logits, vocab_size, work->argmax_scratch, work->argmax, stream),
              "Select greedy token");
    }
}

void Decoder::Model::compute_prefill_logits(Workspace& work, cudaStream_t stream) const {
    normalize_final(work, stream);
    check(text_bf16_gemv(lm_head.values, work.normalized, lm_head.rows, lm_head.columns,
                         work.logits, stream), "Run BF16 projection");
}

void Decoder::Model::enqueue_label_logits(Request& request, cudaStream_t stream) const {
    compute_prefill_logits(request.work, stream);
    auto& logits = request.memory.transfers().logits;
    check(cudaMemcpyAsync(logits.data(), request.work.logits, logits.size() * sizeof(logits[0]),
                          cudaMemcpyDeviceToHost, stream), "Read final prompt logits");
}

void Decoder::Model::compute_logits(Workspace& work, cudaStream_t stream, bool decode,
                                    const TextModelStateRow* state) const {
    if (decode) {
        normalize_final(work, stream, state);
        check(decode_bf16_lm_head(lm_head.values, work.normalized, lm_head.rows, lm_head.columns,
                                  work.logits, stream), "Run decode LM head");
    } else {
        compute_prefill_logits(work, stream);
    }
    check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
          "Select greedy token");
}

}
