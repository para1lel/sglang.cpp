#include "sglang/decoder_mtp.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_prefill.h"
#include "sglang/decoder_verify.h"
#include "sglang/runtime/backend/cuda_backend_policy.h"
#include "sglang/runtime/model/qwen_weight_upload.h"
#include "sglang/text/mtp/draft_depth.h"
#include "sglang/text/mtp/draft_projection.h"
#include "sglang/text/mtp/tc_vocab.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <string>
#include <utility>

namespace sglang::decoder {
namespace {

static_assert(draft_projection_rows % draft_wave_rows == 0);

struct DraftWorkspace {
    static constexpr int score_storage_count = vocab_size * sizeof(Bf16) / sizeof(float);
    static_assert(score_storage_count * sizeof(float) == vocab_size * sizeof(Bf16));

    explicit DraftWorkspace(Allocations& memory)
        : embedding(memory.allocate<Bf16>(hidden_size)),
          concatenated(memory.allocate<Bf16>(2 * hidden_size)),
          residual(memory.allocate<Bf16>(hidden_size)),
          normalized(memory.allocate<Bf16>(hidden_size)),
          branch(memory.allocate<Bf16>(hidden_size)),
          q_gate(memory.allocate<Bf16>(text_gqa_q_gate_width)),
          keys(memory.allocate<Bf16>(text_gqa_kv_width)),
          values(memory.allocate<Bf16>(text_gqa_kv_width)),
          rope_queries(memory.allocate<Bf16>(text_gqa_q_width)),
          rope_keys(memory.allocate<Bf16>(text_gqa_kv_width)),
          attention(memory.allocate<Bf16>(text_gqa_q_width)),
          gate(memory.allocate<Bf16>(intermediate_size)),
          up(memory.allocate<Bf16>(intermediate_size)),
          score_storage(memory.allocate<float>(score_storage_count)),
          logits(reinterpret_cast<Bf16*>(score_storage)),
          activation{memory.allocate<std::uint8_t>(intermediate_size),
                     memory.allocate<float>(text_fp8_group_count(intermediate_size))},
          argmax_scratch(
              memory.allocate<TextLogitCandidate>(text_argmax_scratch_count(vocab_size))),
          argmax(memory.allocate<TextLogitCandidate>(1)) {}

    ExactGqaWorkspace attention_scores() const {
        return {score_storage, score_storage_count / mxfp8_query_heads};
    }

    Bf16* embedding;
    Bf16* concatenated;
    Bf16* residual;
    Bf16* normalized;
    Bf16* branch;
    Bf16* q_gate;
    Bf16* keys;
    Bf16* values;
    Bf16* rope_queries;
    Bf16* rope_keys;
    Bf16* attention;
    Bf16* gate;
    Bf16* up;
    float* score_storage;
    Bf16* logits;
    TextFp8Activation activation;
    TextLogitCandidate* argmax_scratch;
    TextLogitCandidate* argmax;
};

void quantize(const Bf16* input, int width, const DraftWorkspace& work, cudaStream_t stream) {
    check(text_quantize_fp8(input, width, work.activation, stream), "Quantize MTP activation");
}

void project(Fp8Weight weight, const DraftWorkspace& work, Bf16* output, cudaStream_t stream) {
    const TextFp8ActivationView input{work.activation.values, work.activation.scales};
    check(text_fp8_gemv(weight.values, weight.scales, input, weight.rows, weight.columns, output,
                        stream), "Run MTP FP8 projection");
}

}

struct MtpDraftWeights::State {
    State(const std::filesystem::path& directory, int device,
          std::shared_ptr<const Allocations> target_weights, const Bf16* embedding,
          Bf16Weight lm_head, DraftOperatorCapabilities capabilities)
        : device(device), capabilities(capabilities), target_weights(std::move(target_weights)),
          memory(device), embedding(embedding), lm_head(lm_head) {
        if (!this->target_weights || !embedding || !lm_head.values ||
            lm_head.rows != vocab_size || lm_head.columns != hidden_size) {
            throw std::invalid_argument("Invalid MTP target embedding or LM head");
        }
        DeviceScope selected_device(device);
        auto& checkpoint = memory.transfers().checkpoint;
        checkpoint = std::make_unique<Checkpoint>(directory);
        memory.open_stream();
        QwenWeightUpload upload(*checkpoint, memory, memory.stream());
        embedding_norm = upload.bf16("mtp.pre_fc_norm_embedding.weight", {hidden_size});
        hidden_norm = upload.bf16("mtp.pre_fc_norm_hidden.weight", {hidden_size});
        fc = upload.bf16_matrix("mtp.fc.weight", hidden_size, 2 * hidden_size);
        input_norm = upload.bf16("mtp.layers.0.input_layernorm.weight", {hidden_size});
        attention_norm = upload.bf16("mtp.layers.0.post_attention_layernorm.weight", {hidden_size});
        const std::string attention = "mtp.layers.0.self_attn.";
        gqa = {
            upload.fp8_matrix(attention + "q_proj", text_gqa_q_gate_width, hidden_size),
            upload.fp8_matrix(attention + "k_proj", text_gqa_kv_width, hidden_size),
            upload.fp8_matrix(attention + "v_proj", text_gqa_kv_width, hidden_size),
            upload.bf16(attention + "q_norm.weight", {mxfp8_head_dim}),
            upload.bf16(attention + "k_norm.weight", {mxfp8_head_dim}),
            upload.fp8_matrix(attention + "o_proj", hidden_size, text_gqa_q_width),
        };
        const std::string mlp = "mtp.layers.0.mlp.";
        mlp_gate = upload.fp8_matrix(mlp + "gate_proj", intermediate_size, hidden_size);
        mlp_up = upload.fp8_matrix(mlp + "up_proj", intermediate_size, hidden_size);
        mlp_down = upload.fp8_matrix(mlp + "down_proj", hidden_size, intermediate_size);
        final_norm = upload.bf16("mtp.norm.weight", {hidden_size});
        if (capabilities.int8_head) {
            auto* values = memory.allocate<std::uint8_t>(
                static_cast<std::size_t>(vocab_size) * hidden_size);
            auto* maxima = memory.allocate<float>(vocab_size);
            check(quantize_draft_head_int8(lm_head.values, vocab_size, values, maxima,
                                          memory.stream()), "Quantize MTP vocabulary head");
            quantized_head = values;
            quantized_head_maxima = maxima;
            shared_vocab = prepare_tc_vocab(lm_head.values, values, maxima, memory);
        }
        memory.close_stream();
        if (capabilities.int8_head) {
            std::vector<float> maxima(vocab_size);
            check(cudaMemcpy(maxima.data(), quantized_head_maxima, maxima.size() * sizeof(float),
                             cudaMemcpyDeviceToHost), "Read INT8 MTP vocabulary row maxima");
            for (int token = 0; token < vocab_size; ++token) {
                if (!std::isfinite(maxima[token])) {
                    throw std::runtime_error("Nonfinite INT8 MTP vocabulary weight at token " +
                                             std::to_string(token));
                }
            }
        }
        selected_device.restore();
    }

    int device;
    const DraftOperatorCapabilities capabilities;
    std::shared_ptr<const Allocations> target_weights;
    Allocations memory;
    const Bf16* embedding;
    Bf16Weight lm_head;
    const std::uint8_t* quantized_head = nullptr;
    const float* quantized_head_maxima = nullptr;
    std::shared_ptr<const TcVocabWeights> shared_vocab;
    const Bf16* embedding_norm;
    const Bf16* hidden_norm;
    Bf16Weight fc;
    const Bf16* input_norm;
    const Bf16* attention_norm;
    GqaLayer gqa;
    Fp8Weight mlp_gate;
    Fp8Weight mlp_up;
    Fp8Weight mlp_down;
    const Bf16* final_norm;
};

struct MtpDraftRequest::State {
    State(std::shared_ptr<const MtpDraftWeights> weights, int capacity, RequestBlockCache* cache);
    void validate(int position, int token, const Bf16* hidden) const;
    void embed(int shifted_token, const Bf16* hidden, cudaStream_t stream,
               const Bf16* feature = nullptr);
    void prepare_attention(cudaStream_t stream);
    void append(int position, bool score, const MropePosition* mrope, int rotary_position,
                cudaStream_t stream);
    DraftAttentionRow attention_row(int position);
    void gate_attention(cudaStream_t stream);
    void prepare_mlp(cudaStream_t stream);
    void activate_mlp(cudaStream_t stream);
    void normalize_output(cudaStream_t stream);
    void score(cudaStream_t stream, const ConstraintWorkspace* scratch,
               const ConstraintCandidates* allowed);
    void select_score(cudaStream_t stream, const ConstraintWorkspace* scratch,
                      const ConstraintCandidates* allowed);
    bool prepare_mapped_score(cudaStream_t stream);
    int score_token();
    static void score_wave(State* const* batch, int count, cudaStream_t stream);
    static void select_wave(State* const* batch, int count, cudaStream_t stream);

    std::shared_ptr<const MtpDraftWeights> weights;
    Allocations memory;
    DraftWorkspace work;
    int capacity;
    Mxfp8KvCache cache;
    Mxfp8PageTable pages;
    int prefix = 0;
    bool scored = false;
    bool constrained_score = false;
    bool int8_score = false;
    bool retried_score = false;
    bool mapped_score = false;
    TextLogitCandidate* device_score = nullptr;
    TextLogitCandidate retry_winner{};
    bool failed = false;
};

MtpDraftRequest::State::State(std::shared_ptr<const MtpDraftWeights> weights, int capacity,
                              RequestBlockCache* block_cache)
    : weights(std::move(weights)), memory(this->weights->state_->device, block_cache), work(memory),
      capacity(capacity) {
    const int page_count = (capacity + mxfp8_page_size - 1) / mxfp8_page_size;
    const auto payload = page_count * mxfp8_payload_bytes_per_page;
    const auto scales = page_count * mxfp8_scale_bytes_per_page;
    cache = {
        memory.allocate<std::uint8_t>(payload), memory.allocate<std::uint8_t>(payload),
        memory.allocate<std::uint8_t>(scales), memory.allocate<std::uint8_t>(scales), page_count,
    };
    auto* table = memory.allocate<std::int32_t>(page_count);
    pages = {table, page_count};
    auto& host_pages = memory.transfers().logical_pages;
    host_pages.resize(page_count);
    std::iota(host_pages.begin(), host_pages.end(), 0);
    memory.open_stream();
    copy_cuda_async(table, host_pages.data(), host_pages.size() * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice, memory.stream(), "Upload MTP KV page table");
    memory.close_stream();
}

void MtpDraftRequest::State::validate(int position, int token, const Bf16* hidden) const {
    if (failed) {
        throw std::logic_error("MTP draft request failed");
    }
    if (position != prefix || position >= capacity || token < 0 || token >= vocab_size || !hidden) {
        throw std::invalid_argument("Invalid MTP draft input");
    }
}

void MtpDraftRequest::State::embed(int shifted_token, const Bf16* hidden, cudaStream_t stream,
                                  const Bf16* feature) {
    const auto& model = *weights->state_;
    if (!feature) {
        check(text_embedding(model.embedding, vocab_size, hidden_size, shifted_token,
                             work.embedding, stream), "Look up shifted MTP token");
    }
    check(text_rms_norm(feature ? feature : work.embedding, nullptr, model.embedding_norm,
                        hidden_size, work.concatenated, nullptr, stream),
          "Normalize MTP embedding");
    check(text_rms_norm(hidden, nullptr, model.hidden_norm, hidden_size,
                        work.concatenated + hidden_size, nullptr, stream), "Normalize MTP hidden");
}

void MtpDraftRequest::State::prepare_attention(cudaStream_t stream) {
    const auto& model = *weights->state_;
    check(text_rms_norm(work.residual, nullptr, model.input_norm, hidden_size, work.normalized,
                        nullptr, stream), "Normalize MTP layer input");
    quantize(work.normalized, hidden_size, work, stream);
}

void MtpDraftRequest::State::append(int position, bool score, const MropePosition* mrope,
                                   int rotary_position, cudaStream_t stream) {
    const auto& gqa = weights->state_->gqa;
    const int rotary = rotary_position < 0 ? position : rotary_position;
    if (score) {
        const auto status = mrope
            ? mrope_gqa_preprocess(mrope, 1, work.q_gate, work.keys, gqa.query_norm, gqa.key_norm,
                                   work.rope_queries, work.rope_keys, stream)
            : text_gqa_preprocess(rotary, 1, work.q_gate, work.keys, gqa.query_norm, gqa.key_norm,
                                  work.rope_queries, work.rope_keys, stream);
        check(status, "Rotate MTP Q/K");
    } else {
        const auto status = mrope
            ? mrope_gqa_preprocess_keys(mrope, 1, work.keys, gqa.key_norm, work.rope_keys, stream)
            : text_gqa_preprocess_keys(rotary, 1, work.keys, gqa.key_norm, work.rope_keys, stream);
        check(status, "Rotate MTP key");
    }
    check(mxfp8_kv_append(cache, pages, position, 1, work.rope_keys, work.values, stream),
          "Append MTP MXFP8 KV");
}

DraftAttentionRow MtpDraftRequest::State::attention_row(int position) {
    return {cache, pages, position, work.rope_queries, work.attention, work.attention_scores()};
}

void MtpDraftRequest::State::gate_attention(cudaStream_t stream) {
    check(text_gqa_postgate(1, work.q_gate, work.attention, work.attention, stream),
          "Gate MTP attention");
    quantize(work.attention, text_gqa_q_width, work, stream);
}

void MtpDraftRequest::State::prepare_mlp(cudaStream_t stream) {
    const auto& model = *weights->state_;
    check(text_rms_norm(work.branch, work.residual, model.attention_norm, hidden_size,
                        work.normalized, work.residual, stream),
          "Normalize MTP post-attention residual");
    quantize(work.normalized, hidden_size, work, stream);
}

void MtpDraftRequest::State::activate_mlp(cudaStream_t stream) {
    check(text_swiglu(work.gate, work.up, intermediate_size, work.gate, stream),
          "Apply MTP SwiGLU");
    quantize(work.gate, intermediate_size, work, stream);
}

void MtpDraftRequest::State::normalize_output(cudaStream_t stream) {
    const auto& model = *weights->state_;
    check(text_rms_norm(work.branch, work.residual, model.final_norm, hidden_size, work.normalized,
                        nullptr, stream), "Normalize MTP output");
}

void MtpDraftRequest::State::score(cudaStream_t stream, const ConstraintWorkspace* scratch,
                                  const ConstraintCandidates* allowed) {
    const auto& model = *weights->state_;
    const auto& head = model.lm_head;
    if (model.quantized_head && !allowed) {
        check(decode_draft_head_int8(model.quantized_head, model.quantized_head_maxima,
                                     work.normalized, work.activation.values,
                                     work.activation.scales, work.logits, stream),
              "Run INT8 vocabulary head for MTP");
    } else {
        check(text_bf16_gemv(head.values, work.normalized, head.rows, head.columns,
                             work.logits, stream), "Run shared LM head for MTP");
    }
    select_score(stream, scratch, allowed);
}

void MtpDraftRequest::State::select_score(cudaStream_t stream, const ConstraintWorkspace* scratch,
                                         const ConstraintCandidates* allowed) {
    int8_score = weights->state_->quantized_head && !allowed;
    retried_score = false;
    mapped_score = false;
    if (allowed) {
        select_constrained({work.logits, work.argmax_scratch, work.argmax}, *scratch, *allowed,
                           nullptr, 0, nullptr, stream);
    } else if (int8_score) {
        check(text_argmax_finite(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
              "Select finite INT8 MTP token");
    } else {
        check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
              "Select MTP token");
    }
    constrained_score = allowed != nullptr;
    auto& winner = memory.transfers().winner;
    copy_cuda_async(&winner, &work.argmax->token_id, sizeof(winner), cudaMemcpyDeviceToHost,
                    stream, "Copy MTP token");
    scored = true;
}

bool MtpDraftRequest::State::prepare_mapped_score(cudaStream_t stream) {
    if (device_score) return true;
    cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream, &capture), "Check MTP draft capture status");
    if (capture != cudaStreamCaptureStatusNone) return false;
    auto owner = prepare_verification_winners(1);
    if (!owner) return false;
    void* device = nullptr;
    check(cudaHostGetDevicePointer(&device, owner.get(), 0), "Map MTP draft winner storage");
    memory.transfers().verification_results = std::move(owner);
    device_score = static_cast<TextLogitCandidate*>(device);
    return true;
}

int MtpDraftRequest::State::score_token() {
    const auto& transfer = memory.transfers();
    const auto* host = static_cast<const TextLogitCandidate*>(transfer.verification_results.get());
    return mapped_score ? host->token_id : transfer.winner;
}

void MtpDraftRequest::State::score_wave(State* const* batch, int count, cudaStream_t stream) {
    const auto& model = *batch[0]->weights->state_;
    if (model.quantized_head) {
        std::array<DraftInt8Projection, draft_wave_rows> projections;
        for (int index = 0; index < count; ++index) {
            const auto& work = batch[index]->work;
            projections[index] = {work.normalized, work.activation.values,
                                  work.activation.scales, work.logits};
        }
        check(decode_draft_head_int8_batch(model.quantized_head, model.quantized_head_maxima,
                                           projections.data(), count, stream),
              "Run batched INT8 MTP vocabulary head");
    } else {
        std::array<TextBf16Projection, draft_wave_rows> projections;
        for (int index = 0; index < count; ++index) {
            const auto& work = batch[index]->work;
            projections[index] = {work.normalized, work.logits};
        }
        const auto& head = model.lm_head;
        check(text_bf16_gemm(head.values, head.rows, head.columns, projections.data(),
                             count, stream), "Run batched BF16 MTP vocabulary head");
    }
    select_wave(batch, count, stream);
}

void MtpDraftRequest::State::select_wave(State* const* batch, int count, cudaStream_t stream) {
    const auto& model = *batch[0]->weights->state_;
    DraftArgmaxRows winners{};
    if (count > 1 && model.capabilities.grouped_argmax &&
        std::all_of(batch, batch + count, [stream](State* row) {
            return row->prepare_mapped_score(stream);
        })) {
        for (int index = 0; index < count; ++index) {
            auto& row = *batch[index];
            winners.rows[index] = {row.work.logits, row.work.argmax_scratch, row.device_score};
            row.mapped_score = row.scored = true;
            row.int8_score = model.quantized_head != nullptr;
            row.constrained_score = row.retried_score = false;
        }
        check(draft_argmax_wave(winners, count, vocab_size, model.quantized_head != nullptr,
                                 stream), "Select grouped MTP draft winners");
    } else {
        for (int index = 0; index < count; ++index) {
            batch[index]->select_score(stream, nullptr, nullptr);
        }
    }
}

MtpDraftWeights::MtpDraftWeights(const std::filesystem::path& checkpoint, int device,
                                 std::shared_ptr<const Allocations> target_weights,
                                 const Bf16* embedding, Bf16Weight lm_head,
                                 DraftOperatorCapabilities capabilities)
    : state_(std::make_unique<State>(checkpoint, device, std::move(target_weights), embedding,
                                     lm_head, capabilities)) {}

MtpDraftWeights::~MtpDraftWeights() = default;

std::size_t MtpDraftWeights::device_allocation_bytes() const { return state_->memory.bytes(); }

std::shared_ptr<const TcVocabWeights> MtpDraftWeights::shared_vocab() const {
    return state_->shared_vocab;
}

MtpDraftRequest::MtpDraftRequest(std::shared_ptr<const MtpDraftWeights> weights, int capacity,
                                 RequestBlockCache* cache) {
    if (!weights || capacity < 1 || capacity > text_attention_max_tokens) {
        throw std::invalid_argument("Invalid MTP draft weights or capacity");
    }
    DeviceScope selected_device(weights->state_->device);
    state_ = std::make_unique<State>(std::move(weights), capacity, cache);
    selected_device.restore();
}

MtpDraftRequest::~MtpDraftRequest() = default;

void MtpDraftRequest::recycle_after_completion() {
    state_->memory.recycle_after_completion();
}

const Bf16* MtpDraftRequest::enqueue(int position, int shifted_token, const Bf16* hidden,
                                    bool score, cudaStream_t stream,
                                    const ConstraintWorkspace* scratch,
                                    const ConstraintCandidates* allowed,
                                    const MropePosition* mrope, int rotary_position,
                                    const Bf16* feature) {
    return enqueue_impl(position, shifted_token, hidden, score, stream, scratch, allowed, false,
                        mrope, rotary_position, feature);
}

const Bf16* MtpDraftRequest::enqueue_impl(int position, int shifted_token, const Bf16* hidden,
                                        bool score, cudaStream_t stream,
                                        const ConstraintWorkspace* scratch,
                                        const ConstraintCandidates* allowed, bool defer_score,
                                        const MropePosition* mrope, int rotary_position,
                                        const Bf16* feature) {
    auto& request = *state_;
    request.validate(position, shifted_token, hidden);
    auto& work = request.work;
    const auto& model = *request.weights->state_;
    const auto& gqa = model.gqa;
    try {
        request.embed(shifted_token, hidden, stream, feature);
        check(text_bf16_gemv(model.fc.values, work.concatenated, model.fc.rows, model.fc.columns,
                             work.residual, stream), "Run BF16 MTP FC");
        request.prepare_attention(stream);
        if (score) {
            project(gqa.q_gate, work, work.q_gate, stream);
        }
        project(gqa.key, work, work.keys, stream);
        project(gqa.value, work, work.values, stream);
        request.append(position, score, mrope, rotary_position, stream);
        if (score) {
            const DraftAttentionRow attention = request.attention_row(position);
            check(mtp_gqa_attention_batch(&attention, 1, model.capabilities.online_attention,
                                          stream),
                  "Read MTP MXFP8 attention");
            request.gate_attention(stream);
            project(gqa.output, work, work.branch, stream);
            request.prepare_mlp(stream);
            project(model.mlp_gate, work, work.gate, stream);
            project(model.mlp_up, work, work.up, stream);
            request.activate_mlp(stream);
            project(model.mlp_down, work, work.branch, stream);
            request.normalize_output(stream);
            if (!defer_score) request.score(stream, scratch, allowed);
        }
    } catch (...) {
        request.failed = true;
        throw;
    }
    ++request.prefix;
    return score ? request.work.normalized : nullptr;
}

void MtpDraftRequest::enqueue_wave(MtpDraftInput* rows, int count, bool score,
                                   cudaStream_t stream) {
    enqueue_wave_impl(rows, count, score, stream, false);
}

void MtpDraftRequest::enqueue_wave_impl(MtpDraftInput* rows, int count, bool score,
                                       cudaStream_t stream, bool defer_score, Bf16* packed,
                                       bool exact) {
    if (count == 0) return;
    static_assert(draft_wave_rows == decoder_batch_max_mtp_lanes);
    if (!rows || count < 0 || (count > draft_wave_rows &&
                               (!packed || count > draft_projection_rows))) {
        throw std::invalid_argument("Invalid MTP draft wave size");
    }
    if (count == 1) {
        auto& row = rows[0];
        row.hidden = row.request->enqueue_impl(row.position, row.token, row.hidden, score, stream,
                                               nullptr, nullptr, defer_score, row.mrope,
                                               row.rotary_position, nullptr);
        return;
    }
    std::array<State*, draft_projection_rows> states;
    const auto weights = rows[0].request->state_->weights.get();
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        auto& request = *row.request->state_;
        request.validate(row.position, row.token, row.hidden);
        if (request.weights.get() != weights ||
            std::find(states.begin(), states.begin() + index, &request) != states.begin() + index) {
            throw std::invalid_argument("Invalid MTP draft wave lanes");
        }
        states[index] = &request;
    }
    const auto& model = *weights->state_;
    DraftProjectionBatch<State, DraftWorkspace> batch(states.data(), count, packed, stream, exact);
    const auto apply = [&](void (State::* function)(cudaStream_t)) {
        for (int index = 0; index < count; ++index) {
            std::invoke(function, states[index], stream);
        }
    };
    const bool grouped_rope = packed && std::all_of(rows, rows + count,
        [](const MtpDraftInput& row) {
            const int rotary = row.rotary_position < 0 ? row.position : row.rotary_position;
            return !row.mrope && rotary < text_attention_max_tokens;
        });
    try {
        batch.prepare_fc(rows, model.embedding, model.embedding_norm, model.hidden_norm, model.fc);
        if (packed) {
            batch.rms_activation(&DraftWorkspace::residual, model.input_norm, false);
        } else {
            apply(&State::prepare_attention);
        }
        if (score) {
            batch.project({{model.gqa.q_gate, &DraftWorkspace::q_gate},
                           {model.gqa.key, &DraftWorkspace::keys},
                           {model.gqa.value, &DraftWorkspace::values}});
        } else {
            batch.project({{model.gqa.key, &DraftWorkspace::keys},
                           {model.gqa.value, &DraftWorkspace::values}});
        }
        if (grouped_rope) {
            batch.append(rows, model.gqa.query_norm, model.gqa.key_norm, score);
        } else {
            for (int lane = 0; lane < count; ++lane) {
                const auto& row = rows[lane];
                states[lane]->append(row.position, score, row.mrope, row.rotary_position, stream);
            }
        }
        if (score) {
            batch.read_attention(rows, model.capabilities.online_attention);
            if (packed) batch.gate_attention();
            else apply(&State::gate_attention);
            batch.project({{model.gqa.output, &DraftWorkspace::branch}});
            if (packed) {
                batch.rms_activation(&DraftWorkspace::branch, model.attention_norm, true);
            } else {
                apply(&State::prepare_mlp);
            }
            batch.project({{model.mlp_gate, &DraftWorkspace::gate},
                           {model.mlp_up, &DraftWorkspace::up}});
            if (packed) {
                batch.quantize_activation(&DraftWorkspace::gate, &DraftWorkspace::up,
                                          intermediate_size);
            } else {
                apply(&State::activate_mlp);
            }
            batch.project({{model.mlp_down, &DraftWorkspace::branch}});
            if (packed) batch.normalize_output(model.final_norm);
            else apply(&State::normalize_output);
            if (!defer_score) State::score_wave(states.data(), count, stream);
        }
    } catch (...) {
        for (int index = 0; index < count; ++index) {
            states[index]->failed = true;
        }
        throw;
    }
    for (int index = 0; index < count; ++index) {
        ++states[index]->prefix;
        rows[index].hidden = score ? states[index]->work.normalized : nullptr;
    }
}

bool MtpDraftRequest::prepare_depth(MtpDraftRequest& lane, DraftDepthScratch& scratch,
                                    std::uint8_t* activation, cudaStream_t stream) {
    const auto& model = *lane.state_->weights->state_;
    if (!model.quantized_head || !activation) return false;
    if (scratch.activation == activation) return scratch.ready;
    static_assert(wide_mtp_rows * intermediate_size >=
                  decoder_batch_max_lanes * (hidden_size + sizeof(float)));
    static_assert((decoder_batch_max_lanes * hidden_size) % alignof(float) == 0);
    scratch.activation = activation;
    scratch.maxima = reinterpret_cast<float*>(activation + decoder_batch_max_lanes * hidden_size);
    const auto status = prepare_draft_depth(scratch, model.quantized_head, stream);
    if (status && *status != cudaSuccess) {
        scratch.activation = nullptr;
        check(*status, "Prepare depth-major INT8 MTP vocabulary projection");
    }
    scratch.ready = status.has_value();
    return scratch.ready;
}

void MtpDraftRequest::enqueue_depth(MtpDraftInput* rows, int count,
                                    const DraftDepthScratch& scratch, cudaStream_t stream) {
    if (!rows || count < 1 || count > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid wide MTP depth size");
    }
    std::array<State*, decoder_batch_max_lanes> active;
    std::array<int, decoder_batch_max_lanes / draft_wave_rows> packet_sizes{};
    DraftDepthRows vocab{};
    const MtpDraftWeights* weights = nullptr;
    int active_count = 0;
    for (int index = 0; index < count; ++index) {
        if (!rows[index].request) continue;
        auto* state = rows[index].request->state_.get();
        if (!weights) weights = state->weights.get();
        const auto end = active.begin() + active_count;
        if (state->weights.get() != weights || std::find(active.begin(), end, state) != end) {
            throw std::invalid_argument("Invalid wide MTP depth lanes");
        }
        state->validate(rows[index].position, rows[index].token, rows[index].hidden);
        const auto& work = state->work;
        vocab.row[active_count] = {
            work.normalized, work.activation.values, work.activation.scales, work.logits};
        active[active_count++] = state;
        ++packet_sizes[index / draft_wave_rows];
    }
    if (active_count == 0) return;
    try {
        for (int first = 0; first < count;) {
            const int group = first / draft_wave_rows;
            const int projection_rows = count == 2 * draft_wave_rows
                ? count : draft_projection_rows;
            const int groups = projection_rows / draft_wave_rows;
            if (first + projection_rows <= count &&
                std::all_of(packet_sizes.begin() + group, packet_sizes.begin() + group + groups,
                            [](int size) { return size == draft_wave_rows; })) {
                enqueue_wave_impl(rows + first, projection_rows, true, stream, true,
                                  reinterpret_cast<Bf16*>(scratch.activation));
                first += projection_rows;
                continue;
            }
            std::array<MtpDraftInput, draft_wave_rows> packet;
            std::array<int, draft_wave_rows> slots;
            int size = 0;
            for (int index = first; index < std::min(first + draft_wave_rows, count); ++index) {
                if (rows[index].request) {
                    slots[size] = index;
                    packet[size++] = rows[index];
                }
            }
            enqueue_wave_impl(packet.data(), size, true, stream, true);
            for (int index = 0; index < size; ++index) {
                rows[slots[index]].hidden = packet[index].hidden;
            }
            first += draft_wave_rows;
        }
        check(project_draft_depth(scratch, weights->state_->quantized_head_maxima, vocab,
                                  active_count, stream),
              "Run depth-major INT8 MTP vocabulary projection");
        int offset = 0;
        for (const int size : packet_sizes) {
            if (size) State::select_wave(active.data() + offset, size, stream);
            offset += size;
        }
    } catch (...) {
        for (int index = 0; index < active_count; ++index) active[index]->failed = true;
        throw;
    }
}

void MtpDraftRequest::repair_wave(const DraftRepairs* repairs, int count, Bf16* packed,
                                  cudaStream_t stream, bool exact) {
    if (!repairs || count <= 0 || (count > draft_wave_rows &&
                                  (!packed || count > draft_projection_rows ||
                                   count % draft_wave_rows != 0))) {
        throw std::invalid_argument("Invalid MTP draft repair size");
    }
    std::array<MtpDraftInput, draft_projection_rows> rows;
    for (int depth = 0; depth < decoder_batch_max_mtp_steps; ++depth) {
        int size = 0;
        for (int lane = 0; lane < count; ++lane) {
            if (const auto& row = repairs[lane][depth]; row.request) rows[size++] = row;
        }
        if (size == 0) break;
        const bool grouped = !exact || (count == draft_projection_rows &&
            std::all_of(rows.begin(), rows.begin() + size, [](const MtpDraftInput& row) {
                const int rotary = row.rotary_position < 0 ? row.position : row.rotary_position;
                return !row.mrope && rotary < text_attention_max_tokens;
            }));
        if (packed && size == count && grouped) {
            enqueue_wave_impl(rows.data(), size, false, stream, false, packed, exact);
        } else if (count <= draft_wave_rows) {
            enqueue_wave(rows.data(), size, false, stream);
        } else {
            for (int first = 0; first < count; first += draft_wave_rows) {
                size = 0;
                for (int lane = first; lane < first + draft_wave_rows; ++lane) {
                    if (const auto& row = repairs[lane][depth]; row.request) rows[size++] = row;
                }
                enqueue_wave(rows.data(), size, false, stream);
            }
        }
    }
}

void MtpDraftRequest::seed_prefill(int position, const int* shifted_tokens, const Bf16* hidden,
                                   const Bf16* residual, const Bf16* final_norm,
                                   const PrefillProjections& projections,
                                   MtpPrefillBuffers buffers, const MediaPrefill* media) {
    auto& request = *state_;
    if (request.failed) {
        throw std::logic_error("MTP draft request failed");
    }
    const int rows = projections.rows;
    if (position != request.prefix || rows <= 0 || rows > request.capacity - position ||
        !shifted_tokens || !hidden || !residual || !final_norm) {
        throw std::invalid_argument("Invalid MTP draft prefill input");
    }
    const auto& model = *request.weights->state_;
    const auto stream = projections.stream;
    const bool paired = model.capabilities.paired_seed_kv && rows > 1;
    const auto append = [&](int first, int count) {
        const auto status = media
            ? mrope_gqa_preprocess_keys(media->positions + first, count, buffers.keys,
                                        model.gqa.key_norm, buffers.rope_keys, stream)
            : text_gqa_preprocess_keys(first, count, buffers.keys, model.gqa.key_norm,
                                       buffers.rope_keys, stream);
        check(status, "Rotate MTP prefill keys");
        check(mxfp8_kv_append(request.cache, request.pages, first, count, buffers.rope_keys,
                              buffers.values, stream), "Append MTP prefill MXFP8 KV");
        request.prefix += count;
        request.scored = false;
    };
    const auto append_prepared = [&](int count) {
        if (!paired || count == 0) {
            return;
        }
        const TextFp8ActivationView input{projections.activation.values,
                                          projections.activation.scales};
        try {
            check(mtp_seed_project_kv(model.gqa.key, model.gqa.value, input, count,
                                       buffers.keys, buffers.values, stream),
                  "Run exact prefill FP8 projection");
            append(position, count);
        } catch (...) {
            request.failed = true;
            throw;
        }
    };
    for (int offset = 0; offset < rows; offset += prefill_chunk_rows) {
        auto tile = projections;
        tile.rows = std::min(prefill_chunk_rows, rows - offset);
        const int hidden_offset = offset * hidden_size;
        if (paired) {
            tile.activation.values += hidden_offset;
            tile.activation.scales += offset * text_fp8_group_count(hidden_size);
        }
        const auto status = text_rms_norm_rows(
            hidden + hidden_offset, residual + hidden_offset, final_norm, hidden_size, tile.rows,
            buffers.normalized_hidden, nullptr, stream);
        if (status != cudaSuccess) {
            append_prepared(offset);
            check(status, "Normalize target prompt rows for MTP");
        }
        try {
            check(text_embedding_rows(model.embedding, vocab_size, hidden_size,
                                       shifted_tokens + offset, tile.rows, buffers.embedding,
                                       stream), "Gather shifted MTP tokens");
            if (media) {
                media->substitute(buffers.embedding, position + offset + 1, tile.rows, stream);
            }
            check(text_rms_norm_pair_rows(buffers.embedding, buffers.normalized_hidden,
                                          model.embedding_norm, model.hidden_norm, hidden_size,
                                          tile.rows, buffers.concatenated, stream),
                  "Normalize and concatenate MTP prefill inputs");
            tile.project_seed_fc(model.fc, buffers.concatenated, buffers.embedding);
            check(text_rms_norm_rows(buffers.embedding, nullptr, model.input_norm, hidden_size,
                                     tile.rows, buffers.layer_input, nullptr, stream),
                  "Normalize MTP prefill layer input");
            const auto input = tile.quantize(buffers.layer_input, hidden_size);
            if (!paired) {
                tile.project(model.gqa.key, input, buffers.keys);
                tile.project(model.gqa.value, input, buffers.values);
                append(position + offset, tile.rows);
            }
        } catch (...) {
            append_prepared(offset);
            request.failed = true;
            throw;
        }
    }
    append_prepared(rows);
}

int MtpDraftRequest::read_token() const {
    if (state_->failed || !state_->scored) {
        throw std::logic_error("MTP draft has no completed score to read");
    }
    if (state_->retried_score) {
        const auto& winner = state_->retry_winner;
        if (winner.token_id < 0 || winner.token_id >= vocab_size ||
            !std::isfinite(winner.score)) {
            std::fputs("INT8 MTP BF16 retry has no finite winner; dropping proposal\n", stderr);
            return -1;
        }
        return winner.token_id;
    }
    const int token = state_->score_token();
    if (token == -1 && state_->constrained_score) {
        return -1;
    }
    if (token < 0 || token >= vocab_size) {
        throw std::runtime_error("MTP argmax did not return a valid token ID");
    }
    return token;
}

bool MtpDraftRequest::retry_invalid_int8(cudaStream_t stream) {
    auto& request = *state_;
    if (!request.int8_score) return false;
    const int token = request.score_token();
    if (token >= 0 && token < vocab_size) {
        return false;
    }
    std::fputs("INT8 MTP scores are invalid; retrying draft vocabulary head in BF16\n", stderr);
    const auto& head = request.weights->state_->lm_head;
    auto& work = request.work;
    check(text_bf16_gemv(head.values, work.normalized, head.rows, head.columns,
                         work.logits, stream),
          "Retry MTP vocabulary head in BF16");
    check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
          "Select BF16 MTP retry token");
    copy_cuda_async(&request.retry_winner, work.argmax, sizeof(request.retry_winner),
                    cudaMemcpyDeviceToHost, stream, "Copy BF16 MTP retry winner");
    request.int8_score = false;
    request.retried_score = true;
    return true;
}

int MtpDraftRequest::valid_prefix() const { return state_->prefix; }

Mxfp8KvCache MtpDraftRequest::prefix_cache_view(int rows, bool storing) const {
    if (state_->failed || rows < 0 || rows >= state_->capacity ||
        (storing ? rows > state_->prefix : state_->prefix != 0)) {
        throw std::logic_error("Invalid MTP prefix view extent");
    }
    return state_->cache;
}

const Allocations& MtpDraftRequest::prefix_cache_memory() const { return state_->memory; }

void MtpDraftRequest::commit_restored_prefix(int rows) {
    state_->prefix = rows;
}

void MtpDraftRequest::rewind(int valid_tokens) {
    if (state_->failed) {
        throw std::logic_error("MTP draft request failed");
    }
    if (valid_tokens < 0 || valid_tokens > state_->prefix) {
        throw std::invalid_argument("Invalid MTP draft rewind");
    }
    state_->prefix = valid_tokens;
}

std::size_t MtpDraftRequest::device_allocation_bytes() const { return state_->memory.bytes(); }

}
