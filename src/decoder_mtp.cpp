#include "sglang/decoder_mtp.h"

#include "sglang/checkpoint.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_prefill.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <string>
#include <utility>

namespace sglang::decoder {
namespace {

class DraftWeightUpload {
public:
    explicit DraftWeightUpload(Allocations& memory)
        : memory_(memory), checkpoint_(*memory.transfers().checkpoint) {}

    const Bf16* norm(const std::string& name, int width) {
        return bf16(name, {extent(width)});
    }

    Bf16Weight fc(int rows, int columns) {
        return {bf16("mtp.fc.weight", {extent(rows), extent(columns)}), rows, columns};
    }

    Fp8Weight projection(const std::string& name, int rows, int columns) {
        const auto* values = upload(name + ".weight", TensorDtype::F8_E4M3,
                                    {extent(rows), extent(columns)});
        const auto* scales = bf16(name + ".weight_scale_inv",
                                  {extent(text_fp8_group_count(rows)),
                                   extent(text_fp8_group_count(columns))});
        return {values, scales, rows, columns};
    }

private:
    static std::uint64_t extent(int size) { return static_cast<std::uint64_t>(size); }

    const Bf16* bf16(const std::string& name, std::initializer_list<std::uint64_t> shape) {
        return reinterpret_cast<const Bf16*>(upload(name, TensorDtype::BF16, shape));
    }

    const std::uint8_t* upload(const std::string& name, TensorDtype dtype,
                               std::initializer_list<std::uint64_t> shape) {
        const auto& tensor = checkpoint_.tensor(name, dtype, shape);
        auto* destination = memory_.allocate<std::uint8_t>(tensor.size_bytes);
        check(cudaMemcpyAsync(destination, tensor.data, tensor.size_bytes, cudaMemcpyHostToDevice,
                              memory_.stream()), name.c_str());
        return destination;
    }

    Allocations& memory_;
    const Checkpoint& checkpoint_;
};

// The checkpoint ends one row before a page boundary. The final scale group is
// the last group in each head's interleaved scale plane.
void copy_draft_prefix(Mxfp8KvCache dst, Mxfp8KvCache src, int rows, cudaStream_t stream) {
    const int full_pages = rows / mxfp8_page_size;
    const int pages = full_pages + 1;
    if (rows < 1 || rows % mxfp8_page_size != mxfp8_page_size - 1 ||
        pages > dst.page_count || pages > src.page_count) {
        throw std::invalid_argument("Invalid MTP checkpoint extent");
    }
    const auto payload = rows * mxfp8_kv_heads * mxfp8_head_dim;
    const auto scale_offset = full_pages * mxfp8_scale_bytes_per_page;
    constexpr auto head_scales = mxfp8_page_size * mxfp8_groups;
    const auto copy_payload = [&](auto* to, const auto* from) {
        check(cudaMemcpyAsync(to, from, payload, cudaMemcpyDeviceToDevice, stream),
              "Copy MTP prefix payload");
    };
    const auto copy_scales = [&](auto* to, const auto* from) {
        if (full_pages != 0) {
            check(cudaMemcpyAsync(to, from, scale_offset, cudaMemcpyDeviceToDevice, stream),
                  "Copy complete MTP prefix scale pages");
        }
        check(cudaMemcpy2DAsync(to + scale_offset, head_scales, from + scale_offset, head_scales,
                                head_scales - mxfp8_groups, mxfp8_kv_heads,
                                cudaMemcpyDeviceToDevice, stream), "Copy MTP prefix tail scales");
    };
    copy_payload(dst.key_payload, src.key_payload);
    copy_payload(dst.value_payload, src.value_payload);
    copy_scales(dst.key_scales, src.key_scales);
    copy_scales(dst.value_scales, src.value_scales);
}

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
          Bf16Weight lm_head, bool int8_head)
        : device(device), target_weights(std::move(target_weights)), memory(device),
          embedding(embedding), lm_head(lm_head) {
        if (!this->target_weights || !embedding || !lm_head.values ||
            lm_head.rows != vocab_size || lm_head.columns != hidden_size) {
            throw std::invalid_argument("Invalid MTP target embedding or LM head");
        }
        DeviceScope selected_device(device);
        auto& checkpoint = memory.transfers().checkpoint;
        checkpoint = std::make_unique<Checkpoint>(directory);
        memory.open_stream();
        DraftWeightUpload upload(memory);
        embedding_norm = upload.norm("mtp.pre_fc_norm_embedding.weight", hidden_size);
        hidden_norm = upload.norm("mtp.pre_fc_norm_hidden.weight", hidden_size);
        fc = upload.fc(hidden_size, 2 * hidden_size);
        input_norm = upload.norm("mtp.layers.0.input_layernorm.weight", hidden_size);
        attention_norm = upload.norm("mtp.layers.0.post_attention_layernorm.weight", hidden_size);
        const std::string attention = "mtp.layers.0.self_attn.";
        gqa = {
            upload.projection(attention + "q_proj", text_gqa_q_gate_width, hidden_size),
            upload.projection(attention + "k_proj", text_gqa_kv_width, hidden_size),
            upload.projection(attention + "v_proj", text_gqa_kv_width, hidden_size),
            upload.norm(attention + "q_norm.weight", mxfp8_head_dim),
            upload.norm(attention + "k_norm.weight", mxfp8_head_dim),
            upload.projection(attention + "o_proj", hidden_size, text_gqa_q_width),
        };
        const std::string mlp = "mtp.layers.0.mlp.";
        mlp_gate = upload.projection(mlp + "gate_proj", intermediate_size, hidden_size);
        mlp_up = upload.projection(mlp + "up_proj", intermediate_size, hidden_size);
        mlp_down = upload.projection(mlp + "down_proj", hidden_size, intermediate_size);
        final_norm = upload.norm("mtp.norm.weight", hidden_size);
        if (int8_head) {
            auto* values = memory.allocate<std::uint8_t>(
                static_cast<std::size_t>(vocab_size) * hidden_size);
            auto* maxima = memory.allocate<float>(vocab_size);
            check(quantize_draft_head_int8(lm_head.values, vocab_size, values, maxima,
                                          memory.stream()), "Quantize MTP vocabulary head");
            quantized_head = values;
            quantized_head_maxima = maxima;
        }
        memory.close_stream();
        if (int8_head) {
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
    std::shared_ptr<const Allocations> target_weights;
    Allocations memory;
    const Bf16* embedding;
    Bf16Weight lm_head;
    const std::uint8_t* quantized_head = nullptr;
    const float* quantized_head_maxima = nullptr;
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
    void embed(int shifted_token, const Bf16* hidden, cudaStream_t stream);
    void prepare_attention(cudaStream_t stream);
    void append(int position, bool score, cudaStream_t stream);
    DraftAttentionRow attention_row(int position);
    void gate_attention(cudaStream_t stream);
    void prepare_mlp(cudaStream_t stream);
    void activate_mlp(cudaStream_t stream);
    void normalize_output(cudaStream_t stream);
    void score(cudaStream_t stream, const ConstraintWorkspace* scratch,
               const ConstraintCandidates* allowed);
    void select_score(cudaStream_t stream, const ConstraintWorkspace* scratch,
                      const ConstraintCandidates* allowed);
    static void score_wave(State* const* batch, int count, cudaStream_t stream);

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
    check(cudaMemcpyAsync(table, host_pages.data(), host_pages.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice, memory.stream()), "Upload MTP KV page table");
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

void MtpDraftRequest::State::embed(int shifted_token, const Bf16* hidden, cudaStream_t stream) {
    const auto& model = *weights->state_;
    check(text_embedding(model.embedding, vocab_size, hidden_size, shifted_token,
                         work.embedding, stream), "Look up shifted MTP token");
    check(text_rms_norm(work.embedding, nullptr, model.embedding_norm, hidden_size,
                        work.concatenated, nullptr, stream), "Normalize MTP embedding");
    check(text_rms_norm(hidden, nullptr, model.hidden_norm, hidden_size,
                        work.concatenated + hidden_size, nullptr, stream), "Normalize MTP hidden");
}

void MtpDraftRequest::State::prepare_attention(cudaStream_t stream) {
    const auto& model = *weights->state_;
    check(text_rms_norm(work.residual, nullptr, model.input_norm, hidden_size, work.normalized,
                        nullptr, stream), "Normalize MTP layer input");
    quantize(work.normalized, hidden_size, work, stream);
}

void MtpDraftRequest::State::append(int position, bool score, cudaStream_t stream) {
    const auto& gqa = weights->state_->gqa;
    if (score) {
        check(text_gqa_preprocess(position, 1, work.q_gate, work.keys, gqa.query_norm,
                                  gqa.key_norm, work.rope_queries, work.rope_keys, stream),
              "Rotate MTP Q/K");
    } else {
        check(text_gqa_preprocess_keys(position, 1, work.keys, gqa.key_norm, work.rope_keys,
                                      stream), "Rotate MTP key");
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
    check(cudaMemcpyAsync(&winner, &work.argmax->token_id, sizeof(winner), cudaMemcpyDeviceToHost,
                          stream), "Copy MTP token");
    scored = true;
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
    for (int index = 0; index < count; ++index) {
        batch[index]->select_score(stream, nullptr, nullptr);
    }
}

MtpDraftWeights::MtpDraftWeights(const std::filesystem::path& checkpoint, int device,
                                 std::shared_ptr<const Allocations> target_weights,
                                 const Bf16* embedding, Bf16Weight lm_head, bool int8_head)
    : state_(std::make_unique<State>(checkpoint, device, std::move(target_weights), embedding,
                                     lm_head, int8_head)) {}

MtpDraftWeights::~MtpDraftWeights() = default;

std::size_t MtpDraftWeights::device_allocation_bytes() const { return state_->memory.bytes(); }

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
                                    const ConstraintCandidates* allowed) {
    auto& request = *state_;
    request.validate(position, shifted_token, hidden);
    auto& work = request.work;
    const auto& model = *request.weights->state_;
    const auto& gqa = model.gqa;
    try {
        request.embed(shifted_token, hidden, stream);
        check(text_bf16_gemv(model.fc.values, work.concatenated, model.fc.rows, model.fc.columns,
                             work.residual, stream), "Run BF16 MTP FC");
        request.prepare_attention(stream);
        if (score) {
            project(gqa.q_gate, work, work.q_gate, stream);
        }
        project(gqa.key, work, work.keys, stream);
        project(gqa.value, work, work.values, stream);
        request.append(position, score, stream);
        if (score) {
            check(mtp_gqa_attention(request.cache, request.pages, position, work.rope_queries,
                                    work.attention, work.attention_scores(), stream),
                  "Read MTP MXFP8 attention");
            request.gate_attention(stream);
            project(gqa.output, work, work.branch, stream);
            request.prepare_mlp(stream);
            project(model.mlp_gate, work, work.gate, stream);
            project(model.mlp_up, work, work.up, stream);
            request.activate_mlp(stream);
            project(model.mlp_down, work, work.branch, stream);
            request.normalize_output(stream);
            request.score(stream, scratch, allowed);
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
    if (count == 0) {
        return;
    }
    static_assert(draft_wave_rows == decoder_batch_max_mtp_lanes);
    if (!rows || count < 0 || count > draft_wave_rows) {
        throw std::invalid_argument("Invalid MTP draft wave size");
    }
    if (count == 1) {
        auto& row = rows[0];
        row.hidden = row.request->enqueue(row.position, row.token, row.hidden, score, stream);
        return;
    }
    std::array<State*, draft_wave_rows> states{};
    const auto weights = rows[0].request->state_->weights.get();
    std::array<TextBf16Projection, draft_wave_rows> fc;
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        auto& request = *row.request->state_;
        request.validate(row.position, row.token, row.hidden);
        if (request.weights.get() != weights ||
            std::find(states.begin(), states.begin() + index, &request) != states.begin() + index) {
            throw std::invalid_argument("Invalid MTP draft wave lanes");
        }
        states[index] = &request;
        fc[index] = {request.work.concatenated, request.work.residual};
    }
    const auto& model = *weights->state_;
    const auto project_wave = [&](Fp8Weight weight, Bf16* DraftWorkspace::* output) {
        std::array<TextFp8Projection, draft_wave_rows> projections;
        for (int index = 0; index < count; ++index) {
            const auto& work = states[index]->work;
            projections[index] = {{work.activation.values, work.activation.scales}, work.*output};
        }
        check(text_fp8_gemm(weight.values, weight.scales, weight.rows, weight.columns,
                             projections.data(), count, stream), "Run batched MTP FP8");
    };
    const auto apply = [&](void (State::* function)(cudaStream_t)) {
        for (int index = 0; index < count; ++index) {
            std::invoke(function, states[index], stream);
        }
    };
    try {
        for (int index = 0; index < count; ++index) {
            states[index]->embed(rows[index].token, rows[index].hidden, stream);
        }
        check(text_bf16_gemm(model.fc.values, model.fc.rows, model.fc.columns, fc.data(),
                             count, stream), "Run batched BF16 MTP FC");
        apply(&State::prepare_attention);
        if (score) {
            project_wave(model.gqa.q_gate, &DraftWorkspace::q_gate);
        }
        project_wave(model.gqa.key, &DraftWorkspace::keys);
        project_wave(model.gqa.value, &DraftWorkspace::values);
        for (int index = 0; index < count; ++index) {
            states[index]->append(rows[index].position, score, stream);
        }
        if (score) {
            std::array<DraftAttentionRow, draft_wave_rows> attention;
            for (int index = 0; index < count; ++index) {
                attention[index] = states[index]->attention_row(rows[index].position);
            }
            check(mtp_gqa_attention_batch(attention.data(), count, stream),
                  "Read batched MTP MXFP8 attention");
            apply(&State::gate_attention);
            project_wave(model.gqa.output, &DraftWorkspace::branch);
            apply(&State::prepare_mlp);
            project_wave(model.mlp_gate, &DraftWorkspace::gate);
            project_wave(model.mlp_up, &DraftWorkspace::up);
            apply(&State::activate_mlp);
            project_wave(model.mlp_down, &DraftWorkspace::branch);
            apply(&State::normalize_output);
            State::score_wave(states.data(), count, stream);
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

void repair_draft_wave(const std::array<DraftRepairs, decoder_batch_max_mtp_lanes>& repairs,
                       int count, cudaStream_t stream) {
    std::array<MtpDraftInput, decoder_batch_max_mtp_lanes> rows;
    for (int depth = 0; depth < decoder_batch_max_mtp_steps; ++depth) {
        int size = 0;
        for (int index = 0; index < count; ++index) {
            if (const auto& row = repairs[index][depth]; row.request) {
                rows[size++] = row;
            }
        }
        if (size == 0) {
            break;
        }
        MtpDraftRequest::enqueue_wave(rows.data(), size, false, stream);
    }
}

void MtpDraftRequest::seed_prefill(int position, const int* shifted_tokens, const Bf16* hidden,
                                   const PrefillProjections& projections,
                                   MtpPrefillBuffers buffers) {
    auto& request = *state_;
    if (request.failed) {
        throw std::logic_error("MTP draft request failed");
    }
    const int rows = projections.rows;
    if (position != request.prefix || rows <= 0 || rows > request.capacity - position ||
        !shifted_tokens || !hidden) {
        throw std::invalid_argument("Invalid MTP draft prefill input");
    }
    const auto& model = *request.weights->state_;
    const auto stream = projections.stream;
    try {
        check(text_embedding_rows(model.embedding, vocab_size, hidden_size, shifted_tokens, rows,
                                   buffers.embedding, stream), "Gather shifted MTP tokens");
        check(text_rms_norm_pair_rows(buffers.embedding, hidden, model.embedding_norm,
                                      model.hidden_norm, hidden_size, rows, buffers.concatenated,
                                      stream), "Normalize and concatenate MTP prefill inputs");
        projections.project_seed_fc(model.fc, buffers.concatenated, buffers.embedding);
        check(text_rms_norm_rows(buffers.embedding, nullptr, model.input_norm, hidden_size, rows,
                                 buffers.layer_input, nullptr, stream),
              "Normalize MTP prefill layer input");
        const auto input = projections.quantize(buffers.layer_input, hidden_size);
        projections.project(model.gqa.key, input, buffers.keys);
        projections.project(model.gqa.value, input, buffers.values);
        check(text_gqa_preprocess_keys(position, rows, buffers.keys, model.gqa.key_norm,
                                        buffers.rope_keys, stream), "Rotate MTP prefill keys");
        check(mxfp8_kv_append(request.cache, request.pages, position, rows, buffers.rope_keys,
                              buffers.values, stream), "Append MTP prefill MXFP8 KV");
    } catch (...) {
        request.failed = true;
        throw;
    }
    request.prefix += rows;
    request.scored = false;
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
    const int token = state_->memory.transfers().winner;
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
    const int token = request.memory.transfers().winner;
    if (!request.int8_score || (token >= 0 && token < vocab_size)) {
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
    check(cudaMemcpyAsync(&request.retry_winner, work.argmax, sizeof(request.retry_winner),
                          cudaMemcpyDeviceToHost, stream), "Copy BF16 MTP retry winner");
    request.int8_score = false;
    request.retried_score = true;
    return true;
}

int MtpDraftRequest::valid_prefix() const { return state_->prefix; }

void MtpDraftRequest::snapshot_prefix(Mxfp8KvCache snapshot, int rows, cudaStream_t stream) const {
    if (state_->failed || state_->prefix != rows) {
        throw std::logic_error("Draft request is not at its cache checkpoint");
    }
    copy_draft_prefix(snapshot, state_->cache, rows, stream);
}

void MtpDraftRequest::restore_prefix(Mxfp8KvCache snapshot, int rows, cudaStream_t stream) {
    if (state_->failed || state_->prefix != 0 || rows >= state_->capacity) {
        throw std::logic_error("Invalid draft request for prefix restoration");
    }
    copy_draft_prefix(state_->cache, snapshot, rows, stream);
    check(cudaStreamSynchronize(stream), "Complete paired decoder prefix restoration");
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
