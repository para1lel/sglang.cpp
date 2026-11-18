#include "sglang/decoder_mtp.h"

#include "sglang/checkpoint.h"
#include "sglang/decoder_prefill.h"

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

struct DraftWorkspace {
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
          logits(memory.allocate<Bf16>(vocab_size)),
          activation{memory.allocate<std::uint8_t>(intermediate_size),
                     memory.allocate<float>(text_fp8_group_count(intermediate_size))},
          argmax_scratch(
              memory.allocate<TextLogitCandidate>(text_argmax_scratch_count(vocab_size))),
          argmax(memory.allocate<TextLogitCandidate>(1)) {}

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
          Bf16Weight lm_head)
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
        memory.close_stream();
        selected_device.restore();
    }

    int device;
    std::shared_ptr<const Allocations> target_weights;
    Allocations memory;
    const Bf16* embedding;
    Bf16Weight lm_head;
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
    State(std::shared_ptr<const MtpDraftWeights> weights, int capacity);
    void prepare(int shifted_token, const Bf16* hidden, cudaStream_t stream);
    void attend(int position, cudaStream_t stream);
    void run_mlp(cudaStream_t stream);
    void score(cudaStream_t stream);

    std::shared_ptr<const MtpDraftWeights> weights;
    Allocations memory;
    DraftWorkspace work;
    int capacity;
    Mxfp8KvCache cache;
    Mxfp8PageTable pages;
    int prefix = 0;
    bool scored = false;
    bool failed = false;
};

MtpDraftRequest::State::State(std::shared_ptr<const MtpDraftWeights> weights, int capacity)
    : weights(std::move(weights)), memory(this->weights->state_->device), work(memory),
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

void MtpDraftRequest::State::prepare(int shifted_token, const Bf16* hidden, cudaStream_t stream) {
    const auto& model = *weights->state_;
    check(text_embedding(model.embedding, vocab_size, hidden_size, shifted_token,
                         work.embedding, stream), "Look up shifted MTP token");
    check(text_rms_norm(work.embedding, nullptr, model.embedding_norm, hidden_size,
                        work.concatenated, nullptr, stream), "Normalize MTP embedding");
    check(text_rms_norm(hidden, nullptr, model.hidden_norm, hidden_size,
                        work.concatenated + hidden_size, nullptr, stream), "Normalize MTP hidden");
    check(text_bf16_gemv(model.fc.values, work.concatenated, model.fc.rows, model.fc.columns,
                         work.residual, stream), "Run BF16 MTP FC");
    check(text_rms_norm(work.residual, nullptr, model.input_norm, hidden_size, work.normalized,
                        nullptr, stream), "Normalize MTP layer input");
}

void MtpDraftRequest::State::attend(int position, cudaStream_t stream) {
    const auto& model = *weights->state_;
    const auto& gqa = model.gqa;
    quantize(work.normalized, hidden_size, work, stream);
    project(gqa.q_gate, work, work.q_gate, stream);
    project(gqa.key, work, work.keys, stream);
    project(gqa.value, work, work.values, stream);
    check(text_gqa_preprocess(position, 1, work.q_gate, work.keys, gqa.query_norm, gqa.key_norm,
                              work.rope_queries, work.rope_keys, stream), "Rotate MTP Q/K");
    check(mxfp8_kv_append(cache, pages, position, 1, work.rope_keys, work.values, stream),
          "Append MTP MXFP8 KV");
    check(mxfp8_gqa_attention(cache, pages, position + 1, position, 1, work.rope_queries,
                              work.attention, stream), "Read MTP MXFP8 attention");
    check(text_gqa_postgate(1, work.q_gate, work.attention, work.attention, stream),
          "Gate MTP attention");
    quantize(work.attention, text_gqa_q_width, work, stream);
    project(gqa.output, work, work.branch, stream);
    check(text_rms_norm(work.branch, work.residual, model.attention_norm, hidden_size,
                        work.normalized, work.residual, stream),
          "Normalize MTP post-attention residual");
}

void MtpDraftRequest::State::run_mlp(cudaStream_t stream) {
    const auto& model = *weights->state_;
    quantize(work.normalized, hidden_size, work, stream);
    project(model.mlp_gate, work, work.gate, stream);
    project(model.mlp_up, work, work.up, stream);
    check(text_swiglu(work.gate, work.up, intermediate_size, work.gate, stream),
          "Apply MTP SwiGLU");
    quantize(work.gate, intermediate_size, work, stream);
    project(model.mlp_down, work, work.branch, stream);
    check(text_rms_norm(work.branch, work.residual, model.final_norm, hidden_size, work.normalized,
                        nullptr, stream), "Normalize MTP output");
}

void MtpDraftRequest::State::score(cudaStream_t stream) {
    const auto& head = weights->state_->lm_head;
    check(text_bf16_gemv(head.values, work.normalized, head.rows, head.columns,
                         work.logits, stream), "Run shared LM head for MTP");
    check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
          "Select MTP token");
    auto& winner = memory.transfers().winner;
    check(cudaMemcpyAsync(&winner, &work.argmax->token_id, sizeof(winner), cudaMemcpyDeviceToHost,
                          stream), "Copy MTP token");
    scored = true;
}

MtpDraftWeights::MtpDraftWeights(const std::filesystem::path& checkpoint, int device,
                                 std::shared_ptr<const Allocations> target_weights,
                                 const Bf16* embedding, Bf16Weight lm_head)
    : state_(std::make_unique<State>(checkpoint, device, std::move(target_weights), embedding,
                                     lm_head)) {}

MtpDraftWeights::~MtpDraftWeights() = default;

std::size_t MtpDraftWeights::device_allocation_bytes() const { return state_->memory.bytes(); }

MtpDraftRequest::MtpDraftRequest(std::shared_ptr<const MtpDraftWeights> weights, int capacity) {
    if (!weights || capacity < 1 || capacity > text_attention_max_tokens) {
        throw std::invalid_argument("Invalid MTP draft weights or capacity");
    }
    DeviceScope selected_device(weights->state_->device);
    state_ = std::make_unique<State>(std::move(weights), capacity);
    selected_device.restore();
}

MtpDraftRequest::~MtpDraftRequest() = default;

const Bf16* MtpDraftRequest::enqueue(int position, int shifted_token, const Bf16* hidden,
                                    bool score, cudaStream_t stream) {
    auto& request = *state_;
    if (request.failed) {
        throw std::logic_error("MTP draft request failed");
    }
    if (position != request.prefix || position >= request.capacity || shifted_token < 0 ||
        shifted_token >= vocab_size || !hidden) {
        throw std::invalid_argument("Invalid MTP draft input");
    }
    try {
        request.prepare(shifted_token, hidden, stream);
        request.attend(position, stream);
        request.run_mlp(stream);
        if (score) {
            request.score(stream);
        }
    } catch (...) {
        request.failed = true;
        throw;
    }
    ++request.prefix;
    return request.work.normalized;
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
        projections.project(model.fc, buffers.concatenated, buffers.embedding);
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
    const int token = state_->memory.transfers().winner;
    if (token < 0 || token >= vocab_size) {
        throw std::runtime_error("MTP argmax did not return a valid token ID");
    }
    return token;
}

int MtpDraftRequest::valid_prefix() const { return state_->prefix; }

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
