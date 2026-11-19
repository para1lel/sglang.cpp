#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_verify.h"

#include <type_traits>

namespace sglang {
namespace {

using namespace decoder;

void project(Bf16Weight weight, const Bf16* input, Bf16* output, cudaStream_t stream) {
    check(text_bf16_gemv(weight.values, input, weight.rows, weight.columns, output, stream),
          "Run BF16 projection");
}

struct GraphLane {
    Request* request;
    const int* position;
    TextGdnMode mode;
};

DecodeFp8Projection decode_projection(Fp8Weight weight, Bf16* output) {
    return {weight.values, weight.scales, output, weight.rows};
}

template <typename Lane>
Workspace& workspace(const Lane& lane) {
    return lane.request->work;
}

Workspace& workspace(const VerificationRow& row) {
    return *row.work;
}

Workspace& workspace(const BatchGraphLane& lane) {
    return *lane.work;
}

template <typename Lane>
struct LayerBatch {
    const Lane* lanes;
    int count;
    cudaStream_t stream;

    bool single_decode() const {
        if constexpr (!std::is_same_v<Lane, BatchGraphLane>) {
            return count == 1 && lanes[0].mode == TextGdnMode::decode;
        }
        return false;
    }

    template <typename Function>
    void each(Function function) const {
        for (int index = 0; index < count; ++index) {
            function(lanes[index], workspace(lanes[index]));
        }
    }

    void quantize(Bf16* Workspace::*source, int width) const {
        if constexpr (!std::is_same_v<Lane, VerificationRow>) {
            if (single_decode()) {
                auto& work = workspace(lanes[0]);
                check(decode_quantize_fp8(work.*source, width, work.activation, stream),
                      "Quantize decode projection input");
                return;
            }
        }
        each([&](const Lane&, Workspace& work) {
            check(text_quantize_fp8(work.*source, width, work.activation, stream),
                  "Quantize projection input");
        });
    }

    void project(Fp8Weight weight, Bf16* Workspace::*output) const {
        if (count == 1) {
            const auto& work = workspace(lanes[0]);
            const TextFp8ActivationView input{work.activation.values, work.activation.scales};
            const auto status = single_decode()
                ? decode_fp8_project(input, decode_projection(weight, work.*output),
                                     weight.columns, stream)
                : text_fp8_gemv(weight.values, weight.scales, input, weight.rows, weight.columns,
                                 work.*output, stream);
            check(status, "Run FP8 projection");
            return;
        }
        static_assert(decoder_batch_max_lanes <= text_fp8_max_batch);
        std::array<TextFp8Projection, text_fp8_max_batch> projections;
        for (int index = 0; index < count; ++index) {
            const auto& work = workspace(lanes[index]);
            projections[index] = {{work.activation.values, work.activation.scales}, work.*output};
        }
        check(text_fp8_gemm(weight.values, weight.scales, weight.rows, weight.columns,
                            projections.data(), count, stream), "Run batched FP8 projection");
    }
};

void project(Bf16Weight weight, const LayerBatch<VerificationRow>& batch,
             Bf16* Workspace::*output) {
    std::array<TextBf16Projection, target_verify_max_rows> projections;
    for (int index = 0; index < batch.count; ++index) {
        const auto& work = workspace(batch.lanes[index]);
        projections[index] = {work.normalized, work.*output};
    }
    check(text_bf16_gemm(weight.values, weight.rows, weight.columns, projections.data(),
                         batch.count, batch.stream), "Run batched BF16 projection");
}

template <typename Lane>
void run_gdn(const GdnLayer& layer, const LayerBatch<Lane>& batch, int state_index) {
    batch.quantize(&Workspace::normalized, hidden_size);
    batch.project(layer.qkv, &Workspace::gdn_qkv);
    batch.project(layer.z, &Workspace::gdn_z);
    batch.each([&](const Lane& lane, Workspace& work) {
        const auto stream = batch.stream;
        if (batch.single_decode() && layer.b.rows == layer.a.rows &&
            layer.b.columns == layer.a.columns) {
            check(decode_bf16_gdn_gates(layer.b.values, layer.a.values, work.normalized,
                                        layer.b.rows, layer.b.columns, work.gdn_b, work.gdn_a,
                                        stream), "Run decode Gated DeltaNet gate projections");
        } else {
            project(layer.b, work.normalized, work.gdn_b, stream);
            project(layer.a, work.normalized, work.gdn_a, stream);
        }
        if constexpr (std::is_same_v<Lane, BatchGraphLane>) {
            check(decode_gdn(work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a, layer.recurrent,
                             &lane.binding->states[state_index], work.gdn_scratch, work.attention,
                             stream), "Advance bound Gated DeltaNet state");
        } else {
            check(text_gdn_forward(1, lane.mode, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                                   layer.recurrent, lane.request->states[state_index],
                                   work.gdn_scratch, text_gdn_workspace_bytes(1), work.attention,
                                   stream), "Advance Gated DeltaNet state");
        }
    });
    batch.quantize(&Workspace::attention, text_gdn_value_width);
    batch.project(layer.output, &Workspace::projected_attention);
}

void run_gdn(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch, int state_index) {
    auto& verification = *batch.lanes[0].verification;
    verification.bind_gdn(state_index);
    batch.quantize(&Workspace::normalized, hidden_size);
    batch.project(layer.qkv, &Workspace::gdn_qkv);
    batch.project(layer.z, &Workspace::gdn_z);
    project(layer.b, batch, &Workspace::gdn_b);
    project(layer.a, batch, &Workspace::gdn_a);
    verification.advance_gdn(state_index, layer.recurrent);
    batch.quantize(&Workspace::attention, text_gdn_value_width);
    batch.project(layer.output, &Workspace::projected_attention);
}

template <typename Position>
cudaError_t read_gqa(Workspace& work, Mxfp8PageTable pages, Mxfp8KvCache cache,
                     Position position, TextGdnMode mode, cudaStream_t stream) {
    const int capacity = pages.page_count * mxfp8_page_size;
    const auto scores = work.gqa_scores(capacity);
    if constexpr (std::is_same_v<Position, int>) {
        if (mode == TextGdnMode::prefill || position >= scores.capacity) {
            return mxfp8_gqa_attention(cache, pages, position + 1, position, 1,
                                       work.rope_queries, work.attention, stream);
        }
    } else {
        if (capacity > scores.capacity) {
            return decode_gqa_attention(cache, pages, position, work.rope_queries,
                                        work.attention, stream);
        }
    }
    return exact_decode_gqa_attention(cache, pages, position, work.rope_queries, work.attention,
                                      scores, stream);
}

template <typename Position>
void attend_gqa(const GqaLayer& layer, Workspace& work, Mxfp8PageTable pages, Mxfp8KvCache cache,
                Position position, TextGdnMode mode, cudaStream_t stream) {
    if constexpr (std::is_same_v<Position, int>) {
        check(text_gqa_preprocess(position, 1, work.q_gate, work.keys, layer.query_norm,
                                  layer.key_norm, work.rope_queries, work.rope_keys, stream),
              "Normalize and rotate Q/K");
        check(mxfp8_kv_append(cache, pages, position, 1, work.rope_keys, work.values,
                              stream), "Append MXFP8 KV");
    } else {
        check(decode_gqa_preprocess(position, work.q_gate, work.keys, layer.query_norm,
                                    layer.key_norm, work.rope_queries, work.rope_keys, stream),
              "Normalize and rotate decode Q/K");
        check(decode_kv_append(cache, pages, position, work.rope_keys, work.values, stream),
              "Append decode MXFP8 KV");
    }
    check(read_gqa(work, pages, cache, position, mode, stream), "Read causal MXFP8 attention");
    check(text_gqa_postgate(1, work.q_gate, work.attention, work.attention, stream),
          "Gate grouped-query attention");
}

template <typename Lane>
void run_gqa(const GqaLayer& layer, const LayerBatch<Lane>& batch, int cache_index) {
    batch.quantize(&Workspace::normalized, hidden_size);
    if (batch.single_decode() && layer.q_gate.columns == hidden_size &&
        layer.key.columns == hidden_size && layer.value.columns == hidden_size) {
        const auto& work = workspace(batch.lanes[0]);
        check(decode_fp8_qkv({work.activation.values, work.activation.scales},
                             decode_projection(layer.q_gate, work.q_gate),
                             decode_projection(layer.key, work.keys),
                             decode_projection(layer.value, work.values), hidden_size,
                             batch.stream), "Run decode QKV projections");
    } else {
        batch.project(layer.q_gate, &Workspace::q_gate);
        batch.project(layer.key, &Workspace::keys);
        batch.project(layer.value, &Workspace::values);
    }
    batch.each([&](const Lane& lane, Workspace& work) {
        if constexpr (std::is_same_v<Lane, BatchGraphLane>) {
            const auto* binding = lane.binding;
            const auto stream = batch.stream;
            const auto* position = &binding->position;
            check(decode_gqa_preprocess(position, work.q_gate, work.keys, layer.query_norm,
                                        layer.key_norm, work.rope_queries, work.rope_keys, stream),
                  "Normalize and rotate bound decode Q/K");
            check(decode_kv_append(&binding->caches[cache_index], &binding->pages, position,
                                   work.rope_keys, work.values, stream), "Append bound MXFP8 KV");
            check(exact_decode_gqa_attention(&binding->caches[cache_index], &binding->pages,
                                             position, work.rope_queries, work.attention,
                                             &binding->scores, stream),
                  "Read bound causal MXFP8 attention");
            check(text_gqa_postgate(1, work.q_gate, work.attention, work.attention, stream),
                  "Gate grouped-query attention");
        } else {
            auto& request = *lane.request;
            attend_gqa(layer, work, request.pages, request.caches[cache_index], lane.position,
                       lane.mode, batch.stream);
        }
    });
    batch.quantize(&Workspace::attention, text_gqa_q_width);
    batch.project(layer.output, &Workspace::projected_attention);
}

template <typename Lane>
void run_mlp(const Layer& layer, const LayerBatch<Lane>& batch) {
    batch.quantize(&Workspace::normalized, hidden_size);
    if (batch.single_decode() && layer.mlp_gate.columns == hidden_size &&
        layer.mlp_up.columns == hidden_size) {
        const auto& work = workspace(batch.lanes[0]);
        check(decode_fp8_gate_up({work.activation.values, work.activation.scales},
                                 decode_projection(layer.mlp_gate, work.mlp_gate),
                                 decode_projection(layer.mlp_up, work.mlp_up), hidden_size,
                                 batch.stream), "Run decode gate and up projections with SwiGLU");
    } else {
        batch.project(layer.mlp_gate, &Workspace::mlp_gate);
        batch.project(layer.mlp_up, &Workspace::mlp_up);
        batch.each([&](const Lane&, Workspace& work) {
            check(text_swiglu(work.mlp_gate, work.mlp_up, intermediate_size, work.mlp_gate,
                              batch.stream), "Apply SwiGLU");
        });
    }
    batch.quantize(&Workspace::mlp_gate, intermediate_size);
    batch.project(layer.mlp_down, &Workspace::hidden);
}

}

void Decoder::Model::forward(Request& request, int token_id, int position, TextGdnMode mode) const {
    const ForwardLane lane{&request, token_id, position, mode};
    forward(&lane, 1, request.memory.stream());
}

void Decoder::Model::forward(const ForwardLane* lanes, int count, cudaStream_t stream) const {
    if (!lanes || count < 1 || count > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid decoder batch width");
    }
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        check(text_embedding(embedding, vocab_size, hidden_size, lane.token_id,
                             lane.request->work.residual, stream), "Look up token embedding");
    }
    forward_layers(lanes, count, stream);
}

void Decoder::Model::forward_verify(TargetVerification& verification, int position,
                                    const int* tokens, int count, cudaStream_t stream) const {
    verification.begin(position, count, stream);
    std::array<VerificationRow, target_verify_max_rows> rows;
    for (int index = 0; index < count; ++index) {
        auto& row = rows[index];
        row = verification.row(index);
        check(text_embedding(embedding, vocab_size, hidden_size, tokens[index],
                             row.work->residual, stream), "Look up verification input");
    }
    forward_layers(rows.data(), count, stream);
    const LayerBatch<VerificationRow> batch{rows.data(), count, stream};
    batch.each([&](const VerificationRow&, Workspace& work) { normalize_final(work, stream); });
    project(lm_head, batch, &Workspace::logits);
    batch.each([&](const VerificationRow&, Workspace& work) {
        check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
              "Select target verification winner");
    });
}

void Decoder::Model::decode_graph(Request& request) const {
    const auto stream = request.memory.stream();
    if (!request.memory.graph()) {
        request.memory.capture_graph([&] {
            check(decode_embedding(embedding, vocab_size, hidden_size, request.work.argmax,
                                   request.work.residual, stream),
                  "Look up decode token embedding");
            const GraphLane lane{&request, request.decode_position, TextGdnMode::decode};
            forward_layers(&lane, 1, stream);
            compute_logits(request, true);
            check(advance_decode_position(request.decode_position, stream),
                  "Advance decode position");
        });
    }
    check(cudaGraphLaunch(request.memory.graph(), stream), "Replay decode graph");
}

void Decoder::Model::capture_batch_decode(const BatchGraphLane* lanes, int count,
                                          cudaStream_t stream) const {
    const LayerBatch<BatchGraphLane> batch{lanes, count, stream};
    batch.each([&](const BatchGraphLane& lane, Workspace& work) {
        check(decode_embedding(embedding, vocab_size, hidden_size, &lane.binding->input,
                               work.residual, stream), "Look up bound token embedding");
    });
    forward_layers(lanes, count, stream);
    batch.each([&](const BatchGraphLane&, Workspace& work) {
        compute_logits(work, stream);
    });
}

template <typename Lane>
void Decoder::Model::forward_layers(const Lane* lanes, int count, cudaStream_t stream) const {
    const LayerBatch<Lane> batch{lanes, count, stream};
    int gdn_index = 0;
    int gqa_index = 0;
    for (int index = 0; index < layer_count; ++index) {
        const auto& layer = layers[index];
        batch.each([&](const Lane&, Workspace& work) {
            const auto* input = index == 0 ? work.residual : work.hidden;
            auto* residual = index == 0 ? nullptr : work.residual;
            check(text_rms_norm(input, residual, layer.input_norm, hidden_size,
                                work.normalized, residual, stream), "Normalize layer input");
        });
        if (is_full_attention(index)) {
            run_gqa(layer.gqa, batch, gqa_index++);
        } else {
            run_gdn(layer.gdn, batch, gdn_index++);
        }
        batch.each([&](const Lane&, Workspace& work) {
            check(text_rms_norm(work.projected_attention, work.residual, layer.attention_norm,
                                hidden_size, work.normalized, work.residual, stream),
                  "Normalize post-attention residual");
        });
        run_mlp(layer, batch);
    }
}

void Decoder::Model::compute_logits(Request& request, bool decode) const {
    compute_logits(request.work, request.memory.stream(), decode);
}

void Decoder::Model::normalize_final(Workspace& work, cudaStream_t stream) const {
    check(text_rms_norm(work.hidden, work.residual, final_norm, hidden_size,
                        work.normalized, nullptr, stream), "Normalize final residual");
}

void Decoder::Model::compute_logits(Workspace& work, cudaStream_t stream, bool decode) const {
    normalize_final(work, stream);
    if (decode) {
        check(decode_bf16_lm_head(lm_head.values, work.normalized, lm_head.rows, lm_head.columns,
                                  work.logits, stream), "Run decode LM head");
    } else {
        project(lm_head, work.normalized, work.logits, stream);
    }
    check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
          "Select greedy token");
}

}
