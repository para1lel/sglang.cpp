#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"

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

template <typename Lane>
struct LayerBatch {
    const Lane* lanes;
    int count;
    cudaStream_t stream;

    template <typename Function>
    void each(Function function) const {
        for (int index = 0; index < count; ++index) {
            function(lanes[index], lanes[index].request->work);
        }
    }

    void quantize(Bf16* Workspace::*source, int width) const {
        each([&](const Lane&, Workspace& work) {
            check(text_quantize_fp8(work.*source, width, work.activation, stream),
                  "Quantize projection input");
        });
    }

    void project(Fp8Weight weight, Bf16* Workspace::*output) const {
        if (count == 1) {
            const auto& work = lanes[0].request->work;
            const TextFp8ActivationView input{work.activation.values, work.activation.scales};
            check(text_fp8_gemv(weight.values, weight.scales, input, weight.rows, weight.columns,
                                work.*output, stream), "Run FP8 projection");
            return;
        }
        static_assert(decoder_batch_max_lanes <= text_fp8_max_batch);
        std::array<TextFp8Projection, decoder_batch_max_lanes> projections;
        for (int index = 0; index < count; ++index) {
            const auto& work = lanes[index].request->work;
            projections[index] = {{work.activation.values, work.activation.scales}, work.*output};
        }
        check(text_fp8_gemm(weight.values, weight.scales, weight.rows, weight.columns,
                            projections.data(), count, stream), "Run batched FP8 projection");
    }
};

template <typename Lane>
void run_gdn(const GdnLayer& layer, const LayerBatch<Lane>& batch, int state_index) {
    batch.quantize(&Workspace::normalized, hidden_size);
    batch.project(layer.qkv, &Workspace::gdn_qkv);
    batch.project(layer.z, &Workspace::gdn_z);
    batch.each([&](const Lane& lane, Workspace& work) {
        const auto stream = batch.stream;
        project(layer.b, work.normalized, work.gdn_b, stream);
        project(layer.a, work.normalized, work.gdn_a, stream);
        check(text_gdn_forward(1, lane.mode, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                               layer.recurrent, lane.request->states[state_index],
                               work.gdn_scratch, text_gdn_workspace_bytes(1), work.attention,
                               stream), "Advance Gated DeltaNet state");
    });
    batch.quantize(&Workspace::attention, text_gdn_value_width);
    batch.project(layer.output, &Workspace::projected_attention);
}

template <typename Position>
void attend_gqa(const GqaLayer& layer, Request& request, Mxfp8KvCache cache,
                Position position, cudaStream_t stream) {
    auto& work = request.work;
    if constexpr (std::is_same_v<Position, int>) {
        check(text_gqa_preprocess(position, 1, work.q_gate, work.keys, layer.query_norm,
                                  layer.key_norm, work.rope_queries, work.rope_keys, stream),
              "Normalize and rotate Q/K");
        check(mxfp8_kv_append(cache, request.pages, position, 1, work.rope_keys, work.values,
                              stream), "Append MXFP8 KV");
        check(mxfp8_gqa_attention(cache, request.pages, position + 1, position, 1,
                                  work.rope_queries, work.attention, stream),
              "Read causal MXFP8 attention");
    } else {
        check(decode_gqa_preprocess(position, work.q_gate, work.keys, layer.query_norm,
                                    layer.key_norm, work.rope_queries, work.rope_keys, stream),
              "Normalize and rotate decode Q/K");
        check(decode_kv_append(cache, request.pages, position, work.rope_keys, work.values, stream),
              "Append decode MXFP8 KV");
        check(decode_gqa_attention(cache, request.pages, position, work.rope_queries,
                                   work.attention, stream), "Read causal decode MXFP8 attention");
    }
    check(text_gqa_postgate(1, work.q_gate, work.attention, work.attention, stream),
          "Gate grouped-query attention");
}

template <typename Lane>
void run_gqa(const GqaLayer& layer, const LayerBatch<Lane>& batch, int cache_index) {
    batch.quantize(&Workspace::normalized, hidden_size);
    batch.project(layer.q_gate, &Workspace::q_gate);
    batch.project(layer.key, &Workspace::keys);
    batch.project(layer.value, &Workspace::values);
    batch.each([&](const Lane& lane, Workspace&) {
        auto& request = *lane.request;
        attend_gqa(layer, request, request.caches[cache_index], lane.position, batch.stream);
    });
    batch.quantize(&Workspace::attention, text_gqa_q_width);
    batch.project(layer.output, &Workspace::projected_attention);
}

template <typename Lane>
void run_mlp(const Layer& layer, const LayerBatch<Lane>& batch) {
    batch.quantize(&Workspace::normalized, hidden_size);
    batch.project(layer.mlp_gate, &Workspace::mlp_gate);
    batch.project(layer.mlp_up, &Workspace::mlp_up);
    batch.each([&](const Lane&, Workspace& work) {
        check(text_swiglu(work.mlp_gate, work.mlp_up, intermediate_size, work.mlp_gate,
                          batch.stream), "Apply SwiGLU");
    });
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

void Decoder::Model::decode_graph(Request& request) const {
    const auto stream = request.memory.stream();
    if (!request.memory.graph()) {
        request.memory.capture_graph([&] {
            check(decode_embedding(embedding, vocab_size, hidden_size, request.work.argmax,
                                   request.work.residual, stream),
                  "Look up decode token embedding");
            const GraphLane lane{&request, request.decode_position, TextGdnMode::decode};
            forward_layers(&lane, 1, stream);
            compute_logits(request);
            check(advance_decode_position(request.decode_position, stream),
                  "Advance decode position");
        });
    }
    check(cudaGraphLaunch(request.memory.graph(), stream), "Replay decode graph");
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

void Decoder::Model::compute_logits(Request& request) const {
    compute_logits(request.work, request.memory.stream());
}

void Decoder::Model::normalize_final(Workspace& work, cudaStream_t stream) const {
    check(text_rms_norm(work.hidden, work.residual, final_norm, hidden_size,
                        work.normalized, nullptr, stream), "Normalize final residual");
}

void Decoder::Model::compute_logits(Workspace& work, cudaStream_t stream) const {
    normalize_final(work, stream);
    project(lm_head, work.normalized, work.logits, stream);
    check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
          "Select greedy token");
}

}
