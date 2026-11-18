#include "decoder_internal.h"

namespace sglang {
namespace {

using namespace decoder;

TextFp8ActivationView quantize(const Bf16* source, int width, TextFp8Activation scratch,
                               cudaStream_t stream) {
    check(text_quantize_fp8(source, width, scratch, stream), "Quantize projection input");
    return {scratch.values, scratch.scales};
}

void project(Fp8Weight weight, TextFp8ActivationView input, Bf16* output, cudaStream_t stream) {
    check(text_fp8_gemv(weight.values, weight.scales, input, weight.rows, weight.columns,
                        output, stream), "Run FP8 projection");
}

void project(Bf16Weight weight, const Bf16* input, Bf16* output, cudaStream_t stream) {
    check(text_bf16_gemv(weight.values, input, weight.rows, weight.columns, output, stream),
          "Run BF16 projection");
}

void run_gdn(const GdnLayer& layer, Request& request, TextGdnState state, TextGdnMode mode) {
    auto& work = request.work;
    const auto stream = request.memory.stream();
    const auto input = quantize(work.normalized, hidden_size, work.activation, stream);
    project(layer.qkv, input, work.gdn_qkv, stream);
    project(layer.z, input, work.gdn_z, stream);
    project(layer.b, work.normalized, work.gdn_b, stream);
    project(layer.a, work.normalized, work.gdn_a, stream);
    check(text_gdn_forward(1, mode, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                           layer.recurrent, state, work.gdn_scratch, text_gdn_workspace_bytes(1),
                           work.attention, stream), "Advance Gated DeltaNet state");
    const auto attention = quantize(work.attention, text_gdn_value_width, work.activation, stream);
    project(layer.output, attention, work.projected_attention, stream);
}

void run_gqa(const GqaLayer& layer, Request& request, Mxfp8KvCache cache, int position) {
    auto& work = request.work;
    const auto stream = request.memory.stream();
    const auto input = quantize(work.normalized, hidden_size, work.activation, stream);
    project(layer.q_gate, input, work.q_gate, stream);
    project(layer.key, input, work.keys, stream);
    project(layer.value, input, work.values, stream);
    check(text_gqa_preprocess(position, 1, work.q_gate, work.keys, layer.query_norm, layer.key_norm,
                              work.rope_queries, work.rope_keys, stream),
          "Normalize and rotate Q/K");
    check(mxfp8_kv_append(cache, request.pages, position, 1, work.rope_keys, work.values, stream),
          "Append MXFP8 KV");
    check(mxfp8_gqa_attention(cache, request.pages, position + 1, position, 1, work.rope_queries,
                             work.attention, stream), "Read causal MXFP8 attention");
    check(text_gqa_postgate(1, work.q_gate, work.attention, work.attention, stream),
          "Gate grouped-query attention");
    const auto attention = quantize(work.attention, text_gqa_q_width, work.activation, stream);
    project(layer.output, attention, work.projected_attention, stream);
}

void run_mlp(const Layer& layer, Workspace& work, cudaStream_t stream) {
    const auto input = quantize(work.normalized, hidden_size, work.activation, stream);
    project(layer.mlp_gate, input, work.mlp_gate, stream);
    project(layer.mlp_up, input, work.mlp_up, stream);
    check(text_swiglu(work.mlp_gate, work.mlp_up, intermediate_size, work.mlp_gate, stream),
          "Apply SwiGLU");
    const auto activated = quantize(work.mlp_gate, intermediate_size, work.activation, stream);
    project(layer.mlp_down, activated, work.hidden, stream);
}

}

void Decoder::Model::forward(Request& request, int token_id, int position, TextGdnMode mode) const {
    auto& work = request.work;
    const auto stream = request.memory.stream();
    check(text_embedding(embedding, vocab_size, hidden_size, token_id, work.residual, stream),
          "Look up token embedding");
    int gdn_index = 0;
    int gqa_index = 0;
    for (int index = 0; index < layer_count; ++index) {
        const auto& layer = layers[index];
        const auto* input = index == 0 ? work.residual : work.hidden;
        auto* residual = index == 0 ? nullptr : work.residual;
        check(text_rms_norm(input, residual, layer.input_norm, hidden_size,
                            work.normalized, residual, stream), "Normalize layer input");
        if (is_full_attention(index)) {
            run_gqa(layer.gqa, request, request.caches[gqa_index++], position);
        } else {
            run_gdn(layer.gdn, request, request.states[gdn_index++], mode);
        }
        check(text_rms_norm(work.projected_attention, work.residual, layer.attention_norm,
                            hidden_size, work.normalized, work.residual, stream),
              "Normalize post-attention residual");
        run_mlp(layer, work, stream);
    }
}

void Decoder::Model::compute_logits(Request& request) const {
    auto& work = request.work;
    const auto stream = request.memory.stream();
    check(text_rms_norm(work.hidden, work.residual, final_norm, hidden_size,
                        work.normalized, nullptr, stream), "Normalize final residual");
    project(lm_head, work.normalized, work.logits, stream);
    check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
          "Select greedy token");
}

}
