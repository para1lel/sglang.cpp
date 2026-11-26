#pragma once


#include "sglang/decoder_prefill.h"
#include "sglang/decoder_batch.h"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/runtime/cache/prefill_cached_gqa.h"
#include "sglang/prefill_fp8.h"
#include "sglang/runtime/execution/decoder_prefill_plan.h"
#include "sglang/text/decoder_logprobs.h"
#include "sglang/text/gdn_local.h"
#include "sglang/runtime/backend/gdn_vendor_prefill.h"
#include "sglang/text/prefill_attn_sm100.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace sglang {

std::optional<cudaError_t> launch_partial_staged_tmem_prefill(
    const decoder::PagedPrefillSequence*, int, cudaStream_t, decoder::PairedPrefillScratch);
std::optional<cudaError_t> launch_causal_group_staged_tmem_prefill(
    const decoder::PagedPrefillSequence&, cudaStream_t, decoder::PairedPrefillScratch);

namespace decoder::prefill_detail {

struct ScratchLayout {
    std::byte* data = nullptr;
    std::size_t bytes = 0;

    template <typename Element, std::size_t alignment = 16>
    Element* take(std::size_t count) {
        const auto offset = (bytes + alignment - 1) / alignment * alignment;
        bytes = offset + count * sizeof(Element);
        return data ? reinterpret_cast<Element*>(data + offset) : nullptr;
    }
};

struct PrefillDims {
    int q_gate_width = text_gqa_q_gate_width;
    int kv_width = text_gqa_kv_width;
    int q_width = text_gqa_q_width;
    int gdn_qkv_width = text_gdn_qkv_width;
    int gdn_value_width = text_gdn_value_width;
    int gdn_heads = text_gdn_value_heads;
    int intermediate = intermediate_size;

    static PrefillDims from_layers(const std::array<Layer, layer_count>& layers) {
        PrefillDims dims;
        for (const auto& layer : layers) {
            if (layer.gqa.q_gate.rows) {
                dims.q_gate_width = layer.gqa.q_gate.rows;
                dims.kv_width = layer.gqa.key.rows;
                dims.q_width = layer.gqa.output.columns;
            } else if (layer.gdn.qkv.rows) {
                dims.gdn_qkv_width = layer.gdn.qkv.rows;
                dims.gdn_value_width = layer.gdn.z.rows;
                dims.gdn_heads = layer.gdn.b.rows;
            }
            if (layer.mlp_gate.rows) dims.intermediate = layer.mlp_gate.rows;
        }
        dims.validate();
        return dims;
    }

    void validate() const {
        const auto local_gdn = gdn_heads == GdnLocal::value_heads;
        if (q_gate_width != 2 * q_width || q_width <= 0 ||
            !gqa_geometry_valid(GqaGeometry{q_width / mxfp8_head_dim, kv_width / mxfp8_head_dim}) ||
            gdn_heads != text_gdn_value_heads && !local_gdn ||
            gdn_qkv_width != (local_gdn ? GdnLocal::qkv_width : text_gdn_qkv_width) ||
            gdn_value_width != (local_gdn ? GdnLocal::value_width : text_gdn_value_width) ||
            intermediate <= 0 || intermediate % text_fp8_group_size != 0) {
            throw std::logic_error("Inconsistent decoder prefill weight layout");
        }
    }

    int attention_width() const { return std::max(gdn_value_width, q_width); }
    bool is_local_gdn() const { return gdn_heads == GdnLocal::value_heads; }
    GqaGeometry gqa_geometry() const {
        return GqaGeometry{q_width / mxfp8_head_dim, kv_width / mxfp8_head_dim};
    }
    std::size_t gdn_workspace_bytes(int rows) const {
        return is_local_gdn() ? gdn_vendor_prefill_workspace_bytes(rows)
                              : text_gdn_workspace_bytes(rows);
    }
};

struct SeedScratch {
    Bf16* concatenated = nullptr;
    Bf16* keys = nullptr;
    Bf16* values = nullptr;
    Bf16* rope_keys = nullptr;
};

inline std::size_t seed_scratch_bytes(int concat_rows, int kv_rows) {
    return std::size_t(concat_rows) * 2 * hidden_size * sizeof(Bf16) +
           std::size_t(kv_rows) * 3 * text_gqa_kv_width * sizeof(Bf16);
}

inline SeedScratch make_seed_scratch(Bf16* memory, int concat_rows, int kv_rows) {
    const auto concatenated = std::size_t(concat_rows) * 2 * hidden_size;
    const auto stride = std::size_t(kv_rows) * text_gqa_kv_width;
    return {memory, memory + concatenated, memory + concatenated + stride,
            memory + concatenated + 2 * stride};
}

struct TileBuffers {
    TileBuffers(ScratchLayout& layout, int rows, const PrefillDims& tile_dims = PrefillDims{})
        : dims(tile_dims),
          hidden(layout.take<Bf16>(rows * hidden_size)),
          normalized(layout.take<Bf16>(rows * hidden_size)),
          residual(layout.take<Bf16>(rows * hidden_size)),
          projected_attention(layout.take<Bf16>(rows * hidden_size)),
          attention(layout.take<Bf16>(rows * dims.attention_width())),
          mlp_gate(layout.take<Bf16, 32>(rows * dims.intermediate)),
          mlp_up(layout.take<Bf16>(rows * dims.intermediate)),
          q_gate(layout.take<Bf16>(rows * dims.q_gate_width)),
          keys(layout.take<Bf16>(rows * dims.kv_width)),
          values(layout.take<Bf16>(rows * dims.kv_width)),
          rope_queries(layout.take<Bf16>(rows * dims.q_width)),
          rope_keys(layout.take<Bf16>(rows * dims.kv_width)),
          gdn_qkv(layout.take<Bf16>(rows * dims.gdn_qkv_width)),
          gdn_z(layout.take<Bf16>(rows * dims.gdn_value_width)),
          gdn_b(layout.take<Bf16>(rows * dims.gdn_heads)),
          gdn_a(layout.take<Bf16>(rows * dims.gdn_heads)),
          gdn_scratch(layout.take<Bf16>(dims.gdn_workspace_bytes(rows) / sizeof(Bf16))),
          activation{
              layout.take<std::uint8_t>(rows * dims.intermediate),
              layout.take<float>(rows * text_fp8_group_count(dims.intermediate))} {}

    PairedPrefillScratch gqa_scratch() const {
        const auto mlp_begin = reinterpret_cast<std::uintptr_t>(mlp_gate);
        const auto mlp_end = reinterpret_cast<std::uintptr_t>(q_gate);
        const auto gdn_begin = reinterpret_cast<std::uintptr_t>(gdn_qkv);
        const auto gdn_end = reinterpret_cast<std::uintptr_t>(activation.values);
        return {reinterpret_cast<std::byte*>(mlp_gate), mlp_end - mlp_begin,
                reinterpret_cast<std::byte*>(gdn_qkv), gdn_end - gdn_begin};
    }

    TileBuffers attention_segment(int offset) const {
        auto segment = *this;
        segment.attention += offset * dims.attention_width();
        segment.q_gate += offset * dims.q_gate_width;
        segment.keys += offset * dims.kv_width;
        segment.values += offset * dims.kv_width;
        segment.rope_queries += offset * dims.q_width;
        segment.rope_keys += offset * dims.kv_width;
        segment.gdn_qkv += offset * dims.gdn_qkv_width;
        segment.gdn_z += offset * dims.gdn_value_width;
        segment.gdn_b += offset * dims.gdn_heads;
        segment.gdn_a += offset * dims.gdn_heads;
        return segment;
    }

    PrefillDims dims;
    Bf16* hidden;
    Bf16* normalized;
    Bf16* residual;
    Bf16* projected_attention;
    Bf16* attention;
    Bf16* mlp_gate;
    Bf16* mlp_up;
    Bf16* q_gate;
    Bf16* keys;
    Bf16* values;
    Bf16* rope_queries;
    Bf16* rope_keys;
    Bf16* gdn_qkv;
    Bf16* gdn_z;
    Bf16* gdn_b;
    Bf16* gdn_a;
    Bf16* gdn_scratch;
    TextFp8Activation activation;
};

struct Chunk : PrefillProjections {
    TileBuffers work;
    int position;
    const MropePosition* mrope;
    int cached_gqa_tokens = 0;

    bool project_packed(TextFp8ActivationView input,
                        std::initializer_list<PrefillFp8Projection> projections) const {
        if (!tensor_core_fp8) {
            return false;
        }
        const auto status = try_prefill_fp8_packed_sm100(input, rows, hidden_size,
                                                         projections, stream, grouped_projection);
        if (!status) {
            return false;
        }
        check(*status, "Run packed FP8 prefill projections");
        return true;
    }

    void seed_draft(MtpDraftRequest& draft, const int* shifted_tokens, int seed_rows,
                    const Bf16* final_norm, bool wmma, const MediaPrefill* media,
                    const SeedScratch* seed = nullptr) const {
        Bf16* concatenated = work.mlp_gate;
        Bf16* layer_input = work.attention;
        Bf16* keys = work.keys;
        Bf16* values = work.values;
        Bf16* rope_keys = work.rope_keys;
        if (seed) {
            concatenated = seed->concatenated;
            layer_input = work.mlp_gate;
            keys = seed->keys;
            values = seed->values;
            rope_keys = seed->rope_keys;
        } else if (work.dims.intermediate < 2 * hidden_size ||
                   work.dims.attention_width() < hidden_size ||
                   work.dims.kv_width < text_gqa_kv_width) {
            throw std::logic_error("Prefill tile buffers too narrow for MTP draft seeding");
        }
        if (work.dims.intermediate < hidden_size) {
            throw std::logic_error("Prefill tile buffers too narrow for MTP draft seeding");
        }
        const PrefillProjections projections{activation, seed_rows, stream, false, false, wmma};
        draft.seed_prefill(position, shifted_tokens, work.hidden, work.residual, final_norm,
                           projections, {work.normalized, work.projected_attention, concatenated,
                                         layer_input, keys, values, rope_keys},
                           media);
    }

    void gdn_input(const GdnLayer& layer) const {
        const auto& dims = work.dims;
        const TextFp8ActivationView input{activation.values, activation.scales};
        if (!project_packed(input, {
                {layer.qkv.values, layer.qkv.scales, work.gdn_qkv, layer.qkv.rows},
                {layer.z.values, layer.z.scales, work.gdn_z, layer.z.rows},
            })) {
            project(layer.qkv, input, work.gdn_qkv);
            project(layer.z, input, work.gdn_z);
        }
        if (tensor_core_fp8 && layer.b.rows == dims.gdn_heads &&
            layer.a.rows == dims.gdn_heads && layer.b.columns == hidden_size &&
            layer.a.columns == hidden_size) {
            std::optional<cudaError_t> status;
            if (dims.is_local_gdn()) {
                status = try_prefill_bf16_gdn_gates_sm100_local(
                    layer.b.values, layer.a.values, work.normalized, rows, work.gdn_b, work.gdn_a,
                    work.mlp_gate, rows * dims.intermediate * sizeof(Bf16), stream);
                if (!status) status = try_prefill_bf16_gdn_gates_local(
                    layer.b.values, layer.a.values, work.normalized, rows, work.gdn_b, work.gdn_a,
                    stream);
            } else {
                status = try_prefill_bf16_gdn_gates_sm100(
                    layer.b.values, layer.a.values, work.normalized, rows, work.gdn_b, work.gdn_a,
                    work.mlp_gate, rows * dims.intermediate * sizeof(Bf16), stream);
                if (!status) status = try_prefill_bf16_gdn_gates(
                    layer.b.values, layer.a.values, work.normalized, rows, work.gdn_b, work.gdn_a,
                    stream);
            }
            if (status) return check(*status, "Project BF16 prefill Gated DeltaNet gates");
        }
        project(layer.b, work.normalized, work.gdn_b);
        project(layer.a, work.normalized, work.gdn_a);
    }

    void gdn_recur(const GdnLayer& layer, TextGdnState state, int bank,
                   TextFp8Activation quantized) const {
        const auto& dims = work.dims;
        if (std::size_t(dims.kv_width) * sizeof(Bf16) <
            std::size_t(dims.gdn_heads) * sizeof(float2)) {
            throw std::logic_error("Prefill key buffer too narrow for Gated DeltaNet gates");
        }
        const bool chunked_recurrence = position != 0 || bank != 0;
        auto* gates = reinterpret_cast<float2*>(work.keys);
        auto* scratch = chunked_recurrence ? work.mlp_gate : nullptr;
        const auto scratch_bytes = rows * dims.intermediate * sizeof(Bf16);
        const bool local = dims.is_local_gdn();
        if (quantized.values && local && gdn_vendor_prefill_enabled() &&
            gdn_vendor_prefill_eligible_rows(rows)) {
            const auto status = gdn_vendor_prefill_local_fp8(
                rows, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a, layer.recurrent,
                state, work.gdn_scratch, dims.gdn_workspace_bytes(rows), gates, quantized,
                stream);
            check(status, "Advance and quantize prefill Gated DeltaNet (vendor)");
            return;
        }
        if (quantized.values) {
            const auto status = local
                ? tensor_core_prefill_gdn_local(rows, work.gdn_qkv, work.gdn_z, work.gdn_b,
                                                work.gdn_a, layer.recurrent, state,
                                                work.gdn_scratch, gates, quantized, stream,
                                                scratch, scratch_bytes)
                : tensor_core_prefill_gdn(rows, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                                          layer.recurrent, state, work.gdn_scratch, gates,
                                          quantized, stream, scratch, scratch_bytes);
            check(status, "Advance and quantize prefill Gated DeltaNet");
            return;
        }
        if (local && gdn_vendor_prefill_enabled() && gdn_vendor_prefill_eligible_rows(rows)) {
            const auto status = gdn_vendor_prefill_local(
                rows, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a, layer.recurrent,
                state, work.gdn_scratch, dims.gdn_workspace_bytes(rows), gates,
                work.attention, stream);
            check(status, "Advance prefill Gated DeltaNet (vendor)");
            return;
        }
        const auto status = tensor_core_fp8
            ? (local
                   ? tensor_core_prefill_gdn_local(rows, work.gdn_qkv, work.gdn_z, work.gdn_b,
                                                   work.gdn_a, layer.recurrent, state,
                                                   work.gdn_scratch, gates, work.attention,
                                                   stream, scratch, scratch_bytes)
                   : tensor_core_prefill_gdn(rows, work.gdn_qkv, work.gdn_z, work.gdn_b,
                                             work.gdn_a, layer.recurrent, state, work.gdn_scratch,
                                             gates, work.attention, stream, scratch,
                                             scratch_bytes))
            : (local
                   ? text_gdn_forward_local(rows, TextGdnMode::prefill, work.gdn_qkv, work.gdn_z,
                                            work.gdn_b, work.gdn_a, layer.recurrent, state,
                                            work.gdn_scratch, text_gdn_local_workspace_bytes(rows),
                                            work.attention, stream)
                   : text_gdn_forward(rows, TextGdnMode::prefill, work.gdn_qkv, work.gdn_z,
                                      work.gdn_b, work.gdn_a, layer.recurrent, state,
                                      work.gdn_scratch, text_gdn_workspace_bytes(rows),
                                      work.attention, stream));
        check(status, "Advance prefill Gated DeltaNet state");
    }

    void gdn_attention(const GdnLayer& layer, TextGdnState state, int bank) const {
        const bool quantized = tensor_core_fp8;
        gdn_recur(layer, state, bank, quantized ? activation : TextFp8Activation{});
        const auto input = quantized
            ? TextFp8ActivationView{activation.values, activation.scales}
            : quantize(work.attention, work.dims.gdn_value_width);
        project(layer.output, input, work.projected_attention);
    }

    void gqa_input(const GqaLayer& layer, bool reuse_all = false) const {
        const TextFp8ActivationView input{activation.values, activation.scales};
        if (reuse_all || cached_gqa_tokens >= position + rows)
            return project(layer.q_gate, input, work.q_gate);
        if (!project_packed(input, {
                {layer.q_gate.values, layer.q_gate.scales, work.q_gate, layer.q_gate.rows},
                {layer.key.values, layer.key.scales, work.keys, layer.key.rows},
                {layer.value.values, layer.value.scales, work.values, layer.value.rows},
            })) {
            project(layer.q_gate, input, work.q_gate);
            project(layer.key, input, work.keys);
            project(layer.value, input, work.values);
        }
    }

    void gqa_append(const GqaLayer& layer, Mxfp8KvCache cache, Mxfp8PageTable pages) const {
        const auto& dims = work.dims;
        const auto geometry = dims.gqa_geometry();
        const int reused = std::clamp(cached_gqa_tokens - position, 0, rows);
        if (reused == 0) {
            const auto rotation = mrope
                ? mrope_gqa_preprocess(mrope, rows, work.q_gate, work.keys, layer.query_norm,
                                       layer.key_norm, work.rope_queries, work.rope_keys, stream,
                                       geometry)
                : text_gqa_preprocess(position, rows, work.q_gate, work.keys, layer.query_norm,
                                      layer.key_norm, work.rope_queries, work.rope_keys, stream,
                                      geometry);
            check(rotation, "Normalize and rotate prefill Q/K");
        } else {
            check(prefill_gqa_queries(position, mrope, rows, work.q_gate, layer.query_norm,
                                      work.rope_queries, stream, geometry),
                  "Normalize and rotate prefill Q");
            if (reused == rows) return;
            const auto offset = std::size_t(reused) * dims.kv_width;
            const auto rotation = mrope
                ? mrope_gqa_preprocess_keys(mrope + reused, rows - reused, work.keys + offset,
                                            layer.key_norm, work.rope_keys + offset, stream,
                                            geometry)
                : text_gqa_preprocess_keys(position + reused, rows - reused, work.keys + offset,
                                           layer.key_norm, work.rope_keys + offset, stream,
                                           geometry);
            check(rotation, "Normalize and rotate suffix prefill K");
        }
        const auto offset = std::size_t(reused) * dims.kv_width;
        check(mxfp8_kv_append(cache, pages, position + reused, rows - reused,
                              work.rope_keys + offset, work.values + offset, stream, geometry),
              "Append prefill MXFP8 KV");
    }

    void gqa_gate() const {
        check(text_gqa_postgate(rows, work.q_gate, work.attention, work.attention, stream,
                                work.dims.gqa_geometry()),
              "Gate prefill grouped-query attention");
    }

    void gqa_attention(const GqaLayer& layer, Mxfp8KvCache cache, Mxfp8PageTable pages,
                       PairedPrefillScratch scratch, bool postgate = true) const {
        gqa_append(layer, cache, pages);
        const auto status = tensor_core_fp8
            ? tensor_core_model_prefill_gqa(cache, pages, position, rows, work.rope_queries,
                                            work.attention, stream, scratch)
            : mxfp8_gqa_attention(cache, pages, position + rows, position, rows, work.rope_queries,
                                   work.attention, stream, work.dims.gqa_geometry());
        check(status, "Read causal prefill MXFP8 attention");
        if (postgate) gqa_gate();
    }

    void project_scaled(Fp8Weight weight, TextFp8ActivationView input, Bf16* output,
                        bool eligible) const {
        if (eligible) {
            const auto status = try_prefill_scaled_sm100(input, weight.values, weight.scales, rows,
                weight.rows, weight.columns, output, work.mlp_gate, work.mlp_up, stream);
            if (status) return check(*status, "Run scaled FP8 prefill projection");
        }
        project(weight, input, output);
    }

    void mlp(const Layer& layer) const {
        const auto& dims = work.dims;
        const TextFp8ActivationView input{activation.values, activation.scales};
        if (!project_packed(input, {
                {layer.mlp_gate.values, layer.mlp_gate.scales, work.mlp_gate, layer.mlp_gate.rows},
                {layer.mlp_up.values, layer.mlp_up.scales, work.mlp_up, layer.mlp_up.rows},
            })) {
            project(layer.mlp_gate, input, work.mlp_gate);
            project(layer.mlp_up, input, work.mlp_up);
        }
        if (tensor_core_fp8) {
            check(tensor_core_prefill_swiglu_quantize(work.mlp_gate, work.mlp_up,
                                                      dims.intermediate, rows, activation, stream),
                  "Apply and quantize prefill SwiGLU");
            project_scaled(layer.mlp_down, input, work.hidden,
                           rows == 1024 ||
                           (grouped_projection && rows == grouped_prefill_max_rows));
            return;
        }
        check(text_swiglu(work.mlp_gate, work.mlp_up, rows * dims.intermediate,
                          work.mlp_gate, stream), "Apply prefill SwiGLU");
        project(layer.mlp_down, quantize(work.mlp_gate, dims.intermediate), work.hidden);
    }

    void normalize(const Bf16* branch, Bf16* residual, const Bf16* weight,
                   RmsQuantizeLaunch launch = RmsQuantizeLaunch::ordinary) const {
        if (tensor_core_fp8) {
            check(tensor_core_prefill_rms_quantize(branch, residual, weight, hidden_size, rows,
                                                   work.normalized, activation, stream, launch),
                  "Normalize and quantize prefill residual");
            return;
        }
        check(text_rms_norm_rows(branch, residual, weight, hidden_size, rows,
                                 work.normalized, residual, stream), "Normalize prefill residual");
        quantize(work.normalized, hidden_size);
    }

    void normalize_tp(const Bf16* branch, Bf16* residual, const Bf16* weight,
                      RmsQuantizeLaunch launch, bool save_normalized) const {
        if (tensor_core_fp8 && rows == grouped_prefill_max_rows) {
            const auto status = try_prefill_rms_quantize_vector(
                branch, residual, weight, save_normalized ? work.normalized : nullptr,
                activation, stream);
            if (status) return check(*status, "Normalize and quantize TP prefill residual");
        }
        normalize(branch, residual, weight, launch);
    }
};

inline void gather_embeddings(const Bf16* embedding, const int* tokens, int rows,
                              Bf16* residual, cudaStream_t stream) {
    for (int offset = 0; offset < rows; offset += text_embedding_max_rows) {
        const int count = std::min(text_embedding_max_rows, rows - offset);
        check(text_embedding_rows(embedding, vocab_size, hidden_size, tokens + offset, count,
                                  residual + offset * hidden_size, stream),
              "Gather prefill embeddings");
    }
}

inline void stage_prefill_row(Request& request, const TileBuffers& work, int row,
                              cudaStream_t stream) {
    const int offset = row * hidden_size;
    const auto bytes = hidden_size * sizeof(Bf16);
    copy_cuda_async(request.work.hidden, work.hidden + offset, bytes,
                    cudaMemcpyDeviceToDevice, stream, "Stage final prefill hidden row");
    copy_cuda_async(request.work.residual, work.residual + offset, bytes,
                    cudaMemcpyDeviceToDevice, stream, "Stage final prefill residual row");
}

inline void capture_prefill_scores(const DecoderModel& model, const PrefillGrant& grant,
                                   const TileBuffers* buffers, int offset, cudaStream_t stream) {
    if (!grant.scores) return;
    const auto& capture = *grant.scores;
    const int first = std::max(0, capture.start - grant.position);
    const int end = capture.score_end(grant.position, grant.rows) - grant.position;
    for (int start = first; start < end; start += decoder_batch_max_lanes) {
        const int count = std::min(end - start, decoder_batch_max_lanes);
        for (int index = 0; index < count; ++index) {
            const int row = start + index;
            auto work = grant.request->work;
            if (buffers) {
                work.hidden = buffers->hidden + (offset + row) * hidden_size;
                work.residual = buffers->residual + (offset + row) * hidden_size;
            }
            capture.enqueue(model, work, grant.position + row, index, stream);
        }
        capture.collect(count, stream);
    }
}

constexpr int layerwise_row_alignment = 128;
constexpr int max_layerwise_rows = 1023;
static_assert(max_layerwise_rows / 256 <= max_gdn_prefill_segments);
static_assert(text_gqa_kv_width * sizeof(Bf16) >= text_gdn_value_heads * sizeof(float2));

inline int aligned_layerwise_rows(int rows) {
    return (rows + layerwise_row_alignment - 1) / layerwise_row_alignment *
        layerwise_row_alignment;
}

inline PrefillGroup prefill_group(const PrefillGrant* grants, int count) {
    PrefillGroup group{0, 0};
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        if (grant.rows < 256 || grant.rows != grant.tile_rows ||
            aligned_layerwise_rows(grant.rows) != grant.rows ||
            grant.rows > max_layerwise_rows - group.rows) {
            break;
        }
        ++group.grants;
        group.rows += grant.rows;
    }
    return group.grants > 1 ? group : PrefillGroup{1, 0};
}

template <typename Run>
void each_prefill_segment(const PrefillGrant* grants, int count, int offset, Run run) {
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        run(grant, offset);
        offset += grant.rows;
    }
}

template <typename Run>
void each_attention_group(const PrefillGrant* grants, int count, Run run) {
    int offset = 0;
    for (int index = 0; index < count;) {
        const auto group = prefill_group(grants + index, count - index);
        run(grants + index, group.grants, offset);
        offset += group.grants == 1 ? grants[index].rows : group.rows;
        index += group.grants;
    }
}

inline void read_paged_sequences(const PagedPrefillSequence* sequences, int count,
                                 cudaStream_t stream, PairedPrefillScratch scratch) {
    if (const auto batched = try_batched_paged_prefill(sequences, count, stream)) {
        check(*batched, "Read batched causal prefill MXFP8 attention");
        return;
    }
    for (int index = 0; index < count; ++index) {
        const auto& sequence = sequences[index];
        check(tensor_core_model_prefill_gqa(sequence.cache, sequence.pages, sequence.position,
                                           sequence.rows, sequence.queries, sequence.output,
                                           stream, scratch), "Read causal prefill MXFP8 attention");
    }
}

inline Chunk prefill_attention_segment(const Chunk& chunk, const PrefillGrant& grant, int offset) {
    return Chunk{{{}, grant.rows, chunk.stream, true}, chunk.work.attention_segment(offset),
        grant.position, grant.media ? grant.media->positions + grant.position : nullptr,
        grant.request->cached_gqa_tokens};
}

inline bool prefill_grouped_partial_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                                        const PrefillGrant* grants, int count,
                                        PairedPrefillScratch scratch, PairedPrefillScratch staging,
                                        bool postgate) {
    constexpr int width = staged_prefill_batch_size;
    if (count < width) return false;
    const int rows = grants[0].rows;
    const bool ordered = count % width == 0 && (rows == 256 || rows == 512) &&
        std::all_of(grants, grants + count, [rows](const PrefillGrant& grant) {
            return grant.rows == rows && grant.position >= 8192 && grant.position <= 32768 - rows;
        });
    int offset = 0;
    int first = 0;
    for (; first + width <= count; first += width) {
        std::array<PagedPrefillSequence, width> sequences;
        int packet_end = offset;
        for (int lane = 0; lane < width; ++lane) {
            const auto& grant = grants[first + lane];
            const auto segment = prefill_attention_segment(chunk, grant, packet_end);
            const auto cache = grant.request->caches[bank];
            const auto pages = grant.request->pages;
            segment.gqa_append(layer, cache, pages);
            sequences[lane] = {cache, pages, grant.position, grant.rows, segment.work.rope_queries,
                               segment.work.attention};
            packet_end += grant.rows;
        }
        auto submitted = launch_partial_staged_tmem_prefill(sequences.data(), width,
                                                            chunk.stream, staging);
        if (!submitted && ordered) {
            submitted = try_batched_partial_paged_prefill(sequences.data(), width, chunk.stream);
        }
        if (submitted) check(*submitted, "Read grouped ordered partial prefill attention");
        for (int lane = 0; lane < width; ++lane) {
            const auto& grant = grants[first + lane];
            const auto segment = prefill_attention_segment(chunk, grant, offset);
            const auto& sequence = sequences[lane];
            if (!submitted) {
                check(tensor_core_model_prefill_gqa(sequence.cache, sequence.pages,
                    sequence.position, sequence.rows, sequence.queries, sequence.output,
                    chunk.stream, scratch), "Read causal prefill MXFP8 attention");
            }
            if (postgate) segment.gqa_gate();
            offset += grant.rows;
        }
    }
    for (; first < count; ++first) {
        const auto& grant = grants[first];
        const auto segment = prefill_attention_segment(chunk, grant, offset);
        segment.gqa_attention(layer, grant.request->caches[bank], grant.request->pages,
                              scratch, postgate);
        offset += grant.rows;
    }
    return true;
}

inline bool prefill_causal_group_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                                     const PrefillGrant* grants, int count,
                                     PairedPrefillScratch serial_scratch, bool postgate) {
    constexpr int tile_rows = batch_prefill_tile_rows;
    constexpr int group_rows = grouped_prefill_max_rows;
    constexpr int long_context = staged_prefill_batch_long_max_position + tile_rows;
    const auto& first = grants[0];
    if (count != group_rows / tile_rows || first.position < 0 ||
        first.position > long_context - group_rows || first.position % tile_rows != 0) {
        return false;
    }
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        if (grant.request != first.request || grant.media || grant.rows != tile_rows ||
            grant.position != first.position + index * tile_rows) return false;
    }
    const auto cache = first.request->caches[bank];
    const auto pages = first.request->pages;
    auto combined = first;
    combined.rows = group_rows;
    const auto segment = prefill_attention_segment(chunk, combined, 0);
    segment.gqa_append(layer, cache, pages);
    const PagedPrefillSequence sequence{cache, pages, first.position, group_rows,
                                     chunk.work.rope_queries, chunk.work.attention};
    const auto submitted = launch_causal_group_staged_tmem_prefill(
        sequence, chunk.stream, chunk.work.gqa_scratch());
    if (submitted) {
        check(*submitted, "Read staged single-request causal prefill group");
    } else {
        each_prefill_segment(grants, count, 0, [&](const PrefillGrant& grant, int offset) {
            const auto tile = prefill_attention_segment(chunk, grant, offset);
            check(tensor_core_model_prefill_gqa(cache, pages, grant.position, grant.rows,
                      tile.work.rope_queries, tile.work.attention, chunk.stream, serial_scratch),
                  "Read causal prefill MXFP8 attention");
        });
    }
    if (postgate) segment.gqa_gate();
    return true;
}

#include "sglang/runtime/execution/decoder_prefill_staged_gqa.h"

inline void prefill_group_gdn(const Chunk& chunk, const GdnLayer& layer, int bank,
                              const PrefillGrant* grants, int count, int offset) {
    const auto& buffers = chunk.work;
    const auto& dims = buffers.dims;
    if (dims.is_local_gdn() && chunk.tensor_core_fp8 && gdn_vendor_prefill_enabled()) {
        bool eligible = true;
        each_prefill_segment(grants, count, offset, [&](const PrefillGrant& grant, int) {
            eligible = eligible && gdn_vendor_prefill_eligible_rows(grant.rows);
        });
        if (eligible) {
            each_prefill_segment(grants, count, offset, [&](const PrefillGrant& grant, int row) {
                const auto segment = buffers.attention_segment(row);
                const TextFp8Activation fp8{
                    buffers.activation.values + std::size_t(row) * dims.gdn_value_width,
                    buffers.activation.scales + std::size_t(row) * dims.gdn_heads,
                };
                const auto status = gdn_vendor_prefill_local_fp8(
                    grant.rows, segment.gdn_qkv, segment.gdn_z, segment.gdn_b, segment.gdn_a,
                    layer.recurrent, grant.request->states[bank],
                    buffers.gdn_scratch,
                    dims.gdn_workspace_bytes(grant.rows),
                    reinterpret_cast<float2*>(segment.keys), fp8, chunk.stream);
                check(status, "Advance grouped prefill Gated DeltaNet (vendor)");
            });
            return;
        }
    }
    GdnPrefillSegment segments[max_gdn_prefill_segments]{};
    bool grouped_recurrence = count > 1;
    int index = 0;
    each_prefill_segment(grants, count, offset, [&](const PrefillGrant& grant, int row) {
        const auto segment = buffers.attention_segment(row);
        const TextFp8Activation fp8{
            buffers.activation.values + std::size_t(row) * dims.gdn_value_width,
            buffers.activation.scales + std::size_t(row) * dims.gdn_heads,
        };
        grouped_recurrence = grouped_recurrence &&
            grant.rows % layerwise_row_alignment == 0 && (grant.position != 0 || bank != 0);
        segments[index++] = {
            grant.rows, segment.gdn_qkv, segment.gdn_z, segment.gdn_b, segment.gdn_a,
            grant.request->states[bank],
            buffers.gdn_scratch +
                (dims.is_local_gdn() ? text_gdn_local_workspace_bytes(row)
                                     : text_gdn_workspace_bytes(row)) / sizeof(Bf16),
            reinterpret_cast<float2*>(segment.keys), fp8,
            buffers.mlp_gate + row * dims.intermediate,
            std::size_t(grant.rows) * dims.intermediate * sizeof(Bf16),
        };
    });
    const auto status = grouped_recurrence
        ? (dims.is_local_gdn()
               ? tensor_core_prefill_gdn_group_local(segments, count, layer.recurrent,
                                                     chunk.stream)
               : tensor_core_prefill_gdn_group(segments, count, layer.recurrent, chunk.stream))
        : std::nullopt;
    if (status) {
        if (*status != cudaSuccess && cudaPeekAtLastError() == *status) {
            (void)cudaGetLastError();
        }
        check(*status, "Advance grouped prefill Gated DeltaNet");
        return;
    }
    index = 0;
    each_prefill_segment(grants, count, offset, [&](const PrefillGrant& grant, int row) {
        const auto& segment = segments[index++];
        prefill_attention_segment(chunk, grant, row).gdn_recur(
            layer, segment.state, bank, segment.output);
    });
}

inline PairedPrefillScratch serial_gqa_scratch(const TileBuffers& buffers, int serial_capacity) {
    ScratchLayout serial_layout{reinterpret_cast<std::byte*>(buffers.hidden)};
    const auto sizes = TileBuffers(serial_layout, serial_capacity, buffers.dims).gqa_scratch();
    auto scratch = buffers.gqa_scratch();
    scratch.key_bytes = sizes.key_bytes;
    scratch.value_bytes = sizes.value_bytes;
    return scratch;
}

inline bool is_full_partial_prefill_group(const PrefillGrant* grants, PrefillGroup group) {
    const int rows = grants[0].rows;
    if (group.rows != grouped_prefill_max_rows || (rows != 256 && rows != 512) ||
        group.grants % max_gdn_prefill_segments != 0) {
        return false;
    }
    for (int index = 1; index < group.grants; ++index) {
        if (grants[index].rows != rows) return false;
    }
    return true;
}

inline void prefill_partial_gdn(const Chunk& chunk, const GdnLayer& layer, int bank,
                                const PrefillGrant* grants, int count, bool grouped) {
    const auto& activation = chunk.work.activation;
    const auto& dims = chunk.work.dims;
    if (grouped) {
        int offset = 0;
        for (int index = 0; index < count; index += max_gdn_prefill_segments) {
            prefill_group_gdn(chunk, layer, bank, grants + index, max_gdn_prefill_segments, offset);
            offset += max_gdn_prefill_segments * grants[index].rows;
        }
        return;
    }
    if (std::size_t(dims.kv_width) * sizeof(Bf16) <
        std::size_t(dims.gdn_heads) * sizeof(float2)) {
        throw std::logic_error("Prefill key buffer too narrow for Gated DeltaNet gates");
    }
    const bool vendor =
        chunk.tensor_core_fp8 && dims.is_local_gdn() && gdn_vendor_prefill_enabled();
    int index = 0;
    int row = 0;
    while (index < count) {
        const PrefillGrant& first = grants[index];
        int run = 1;
        while (index + run < count && grants[index + run].request == first.request &&
               !grants[index + run].media && grants[index + run].rows == first.rows &&
               grants[index + run].position == first.position + run * first.rows) {
            ++run;
        }
        const int rows = run * first.rows;
        if (vendor && activation.values && gdn_vendor_prefill_eligible_rows(rows)) {
            const auto segment = prefill_attention_segment(chunk, first, row);
            const TextFp8Activation quantized{
                activation.values + std::size_t(row) * dims.gdn_value_width,
                activation.scales + std::size_t(row) * dims.gdn_heads,
            };
            check(gdn_vendor_prefill_local_fp8(
                      rows, segment.work.gdn_qkv, segment.work.gdn_z, segment.work.gdn_b,
                      segment.work.gdn_a, layer.recurrent, first.request->states[bank],
                      chunk.work.gdn_scratch, dims.gdn_workspace_bytes(rows),
                      reinterpret_cast<float2*>(segment.work.keys), quantized, chunk.stream),
                  "Advance prefill Gated DeltaNet (vendor)");
        } else {
            for (int step = 0; step < run; ++step) {
                const auto segment =
                    prefill_attention_segment(chunk, grants[index + step], row);
                const TextFp8Activation quantized{
                    activation.values + std::size_t(row) * dims.gdn_value_width,
                    activation.scales + std::size_t(row) * dims.gdn_heads,
                };
                segment.gdn_recur(layer, grants[index + step].request->states[bank], bank,
                                  quantized);
                row += grants[index + step].rows;
            }
            index += run;
            continue;
        }
        row += rows;
        index += run;
    }
}

inline void prefill_serial_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                               const PrefillGrant* grants, int count, int serial_capacity,
                               bool causal, bool postgate) {
    const auto scratch = serial_gqa_scratch(chunk.work, serial_capacity);
    if (causal) {
        if (prefill_causal_group_gqa(chunk, layer, bank, grants, count, scratch, postgate)) return;
    } else if (prefill_grouped_partial_gqa(
                   chunk, layer, bank, grants, count, scratch, chunk.work.gqa_scratch(),
                   postgate)) {
        return;
    }
    each_prefill_segment(grants, count, 0, [&](const PrefillGrant& grant, int offset) {
        const auto segment = prefill_attention_segment(chunk, grant, offset);
        segment.gqa_attention(layer, grant.request->caches[bank], grant.request->pages,
                              scratch, postgate);
    });
}

}
}
