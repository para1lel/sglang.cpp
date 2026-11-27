#include "sglang/runtime/execution/decoder_prefill_tiles.h"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/text/decoder_logprobs.h"
#include "sglang/text/gdn_local.h"
#include "sglang/prefill_fp8.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sglang::decoder::prefill_detail {

namespace {

template <typename Element, std::size_t alignment = 16>
Element* take(ScratchLayout& layout, std::size_t count) {
    const auto offset = (layout.bytes + alignment - 1) / alignment * alignment;
    layout.bytes = offset + (count * sizeof(Element));
    return layout.data ? reinterpret_cast<Element*>(layout.data + offset) : nullptr;
}

}

Bf16* ScratchLayout::take_bf16(std::size_t count) {
    return take<Bf16>(*this, count);
}

PrefillDims PrefillDims::from_layers(const std::array<Layer, layer_count>& layers) {
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

void PrefillDims::validate() const {
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

int PrefillDims::attention_width() const { return std::max(gdn_value_width, q_width); }

bool PrefillDims::is_local_gdn() const { return gdn_heads == GdnLocal::value_heads; }

GqaGeometry PrefillDims::gqa_geometry() const {
    return GqaGeometry{q_width / mxfp8_head_dim, kv_width / mxfp8_head_dim};
}

std::size_t PrefillDims::gdn_workspace_bytes(int rows) const {
    return is_local_gdn() ? text_gdn_local_workspace_bytes(rows)
                          : text_gdn_workspace_bytes(rows);
}

std::size_t seed_scratch_bytes(int concat_rows, int kv_rows) {
    return std::size_t(concat_rows) * 2 * hidden_size * sizeof(Bf16) +
           std::size_t(kv_rows) * 3 * text_gqa_kv_width * sizeof(Bf16);
}

SeedScratch make_seed_scratch(Bf16* memory, int concat_rows, int kv_rows) {
    const auto concatenated = std::size_t(concat_rows) * 2 * hidden_size;
    const auto stride = std::size_t(kv_rows) * text_gqa_kv_width;
    return {memory, memory + concatenated, memory + concatenated + stride,
            memory + concatenated + 2 * stride};
}

TileBuffers::TileBuffers(ScratchLayout& layout, int rows, const PrefillDims& tile_dims)
    : dims(tile_dims), hidden(take<Bf16>(layout, static_cast<std::size_t>(rows) * hidden_size)),
      normalized(take<Bf16>(layout, static_cast<std::size_t>(rows) * hidden_size)),
      residual(take<Bf16>(layout, static_cast<std::size_t>(rows) * hidden_size)),
      projected_attention(take<Bf16>(layout, static_cast<std::size_t>(rows) * hidden_size)),
      attention(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.attention_width())),
      mlp_gate(take<Bf16, 32>(layout, static_cast<std::size_t>(rows) * dims.intermediate)),
      mlp_up(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.intermediate)),
      q_gate(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.q_gate_width)),
      keys(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.kv_width)),
      values(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.kv_width)),
      rope_queries(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.q_width)),
      rope_keys(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.kv_width)),
      gdn_qkv(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.gdn_qkv_width)),
      gdn_z(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.gdn_value_width)),
      gdn_b(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.gdn_heads)),
      gdn_a(take<Bf16>(layout, static_cast<std::size_t>(rows) * dims.gdn_heads)),
      gdn_scratch(take<Bf16>(layout, dims.gdn_workspace_bytes(rows) / sizeof(Bf16))),
      activation{
          take<std::uint8_t>(layout, static_cast<std::size_t>(rows) * dims.intermediate),
          take<float>(layout,
                      static_cast<std::size_t>(rows) * text_fp8_group_count(dims.intermediate)),
      } {
}

PairedPrefillScratch TileBuffers::gqa_scratch() const {
    const auto mlp_begin = reinterpret_cast<std::uintptr_t>(mlp_gate);
    const auto mlp_end = reinterpret_cast<std::uintptr_t>(q_gate);
    const auto gdn_begin = reinterpret_cast<std::uintptr_t>(gdn_qkv);
    const auto gdn_end = reinterpret_cast<std::uintptr_t>(activation.values);
    return {reinterpret_cast<std::byte*>(mlp_gate), mlp_end - mlp_begin,
            reinterpret_cast<std::byte*>(gdn_qkv), gdn_end - gdn_begin};
}

TileBuffers TileBuffers::attention_segment(int offset) const {
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

bool Chunk::project_packed(TextFp8ActivationView input,
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

void Chunk::seed_draft(MtpDraftRequest& draft, const int* shifted_tokens, int seed_rows,
                const Bf16* final_norm, bool wmma, const MediaPrefill* media,
                const SeedScratch* seed) const {
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

void Chunk::gdn_input(const GdnLayer& layer) const {
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

void Chunk::gdn_recur(const GdnLayer& layer, TextGdnState state, int bank,
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

void Chunk::gdn_attention(const GdnLayer& layer, TextGdnState state, int bank) const {
    const bool quantized = tensor_core_fp8;
    gdn_recur(layer, state, bank, quantized ? activation : TextFp8Activation{});
    const auto input = quantized
        ? TextFp8ActivationView{activation.values, activation.scales}
        : quantize(work.attention, work.dims.gdn_value_width);
    project(layer.output, input, work.projected_attention);
}

void Chunk::gqa_input(const GqaLayer& layer, bool reuse_all) const {
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

void Chunk::gqa_append(const GqaLayer& layer, Mxfp8KvCache cache, Mxfp8PageTable pages) const {
    const auto geometry = work.dims.gqa_geometry();
    const int kv_width = geometry.kv_heads * mxfp8_head_dim;
    const int reused = std::clamp(cached_gqa_tokens - position, 0, rows);
    const auto offset = std::size_t(reused) * kv_width;
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
        const auto rotation = mrope
            ? mrope_gqa_preprocess_keys(mrope + reused, rows - reused, work.keys + offset,
                                        layer.key_norm, work.rope_keys + offset, stream,
                                        geometry)
            : text_gqa_preprocess_keys(position + reused, rows - reused, work.keys + offset,
                                       layer.key_norm, work.rope_keys + offset, stream,
                                       geometry);
        check(rotation, "Normalize and rotate suffix prefill K");
    }
    check(mxfp8_kv_append(cache, pages, position + reused, rows - reused,
                          work.rope_keys + offset, work.values + offset, stream, geometry),
          "Append prefill MXFP8 KV");
}

void Chunk::gqa_gate() const {
    check(text_gqa_postgate(rows, work.q_gate, work.attention, work.attention, stream,
                            work.dims.gqa_geometry()),
          "Gate prefill grouped-query attention");
}

void Chunk::gqa_read(Mxfp8KvCache cache, Mxfp8PageTable pages, PairedPrefillScratch scratch) const {
    auto staged = long_staging.keys && tensor_core_fp8
        ? try_long_serial_staged_paged_prefill(
              {cache, pages, position, rows, work.rope_queries, work.attention}, stream,
              long_staging)
        : std::nullopt;
    if (!staged) {
        staged = tensor_core_fp8
            ? tensor_core_model_prefill_gqa(cache, pages, position, rows, work.rope_queries,
                                            work.attention, stream, scratch)
            : mxfp8_gqa_attention(cache, pages, position + rows, position, rows,
                work.rope_queries, work.attention, stream, work.dims.gqa_geometry());
    }
    check(*staged, "Read causal prefill MXFP8 attention");
}

void Chunk::gqa_attention(const GqaLayer& layer, Mxfp8KvCache cache, Mxfp8PageTable pages,
                   PairedPrefillScratch scratch, bool postgate) const {
    gqa_append(layer, cache, pages);
    gqa_read(cache, pages, scratch);
    if (postgate) gqa_gate();
}

void Chunk::project_scaled(Fp8Weight weight, TextFp8ActivationView input, Bf16* output,
                    bool eligible) const {
    if (eligible) {
        const auto status = try_prefill_scaled_sm100(input, weight.values, weight.scales, rows,
            weight.rows, weight.columns, output, work.mlp_gate, work.mlp_up, stream);
        if (status) return check(*status, "Run scaled FP8 prefill projection");
    }
    project(weight, input, output);
}

void Chunk::mlp(const Layer& layer) const {
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

void Chunk::normalize(const Bf16* branch, Bf16* residual, const Bf16* weight,
               RmsQuantizeLaunch launch) const {
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

void Chunk::normalize_tp(const Bf16* branch, Bf16* residual, const Bf16* weight,
                  RmsQuantizeLaunch launch, bool save_normalized) const {
    if (tensor_core_fp8 && rows == grouped_prefill_max_rows) {
        const auto status = try_prefill_rms_quantize_vector(
            branch, residual, weight, save_normalized ? work.normalized : nullptr,
            activation, stream);
        if (status) return check(*status, "Normalize and quantize TP prefill residual");
    }
    normalize(branch, residual, weight, launch);
}

void gather_embeddings(const Bf16* embedding, const int* tokens, int rows,
                              Bf16* residual, cudaStream_t stream) {
    for (int offset = 0; offset < rows; offset += text_embedding_max_rows) {
        const int count = std::min(text_embedding_max_rows, rows - offset);
        check(text_embedding_rows(embedding, vocab_size, hidden_size, tokens + offset, count,
                                  residual + offset * hidden_size, stream),
              "Gather prefill embeddings");
    }
}

void stage_prefill_row(Request& request, const TileBuffers& work, int row,
                              cudaStream_t stream) {
    const int offset = row * hidden_size;
    const auto bytes = hidden_size * sizeof(Bf16);
    copy_cuda_async(request.work.hidden, work.hidden + offset, bytes,
                    cudaMemcpyDeviceToDevice, stream, "Stage final prefill hidden row");
    copy_cuda_async(request.work.residual, work.residual + offset, bytes,
                    cudaMemcpyDeviceToDevice, stream, "Stage final prefill residual row");
}

void capture_prefill_scores(const DecoderModel& model, const PrefillGrant& grant,
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

namespace {

int aligned_layerwise_rows(int rows) {
    return (rows + layerwise_row_alignment - 1) / layerwise_row_alignment *
        layerwise_row_alignment;
}

}

PrefillGroup prefill_group(const PrefillGrant* grants, int count) {
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

Chunk prefill_attention_segment(const Chunk& chunk, const PrefillGrant& grant, int offset) {
    return Chunk{{{}, grant.rows, chunk.stream, true}, chunk.work.attention_segment(offset),
        grant.position, grant.media ? grant.media->positions + grant.position : nullptr,
        grant.request->cached_gqa_tokens, chunk.long_staging};
}

}

namespace sglang::decoder {

std::size_t prefill_workspace_bytes(int rows) {
    prefill_detail::ScratchLayout layout;
    const prefill_detail::TileBuffers buffers(layout, rows);
    return layout.bytes;
}

TextFp8ActivationView PrefillProjections::quantize(const Bf16* input, int width) const {
    check(text_quantize_fp8_rows(input, width, rows, activation, stream),
          "Quantize prefill projection input");
    return {activation.values, activation.scales};
}

void PrefillProjections::project(Fp8Weight weight, TextFp8ActivationView input,
                                 Bf16* output) const {
    if (tensor_core_fp8) {
        const auto status = try_text_fp8_gemm_sm100(
            weight.values, weight.scales, input, rows, weight.rows, weight.columns,
            output, stream, rows > prefill_chunk_rows, grouped_projection);
        if (status) {
            check(*status, "Run tensor-core prefill FP8 projection");
            return;
        }
    }
    check(text_fp8_gemv_rows(weight.values, weight.scales, input, weight.rows, weight.columns,
                             rows, output, stream), "Run exact prefill FP8 projection");
}

void PrefillProjections::project(Bf16Weight weight, const Bf16* input, Bf16* output) const {
    check(text_bf16_gemv_rows(weight.values, input, weight.rows, weight.columns, rows, output,
                              stream), "Run exact prefill BF16 projection");
}

int PromptLogprobCapture::score_end(int position, int rows) const {
    return std::min(position + rows, static_cast<int>(prompt->size()) - 1);
}

void PromptLogprobCapture::enqueue(const DecoderModel& model, Workspace& work, int position,
                                  int index, cudaStream_t stream) const {
    model.compute_prefill_logits(work, stream);
    const int token = (*prompt)[position + 1];
    check(enqueue_prompt_logprobs(work.logits, vocab_size, token, top, workspace,
                                  readback + index, stream), "Score prompt token");
}

void PromptLogprobCapture::collect(int rows, cudaStream_t stream) const {
    if (!rows) return;
    synchronize_cuda(stream, "Collect prompt token scores");
    for (int index = 0; index < rows; ++index) {
        const auto& row = readback[index];
        output->push_back(
            row.selected.token_id < 0 ? nullptr : collect_generated_logprobs(row, top));
    }
}

void finish_prompt_logprobs(const DecoderOptions& options, DecoderPromptLogprobs& scores,
                                     DecoderBatchStep& result) {
    result.sampling_failed = std::any_of(scores.begin(), scores.end(),
                                        [](const auto& row) { return !row; });
    if (!result.sampling_failed) {
        result.prompt_logprobs = std::make_shared<DecoderPromptLogprobs>(std::move(scores));
    }
    if (options.max_new_tokens == 0 || result.sampling_failed) {
        result.finish_reason = DecoderFinishReason::length;
    }
}

void PrefillProjections::project_seed_fc(
    Bf16Weight weight, const Bf16* input, Bf16* output) const {
    if (!wmma_draft_seed) {
        project(weight, input, output);
        return;
    }
    check(text_bf16_seed_fc(weight.values, input, weight.rows, weight.columns, rows, output,
                            stream), "Run MTP prefill seed FC");
}

}

namespace sglang {

using namespace decoder;

void Decoder::Model::seed_draft(Workspace& work, MtpDraftRequest& draft, int position,
                                int shifted_token, cudaStream_t stream,
                                const MediaPrefill* media) const {
    normalize_final(work, stream);
    draft.enqueue(position, shifted_token, work.normalized, false, stream, nullptr, nullptr,
                  media ? media->positions + position : nullptr, -1,
                  media ? media->feature(position + 1) : nullptr);
}

}
