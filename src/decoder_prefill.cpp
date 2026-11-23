#include "sglang/decoder_prefill.h"
#include "sglang/decoder_batch.h"

#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/decoder_prefix_cache.h"
#include "sglang/prefill_fp8.h"
#include "sglang/text/decoder_logprobs.h"
#include "sglang/text/prefill_attn_sm100.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string_view>
#include <utility>

namespace sglang {
namespace {

using namespace decoder;

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

struct TileBuffers {
    TileBuffers(ScratchLayout& layout, int rows)
        : hidden(layout.take<Bf16>(rows * hidden_size)),
          normalized(layout.take<Bf16>(rows * hidden_size)),
          residual(layout.take<Bf16>(rows * hidden_size)),
          projected_attention(layout.take<Bf16>(rows * hidden_size)),
          attention(layout.take<Bf16>(rows * std::max(text_gdn_value_width, text_gqa_q_width))),
          mlp_gate(layout.take<Bf16, 32>(rows * intermediate_size)),
          mlp_up(layout.take<Bf16>(rows * intermediate_size)),
          q_gate(layout.take<Bf16>(rows * text_gqa_q_gate_width)),
          keys(layout.take<Bf16>(rows * text_gqa_kv_width)),
          values(layout.take<Bf16>(rows * text_gqa_kv_width)),
          rope_queries(layout.take<Bf16>(rows * text_gqa_q_width)),
          rope_keys(layout.take<Bf16>(rows * text_gqa_kv_width)),
          gdn_qkv(layout.take<Bf16>(rows * text_gdn_qkv_width)),
          gdn_z(layout.take<Bf16>(rows * text_gdn_value_width)),
          gdn_b(layout.take<Bf16>(rows * text_gdn_value_heads)),
          gdn_a(layout.take<Bf16>(rows * text_gdn_value_heads)),
          gdn_scratch(layout.take<Bf16>(text_gdn_workspace_bytes(rows) / sizeof(Bf16))),
          activation{
              layout.take<std::uint8_t>(rows * intermediate_size),
              layout.take<float>(rows * text_fp8_group_count(intermediate_size))} {}

    PairedPrefillScratch gqa_scratch() const {
        const auto mlp_begin = reinterpret_cast<std::uintptr_t>(mlp_gate);
        const auto mlp_end = reinterpret_cast<std::uintptr_t>(q_gate);
        const auto gdn_begin = reinterpret_cast<std::uintptr_t>(gdn_qkv);
        const auto gdn_end = reinterpret_cast<std::uintptr_t>(activation.values);
        return {reinterpret_cast<std::byte*>(mlp_gate), mlp_end - mlp_begin,
                reinterpret_cast<std::byte*>(gdn_qkv), gdn_end - gdn_begin};
    }

    TileBuffers attention_segment(int offset) const {
        static_assert(text_gdn_value_width == text_gqa_q_width);
        auto segment = *this;
        segment.attention += offset * text_gqa_q_width;
        segment.q_gate += offset * text_gqa_q_gate_width;
        segment.keys += offset * text_gqa_kv_width;
        segment.values += offset * text_gqa_kv_width;
        segment.rope_queries += offset * text_gqa_q_width;
        segment.rope_keys += offset * text_gqa_kv_width;
        segment.gdn_qkv += offset * text_gdn_qkv_width;
        segment.gdn_z += offset * text_gdn_value_width;
        segment.gdn_b += offset * text_gdn_value_heads;
        segment.gdn_a += offset * text_gdn_value_heads;
        return segment;
    }

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
                    const Bf16* final_norm, bool wmma, const MediaPrefill* media) const {
        static_assert(intermediate_size >= 2 * hidden_size);
        static_assert(std::max(text_gdn_value_width, text_gqa_q_width) >= hidden_size);
        const PrefillProjections projections{activation, seed_rows, stream, false, false, wmma};
        draft.seed_prefill(position, shifted_tokens, work.hidden, work.residual, final_norm,
                           projections, {work.normalized, work.projected_attention, work.mlp_gate,
                                         work.attention, work.keys, work.values, work.rope_keys},
                           media);
    }

    void gdn_input(const GdnLayer& layer) const {
        const TextFp8ActivationView input{activation.values, activation.scales};
        if (!project_packed(input, {
                {layer.qkv.values, layer.qkv.scales, work.gdn_qkv, layer.qkv.rows},
                {layer.z.values, layer.z.scales, work.gdn_z, layer.z.rows},
            })) {
            project(layer.qkv, input, work.gdn_qkv);
            project(layer.z, input, work.gdn_z);
        }
        if (tensor_core_fp8 && layer.b.rows == text_gdn_value_heads &&
            layer.a.rows == text_gdn_value_heads && layer.b.columns == hidden_size &&
            layer.a.columns == hidden_size) {
            const auto status = try_prefill_bf16_gdn_gates(
                layer.b.values, layer.a.values, work.normalized, rows, work.gdn_b, work.gdn_a,
                stream);
            if (status) {
                check(*status, "Project BF16 prefill Gated DeltaNet gates");
                return;
            }
        }
        project(layer.b, work.normalized, work.gdn_b);
        project(layer.a, work.normalized, work.gdn_a);
    }

    void gdn_recur(const GdnLayer& layer, TextGdnState state, int bank,
                   TextFp8Activation quantized) const {
        static_assert(text_gqa_kv_width * sizeof(Bf16) >= text_gdn_value_heads * sizeof(float2));
        const bool chunked_recurrence = position != 0 || bank != 0;
        auto* gates = reinterpret_cast<float2*>(work.keys);
        auto* scratch = chunked_recurrence ? work.mlp_gate : nullptr;
        const auto scratch_bytes = rows * intermediate_size * sizeof(Bf16);
        if (quantized.values) {
            check(tensor_core_prefill_gdn(rows, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                                          layer.recurrent, state, work.gdn_scratch, gates,
                                          quantized, stream, scratch, scratch_bytes),
                  "Advance and quantize prefill Gated DeltaNet");
            return;
        }
        const auto status = tensor_core_fp8
            ? tensor_core_prefill_gdn(rows, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                                      layer.recurrent, state, work.gdn_scratch, gates,
                                      work.attention, stream, scratch, scratch_bytes)
            : text_gdn_forward(rows, TextGdnMode::prefill, work.gdn_qkv, work.gdn_z, work.gdn_b,
                               work.gdn_a, layer.recurrent, state, work.gdn_scratch,
                               text_gdn_workspace_bytes(rows), work.attention, stream);
        check(status, "Advance prefill Gated DeltaNet state");
    }

    void gdn_attention(const GdnLayer& layer, TextGdnState state, int bank) const {
        const bool quantized = tensor_core_fp8;
        gdn_recur(layer, state, bank, quantized ? activation : TextFp8Activation{});
        const auto input = quantized
            ? TextFp8ActivationView{activation.values, activation.scales}
            : quantize(work.attention, text_gdn_value_width);
        project(layer.output, input, work.projected_attention);
    }

    void gqa_input(const GqaLayer& layer) const {
        const TextFp8ActivationView input{activation.values, activation.scales};
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
        const auto rotation = mrope
            ? mrope_gqa_preprocess(mrope, rows, work.q_gate, work.keys, layer.query_norm,
                                    layer.key_norm, work.rope_queries, work.rope_keys, stream)
            : text_gqa_preprocess(position, rows, work.q_gate, work.keys, layer.query_norm,
                                   layer.key_norm, work.rope_queries, work.rope_keys, stream);
        check(rotation, "Normalize and rotate prefill Q/K");
        check(mxfp8_kv_append(cache, pages, position, rows, work.rope_keys, work.values, stream),
              "Append prefill MXFP8 KV");
    }

    void gqa_gate() const {
        check(text_gqa_postgate(rows, work.q_gate, work.attention, work.attention, stream),
              "Gate prefill grouped-query attention");
    }

    void gqa_attention(const GqaLayer& layer, Mxfp8KvCache cache, Mxfp8PageTable pages,
                       PairedPrefillScratch scratch, bool postgate = true) const {
        gqa_append(layer, cache, pages);
        const auto status = tensor_core_fp8
            ? tensor_core_model_prefill_gqa(cache, pages, position, rows, work.rope_queries,
                                            work.attention, stream, scratch)
            : mxfp8_gqa_attention(cache, pages, position + rows, position, rows, work.rope_queries,
                                   work.attention, stream);
        check(status, "Read causal prefill MXFP8 attention");
        if (postgate) gqa_gate();
    }

    void attention_output(Fp8Weight output, int width) const {
        project(output, quantize(work.attention, width), work.projected_attention);
    }

    void mlp(const Layer& layer) const {
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
                                                      intermediate_size, rows, activation, stream),
                  "Apply and quantize prefill SwiGLU");
            project(layer.mlp_down, input, work.hidden);
            return;
        }
        check(text_swiglu(work.mlp_gate, work.mlp_up, rows * intermediate_size,
                          work.mlp_gate, stream), "Apply prefill SwiGLU");
        project(layer.mlp_down, quantize(work.mlp_gate, intermediate_size), work.hidden);
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
};

void gather_embeddings(const Bf16* embedding, const int* tokens, int rows,
                       Bf16* residual, cudaStream_t stream) {
    for (int offset = 0; offset < rows; offset += text_embedding_max_rows) {
        const int count = std::min(text_embedding_max_rows, rows - offset);
        check(text_embedding_rows(embedding, vocab_size, hidden_size, tokens + offset, count,
                                  residual + offset * hidden_size, stream),
              "Gather prefill embeddings");
    }
}

void stage_prefill_row(Request& request, const TileBuffers& work, int row, cudaStream_t stream) {
    const int offset = row * hidden_size;
    const auto bytes = hidden_size * sizeof(Bf16);
    check(cudaMemcpyAsync(request.work.hidden, work.hidden + offset, bytes,
                          cudaMemcpyDeviceToDevice, stream), "Stage final prefill hidden row");
    check(cudaMemcpyAsync(request.work.residual, work.residual + offset, bytes,
                          cudaMemcpyDeviceToDevice, stream), "Stage final prefill residual row");
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

void publish_prefill_snapshot(const PrefillGrant& grant, cudaStream_t stream) {
    const auto* snapshot = grant.snapshot;
    if (!snapshot || grant.position + grant.rows != snapshot->position) return;
    snapshot->publishing = true;
    snapshot->cache.store(*grant.request, snapshot->prompt, snapshot->options.prefix_cache_salt,
                          snapshot->position, stream, grant.draft, snapshot->media.get());
    snapshot->publishing = false;
}

constexpr int layerwise_row_alignment = 128;
constexpr int max_layerwise_rows = 1023;
constexpr int min_partial_prefill_rows = 256;
static_assert(max_layerwise_rows / 256 <= max_gdn_prefill_segments);
static_assert(text_gqa_kv_width * sizeof(Bf16) >= text_gdn_value_heads * sizeof(float2));

struct PrefillGroup {
    int grants;
    int rows;
};

int aligned_layerwise_rows(int rows) {
    return (rows + layerwise_row_alignment - 1) / layerwise_row_alignment *
        layerwise_row_alignment;
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

void read_paged_sequences(const PagedPrefillSequence* sequences, int count, cudaStream_t stream,
                          PairedPrefillScratch scratch) {
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

Chunk prefill_attention_segment(const Chunk& chunk, const PrefillGrant& grant, int offset) {
    return Chunk{{{}, grant.rows, chunk.stream, true}, chunk.work.attention_segment(offset),
                 grant.position, grant.media ? grant.media->positions + grant.position : nullptr};
}

bool prefill_grouped_partial_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                                 const PrefillGrant* grants, int count,
                                 PairedPrefillScratch scratch, bool postgate) {
    constexpr int width = staged_prefill_batch_size;
    const int rows = grants[0].rows;
    if (count % width != 0 || (rows != 256 && rows != 512) ||
        !std::all_of(grants, grants + count, [rows](const PrefillGrant& grant) {
            return grant.rows == rows && grant.position >= 8192 && grant.position <= 32768 - rows;
        })) {
        return false;
    }
    for (int first = 0; first < count; first += width) {
        std::array<PagedPrefillSequence, width> sequences;
        for (int lane = 0; lane < width; ++lane) {
            const auto& grant = grants[first + lane];
            const auto segment = prefill_attention_segment(chunk, grant, (first + lane) * rows);
            const auto cache = grant.request->caches[bank];
            const auto pages = grant.request->pages;
            segment.gqa_append(layer, cache, pages);
            sequences[lane] = {cache, pages, grant.position, rows, segment.work.rope_queries,
                               segment.work.attention};
        }
        const auto submitted = try_batched_partial_paged_prefill(sequences.data(), width,
                                                                 chunk.stream);
        if (submitted) check(*submitted, "Read grouped ordered partial prefill attention");
        for (int lane = 0; lane < width; ++lane) {
            const auto segment = prefill_attention_segment(chunk, grants[first + lane],
                                                           (first + lane) * rows);
            const auto& sequence = sequences[lane];
            if (!submitted) {
                check(tensor_core_model_prefill_gqa(sequence.cache, sequence.pages,
                    sequence.position, sequence.rows, sequence.queries, sequence.output,
                    chunk.stream, scratch), "Read causal prefill MXFP8 attention");
            }
            if (postgate) segment.gqa_gate();
        }
    }
    return true;
}

void prefill_group_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                       const PrefillGrant* grants, int count, int offset,
                       bool staged_batch = false) {
    static_assert(staged_prefill_batch_size >= max_layerwise_rows / 256);
    std::array<PagedPrefillSequence, staged_prefill_batch_size> sequences;
    int index = 0;
    each_prefill_segment(grants, count, offset, [&](const PrefillGrant& grant, int row) {
        const auto segment = prefill_attention_segment(chunk, grant, row);
        const auto cache = grant.request->caches[bank];
        const auto pages = grant.request->pages;
        segment.gqa_append(layer, cache, pages);
        sequences[index++] = {cache, pages, grant.position, grant.rows, segment.work.rope_queries,
                              segment.work.attention};
    });
    if (staged_batch) {
        const auto scratch = chunk.work.gqa_scratch();
        const int position = sequences.front().position;
        if (position <= staged_prefill_batch_max_position) {
            if (const auto staged = try_batched_staged_paged_prefill(sequences.data(), count,
                                                                    chunk.stream, scratch)) {
                check(*staged, "Read staged batched causal prefill MXFP8 attention");
                return;
            }
            for (int lane = 0; lane < count; ++lane) {
                read_paged_sequences(sequences.data() + lane, 1, chunk.stream, scratch);
            }
            return;
        }
        const int visible = position + batch_prefill_tile_rows;
        const int subgroup = visible <= 32768 ? 4 : visible <= 65536 ? 2 : 1;
        for (int lane = 0; lane < count; lane += subgroup) {
            const auto* first = sequences.data() + lane;
            if (const auto staged = try_long_staged_paged_prefill(first, subgroup, chunk.stream,
                                                                  scratch)) {
                check(*staged, "Read long staged causal prefill MXFP8 attention");
                continue;
            }
            for (int index = 0; index < subgroup; ++index) {
                read_paged_sequences(first + index, 1, chunk.stream, scratch);
            }
        }
        return;
    }
    read_paged_sequences(sequences.data(), count, chunk.stream,
                         count == 1 ? chunk.work.gqa_scratch() : PairedPrefillScratch{});
}

bool can_batch_staged_gqa(const PrefillGrant* grants, int count) {
    if (count != staged_prefill_batch_size) {
        return false;
    }
    const int position = grants[0].position;
    return position >= staged_prefill_batch_min_position &&
        position <= staged_prefill_batch_long_max_position &&
        position % batch_prefill_tile_rows == 0 &&
        std::all_of(grants, grants + count, [&](const PrefillGrant& grant) {
            return grant.position == position && grant.rows == batch_prefill_tile_rows;
        });
}

void prefill_group_gdn(const Chunk& chunk, const GdnLayer& layer, int bank,
                       const PrefillGrant* grants, int count, int offset) {
    const auto& buffers = chunk.work;
    GdnPrefillSegment segments[max_gdn_prefill_segments]{};
    bool grouped_recurrence = count > 1;
    int index = 0;
    each_prefill_segment(grants, count, offset, [&](const PrefillGrant& grant, int row) {
        const auto segment = buffers.attention_segment(row);
        const TextFp8Activation fp8{
            buffers.activation.values + std::size_t(row) * text_gdn_value_width,
            buffers.activation.scales + std::size_t(row) * text_gdn_value_heads,
        };
        grouped_recurrence = grouped_recurrence &&
            grant.rows % layerwise_row_alignment == 0 && (grant.position != 0 || bank != 0);
        segments[index++] = {
            grant.rows, segment.gdn_qkv, segment.gdn_z, segment.gdn_b, segment.gdn_a,
            grant.request->states[bank],
            buffers.gdn_scratch + text_gdn_workspace_bytes(row) / sizeof(Bf16),
            reinterpret_cast<float2*>(segment.keys), fp8,
            buffers.mlp_gate + row * intermediate_size,
            std::size_t(grant.rows) * intermediate_size * sizeof(Bf16),
        };
    });
    const auto status = grouped_recurrence
        ? tensor_core_prefill_gdn_group(segments, count, layer.recurrent, chunk.stream)
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

PairedPrefillScratch serial_gqa_scratch(const TileBuffers& buffers, int serial_capacity) {
    ScratchLayout serial_layout{reinterpret_cast<std::byte*>(buffers.hidden)};
    const auto sizes = TileBuffers(serial_layout, serial_capacity).gqa_scratch();
    auto scratch = buffers.gqa_scratch();
    scratch.key_bytes = sizes.key_bytes;
    scratch.value_bytes = sizes.value_bytes;
    return scratch;
}

enum class PrefillAttentionMode { batched, full_group, serial_segments };

bool is_full_partial_prefill_group(const PrefillGrant* grants, PrefillGroup group) {
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

void prefill_partial_gdn(const Chunk& chunk, const GdnLayer& layer, int bank,
                         const PrefillGrant* grants, int count, bool grouped) {
    if (grouped) {
        int offset = 0;
        for (int index = 0; index < count; index += max_gdn_prefill_segments) {
            prefill_group_gdn(chunk, layer, bank, grants + index, max_gdn_prefill_segments, offset);
            offset += max_gdn_prefill_segments * grants[index].rows;
        }
        return;
    }
    const auto& activation = chunk.work.activation;
    each_prefill_segment(grants, count, 0, [&](const PrefillGrant& grant, int row) {
        const auto segment = prefill_attention_segment(chunk, grant, row);
        const TextFp8Activation quantized{
            activation.values + std::size_t(row) * text_gdn_value_width,
            activation.scales + std::size_t(row) * text_gdn_value_heads,
        };
        segment.gdn_recur(layer, grant.request->states[bank], bank, quantized);
    });
}

void prefill_layerwise(const DecoderModel& model, const PrefillGrant* grants, PrefillGroup group,
                       PrefillWorkspace& workspace, PrefillAttentionMode attention_mode,
                       int serial_capacity = 0) {
    const auto stream = workspace.stream();
    ScratchLayout layout{static_cast<std::byte*>(workspace.data())};
    const TileBuffers buffers(layout, workspace.capacity());
    const bool grouped = group.rows > max_layerwise_rows;
    const bool serial_attention = attention_mode == PrefillAttentionMode::serial_segments;
    const bool wide_partial = serial_attention && is_full_partial_prefill_group(grants, group);
    const bool full_group = attention_mode == PrefillAttentionMode::full_group;
    const bool packed_preparation = full_group || wide_partial;
    const auto rms_launch = packed_preparation ? RmsQuantizeLaunch::full_group_prefill
                                               : RmsQuantizeLaunch::ordinary;
    const Chunk chunk{{buffers.activation, group.rows, stream, true,
                       grouped || serial_attention}, buffers, 0, nullptr};
    each_prefill_segment(grants, group.grants, 0, [&](const PrefillGrant& grant, int offset) {
        auto* residual = buffers.residual + offset * hidden_size;
        gather_embeddings(model.embedding, grant.tokens, grant.rows, residual, stream);
        if (grant.media) grant.media->substitute(residual, grant.position, grant.rows, stream);
    });
    execute_layers(
        model.layers,
        [&](const Bf16* weight, const char*, const auto& residuals) {
            const auto [branch, residual] = residuals(buffers);
            chunk.normalize(branch, residual, weight, rms_launch);
        },
        [&](const GqaLayer& layer, int bank) {
            chunk.gqa_input(layer);
            if (serial_attention) {
                const auto scratch = serial_gqa_scratch(buffers, serial_capacity);
                if (!prefill_grouped_partial_gqa(chunk, layer, bank, grants, group.grants,
                                                 scratch, !wide_partial)) {
                    each_prefill_segment(grants, group.grants, 0,
                                         [&](const PrefillGrant& grant, int row) {
                        const auto segment = prefill_attention_segment(chunk, grant, row);
                        segment.gqa_attention(layer, grant.request->caches[bank],
                                              grant.request->pages, scratch, !wide_partial);
                    });
                }
            } else if (grouped && can_batch_staged_gqa(grants, group.grants)) {
                prefill_group_gqa(chunk, layer, bank, grants, group.grants, 0, true);
            } else {
                each_attention_group(grants, group.grants,
                                     [&](const PrefillGrant* first, int count, int offset) {
                    prefill_group_gqa(chunk, layer, bank, first, count, offset);
                });
            }
            if (packed_preparation) {
                if (const auto status = try_prefill_gqa_postgate_quantize(
                        group.rows, buffers.q_gate, buffers.attention, buffers.activation,
                        stream)) {
                    check(*status, "Gate and quantize prefill grouped-query attention");
                    const TextFp8ActivationView input{
                        buffers.activation.values, buffers.activation.scales};
                    chunk.project(layer.output, input, buffers.projected_attention);
                    return;
                }
            }
            if (wide_partial) {
                each_prefill_segment(grants, group.grants, 0,
                                     [&](const PrefillGrant& grant, int row) {
                    prefill_attention_segment(chunk, grant, row).gqa_gate();
                });
            }
            if (!serial_attention) chunk.gqa_gate();
            chunk.attention_output(layer.output, text_gqa_q_width);
        },
        [&](const GdnLayer& layer, int bank) {
            chunk.gdn_input(layer);
            if (serial_attention) {
                prefill_partial_gdn(chunk, layer, bank, grants, group.grants, wide_partial);
            } else if (grouped) {
                prefill_group_gdn(chunk, layer, bank, grants, group.grants, 0);
            } else {
                each_attention_group(grants, group.grants,
                                     [&](const PrefillGrant* first, int count, int offset) {
                    prefill_group_gdn(chunk, layer, bank, first, count, offset);
                });
            }
            chunk.project(layer.output, {buffers.activation.values, buffers.activation.scales},
                          buffers.projected_attention);
        },
        [&](const Layer& layer) { chunk.mlp(layer); });
    each_prefill_segment(grants, group.grants, 0, [&](const PrefillGrant& grant, int offset) {
        if (grant.draft && grant.prompt_remaining > 1) {
            auto seed = chunk;
            seed.position = grant.position;
            seed.work.hidden += offset * hidden_size;
            seed.work.residual += offset * hidden_size;
            const bool wmma = model.backend.draft.wmma_seed_fc &&
                grant.position + grant.prompt_remaining >= wmma_seed_min_prompt_tokens;
            seed.seed_draft(*grant.draft, grant.tokens + 1,
                            std::min(grant.rows, grant.prompt_remaining - 1),
                            model.final_norm, wmma, grant.media);
        }
        capture_prefill_scores(model, grant, &buffers, offset, stream);
        stage_prefill_row(*grant.request, buffers, offset + grant.rows - 1, stream);
        publish_prefill_snapshot(grant, stream);
    });
}

bool prefill_projection_tiles(const DecoderModel& model, const PrefillGrant* grants, int count,
                              int lanes, cudaStream_t stream,
                              DecoderBatchStepStats& stats) {
    const int rows = lanes * batch_prefill_tile_rows;
    const auto attention_mode = lanes == grouped_prefill_max_rows / batch_prefill_tile_rows
        ? PrefillAttentionMode::full_group : PrefillAttentionMode::batched;
    PrefillWorkspace workspace(stream, rows);
    if (!workspace.data()) {
        return false;
    }
    stats.prefill_workspace_bytes = std::max(stats.prefill_workspace_bytes, workspace.bytes());
    for (int offset = 0; offset < grants[0].rows; offset += batch_prefill_tile_rows) {
        for (int index = 0; index < count; index += lanes) {
            std::array<PrefillGrant, grouped_prefill_max_rows / batch_prefill_tile_rows> group;
            for (int lane = 0; lane < lanes; ++lane) {
                group[lane] = prefill_tile(grants[index + lane], offset);
            }
            prefill_layerwise(model, group.data(), {lanes, rows}, workspace, attention_mode);
            ++stats.layerwise_prefill_groups;
            stats.layerwise_prefill_rows += rows;
        }
    }
    workspace.close();
    return true;
}

PrefillGroup partial_projection_group(const PrefillGrant* grants, int count) {
    PrefillGroup group{};
    while (group.grants < count) {
        const auto& grant = grants[group.grants];
        if (grant.rows < min_partial_prefill_rows || grant.rows > batch_prefill_tile_rows ||
            grant.rows != grant.tile_rows || grant.rows > grouped_prefill_max_rows - group.rows) {
            break;
        }
        ++group.grants;
        group.rows += grant.rows;
    }
    return group;
}

void enqueue_native_prefill_grants(const DecoderModel& model, const PrefillGrant* grants,
                                   int count, PrefillWorkspace& workspace, int capacity);

bool prefill_partial_projection_tiles(const DecoderModel& model, const PrefillGrant* grants,
                                      int count, cudaStream_t stream,
                                      DecoderBatchStepStats& stats) {
    if (!model.backend.tensor_core_prefill || count < 2) return false;
    int serial_capacity = 1;
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        serial_capacity = std::max(serial_capacity, grant.tile_rows);
    }
    int capacity = serial_capacity;
    bool grouped = false;
    for (int index = 0; index < count;) {
        const auto group = partial_projection_group(grants + index, count - index);
        if (group.grants >= 2) {
            grouped = true;
            capacity = std::max(capacity, group.rows);
        }
        index += std::max(1, group.grants);
    }
    if (!grouped) return false;

    PrefillWorkspace workspace(stream, capacity);
    if (!workspace.data()) return false;
    stats.prefill_workspace_bytes = std::max(stats.prefill_workspace_bytes, workspace.bytes());
    for (int index = 0; index < count;) {
        const auto group = partial_projection_group(grants + index, count - index);
        if (group.grants >= 2) {
            prefill_layerwise(model, grants + index, group, workspace,
                              PrefillAttentionMode::serial_segments, serial_capacity);
            ++stats.layerwise_prefill_groups;
            stats.layerwise_prefill_rows += group.rows;
            index += group.grants;
            continue;
        }
        enqueue_native_prefill_grants(model, grants + index, 1, workspace, serial_capacity);
        ++index;
    }
    workspace.close();
    return true;
}

}

std::size_t decoder::prefill_workspace_bytes(int rows) {
    ScratchLayout layout;
    const TileBuffers buffers(layout, rows);
    return layout.bytes;
}

bool decoder::try_prefill_projection_groups(
    const DecoderModel& model, const PrefillGrant* grants, int count, cudaStream_t stream,
    DecoderBatchStepStats& stats) {
    if (model.backend.paired_prefill && pairable_prefill_grants(grants, count)) {
        if (count == 4 && prefill_projection_tiles(model, grants, count, 4, stream, stats)) {
            return true;
        }
        return prefill_projection_tiles(model, grants, count, 2, stream, stats);
    }
    return prefill_partial_projection_tiles(model, grants, count, stream, stats);
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

void Decoder::Model::seed_draft(Workspace& work, MtpDraftRequest& draft, int position,
                                int shifted_token, cudaStream_t stream,
                                const MediaPrefill* media) const {
    normalize_final(work, stream);
    draft.enqueue(position, shifted_token, work.normalized, false, stream, nullptr, nullptr,
                  media ? media->positions + position : nullptr, -1,
                  media ? media->feature(position + 1) : nullptr);
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
    check(cudaStreamSynchronize(stream), "Collect prompt token scores");
    for (int index = 0; index < rows; ++index) {
        const auto& row = readback[index];
        output->push_back(
            row.selected.token_id < 0 ? nullptr : collect_generated_logprobs(row, top));
    }
}

void decoder::finish_prompt_logprobs(const DecoderOptions& options, DecoderPromptLogprobs& scores,
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

void Decoder::Model::prefill_chunk(Request& request, const int* tokens, int rows, int position,
                                   PrefillWorkspace& workspace, MtpDraftRequest* draft,
                                   int prompt_remaining, const MediaPrefill* media,
                                   const PromptLogprobCapture* scores,
                                   const PrefillSnapshotCapture* snapshot,
                                   int logical_capacity) const {
    const auto stream = workspace.stream();
    const int capacity = logical_capacity ? logical_capacity : workspace.capacity();
    if (!stream || !tokens || rows < 1 || rows > capacity || capacity > workspace.capacity() ||
        position < 0 ||
        position > text_attention_max_tokens - rows ||
        (position + rows - 1) / mxfp8_page_size >= request.pages.page_count ||
        (draft && prompt_remaining < rows)) {
        throw std::invalid_argument("Invalid decoder prefill tile");
    }
    const PrefillGrant grant{&request, tokens, position, rows, rows, prompt_remaining,
                             draft, media, scores, snapshot};
    if (rows == 1 || !workspace.data()) {
        if (!std::all_of(tokens, tokens + rows,
                         [](int token) { return token >= 0 && token < vocab_size; })) {
            throw std::invalid_argument("Invalid decoder prefill token");
        }
        int pending = 0;
        const int first = scores ? scores->start : position;
        const int end = scores ? scores->score_end(position, rows) : position;
        for (int row = 0; row < rows; ++row) {
            const int token_position = position + row;
            ForwardLane lane{
                &request, tokens[row], token_position, TextGdnMode::prefill, false,
                media ? media->feature(token_position) : nullptr,
                media ? media->positions + token_position : nullptr};
            forward(&lane, 1, stream);
            if (token_position >= first && token_position < end) {
                scores->enqueue(*this, request.work, token_position, pending++, stream);
                if (pending == decoder_batch_max_lanes) {
                    scores->collect(std::exchange(pending, 0), stream);
                }
            }
            if (draft && row + 1 < prompt_remaining) {
                seed_draft(request.work, *draft, token_position, tokens[row + 1], stream, media);
            }
        }
        if (scores) scores->collect(pending, stream);
        publish_prefill_snapshot(grant, stream);
        return;
    }
    ScratchLayout layout{static_cast<std::byte*>(workspace.data())};
    const TileBuffers buffers(layout, capacity);
    const bool tensor_core = backend.tensor_core_prefill;
    const Chunk chunk{{buffers.activation, rows, stream, tensor_core}, buffers, position,
                      media ? media->positions + position : nullptr};
    const auto& work = chunk.work;
    gather_embeddings(embedding, tokens, rows, work.residual, stream);
    if (media) {
        media->substitute(work.residual, position, rows, stream);
    }
    const auto normalize = [&](const Bf16* weight, const char*, const auto& residuals) {
        const auto [branch, residual] = residuals(work);
        chunk.normalize(branch, residual, weight);
    };
    execute_layers(layers, normalize,
                   [&](const GqaLayer& layer, int bank) {
                       chunk.gqa_input(layer);
                       chunk.gqa_attention(layer, request.caches[bank], request.pages,
                                           work.gqa_scratch());
                       chunk.attention_output(layer.output, text_gqa_q_width);
                   },
                   [&](const GdnLayer& layer, int bank) {
                       chunk.gdn_input(layer);
                       chunk.gdn_attention(layer, request.states[bank], bank);
                   },
                   [&](const Layer& layer) { chunk.mlp(layer); });
    if (draft && prompt_remaining > 1) {
        const bool wmma = tensor_core && backend.draft.wmma_seed_fc &&
                          position + prompt_remaining >= wmma_seed_min_prompt_tokens;
        chunk.seed_draft(*draft, tokens + 1, std::min(rows, prompt_remaining - 1),
                         final_norm, wmma, media);
    }
    capture_prefill_scores(*this, grant, &work, 0, stream);
    stage_prefill_row(request, work, rows - 1, stream);
    publish_prefill_snapshot(grant, stream);
}

namespace {

void enqueue_native_prefill_grants(const DecoderModel& model, const PrefillGrant* grants,
                                   int count, PrefillWorkspace& workspace, int capacity) {
    const auto stream = workspace.stream();
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        if (grant.rows == 1) {
            if (grant.draft && grant.prompt_remaining > 1) {
                model.seed_draft(grant.request->work, *grant.draft, grant.position, grant.tokens[1],
                                 stream, grant.media);
            }
            capture_prefill_scores(model, grant, nullptr, 0, stream);
            publish_prefill_snapshot(grant, stream);
        } else {
            for (int offset = 0; offset < grant.rows;) {
                const int rows = std::min({grant.tile_rows, capacity, grant.rows - offset});
                model.prefill_chunk(*grant.request, grant.tokens + offset, rows,
                                    grant.position + offset, workspace, grant.draft,
                                    grant.prompt_remaining - offset, grant.media, grant.scores,
                                    grant.snapshot, capacity);
                offset += rows;
            }
        }
    }
}

}

void Decoder::Model::prefill_grants(const PrefillGrant* grants, int count, cudaStream_t stream,
                                   DecoderBatchStepStats& stats) const {
    if (count == 0 || try_prefill_projection_groups(*this, grants, count, stream, stats)) return;
    int capacity = 1;
    for (int index = 0; index < count; ++index) {
        capacity = std::max(capacity, grants[index].tile_rows);
    }
    std::optional<PrefillWorkspace> workspace(std::in_place, stream, capacity);
    if (!workspace->data() && capacity > prefill_chunk_rows) {
        workspace.emplace(stream, prefill_chunk_rows);
    }
    stats.prefill_workspace_bytes = std::max(stats.prefill_workspace_bytes, workspace->bytes());
    enqueue_native_prefill_grants(*this, grants, count, *workspace, workspace->capacity());
    workspace->close();
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
