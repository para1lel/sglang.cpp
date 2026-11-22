#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_verify.h"
#include "sglang/text/target_fp8.h"
#include "sglang/text/mtp/tc_vocab.h"

#include <algorithm>
#include <initializer_list>
#include <type_traits>
#include <utility>

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
    static constexpr bool narrow_gqa = true;
};

DecodeFp8Projection decode_projection(Fp8Weight weight, Bf16* output) {
    return {weight.values, weight.scales, output, weight.rows};
}

struct LayerProjection {
    Fp8Weight weight;
    decltype(&Workspace::hidden) output;
};

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
    bool ordered_attention = false;
    bool target_tc = false;
    bool vector_attention = false;
    int target_split_end = 0;

    bool tensor_core_lane(int index) const {
        if constexpr (std::is_same_v<Lane, ForwardLane>) {
            return target_tc && lanes[index].mode == TextGdnMode::decode &&
                !lanes[index].mrope && lanes[index].rotary_position < 0;
        }
        return target_tc;
    }

    bool single_decode() const {
        if constexpr (std::is_same_v<Lane, BatchGraphLane>) {
            return count == 1;
        } else {
            return count == 1 && lanes[0].mode == TextGdnMode::decode;
        }
    }

    bool fuse_verify_activation() const {
        return std::is_same_v<Lane, VerificationRow> && count == fused_verify_rows;
    }

    bool fuse_wide_verify_norm() const {
        static_assert(WideVerifyRmsQuantizeRows::capacity ==
                      target_verify_max_rows * decoder_batch_max_mtp_lanes);
        return std::is_same_v<Lane, VerificationRow> && count > fused_verify_rows;
    }

    TextFp8Activation verification_activation(int index, int width) const {
        auto activation = workspace(lanes[index]).activation;
        if (target_tc) {
            static_assert(fused_verify_rows == text_fp8_max_batch);
            static_assert(fused_verify_rows * intermediate_size <= vocab_size * sizeof(Bf16));
            auto* packed = reinterpret_cast<std::uint8_t*>(workspace(lanes[0]).logits);
            activation.values = packed + index * width;
        }
        return activation;
    }

    int projection_rows(int group) const {
        const int groups = (count + text_fp8_max_batch - 1) / text_fp8_max_batch;
        return count / groups + (group < count % groups);
    }

    template <typename Function>
    void each(Function function) const {
        for (int index = 0; index < count; ++index) {
            function(lanes[index], workspace(lanes[index]));
        }
    }

    void quantize(Bf16* Workspace::*source, int width) const {
        if (fuse_verify_activation() && width % text_fp8_group_size == 0) {
            quantize_verification(source, width);
            return;
        }
        if (fuse_wide_verify_norm() && width % text_fp8_group_size == 0) {
            quantize_wide_verification(source, width);
            return;
        }
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

    void quantize_verification(Bf16* Workspace::*source, int width,
                               Bf16* Workspace::*up = nullptr) const {
        VerifyQuantizeRows rows;
        for (int index = 0; index < fused_verify_rows; ++index) {
            auto& work = workspace(lanes[index]);
            rows.entries[index] = {
                work.*source, up ? work.*up : nullptr, verification_activation(index, width)};
        }
        check(verify_quantize_fp8(rows, width, stream), "Quantize five verification rows");
    }

    void quantize_wide_verification(Bf16* Workspace::*source, int width,
                                    Bf16* Workspace::*up = nullptr) const {
        static_assert(WideVerifyQuantizeRows::capacity ==
                      target_verify_max_rows * decoder_batch_max_mtp_lanes);
        WideVerifyQuantizeRows rows{};
        for (int index = 0; index < count; ++index) {
            auto& work = workspace(lanes[index]);
            rows.entries[index] = {work.*source, up ? work.*up : nullptr, work.activation};
        }
        check(verify_quantize_fp8_wide(rows, count, width, stream),
              "Quantize wide verification rows");
    }

    void project(Fp8Weight weight, Bf16* Workspace::*output) const {
        if (target_tc && (count > 1 || tensor_core_lane(0)) && target_shape(weight)) {
            project_target({{weight, output}});
            return;
        }
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
        std::array<TextFp8Projection, text_fp8_max_batch> projections;
        for (int first = 0, group = 0; first < count; ++group) {
            const int rows = projection_rows(group);
            for (int index = 0; index < rows; ++index) {
                const auto& work = workspace(lanes[first + index]);
                const auto activation = fuse_verify_activation()
                    ? verification_activation(first + index, weight.columns) : work.activation;
                projections[index] = {{activation.values, activation.scales}, work.*output};
            }
            check(text_fp8_gemm(weight.values, weight.scales, weight.rows, weight.columns,
                                projections.data(), rows, stream), "Run batched FP8 projection");
            first += rows;
        }
    }

    static bool target_shape(Fp8Weight weight) {
        return weight.rows > 0 && weight.rows <= intermediate_size &&
               weight.columns > 0 && weight.columns <= intermediate_size &&
               weight.rows % text_fp8_group_size == 0 &&
               weight.columns % text_fp8_group_size == 0;
    }

    void project_group(std::initializer_list<LayerProjection> layers) const {
        const int columns = layers.begin()->weight.columns;
        if (target_tc && (count > 1 || tensor_core_lane(0)) &&
            std::all_of(layers.begin(), layers.end(), [columns](const auto& layer) {
                return target_shape(layer.weight) && layer.weight.columns == columns;
            })) {
            project_target(layers);
            return;
        }
        for (const auto& layer : layers) {
            project(layer.weight, layer.output);
        }
    }

    bool project_target_chunk(std::initializer_list<LayerProjection> layers, int first,
                              int rows, bool allow_fallback) const {
        std::array<TextFp8Projection, text_decode_max_targets * text_target_fp8_max_rows> operands;
        std::array<TextFp8DecodeTarget, text_decode_max_targets> targets;
        const int columns = layers.begin()->weight.columns;
        int target = 0;
        for (const auto& layer : layers) {
            auto* entries = operands.data() + target * rows;
            for (int index = 0; index < rows; ++index) {
                const auto& work = workspace(lanes[first + index]);
                const auto activation = fuse_verify_activation()
                    ? verification_activation(first + index, columns) : work.activation;
                entries[index] = {{activation.values, activation.scales}, work.*layer.output};
            }
            const auto weight = layer.weight;
            targets[target++] = {weight.values, weight.scales, weight.rows, entries[0].output};
        }
        auto* scratch = workspace(lanes[first]).logits;
        static_assert(text_target_fp8_max_rows * intermediate_size <= vocab_size * sizeof(Bf16));
        const auto status = tensor_core_lane(first)
            ? try_target_fp8_tc(targets.data(), target, columns, operands.data(),
                                rows, scratch, stream)
            : std::nullopt;
        if (status) {
            check(*status, "Run target FP8 projection");
            return true;
        }
        if (!allow_fallback) {
            return false;
        }
        for (int index = 0; index < target; ++index) {
            const auto& split = targets[index];
            check(text_fp8_gemm(split.weight, split.weight_scales, split.out_width, columns,
                                operands.data() + index * rows, rows, stream),
                  "Run fallback target FP8 projection");
        }
        return true;
    }

    void project_target(std::initializer_list<LayerProjection> layers) const {
        constexpr bool verification = std::is_same_v<Lane, VerificationRow>;
        int first = 0;
        if constexpr (verification) {
            const int groups = (count - 1) / text_target_fp8_max_rows + 1;
            for (int group = 0; group < groups && count > text_fp8_max_batch; ++group) {
                const int rows = count / groups + (group < count % groups);
                if (!project_target_chunk(layers, first, rows, false)) {
                    break;
                }
                first += rows;
            }
        }
        const int remaining = count - first;
        const int groups = (remaining + text_fp8_max_batch - 1) / text_fp8_max_batch;
        for (int group = 0; first < count; ++group) {
            const int rows = verification ? remaining / groups + (group < remaining % groups) : 1;
            project_target_chunk(layers, first, rows, true);
            first += rows;
        }
    }
};

void project(Bf16Weight weight, const LayerBatch<VerificationRow>& batch,
             Bf16* Workspace::*output) {
    std::array<TextBf16Projection, text_fp8_max_batch> projections;
    for (int first = 0, group = 0; first < batch.count; ++group) {
        const int rows = batch.projection_rows(group);
        for (int index = 0; index < rows; ++index) {
            const auto& work = workspace(batch.lanes[first + index]);
            projections[index] = {work.normalized, work.*output};
        }
        check(text_bf16_gemm(weight.values, weight.rows, weight.columns, projections.data(),
                             rows, batch.stream), "Run batched BF16 projection");
        first += rows;
    }
}

constexpr int wide_gdn_rows = GdnVerificationBatch::lanes * GdnVerificationBatch::rows;
static_assert(GdnVerificationBatch::lanes == decoder_batch_max_mtp_lanes);
static_assert(GdnVerificationBatch::rows == target_verify_max_rows);

void project_gdn_gates(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch,
                       bool batched) {
    if (!batch.target_tc || batch.count < 2 ||
        layer.b.rows != text_gdn_value_heads || layer.a.rows != text_gdn_value_heads ||
        layer.b.columns != hidden_size || layer.a.columns != hidden_size) {
        project(layer.b, batch, &Workspace::gdn_b);
        project(layer.a, batch, &Workspace::gdn_a);
        return;
    }
    std::array<VerifyGdnGateRow, wide_gdn_rows> gates;
    const auto fill = [&](int first, int rows) {
        for (int index = 0; index < rows; ++index) {
            const auto& work = workspace(batch.lanes[first + index]);
            gates[index] = {work.normalized, work.gdn_b, work.gdn_a};
        }
        return try_verify_bf16_gdn_gates(layer.b.values, layer.a.values, gates.data(),
                                         rows, batch.stream);
    };
    if (batched) {
        if (const auto status = fill(0, batch.count)) {
            check(*status, "Run wide verification GDN gate projections");
            return;
        }
    }
    std::array<TextBf16Projection, text_fp8_max_batch> gate_b, gate_a;
    for (int first = 0, group = 0; first < batch.count; ++group) {
        const int rows = batch.projection_rows(group);
        if (const auto status = fill(first, rows)) {
            check(*status, "Run grouped BF16 verification GDN gate projections");
        } else {
            for (int index = 0; index < rows; ++index) {
                const auto& gate = gates[index];
                gate_b[index] = {gate.input, gate.output_b};
                gate_a[index] = {gate.input, gate.output_a};
            }
            check(text_bf16_gemm(layer.b.values, layer.b.rows, layer.b.columns, gate_b.data(),
                                rows, batch.stream), "Run batched BF16 GDN b projection");
            check(text_bf16_gemm(layer.a.values, layer.a.rows, layer.a.columns, gate_a.data(),
                                rows, batch.stream), "Run batched BF16 GDN a projection");
        }
        first += rows;
    }
}

template <typename Lane>
void run_gdn(const GdnLayer& layer, const LayerBatch<Lane>& batch, int state_index) {
    batch.quantize(&Workspace::normalized, hidden_size);
    batch.project_group({{layer.qkv, &Workspace::gdn_qkv}, {layer.z, &Workspace::gdn_z}});
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
    const auto& first = batch.lanes[0];
    if (first.binding) {
        auto& scratch = *first.graph_workspace;
        scratch.bind_gdn(state_index, batch.count);
        scratch.replay[state_index].weights = layer.recurrent;
    }
    batch.each([&](const VerificationRow& row, Workspace&) {
        if (row.first && !row.binding) {
            row.verification->bind_gdn(state_index);
        }
    });
    if (!batch.fuse_verify_activation() && !batch.fuse_wide_verify_norm()) {
        batch.quantize(&Workspace::normalized, hidden_size);
    }
    batch.project_group({{layer.qkv, &Workspace::gdn_qkv}, {layer.z, &Workspace::gdn_z}});
    const bool batched = batch.target_tc && !first.binding && batch.count == wide_gdn_rows;
    project_gdn_gates(layer, batch, batched);
    if (first.binding) {
        auto& scratch = *first.graph_workspace;
        check(decode_verify_gdn(batch.count, scratch.replay[state_index].inputs,
                                first.work->gdn_z, layer.recurrent,
                                &first.binding->initial[state_index],
                                &first.binding->tail[state_index], scratch.observable,
                                first.work->attention, batch.stream),
              "Advance bound verification GDN");
    } else if (batched) {
        GdnVerificationBatch states;
        for (int lane = 0; lane < GdnVerificationBatch::lanes; ++lane) {
            states.entries[lane] = batch.lanes[lane].verification->gdn_lane(state_index,
                                                                            layer.recurrent);
        }
        check(decode_verify_gdn_batch(states, layer.recurrent, batch.stream),
              "Advance batched target verification GDN");
    } else {
        batch.each([&](const VerificationRow& row, Workspace&) {
            if (row.first) {
                row.verification->advance_gdn(state_index, layer.recurrent);
            }
        });
    }
    batch.quantize(&Workspace::attention, text_gdn_value_width);
    batch.project(layer.output, &Workspace::projected_attention);
}

template <typename Position>
cudaError_t read_gqa(Workspace& work, Mxfp8PageTable pages, Mxfp8KvCache cache,
                     Position position, TextGdnMode mode, bool narrow_gqa,
                     ExactGqaWorkspace singleton_scores, bool vector, cudaStream_t stream) {
    const int capacity = pages.page_count * mxfp8_page_size;
    const auto scores = work.gqa_scores(capacity);
    if constexpr (std::is_same_v<Position, int>) {
        if (singleton_scores.scores) {
            return exact_decode_gqa_attention(cache, pages, position, work.rope_queries,
                work.attention, singleton_scores, stream, 0, mode == TextGdnMode::prefill);
        }
        if (mode == TextGdnMode::prefill || position >= scores.capacity) {
            return mxfp8_gqa_attention(cache, pages, position + 1, position, 1,
                                       work.rope_queries, work.attention, stream);
        }
        if (vector) {
            return vector_decode_gqa_attention(cache, pages, position, work.rope_queries,
                                               work.attention, scores, stream);
        }
    } else {
        if (capacity > scores.capacity) {
            return decode_gqa_attention(cache, pages, position, work.rope_queries,
                                        work.attention, stream);
        }
    }
    return exact_decode_gqa_attention(cache, pages, position, work.rope_queries, work.attention,
                                      scores, stream, narrow_gqa ? vocab_size * sizeof(Bf16) : 0);
}

template <typename Position>
void append_gqa(const GqaLayer& layer, Workspace& work, Mxfp8PageTable pages, Mxfp8KvCache cache,
                Position position, cudaStream_t stream, const MropePosition* mrope,
                int rotary_position) {
    if constexpr (std::is_same_v<Position, int>) {
        const auto status = mrope
            ? mrope_gqa_preprocess(mrope, 1, work.q_gate, work.keys, layer.query_norm,
                                    layer.key_norm, work.rope_queries, work.rope_keys, stream)
            : text_gqa_preprocess(rotary_position < 0 ? position : rotary_position, 1, work.q_gate,
                                   work.keys, layer.query_norm, layer.key_norm, work.rope_queries,
                                   work.rope_keys, stream);
        check(status, "Normalize and rotate Q/K");
        check(mxfp8_kv_append(cache, pages, position, 1, work.rope_keys, work.values,
                              stream), "Append MXFP8 KV");
    } else {
        check(decode_gqa_preprocess(position, work.q_gate, work.keys, layer.query_norm,
                                    layer.key_norm, work.rope_queries, work.rope_keys, stream),
              "Normalize and rotate decode Q/K");
        check(decode_kv_append(cache, pages, position, work.rope_keys, work.values, stream),
              "Append decode MXFP8 KV");
    }
}

void gate_gqa(Workspace& work, cudaStream_t stream) {
    check(text_gqa_postgate(1, work.q_gate, work.attention, work.attention, stream),
          "Gate grouped-query attention");
}

PagedDecodeBuffers paged_gqa_buffers(Workspace& work) {
    static_assert(vocab_size * sizeof(Bf16) >= paged_decode_scratch_bytes);
    return {work.rope_queries, work.attention, work.logits};
}

void attend_bound_gqa(const GqaLayer& layer, const LayerBatch<BatchGraphLane>& batch,
                      int cache_index) {
    static_assert(paged_decode_lanes == decoder_batch_max_graph_lanes);
    std::array<BoundPagedDecodeLane, paged_decode_lanes> reads;
    for (int index = 0; index < batch.count; ++index) {
        const auto& lane = batch.lanes[index];
        const auto* binding = lane.binding;
        const auto* position = &binding->position;
        auto& work = workspace(lane);
        check(decode_gqa_preprocess(position, work.q_gate, work.keys, layer.query_norm,
                                   layer.key_norm, work.rope_queries, work.rope_keys,
                                   batch.stream), "Normalize and rotate bound decode Q/K");
        check(decode_kv_append(&binding->caches[cache_index], &binding->pages, position,
                              work.rope_keys, work.values, batch.stream),
              "Append bound decode MXFP8 KV");
        reads[index] = {&binding->caches[cache_index], &binding->pages, position,
                        &binding->scores, paged_gqa_buffers(work)};
    }
    if (batch.single_decode()) {
        const auto& read = reads[0];
        auto status = cudaSuccess;
        if (batch.target_split_end) {
            status = target_split_gqa(read, batch.stream, batch.target_split_end);
        } else if (batch.vector_attention) {
            status = vector_decode_gqa_attention(read.cache, read.pages, read.position,
                read.buffers.queries, read.buffers.output, read.scores, batch.stream);
        } else {
            status = exact_decode_gqa_attention(read.cache, read.pages, read.position,
                read.buffers.queries, read.buffers.output, read.scores, batch.stream,
                batch.ordered_attention ? 0 : vocab_size * sizeof(Bf16));
        }
        check(status, "Read bound singleton MXFP8 attention");
    } else {
        check(paged_decode_gqa(reads.data(), batch.count, batch.stream),
              "Read bound causal MXFP8 attention");
    }
    batch.each([&](const BatchGraphLane&, Workspace& work) { gate_gqa(work, batch.stream); });
}

void attend_verify_gqa(const GqaLayer& layer, const LayerBatch<VerificationRow>& batch,
                       int cache_index) {
    const auto& binding = *batch.lanes[0].binding;
    const auto* cache = &binding.caches[cache_index];
    const bool grouped = batch.target_split_end == text_attention_max_tokens && batch.count > 1;
    std::array<BoundPagedDecodeLane, target_verify_max_rows> reads;
    for (int index = 0; index < batch.count; ++index) {
        auto& work = workspace(batch.lanes[index]);
        const auto* position = &binding.positions[index];
        check(decode_gqa_preprocess(position, work.q_gate, work.keys, layer.query_norm,
                                    layer.key_norm, work.rope_queries, work.rope_keys,
                                    batch.stream), "Normalize and rotate bound verification Q/K");
        check(decode_kv_append(cache, &binding.pages, position, work.rope_keys, work.values,
                               batch.stream), "Append bound verification MXFP8 KV");
        auto status = cudaSuccess;
        if (grouped) {
            reads[index] = {cache, &binding.pages, position, &binding.scores[index],
                            paged_gqa_buffers(work)};
            continue;
        }
        if (batch.target_split_end) {
            status = target_split_gqa({cache, &binding.pages, position, &binding.scores[index],
                                       paged_gqa_buffers(work)}, batch.stream,
                                      batch.target_split_end);
        } else if (batch.vector_attention) {
            status = vector_decode_gqa_attention(cache, &binding.pages, position, work.rope_queries,
                work.attention, &binding.scores[index], batch.stream);
        } else {
            status = exact_decode_gqa_attention(cache, &binding.pages, position, work.rope_queries,
                work.attention, &binding.scores[index], batch.stream);
        }
        check(status, "Read bound verification MXFP8 attention");
        gate_gqa(work, batch.stream);
    }
    if (grouped) {
        check(target_split_gqa(reads.data(), batch.count, batch.target_split_end, batch.stream),
              "Read grouped verification MXFP8 attention");
        batch.each([&](const VerificationRow&, Workspace& work) { gate_gqa(work, batch.stream); });
    }
}

template <typename Lane>
void attend_eager_gqa(const GqaLayer& layer, const LayerBatch<Lane>& batch, int cache_index) {
    std::array<PagedDecodeLane, decoder_batch_max_lanes> reads;
    std::array<Workspace*, decoder_batch_max_lanes> pending;
    std::array<PagedDecodeLane, target_split_eager_max_rows> split_reads;
    std::array<Workspace*, target_split_eager_max_rows> split_pending;
    const bool wide = batch.count > paged_decode_lanes;
    const bool grouped = std::is_same_v<Lane, VerificationRow> &&
                         batch.target_split_end == text_attention_max_tokens && batch.count > 1;
    int count = 0;
    int split_count = 0;
    const auto flush_split = [&] {
        if (split_count == 0) {
            return;
        }
        check(target_split_gqa(split_reads.data(), split_count, batch.target_split_end,
                               batch.stream), "Read grouped target MXFP8 attention");
        for (int index = 0; index < split_count; ++index) {
            gate_gqa(*split_pending[index], batch.stream);
        }
        split_count = 0;
    };
    batch.each([&](const Lane& lane, Workspace& work) {
        auto& request = *lane.request;
        const auto cache = request.caches[cache_index];
        bool narrow_gqa = false;
        ExactGqaWorkspace singleton_scores{};
        const MropePosition* mrope = nullptr;
        int rotary_position = -1;
        if constexpr (!std::is_same_v<Lane, VerificationRow>) {
            narrow_gqa = lane.narrow_gqa && !batch.ordered_attention;
        }
        if constexpr (std::is_same_v<Lane, ForwardLane>) {
            mrope = lane.mrope;
            rotary_position = lane.rotary_position;
            singleton_scores = lane.singleton_scores;
        }
        append_gqa(layer, work, request.pages, cache, lane.position, batch.stream, mrope,
                   rotary_position);
        if constexpr (std::is_same_v<Lane, ForwardLane>) {
            const bool ordered = batch.ordered_attention && rotary_position < 0;
            if (!ordered && lane.mode == TextGdnMode::decode && (wide
                    ? lane.long_gqa && long_paged_decode_position(lane.position)
                    : batch.count > 1 && paged_decode_position(lane.position))) {
                reads[count] = {cache, request.pages, lane.position, paged_gqa_buffers(work)};
                pending[count++] = &work;
                return;
            }
        }
        const bool vector = batch.vector_attention && !mrope && rotary_position < 0;
        if constexpr (std::is_same_v<decltype(lane.position), int>) {
            if (batch.target_split_end && vector && lane.mode == TextGdnMode::decode &&
                target_split_position(lane.position, batch.target_split_end)) {
                const PagedDecodeLane read{cache, request.pages, lane.position,
                                           paged_gqa_buffers(work)};
                if (grouped) {
                    split_reads[split_count] = read;
                    split_pending[split_count++] = &work;
                    if (split_count == target_split_eager_max_rows) {
                        flush_split();
                    }
                    return;
                }
                check(target_split_gqa(read, batch.stream, batch.target_split_end),
                      "Read split target MXFP8 attention");
                gate_gqa(work, batch.stream);
                return;
            }
        }
        check(read_gqa(work, request.pages, cache, lane.position, lane.mode, narrow_gqa,
                       singleton_scores, vector, batch.stream), "Read causal MXFP8 attention");
        gate_gqa(work, batch.stream);
    });
    flush_split();
    if (count > 0) {
        const auto status = wide ? long_paged_decode_gqa(reads.data(), count, batch.stream)
                                 : paged_decode_gqa(reads.data(), count, batch.stream);
        check(status, "Read paged MXFP8 attention");
        for (int index = 0; index < count; ++index) {
            gate_gqa(*pending[index], batch.stream);
        }
    }
}

template <typename Lane>
void run_gqa(const GqaLayer& layer, const LayerBatch<Lane>& batch, int cache_index) {
    if (!batch.fuse_verify_activation() && !batch.fuse_wide_verify_norm()) {
        batch.quantize(&Workspace::normalized, hidden_size);
    }
    if (!batch.tensor_core_lane(0) && batch.single_decode() &&
        layer.q_gate.columns == hidden_size &&
        layer.key.columns == hidden_size && layer.value.columns == hidden_size) {
        const auto& work = workspace(batch.lanes[0]);
        check(decode_fp8_qkv({work.activation.values, work.activation.scales},
                             decode_projection(layer.q_gate, work.q_gate),
                             decode_projection(layer.key, work.keys),
                             decode_projection(layer.value, work.values), hidden_size,
                             batch.stream), "Run decode QKV projections");
    } else {
        batch.project_group({{layer.q_gate, &Workspace::q_gate}, {layer.key, &Workspace::keys},
                             {layer.value, &Workspace::values}});
    }
    if constexpr (std::is_same_v<Lane, BatchGraphLane>) {
        attend_bound_gqa(layer, batch, cache_index);
    } else if constexpr (std::is_same_v<Lane, VerificationRow>) {
        if (batch.lanes[0].binding) {
            attend_verify_gqa(layer, batch, cache_index);
        } else {
            attend_eager_gqa(layer, batch, cache_index);
        }
    } else {
        attend_eager_gqa(layer, batch, cache_index);
    }
    batch.quantize(&Workspace::attention, text_gqa_q_width);
    batch.project(layer.output, &Workspace::projected_attention);
}

template <typename Lane>
void run_mlp(const Layer& layer, const LayerBatch<Lane>& batch) {
    const bool five = batch.fuse_verify_activation();
    const bool wide = batch.fuse_wide_verify_norm();
    if (five || wide) {
        batch.project_group({{layer.mlp_gate, &Workspace::mlp_gate},
                             {layer.mlp_up, &Workspace::mlp_up}});
        if (five) {
            batch.quantize_verification(&Workspace::mlp_gate, intermediate_size,
                                        &Workspace::mlp_up);
        } else {
            batch.quantize_wide_verification(&Workspace::mlp_gate, intermediate_size,
                                             &Workspace::mlp_up);
        }
        batch.project(layer.mlp_down, &Workspace::hidden);
        return;
    }
    batch.quantize(&Workspace::normalized, hidden_size);
    if (!batch.tensor_core_lane(0) && batch.single_decode() &&
        layer.mlp_gate.columns == hidden_size &&
        layer.mlp_up.columns == hidden_size) {
        const auto& work = workspace(batch.lanes[0]);
        check(decode_fp8_gate_up({work.activation.values, work.activation.scales},
                                 decode_projection(layer.mlp_gate, work.mlp_gate),
                                 decode_projection(layer.mlp_up, work.mlp_up), hidden_size,
                                 batch.stream), "Run decode gate and up projections with SwiGLU");
    } else {
        batch.project_group({{layer.mlp_gate, &Workspace::mlp_gate},
                             {layer.mlp_up, &Workspace::mlp_up}});
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
    const ForwardLane lane{&request, token_id, position, mode, mode == TextGdnMode::decode};
    forward(&lane, 1, request.memory.stream());
}

void Decoder::Model::forward(const ForwardLane* lanes, int count, cudaStream_t stream) const {
    if (!lanes || count < 1 || count > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid decoder batch width");
    }
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        auto* residual = lane.request->work.residual;
        if (lane.feature) {
            check(cudaMemcpyAsync(residual, lane.feature, hidden_size * sizeof(Bf16),
                                  cudaMemcpyDeviceToDevice, stream), "Substitute visual embedding");
        } else {
            check(text_embedding(embedding, vocab_size, hidden_size, lane.token_id, residual,
                                 stream), "Look up token embedding");
        }
    }
    forward_layers(lanes, count, stream);
}

bool Decoder::Model::verify_mtp(MtpRound* rounds, int count, const ConstraintWorkspace& scratch,
                                void* sampling_scratch, cudaStream_t stream,
                                VerificationGraph* graph, const TcVocabScratch* vocabulary) const {
    std::array<VerificationSequence, decoder_batch_max_mtp_lanes> sequences;
    std::array<TargetVerification*, decoder_batch_max_mtp_lanes> verifiers;
    if (count < 1 || count > decoder_batch_max_mtp_lanes) {
        throw std::invalid_argument("Invalid number of MTP rounds");
    }
    for (int index = 0; index < count; ++index) {
        const auto& round = rounds[index];
        const int rows = round.rows();
        verifiers[index] = round.target;
        sequences[index] = {round.target, round.position, round.tokens.data(), rows,
                            round.constraint != nullptr};
        if (round.constraint || round.sampling || round.target->needs_logits() ||
            rows != target_verify_max_rows) {
            vocabulary = nullptr;
        }
    }
    const bool replayed = forward_verify(sequences.data(), count, stream, graph, vocabulary);
    for (int index = 0; index < count; ++index) {
        const auto& round = rounds[index];
        if (round.constraint) {
            round.target->constrain(*round.constraint, scratch, round.sampling, round.first_output,
                                    sampling_scratch);
        } else if (round.sampling) {
            round.target->sample(*round.sampling, round.first_output, sampling_scratch);
        }
    }
    TargetVerification::read_all(verifiers.data(), count, stream);
    return replayed;
}

bool Decoder::Model::forward_verify(const VerificationSequence* sequences, int count,
                                    cudaStream_t stream, VerificationGraph* graph,
                                    const TcVocabScratch* vocabulary) const {
    if (!sequences || count < 1 || count > decoder_batch_max_mtp_lanes) {
        throw std::invalid_argument("Invalid number of verification sequences");
    }
    constexpr int max_rows = target_verify_max_rows * decoder_batch_max_mtp_lanes;
    int total = 0;
    int longest = 0;
    for (int index = 0; index < count; ++index) {
        const auto& sequence = sequences[index];
        if (!sequence.verification || !sequence.tokens ||
            sequence.count < 1 || sequence.count > target_verify_max_rows) {
            throw std::invalid_argument("Invalid number of verification rows");
        }
        total += sequence.count;
        longest = std::max(longest, sequence.count);
        sequence.verification->begin(sequence.position, sequence.count, stream);
    }
    const auto& first = sequences[0];
    if (graph && count == 1 && !first.constrained && first.verification->graph_eligible(first) &&
        std::exchange(graph->warmed, true)) {
        const bool replayed = graph->execution.replay_batch_graph(1, [&] {
            graph->scratch = std::make_unique<VerificationGraphScratch>(device);
        }, [&] {
            auto& scratch = *graph->scratch;
            auto* binding = scratch.device_binding;
            std::array<VerificationRow, k1_verification_rows> rows;
            for (int index = 0; index < k1_verification_rows; ++index) {
                auto& work = scratch.workspace.work[index];
                rows[index] = {nullptr, &work, 0, nullptr, index == 0, binding, &scratch.workspace};
                check(decode_embedding(embedding, vocab_size, hidden_size, &binding->inputs[index],
                                       work.residual, stream), "Look up bound verification input");
            }
            enqueue_verification(rows.data(), k1_verification_rows, stream);
        }, [&] {
            graph->scratch->upload(first, stream);
            return true;
        }, true);
        if (replayed) {
            first.verification->use_graph_workspace(graph->scratch->workspace);
            return true;
        }
        graph->scratch.reset();
    }
    std::array<VerificationRow, max_rows> rows;
    int next = 0;
    for (int index = 0; index < longest; ++index) {
        for (int lane = 0; lane < count; ++lane) {
            const auto& sequence = sequences[lane];
            if (index < sequence.count) {
                auto row = sequence.verification->row(index);
                row.constrained = sequence.constrained;
                rows[next++] = row;
                check(text_embedding(embedding, vocab_size, hidden_size, sequence.tokens[index],
                                     row.work->residual, stream), "Look up verification input");
            }
        }
    }
    enqueue_verification(rows.data(), total, stream, vocabulary);
    return false;
}

void Decoder::Model::enqueue_verification(const VerificationRow* rows, int count,
                                          cudaStream_t stream,
                                          const TcVocabScratch* vocabulary) const {
    forward_layers(rows, count, stream);
    const LayerBatch<VerificationRow> batch{rows, count, stream};
    batch.each([&](const VerificationRow&, Workspace& work) { normalize_final(work, stream); });
    if (vocabulary && count == tc_vocab_rows) {
        std::array<TcVocabRow, tc_vocab_rows> projection;
        for (int index = 0; index < count; ++index) {
            const auto& work = *rows[index].work;
            projection[index] = {work.normalized, work.argmax};
        }
        if (try_tc_vocab(*tc_vocab, *vocabulary, projection, stream)) {
            return;
        }
    }
    project(lm_head, batch, &Workspace::logits);
    batch.each([&](const VerificationRow& row, Workspace& work) {
        if (!row.constrained) {
            check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
                  "Select target verification winner");
        }
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
    DecodeHeadRows rows;
    if (count == static_cast<int>(rows.size())) {
        for (int index = 0; index < count; ++index) {
            rows[index] = lanes[index].work;
        }
        compute_decode_logits(rows, stream);
        return;
    }
    batch.each([&](const BatchGraphLane&, Workspace& work) {
        compute_logits(work, stream);
    });
}

template <typename Lane>
void Decoder::Model::forward_layers(const Lane* lanes, int count, cudaStream_t stream) const {
    const auto policy = fp8_prefill();
    const LayerBatch<Lane> batch{lanes, count, stream, token_ordered_decode(policy),
                                 is_b200() && tensor_core_target(policy),
                                 is_b200() && policy == Fp8PrefillPolicy::tensor_core_mtp_target,
                                 target_split_end_};
    const auto normalize = [&](const Bf16* weight, const char* operation, const auto& residuals) {
        if (batch.fuse_verify_activation()) {
            VerifyRmsQuantizeRows rows;
            for (int index = 0; index < fused_verify_rows; ++index) {
                auto& work = workspace(lanes[index]);
                const auto [branch, residual] = residuals(work);
                rows.entries[index] = {
                    branch, residual, work.normalized,
                    batch.verification_activation(index, hidden_size)};
            }
            check(verify_rms_quantize(rows, weight, stream), operation);
            return;
        }
        if (batch.fuse_wide_verify_norm()) {
            WideVerifyRmsQuantizeRows rows{};
            for (int index = 0; index < count; ++index) {
                auto& work = workspace(lanes[index]);
                const auto [branch, residual] = residuals(work);
                rows.entries[index] = {branch, residual, work.normalized, work.activation};
            }
            check(verify_rms_quantize_wide(rows, count, weight, stream), operation);
            return;
        }
        batch.each([&](const Lane&, Workspace& work) {
            const auto [branch, residual] = residuals(work);
            check(text_rms_norm(branch, residual, weight, hidden_size, work.normalized,
                                residual, stream), operation);
        });
    };
    execute_layers(layers, normalize,
                   [&](const GqaLayer& layer, int bank) { run_gqa(layer, batch, bank); },
                   [&](const GdnLayer& layer, int bank) { run_gdn(layer, batch, bank); },
                   [&](const Layer& layer) { run_mlp(layer, batch); });
}

void Decoder::Model::compute_logits(Request& request, bool decode) const {
    compute_logits(request.work, request.memory.stream(), decode);
}

void Decoder::Model::normalize_final(Workspace& work, cudaStream_t stream) const {
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

void Decoder::Model::compute_decode_logits(const DecodeHeadRows& rows, cudaStream_t stream) const {
    DecodeHeadProjections projections;
    static_assert(std::tuple_size_v<DecodeHeadRows> == std::tuple_size_v<DecodeHeadProjections>);
    for (std::size_t index = 0; index < rows.size(); ++index) {
        auto& work = *rows[index];
        normalize_final(work, stream);
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
    project(lm_head, work.normalized, work.logits, stream);
}

void Decoder::Model::enqueue_label_logits(Request& request, cudaStream_t stream) const {
    compute_prefill_logits(request.work, stream);
    auto& logits = request.memory.transfers().logits;
    check(cudaMemcpyAsync(logits.data(), request.work.logits, logits.size() * sizeof(logits[0]),
                          cudaMemcpyDeviceToHost, stream), "Read final prompt logits");
}

void Decoder::Model::compute_logits(Workspace& work, cudaStream_t stream, bool decode) const {
    if (decode) {
        normalize_final(work, stream);
        check(decode_bf16_lm_head(lm_head.values, work.normalized, lm_head.rows, lm_head.columns,
                                  work.logits, stream), "Run decode LM head");
    } else {
        compute_prefill_logits(work, stream);
    }
    check(text_argmax(work.logits, vocab_size, work.argmax_scratch, work.argmax, stream),
          "Select greedy token");
}

}
