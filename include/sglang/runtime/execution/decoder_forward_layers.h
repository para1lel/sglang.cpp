#pragma once

// Shared per-layer execution machinery for the decoder forward paths
// (TP1 forward_layers and the tensor-parallel forward_layers_tp). Split out
// of src/decoder_forward.cpp so the TP drivers live in their own TU; the
// kernels, widths, and dispatch rules here are unchanged.

#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_verify.h"
#include "sglang/text/gdn_local.h"
#include "sglang/text/gqa_verify_batch.h"
#include "sglang/text/target_fp8.h"

#include <algorithm>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace sglang {
namespace decoder {
namespace forward_detail {

// Two-rank sharding for rank-local width lookups; widths are rank-blind, so
// the rank field never matters here.
inline constexpr WeightSharding tp2_sharding{0, 2};

inline void project(Bf16Weight weight, const Bf16* input, Bf16* output, cudaStream_t stream) {
    check(text_bf16_gemv(weight.values, input, weight.rows, weight.columns, output, stream),
          "Run BF16 projection");
}

struct GraphLane {
    Request* request;
    const int* position;
    const int* rotary_position;
    TextGdnMode mode;
    static constexpr bool narrow_gqa = true;
};

inline DecodeFp8Projection decode_projection(Fp8Weight weight, Bf16* output) {
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

inline Workspace& workspace(const VerificationRow& row) {
    return *row.work;
}

inline Workspace& workspace(const BatchGraphLane& lane) {
    return *lane.work;
}

static_assert(wide_mtp_rows == text_target_fp8_m40_rows);
static_assert(2 * wide_mtp_rows == text_target_fp8_m80_rows);
static_assert(max_wide_mtp_rows == text_target_fp8_m160_rows);

template <typename Lane>
struct LayerBatch {
    const Lane* lanes;
    int count;
    cudaStream_t stream;
    bool ordered_attention = false;
    bool target_tc = false;
    bool vector_attention = false;
    bool grouped_target_attention = false;
    int target_split_end = 0;
    std::uint8_t* packed_activation = nullptr;
    // Tensor-parallel output redirection: when set, row-parallel projections
    // write each lane's partial into a contiguous reduction slab instead of
    // the per-lane workspace member. Only Workspace::projected_attention and
    // Workspace::hidden are ever redirected.
    Bf16* const* attention_output = nullptr;
    Bf16* const* hidden_output = nullptr;
    // True when this batch runs rank-local (TP2) weights; kernels and widths
    // below then follow the rank-local geometry. Never infer this from buffer
    // widths: the local q_gate width equals the global query width, and the
    // local gate+up rows sum to the global intermediate size.
    bool tensor_parallel = false;

    // Rank-local widths come from WeightSharding, the loader's single source
    // of truth for the sharded geometry.
    int gqa_attention_width() const {
        return tensor_parallel ? tp2_sharding.local_q_width() : text_gqa_q_width;
    }
    int gdn_attention_width() const {
        return tensor_parallel ? tp2_sharding.local_gdn_value_width() : text_gdn_value_width;
    }
    int gdn_gate_heads() const {
        return tensor_parallel ? tp2_sharding.local_gdn_value_heads() : text_gdn_value_heads;
    }
    int intermediate_width() const {
        return tensor_parallel ? tp2_sharding.local_intermediate_size() : intermediate_size;
    }
    GqaGeometry gqa_geometry() const {
        return tensor_parallel ? gqa_rank_geometry : gqa_full_geometry;
    }

    Bf16* lane_output(int index, Bf16* Workspace::*output) const {
        if (output == &Workspace::projected_attention && attention_output) {
            return attention_output[index];
        }
        if (output == &Workspace::hidden && hidden_output) {
            return hidden_output[index];
        }
        return workspace(lanes[index]).*output;
    }
    LayerBatch verification_packet(int first) const {
        auto packet = *this;
        packet.lanes += first;
        packet.count = verification_packet_rows;
        return packet;
    }

    bool tensor_core_lane(int index) const {
        if constexpr (std::is_same_v<Lane, ForwardLane>) {
            return target_tc && lanes[index].mode == TextGdnMode::decode;
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
        static_assert(WideVerifyRmsQuantizeRows::capacity == max_wide_mtp_rows);
        return std::is_same_v<Lane, VerificationRow> && count > fused_verify_rows;
    }

    bool packs_projection_activation(int width) const {
        return target_tc && (fuse_verify_activation() || fuse_wide_verify_norm()) &&
               width > 0 && width <= intermediate_size && width % text_fp8_group_size == 0;
    }

    TextFp8Activation projection_activation(int index, int width) const {
        auto activation = workspace(lanes[index]).activation;
        if (packs_projection_activation(width)) {
            static_assert(verification_packet_rows * intermediate_size <=
                          vocab_size * sizeof(Bf16));
            // Each projection consumes this input before the stream reuses logits scratch.
            auto* packed = packed_activation ? packed_activation
                : reinterpret_cast<std::uint8_t*>(workspace(lanes[0]).logits);
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
                work.*source, up ? work.*up : nullptr, projection_activation(index, width)};
        }
        check(verify_quantize_fp8(rows, width, stream), "Quantize five verification rows");
    }

    void quantize_wide_verification(Bf16* Workspace::*source, int width,
                                    Bf16* Workspace::*up = nullptr) const {
        const int capacity = packed_activation ? WideVerifyQuantizeRows::capacity
                                               : verification_packet_rows;
        for (int first = 0; first < count; first += capacity) {
            WideVerifyQuantizeRows rows{};
            const int size = std::min(count - first, capacity);
            for (int index = 0; index < size; ++index) {
                auto& work = workspace(lanes[first + index]);
                rows.entries[index] = {work.*source, up ? work.*up : nullptr,
                                       projection_activation(first + index, width)};
            }
            check(verify_quantize_fp8_wide(rows, size, width, stream),
                  "Quantize wide verification rows");
        }
    }

    void project(Fp8Weight weight, Bf16* Workspace::*output) const {
        if (target_tc && (count > 1 || tensor_core_lane(0)) && target_shape(weight)) {
            project_target({{weight, output}});
            return;
        }
        if (count == 1) {
            const auto& work = workspace(lanes[0]);
            const TextFp8ActivationView input{work.activation.values, work.activation.scales};
            Bf16* output_row = lane_output(0, output);
            const auto status = single_decode()
                ? decode_fp8_project(input, decode_projection(weight, output_row),
                                     weight.columns, stream)
                : text_fp8_gemv(weight.values, weight.scales, input, weight.rows, weight.columns,
                                output_row, stream);
            check(status, "Run FP8 projection");
            return;
        }
        std::array<TextFp8Projection, text_fp8_max_batch> projections;
        for (int first = 0, group = 0; first < count; ++group) {
            const int rows = projection_rows(group);
            for (int index = 0; index < rows; ++index) {
                const auto activation = projection_activation(first + index, weight.columns);
                projections[index] = {{activation.values, activation.scales},
                                      lane_output(first + index, output)};
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

    template <int Capacity = text_target_fp8_max_rows>
    bool project_target_chunk(std::initializer_list<LayerProjection> layers, int first,
                              int rows, bool allow_fallback) const {
        std::array<TextFp8Projection, text_decode_max_targets * Capacity> operands;
        std::array<TextFp8DecodeTarget, text_decode_max_targets> targets;
        const int columns = layers.begin()->weight.columns;
        int target = 0;
        for (const auto& layer : layers) {
            auto* entries = operands.data() + target * rows;
            for (int index = 0; index < rows; ++index) {
                const auto activation = projection_activation(first + index, columns);
                entries[index] = {{activation.values, activation.scales},
                                  lane_output(first + index, layer.output)};
            }
            const auto weight = layer.weight;
            targets[target++] = {weight.values, weight.scales, weight.rows, entries[0].output};
        }
        auto* scratch = packs_projection_activation(columns)
            ? reinterpret_cast<Bf16*>(projection_activation(first, columns).values)
            : workspace(lanes[first]).logits;
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
        if constexpr (verification) {
            if (packed_activation) {
                if (project_target_chunk<max_wide_mtp_rows>(layers, 0, count, false)) return;
                if (count == wide_mtp_rows) {
                    project_verification_packets(layers, 0, count);
                } else {
                    for (int first = 0; first < count; first += wide_mtp_rows) {
                        if (!project_target_chunk<wide_mtp_rows>(layers, first, wide_mtp_rows,
                                                                 false)) {
                            project_verification_packets(layers, first, wide_mtp_rows);
                        }
                    }
                }
                return;
            }
        }
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

    void project_verification_packets(std::initializer_list<LayerProjection> layers, int first,
                                      int size) const {
        for (int end = first + size; first < end; first += verification_packet_rows) {
            if (!project_target_chunk(layers, first, verification_packet_rows, false)) {
                for (int row = first; row < first + verification_packet_rows;
                     row += text_fp8_max_batch) {
                    project_target_chunk(layers, row, text_fp8_max_batch, true);
                }
            }
        }
    }
};

inline constexpr int gdn_packet_rows = GdnVerificationBatch::lanes * GdnVerificationBatch::rows;
static_assert(GdnVerificationBatch::lanes == decoder_batch_max_mtp_lanes);
static_assert(GdnVerificationBatch::rows == target_verify_max_rows);

inline void project_gdn_gates(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch,
                              bool batched) {
    if (!batch.target_tc || batch.count < 2 ||
        layer.b.rows != batch.gdn_gate_heads() || layer.a.rows != batch.gdn_gate_heads() ||
        layer.b.columns != hidden_size || layer.a.columns != hidden_size) {
        project_verification_bf16(layer.b, batch.lanes, batch.count, &Workspace::gdn_b,
                                  batch.stream);
        project_verification_bf16(layer.a, batch.lanes, batch.count, &Workspace::gdn_a,
                                  batch.stream);
        return;
    }
    std::array<VerifyGdnGateRow, gdn_packet_rows> gates;
    const auto fill = [&](int first, int rows) {
        for (int index = 0; index < rows; ++index) {
            const auto& work = workspace(batch.lanes[first + index]);
            gates[index] = {work.normalized, work.gdn_b, work.gdn_a};
        }
        const auto gates_kernel =
            batch.tensor_parallel ? try_verify_bf16_gdn_gates_local : try_verify_bf16_gdn_gates;
        return gates_kernel(layer.b.values, layer.a.values, gates.data(), rows, batch.stream);
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

inline void run_gdn_packet(const GdnLayer& layer, const LayerBatch<VerificationRow>& packet,
                           int state_index) {
    project_gdn_gates(layer, packet, true);
    GdnVerificationBatch states;
    for (int lane = 0; lane < GdnVerificationBatch::lanes; ++lane) {
        states.entries[lane] = packet.lanes[lane].verification->gdn_lane(
            state_index, layer.recurrent);
    }
    check(packet.tensor_parallel
              ? decode_verify_gdn_batch_local(states, layer.recurrent, packet.stream)
              : decode_verify_gdn_batch(states, layer.recurrent, packet.stream),
          "Advance batched target verification GDN packet");
}

inline bool run_compact_gdn(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch,
                            int state_index) {
    constexpr int rows_per_lane = CompactGdnVerificationBatch::rows;
    constexpr int row_capacity = CompactGdnVerificationBatch::lanes * rows_per_lane;
    if (!batch.packed_activation || !batch.target_tc || batch.lanes[0].binding ||
        (batch.count != row_capacity / 2 && batch.count != row_capacity) ||
        layer.b.rows != batch.gdn_gate_heads() || layer.a.rows != batch.gdn_gate_heads() ||
        layer.b.columns != hidden_size || layer.a.columns != hidden_size) {
        return false;
    }
    std::array<VerifyGdnGateRow, row_capacity> gates;
    for (int index = 0; index < batch.count; ++index) {
        const auto& work = workspace(batch.lanes[index]);
        gates[index] = {work.normalized, work.gdn_b, work.gdn_a};
    }
    const auto gates_kernel = batch.tensor_parallel ? try_verify_bf16_gdn_gates_compact_local
                                                    : try_verify_bf16_gdn_gates_compact;
    const auto status = gates_kernel(layer.b.values, layer.a.values, gates.data(), batch.count,
                                     batch.stream);
    if (!status) {
        return false;
    }
    check(*status, "Run compact verification GDN gate projections");
    CompactGdnVerificationBatch states{};
    CompactGdnFp8Rows output{batch.packed_activation, {}};
    const int lanes = batch.count / rows_per_lane;
    for (int lane = 0; lane < lanes; ++lane) {
        const int first = compact_gdn_row(lane, 0);
        states.entries[lane] = batch.lanes[first].verification->gdn_lane(
            state_index, layer.recurrent);
        for (int step = 0; step < rows_per_lane; ++step) {
            const int row = compact_gdn_row(lane, step);
            output.scales[row] = batch.projection_activation(row, batch.gdn_attention_width())
                                     .scales;
        }
    }
    check(batch.tensor_parallel
              ? decode_verify_gdn_compact_local(states, lanes, layer.recurrent, output,
                                                batch.stream)
              : decode_verify_gdn_compact(states, lanes, layer.recurrent, output, batch.stream),
          "Advance and quantize compact verification GDN");
    return true;
}

template <typename Lane>
void run_gdn(const GdnLayer& layer, const LayerBatch<Lane>& batch, int state_index) {
    batch.quantize(&Workspace::normalized, hidden_size);
    batch.project_group({{layer.qkv, &Workspace::gdn_qkv}, {layer.z, &Workspace::gdn_z}});
    batch.each([&](const Lane& lane, Workspace& work) {
        const auto stream = batch.stream;
        if (batch.single_decode() && layer.b.rows == layer.a.rows &&
            layer.b.columns == layer.a.columns) {
            const auto gate_kernel =
                batch.tensor_parallel ? decode_bf16_gdn_gates_local : decode_bf16_gdn_gates;
            check(gate_kernel(layer.b.values, layer.a.values, work.normalized, layer.b.rows,
                              layer.b.columns, work.gdn_b, work.gdn_a, stream),
                  "Run decode Gated DeltaNet gate projections");
        } else {
            project(layer.b, work.normalized, work.gdn_b, stream);
            project(layer.a, work.normalized, work.gdn_a, stream);
        }
        if constexpr (std::is_same_v<Lane, BatchGraphLane>) {
            check(batch.tensor_parallel
                      ? decode_gdn_local(work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                                         layer.recurrent, &lane.binding->states[state_index],
                                         work.gdn_scratch, work.attention, stream)
                      : decode_gdn(work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                                   layer.recurrent, &lane.binding->states[state_index],
                                   work.gdn_scratch, work.attention, stream),
                  "Advance bound Gated DeltaNet state");
        } else {
            check(batch.tensor_parallel
                      ? text_gdn_forward_local(1, lane.mode, work.gdn_qkv, work.gdn_z,
                                               work.gdn_b, work.gdn_a, layer.recurrent,
                                               lane.request->states[state_index],
                                               work.gdn_scratch,
                                               text_gdn_local_workspace_bytes(1),
                                               work.attention, stream)
                      : text_gdn_forward(1, lane.mode, work.gdn_qkv, work.gdn_z, work.gdn_b,
                                         work.gdn_a, layer.recurrent,
                                         lane.request->states[state_index], work.gdn_scratch,
                                         text_gdn_workspace_bytes(1), work.attention, stream),
                  "Advance Gated DeltaNet state");
        }
    });
    batch.quantize(&Workspace::attention, batch.gdn_attention_width());
    batch.project(layer.output, &Workspace::projected_attention);
}

inline void run_gdn(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch,
                    int state_index) {
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
    if (run_compact_gdn(layer, batch, state_index)) {
        batch.project(layer.output, &Workspace::projected_attention);
        return;
    }
    if (batch.packed_activation) {
        for (int start = 0; start < batch.count; start += gdn_packet_rows) {
            run_gdn_packet(layer, batch.verification_packet(start), state_index);
        }
    } else if (batch.target_tc && !first.binding && batch.count == gdn_packet_rows) {
        run_gdn_packet(layer, batch, state_index);
    } else {
        project_gdn_gates(layer, batch, false);
        if (first.binding) {
            auto& scratch = *first.graph_workspace;
            check(batch.tensor_parallel
                      ? decode_verify_gdn_local(batch.count, scratch.replay[state_index].inputs,
                                                first.work->gdn_z, layer.recurrent,
                                                &first.binding->initial[state_index],
                                                &first.binding->tail[state_index],
                                                scratch.observable, first.work->attention,
                                                batch.stream)
                      : decode_verify_gdn(batch.count, scratch.replay[state_index].inputs,
                                          first.work->gdn_z, layer.recurrent,
                                          &first.binding->initial[state_index],
                                          &first.binding->tail[state_index], scratch.observable,
                                          first.work->attention, batch.stream),
                  "Advance bound verification GDN");
        } else {
            batch.each([&](const VerificationRow& row, Workspace&) {
                if (row.first) {
                    row.verification->advance_gdn(state_index, layer.recurrent);
                }
            });
        }
    }
    batch.quantize(&Workspace::attention, batch.gdn_attention_width());
    batch.project(layer.output, &Workspace::projected_attention);
}

template <typename Position>
cudaError_t read_gqa(Workspace& work, Mxfp8PageTable pages, Mxfp8KvCache cache,
                     Position position, TextGdnMode mode, bool narrow_gqa,
                     ExactGqaWorkspace singleton_scores, bool vector, cudaStream_t stream) {
    const int capacity = pages.page_count * mxfp8_page_size;
    const auto scores = work.gqa_scores(capacity);
    // Geometry is cache-inferred: TP2 caches carry kv_heads = 2, TP1 caches 4.
    const GqaGeometry geometry{gqa_query_heads(cache.kv_heads), cache.kv_heads};
    if constexpr (std::is_same_v<Position, int>) {
        if (singleton_scores.scores) {
            return exact_decode_gqa_attention(cache, pages, position, work.rope_queries,
                work.attention, singleton_scores, stream, 0, mode == TextGdnMode::prefill);
        }
        if (mode == TextGdnMode::prefill || position >= scores.capacity) {
            return mxfp8_gqa_attention(cache, pages, position + 1, position, 1,
                                       work.rope_queries, work.attention, stream, geometry);
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
                Position rotary_position, GqaGeometry geometry) {
    if constexpr (std::is_same_v<Position, int>) {
        const auto status = mrope
            ? mrope_gqa_preprocess(mrope, 1, work.q_gate, work.keys, layer.query_norm,
                                   layer.key_norm, work.rope_queries, work.rope_keys, stream,
                                   geometry)
            : text_gqa_preprocess(rotary_position < 0 ? position : rotary_position, 1,
                                  work.q_gate, work.keys, layer.query_norm, layer.key_norm,
                                  work.rope_queries, work.rope_keys, stream, geometry);
        check(status, "Normalize and rotate Q/K");
        check(mxfp8_kv_append(cache, pages, position, 1, work.rope_keys, work.values, stream,
                              geometry), "Append MXFP8 KV");
    } else {
        check(decode_gqa_preprocess(rotary_position, work.q_gate, work.keys, layer.query_norm,
                                    layer.key_norm, work.rope_queries, work.rope_keys, stream,
                                    geometry),
              "Normalize and rotate decode Q/K");
        check(decode_kv_append(cache, pages, position, work.rope_keys, work.values, stream),
              "Append decode MXFP8 KV");
    }
}

inline void gate_gqa(Workspace& work, cudaStream_t stream, GqaGeometry geometry) {
    check(text_gqa_postgate(1, work.q_gate, work.attention, work.attention, stream, geometry),
          "Gate grouped-query attention");
}

inline void attend_bound_gqa(const GqaLayer& layer, const LayerBatch<BatchGraphLane>& batch,
                             int cache_index) {
    static_assert(paged_decode_lanes == decoder_batch_max_graph_lanes);
    std::array<BoundPagedDecodeLane, paged_decode_lanes> reads;
    for (int index = 0; index < batch.count; ++index) {
        const auto& lane = batch.lanes[index];
        const auto* binding = lane.binding;
        const auto* position = &binding->position;
        auto& work = workspace(lane);
        check(decode_gqa_preprocess(&binding->rotary_position, work.q_gate, work.keys,
                                    layer.query_norm, layer.key_norm, work.rope_queries,
                                    work.rope_keys, batch.stream, batch.gqa_geometry()),
              "Normalize and rotate bound Q/K");
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
    batch.each([&](const BatchGraphLane&, Workspace& work) {
        gate_gqa(work, batch.stream, batch.gqa_geometry());
    });
}

inline void attend_verify_gqa(const GqaLayer& layer, const LayerBatch<VerificationRow>& batch,
                              int cache_index) {
    const auto& binding = *batch.lanes[0].binding;
    const auto* cache = &binding.caches[cache_index];
    const bool grouped = batch.grouped_target_attention && batch.count > 1;
    std::array<BoundPagedDecodeLane, target_verify_max_rows> reads;
    for (int index = 0; index < batch.count; ++index) {
        auto& work = workspace(batch.lanes[index]);
        const auto* position = &binding.positions[index];
        check(decode_gqa_preprocess(&binding.rotary_positions[index], work.q_gate, work.keys,
                                    layer.query_norm, layer.key_norm, work.rope_queries,
                                    work.rope_keys, batch.stream, batch.gqa_geometry()),
              "Normalize and rotate bound verification Q/K");
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
        gate_gqa(work, batch.stream, batch.gqa_geometry());
    }
    if (grouped) {
        check(target_split_gqa(reads.data(), batch.count, batch.target_split_end, batch.stream),
              "Read grouped verification MXFP8 attention");
        batch.each([&](const VerificationRow&, Workspace& work) {
            gate_gqa(work, batch.stream, batch.gqa_geometry());
        });
    }
}

template <typename Lane>
void attend_eager_gqa(const GqaLayer& layer, const LayerBatch<Lane>& batch, int cache_index) {
    std::array<PagedDecodeLane, decoder_batch_max_lanes> reads;
    std::array<Workspace*, decoder_batch_max_lanes> pending;
    std::array<PagedDecodeLane, target_split_eager_max_rows> split_reads;
    std::array<Workspace*, target_split_eager_max_rows> split_pending;
    const bool grouped = std::is_same_v<Lane, VerificationRow> &&
                         batch.grouped_target_attention && batch.count > 1;
    int count = 0;
    int split_count = 0;
    const auto flush_split = [&] {
        if (split_count == 0) {
            return;
        }
        check(target_split_gqa(split_reads.data(), split_count, batch.target_split_end,
                               batch.stream), "Read grouped target MXFP8 attention");
        for (int index = 0; index < split_count; ++index) {
            gate_gqa(*split_pending[index], batch.stream, batch.gqa_geometry());
        }
        split_count = 0;
    };
    batch.each([&](const Lane& lane, Workspace& work) {
        auto& request = *lane.request;
        const auto cache = request.caches[cache_index];
        bool narrow_gqa = false;
        ExactGqaWorkspace singleton_scores{};
        bool reuse = false;
        int cached_rotary_position = 0;
        const MropePosition* mrope = nullptr;
        if constexpr (!std::is_same_v<Lane, VerificationRow>) {
            narrow_gqa = lane.narrow_gqa && !batch.ordered_attention;
        }
        if constexpr (std::is_same_v<Lane, ForwardLane>) {
            mrope = lane.mrope;
            singleton_scores = lane.singleton_scores;
            reuse = lane.mode == TextGdnMode::prefill &&
                lane.position < request.cached_gqa_tokens;
            cached_rotary_position = lane.rotary_position < 0 ? lane.position
                                                              : lane.rotary_position;
        }
        if (reuse) {
            check(prefill_gqa_queries(cached_rotary_position, mrope, 1, work.q_gate,
                                      layer.query_norm, work.rope_queries, batch.stream,
                                      batch.gqa_geometry()),
                  "Normalize cached prefill Q");
        } else {
            append_gqa(layer, work, request.pages, cache, lane.position, batch.stream, mrope,
                       lane.rotary_position, batch.gqa_geometry());
        }
        if constexpr (std::is_same_v<Lane, ForwardLane>) {
            if (!batch.ordered_attention && lane.mode == TextGdnMode::decode &&
                batch.count > 1 && batch.count <= paged_decode_lanes &&
                paged_decode_position(lane.position)) {
                reads[count] = {cache, request.pages, lane.position, paged_gqa_buffers(work)};
                pending[count++] = &work;
                return;
            }
        }
        const bool vector = batch.vector_attention;
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
                gate_gqa(work, batch.stream, batch.gqa_geometry());
                return;
            }
        }
        check(read_gqa(work, request.pages, cache, lane.position, lane.mode, narrow_gqa,
                       singleton_scores, vector, batch.stream), "Read causal MXFP8 attention");
        gate_gqa(work, batch.stream, batch.gqa_geometry());
    });
    flush_split();
    if (count > 0) {
        const auto status = paged_decode_gqa(reads.data(), count, batch.stream);
        check(status, "Read paged MXFP8 attention");
        for (int index = 0; index < count; ++index) {
            gate_gqa(*pending[index], batch.stream, batch.gqa_geometry());
        }
    }
}

inline void attend_eager_verification(const GqaLayer& layer,
                                      const LayerBatch<VerificationRow>& batch, int cache_index) {
    const auto attend = [&](const auto& packet) {
        if (packet.tensor_parallel &&
            try_gqa_verify_short(layer, packet.lanes, packet.count, cache_index,
                                 packet.vector_attention, packet.stream)) {
            return;
        }
        if (!try_gqa_verify_batch(layer, packet.lanes, packet.count, cache_index,
                                  packet.vector_attention, packet.target_split_end,
                                  packet.stream)) {
            attend_eager_gqa(layer, packet, cache_index);
        }
    };
    const int width = batch.packed_activation ? verification_packet_rows : batch.count;
    const auto remainder = [&](const auto& selected) {
        std::array<VerificationRow, verification_packet_rows> rows;
        for (int first = 0; first < batch.count; first += width) {
            auto packet = batch;
            packet.lanes = rows.data();
            packet.count = 0;
            for (int index = first; index < std::min(first + width, batch.count); ++index) {
                if (!selected[index]) rows[packet.count++] = batch.lanes[index];
            }
            if (packet.count) attend(packet);
        }
    };
    if (try_gqa_verify_long(layer, batch.lanes, batch.count, cache_index, batch.vector_attention,
                            batch.target_split_end, batch.stream, remainder)) return;
    if (!batch.packed_activation) {
        attend(batch);
        return;
    }
    for (int first = 0; first < batch.count; first += width) {
        attend(batch.verification_packet(first));
    }
}

template <typename Lane>
void run_gqa(const GqaLayer& layer, const LayerBatch<Lane>& batch, int cache_index) {
    if (!batch.fuse_verify_activation() && !batch.fuse_wide_verify_norm()) {
        batch.quantize(&Workspace::normalized, hidden_size);
    }
    const bool reuse = [&] {
        if constexpr (std::is_same_v<Lane, ForwardLane>) {
            return std::all_of(batch.lanes, batch.lanes + batch.count, [](const ForwardLane& lane) {
                return lane.mode == TextGdnMode::prefill &&
                    lane.position < lane.request->cached_gqa_tokens;
            });
        }
        return false;
    }();
    if (reuse) {
        batch.project(layer.q_gate, &Workspace::q_gate);
    } else if (!batch.tensor_core_lane(0) && batch.single_decode() &&
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
            attend_eager_verification(layer, batch, cache_index);
        }
    } else {
        attend_eager_gqa(layer, batch, cache_index);
    }
    batch.quantize(&Workspace::attention, batch.gqa_attention_width());
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
            batch.quantize_verification(&Workspace::mlp_gate, batch.intermediate_width(),
                                        &Workspace::mlp_up);
        } else {
            batch.quantize_wide_verification(&Workspace::mlp_gate, batch.intermediate_width(),
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
            check(text_swiglu(work.mlp_gate, work.mlp_up, batch.intermediate_width(),
                              work.mlp_gate, batch.stream), "Apply SwiGLU");
        });
    }
    batch.quantize(&Workspace::mlp_gate, batch.intermediate_width());
    batch.project(layer.mlp_down, &Workspace::hidden);
}

}  // namespace forward_detail
}  // namespace decoder
}  // namespace sglang
