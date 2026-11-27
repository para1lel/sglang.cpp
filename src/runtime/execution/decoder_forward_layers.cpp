#include "sglang/runtime/execution/decoder_forward_layers.h"
#include <algorithm>
#include <type_traits>

namespace sglang::decoder::forward_detail {

void project(Bf16Weight weight, const Bf16* input, Bf16* output, cudaStream_t stream) {
    check(text_bf16_gemv(weight.values, input, weight.rows, weight.columns, output, stream),
          "Run BF16 projection");
}

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
int LayerBatch<Lane>::gqa_attention_width() const {
    return tensor_parallel ? gqa_rank_geometry.query_heads * mxfp8_head_dim : text_gqa_q_width;
}

template <typename Lane>
int LayerBatch<Lane>::gdn_attention_width() const {
    return tensor_parallel ? GdnLocal::value_width : text_gdn_value_width;
}

template <typename Lane>
int LayerBatch<Lane>::gdn_gate_heads() const {
    return tensor_parallel ? GdnLocal::value_heads : text_gdn_value_heads;
}

template <typename Lane>
int LayerBatch<Lane>::intermediate_width() const {
    return tensor_parallel ? intermediate_size / 2 : intermediate_size;
}

template <typename Lane>
GqaGeometry LayerBatch<Lane>::gqa_geometry() const {
    return tensor_parallel ? gqa_rank_geometry : gqa_full_geometry;
}

template <typename Lane>
Bf16* LayerBatch<Lane>::lane_output(int index, Bf16* Workspace::*output) const {
    if (output == &Workspace::projected_attention && attention_output) {
        return attention_output[index];
    }
    if (output == &Workspace::hidden && hidden_output) {
        return hidden_output[index];
    }
    return workspace(lanes[index]).*output;
}

template <typename Lane>
LayerBatch<Lane> LayerBatch<Lane>::verification_packet(int first) const {
    auto packet = *this;
    packet.lanes += first;
    packet.count = std::min(verification_packet_rows, count - first);
    return packet;
}

template <typename Lane>
bool LayerBatch<Lane>::tensor_core_lane(int index) const {
    if constexpr (std::is_same_v<Lane, ForwardLane>) {
        return target_tc && lanes[index].mode == TextGdnMode::decode;
    }
    return target_tc;
}

template <typename Lane>
bool LayerBatch<Lane>::single_decode() const {
    if constexpr (std::is_same_v<Lane, BatchGraphLane>) {
        return count == 1;
    } else {
        return count == 1 && lanes[0].mode == TextGdnMode::decode;
    }
}

template <typename Lane>
bool LayerBatch<Lane>::fuse_verify_activation() const {
    return std::is_same_v<Lane, VerificationRow> &&
           (count == fused_verify_rows || singleton_verify_activation);
}

template <typename Lane>
bool LayerBatch<Lane>::fuse_wide_verify_norm() const {
    static_assert(WideVerifyRmsQuantizeRows::capacity == max_wide_mtp_rows);
    return std::is_same_v<Lane, VerificationRow> && count > fused_verify_rows;
}

template <typename Lane>
bool LayerBatch<Lane>::packs_projection_activation(int width) const {
    return target_tc && (fuse_verify_activation() || fuse_wide_verify_norm()) &&
           width > 0 && width <= intermediate_size && width % text_fp8_group_size == 0;
}

template <typename Lane>
TextFp8Activation LayerBatch<Lane>::projection_activation(int index, int width) const {
    if (projection_override) return projection_override[index];
    auto activation = workspace(lanes[index]).activation;
    if (packs_projection_activation(width)) {
        static_assert(verification_packet_rows * intermediate_size <=
                      vocab_size * sizeof(Bf16));

        auto* packed = packed_activation ? packed_activation
            : reinterpret_cast<std::uint8_t*>(workspace(lanes[0]).logits);
        activation.values = packed + index * width;
    }
    return activation;
}

template <typename Lane>
int LayerBatch<Lane>::projection_rows(int group) const {
    const int groups = (count + text_fp8_max_batch - 1) / text_fp8_max_batch;
    return count / groups + (group < count % groups);
}

template <typename Lane>
void LayerBatch<Lane>::quantize(Bf16* Workspace::*source, int width) const {
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
    for (int index = 0; index < count; ++index) {
        auto& work = workspace(lanes[index]);
        check(text_quantize_fp8(work.*source, width, work.activation, stream),
              "Quantize projection input");
    }
}

template <typename Lane>
void LayerBatch<Lane>::quantize_verification(Bf16* Workspace::*source, int width,
                           Bf16* Workspace::*up) const {
    VerifyQuantizeRows rows{};
    for (int index = 0; index < count; ++index) {
        auto& work = workspace(lanes[index]);
        rows.entries[index] = {
            work.*source, up ? work.*up : nullptr, projection_activation(index, width)};
    }
    const auto status = count == fused_verify_rows
        ? verify_quantize_fp8(rows, width, stream)
        : verify_quantize_partial_fp8(rows, count, width, stream);
    check(status, "Quantize verification activation");
}

template <typename Lane>
void LayerBatch<Lane>::quantize_wide_verification(Bf16* Workspace::*source, int width,
                                Bf16* Workspace::*up) const {
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

template <typename Lane>
bool LayerBatch<Lane>::project_fused_mlp(const Layer& layer) const {
    if constexpr (!std::is_same_v<Lane, VerificationRow>) return false;
    constexpr int rows = text_target_fp8_m10_rows;
    constexpr int width = intermediate_size / 2;
    if (!tensor_parallel || !target_tc || count != rows || packed_activation ||
        layer.mlp_gate.columns != hidden_size || layer.mlp_up.columns != hidden_size ||
        layer.mlp_down.columns != width) return false;
    auto* logits = workspace(lanes[0]).logits;
    if (!logits) return false;
    constexpr int groups = width / text_fp8_group_size;
    constexpr int norm_value_bytes = rows * hidden_size;
    constexpr int down_bytes = rows * width;
    static_assert(norm_value_bytes % 16 == 0 && down_bytes % 16 == 0);
    static_assert(norm_value_bytes + down_bytes + rows * groups * sizeof(float) <=
                  vocab_size * sizeof(Bf16));
    auto* down_values = reinterpret_cast<std::uint8_t*>(logits) + norm_value_bytes;
    auto* down_scales = reinterpret_cast<float*>(down_values + down_bytes);
    TextFp8Activation down[rows];
    TextFp8Activation inputs[rows];
    for (int row = 0; row < rows; ++row) {
        inputs[row] = projection_activation(row, hidden_size);
        down[row] = {down_values + row * width, down_scales + row * groups};
    }
    const TextFp8DecodeTarget targets[] = {
        {layer.mlp_gate.values, layer.mlp_gate.scales, layer.mlp_gate.rows, nullptr},
        {layer.mlp_up.values, layer.mlp_up.scales, layer.mlp_up.rows, nullptr},
    };
    const auto result = try_target_fp8_gate_up(targets, hidden_size, inputs, down, stream);
    if (!result) return false;
    check(*result, "Run fused target FP8 gate, up, and SwiGLU");
    auto projection = *this;
    projection.projection_override = down;
    projection.project(layer.mlp_down, &Workspace::hidden);
    return true;
}

template <typename Lane>
void LayerBatch<Lane>::project(Fp8Weight weight, Bf16* Workspace::*output) const {
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

template <typename Lane>
bool LayerBatch<Lane>::target_shape(Fp8Weight weight) {
    return weight.rows > 0 && weight.rows <= intermediate_size &&
           weight.columns > 0 && weight.columns <= intermediate_size &&
           weight.rows % text_fp8_group_size == 0 &&
           weight.columns % text_fp8_group_size == 0;
}

template <typename Lane>
void LayerBatch<Lane>::project_group(std::initializer_list<LayerProjection> layers) const {
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

template <typename Lane>
template <int Capacity>
bool LayerBatch<Lane>::project_target_chunk(std::initializer_list<LayerProjection> layers,
                                           int first, int rows, bool allow_fallback) const {
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

template <typename Lane>
void LayerBatch<Lane>::project_target(std::initializer_list<LayerProjection> layers) const {
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

template <typename Lane>
void LayerBatch<Lane>::project_verification_packets(std::initializer_list<LayerProjection> layers,
                                                   int first, int size) const {
    for (int end = first + size; first < end; first += verification_packet_rows) {
        if (!project_target_chunk(layers, first, verification_packet_rows, false)) {
            for (int row = first; row < first + verification_packet_rows;
                 row += text_fp8_max_batch) {
                project_target_chunk(layers, row, text_fp8_max_batch, true);
            }
        }
    }
}

template <typename Lane>
void run_mlp(const Layer& layer, const LayerBatch<Lane>& batch) {
    const bool narrow = batch.fuse_verify_activation();
    const bool wide = batch.fuse_wide_verify_norm();
    if (narrow || wide) {
        if (wide && batch.project_fused_mlp(layer)) return;
        batch.project_group({{layer.mlp_gate, &Workspace::mlp_gate},
                             {layer.mlp_up, &Workspace::mlp_up}});
        if (narrow) {
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
        for (int index = 0; index < batch.count; ++index) {
            auto& work = workspace(batch.lanes[index]);
            check(text_swiglu(work.mlp_gate, work.mlp_up, batch.intermediate_width(),
                              work.mlp_gate, batch.stream), "Apply SwiGLU");
        }
    }
    batch.quantize(&Workspace::mlp_gate, batch.intermediate_width());
    batch.project(layer.mlp_down, &Workspace::hidden);
}

template struct LayerBatch<ForwardLane>;
template void run_mlp(const Layer&, const LayerBatch<ForwardLane>&);
template Workspace& workspace(const ForwardLane&);
template struct LayerBatch<GraphLane>;
template void run_mlp(const Layer&, const LayerBatch<GraphLane>&);
template Workspace& workspace(const GraphLane&);
template struct LayerBatch<VerificationRow>;
template void run_mlp(const Layer&, const LayerBatch<VerificationRow>&);
template struct LayerBatch<BatchGraphLane>;
template void run_mlp(const Layer&, const LayerBatch<BatchGraphLane>&);

}
