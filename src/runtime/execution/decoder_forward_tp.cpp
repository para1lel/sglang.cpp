#include "sglang/decoder_internal.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/backend/mnnvl_rms_quantize.h"
#include "sglang/runtime/backend/tensor_parallel.h"
#include "sglang/runtime/execution/decoder_forward_layers.h"
#include "sglang/runtime/execution/verification_row_copy.h"
#include "sglang/text/mtp/tc_vocab.h"

#include <type_traits>

namespace sglang {
namespace {

using namespace decoder;
using namespace decoder::forward_detail;

template <typename Lane>
LayerBatch<Lane> tp_build_batch(const Lane* lanes, int count, cudaStream_t lane_stream,
                                const CudaBackendPolicy& policy, Bf16* const* attention_output,
                                Bf16* const* hidden_output, bool tensor_parallel,
                                std::uint8_t* packed_activation) {
    LayerBatch<Lane> batch{
        lanes, count, lane_stream,
        policy.ordered_attention,
        policy.tensor_core_decode,
        policy.vector_decode_attention,
        policy.grouped_target_attention,
        policy.target_split_end,
        packed_activation,
    };
    batch.attention_output = attention_output;
    batch.hidden_output = hidden_output;
    batch.tensor_parallel = tensor_parallel;
    return batch;
}

template <typename Rows = MnnvlRmsQuantizeRows>
Rows tp_mnnvl_rms_rows(const LayerBatch<VerificationRow>& batch) {
    Rows rows{};
    if constexpr (std::is_same_v<Rows, MnnvlRmsQuantizeRows>) rows.count = batch.count;
    for (int index = 0; index < batch.count; ++index) {
        auto& work = workspace(batch.lanes[index]);
        rows.entries[index] = {work.residual, work.normalized,
                               batch.projection_activation(index, hidden_size)};
    }
    return rows;
}

struct TpReduceNorm {
    TensorParallel& group;
    cudaStream_t streams[2];
    std::size_t count;
    bool fused = false;
    bool all_replicated = true;
    MnnvlRmsQuantizeRows rows[2];

    template <typename Lane>
    void prepare(const LayerBatch<Lane>& primary, const LayerBatch<Lane>& peer) {
        if constexpr (std::is_same_v<Lane, VerificationRow>) {
            fused = (primary.count == MnnvlRmsQuantizeRows::size ||
                     (primary.singleton_verify_activation && peer.singleton_verify_activation)) &&
                    primary.packs_projection_activation(hidden_size) &&
                    peer.packs_projection_activation(hidden_size);
            if (fused) {
                rows[0] = tp_mnnvl_rms_rows(primary);
                rows[1] = tp_mnnvl_rms_rows(peer);
            }
        }
    }

    bool operator()(Bf16* const buffers[], const Bf16* gain, const Bf16* peer_gain) {
        if (fused && gain) {
            const Bf16* gains[] = {gain, peer_gain};
            fused = group.try_allreduce_rms_quantize(buffers, streams, rows, gains);
            if (fused) {
                all_replicated = false;
                return true;
            }
        }
        constexpr auto one_shot_count = std::size_t(tc_vocab_wide_rows) * hidden_size;
        constexpr auto two_shot_count = 2 * one_shot_count;
        const auto route = group.allreduce_bf16(buffers, streams, count);
        all_replicated &= (count == one_shot_count && route == TpAllReduceRoute::mnnvl_one_shot) ||
                          (count == two_shot_count && route == TpAllReduceRoute::mnnvl_two_shot);
        return false;
    }
};

struct TpReduceNorm40 {
    TensorParallel& group;
    cudaStream_t streams[2];
    MnnvlRmsQuantizeRows40 rows[2];
    bool fused = true;
    int fused_calls = 0;
    int ordinary_calls = 0;

    bool replicated() const {
        constexpr int reductions = 2 * layer_count;
        return (fused_calls == reductions - 1 && ordinary_calls == 1) ||
               (fused_calls == 0 && ordinary_calls == reductions);
    }

    bool operator()(Bf16* const buffers[], const Bf16* gain, const Bf16* peer_gain) {
        if (fused && gain) {
            const Bf16* gains[] = {gain, peer_gain};
            fused = group.try_allreduce_rms_quantize_m40(buffers, streams, rows, gains);
            if (fused) {
                ++fused_calls;
                return true;
            }
        }
        const auto route = group.allreduce_bf16(buffers, streams,
                                                MnnvlRmsQuantizeRows40::size * hidden_size);
        ordinary_calls += route == TpAllReduceRoute::mnnvl_one_shot;
        return false;
    }
};

template <typename Run>
[[gnu::noinline]] bool tp_run_m40(TensorParallel& group,
                                  const LayerBatch<VerificationRow>& primary,
                                  const LayerBatch<VerificationRow>& peer, const Run& run) {
    TpReduceNorm40 reduce{group, {primary.stream, peer.stream},
        {tp_mnnvl_rms_rows<MnnvlRmsQuantizeRows40>(primary),
         tp_mnnvl_rms_rows<MnnvlRmsQuantizeRows40>(peer)}};
    run(reduce);
    return reduce.replicated();
}

template <typename Lane, typename Run>
bool tp_run_reductions(TensorParallel& group, const LayerBatch<Lane>& primary,
                       const LayerBatch<Lane>& peer, const Run& run) {
    if constexpr (std::is_same_v<Lane, VerificationRow>) {
        static_assert(MnnvlRmsQuantizeRows40::size == wide_mtp_rows);
        if (primary.count == wide_mtp_rows && primary.packed_activation &&
            peer.packed_activation && primary.packs_projection_activation(hidden_size) &&
            peer.packs_projection_activation(hidden_size)) {
            return tp_run_m40(group, primary, peer, run);
        }
    }
    TpReduceNorm reduce{group, {primary.stream, peer.stream},
                        static_cast<std::size_t>(primary.count) * hidden_size};
    reduce.prepare(primary, peer);
    run(reduce);
    return reduce.all_replicated;
}

template <typename Lane>
void tp_scatter(LayerBatch<Lane>& batch, int count, cudaStream_t lane_stream,
                const Bf16* reduced) {
    if constexpr (std::is_same_v<Lane, VerificationRow>) {
        static_assert(tp_max_forward_lanes <= VerificationCopyRows::capacity);
        VerificationCopyRows destinations{};
        for (int lane = 0; lane < count; ++lane) {
            destinations.destination[lane] = workspace(batch.lanes[lane]).hidden;
        }
        check(scatter_verification_hidden(reduced, destinations, count, lane_stream),
              "Scatter tensor-parallel verification hidden states");
    } else {
        for (int lane = 0; lane < count; ++lane) {
            copy_cuda_async(workspace(batch.lanes[lane]).hidden, reduced + lane * hidden_size,
                            hidden_size * sizeof(Bf16), cudaMemcpyDeviceToDevice, lane_stream,
                            "Scatter tensor-parallel hidden state");
        }
    }
}

}

void Decoder::Model::forward_tp(
    const Model& peer, const decoder::ForwardLane* primary_lanes,
    const decoder::ForwardLane* peer_lanes, int count, cudaStream_t stream,
    cudaStream_t peer_stream, decoder::TensorParallel& group,
    const decoder::TpPartials& primary_partials, const decoder::TpPartials& peer_partials) const {
    if (!primary_lanes || !peer_lanes || count < 1 || count > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid decoder batch width");
    }
    embed_forward_lanes(embedding, primary_lanes, count, stream, "Look up token embedding");
    decoder::DeviceScope peer_scope(peer.device);
    embed_forward_lanes(peer.embedding, peer_lanes, count, peer_stream,
                        "Look up peer token embedding");
    peer_scope.restore();
    forward_layers_tp(peer, primary_lanes, peer_lanes, count, stream, peer_stream, group,
                      primary_partials, peer_partials);
}

template <typename Lane>
bool Decoder::Model::forward_layers_tp(
    const Model& peer, const Lane* primary_lanes, const Lane* peer_lanes, int count,
    cudaStream_t stream, cudaStream_t peer_stream, decoder::TensorParallel& group,
    const decoder::TpPartials& primary_partials, const decoder::TpPartials& peer_partials,
    std::uint8_t* primary_packed, std::uint8_t* peer_packed, bool singleton_verify) const {
    if (!primary_lanes || !peer_lanes || count < 1 || count > tp_max_forward_lanes) {
        throw std::invalid_argument("Invalid tensor-parallel batch width");
    }
    static_assert(tp_max_forward_lanes >= verification_packet_rows);
    std::array<Bf16*, tp_max_forward_lanes> primary_attention{};
    std::array<Bf16*, tp_max_forward_lanes> primary_hidden{};
    std::array<Bf16*, tp_max_forward_lanes> peer_attention{};
    std::array<Bf16*, tp_max_forward_lanes> peer_hidden{};
    for (int index = 0; index < count; ++index) {
        primary_attention[index] = primary_partials.attention + index * hidden_size;
        primary_hidden[index] = primary_partials.hidden + index * hidden_size;
        peer_attention[index] = peer_partials.attention + index * hidden_size;
        peer_hidden[index] = peer_partials.hidden + index * hidden_size;
    }
    auto primary = tp_build_batch(primary_lanes, count, stream, backend,
                                  primary_attention.data(), primary_hidden.data(),
                                  tp != nullptr, primary_packed);
    auto peer_batch = tp_build_batch(peer_lanes, count, peer_stream, peer.backend,
                                     peer_attention.data(), peer_hidden.data(),
                                     tp != nullptr, peer_packed);
    if constexpr (std::is_same_v<Lane, VerificationRow>) {
        const bool narrow = singleton_verify && count >= 3 && count <= fused_verify_rows &&
            primary.tensor_parallel && peer_batch.tensor_parallel && primary.target_tc &&
            peer_batch.target_tc && !primary_packed && !peer_packed &&
            !primary_lanes[0].binding && !peer_lanes[0].binding;
        primary.singleton_verify_activation = narrow;
        peer_batch.singleton_verify_activation = narrow;
    }
    Bf16* attention_reduce[2] = {primary_partials.attention, peer_partials.attention};
    Bf16* hidden_reduce[2] = {primary_partials.hidden, peer_partials.hidden};
    const auto run = [&](auto& reduce_norm) {
        bool input_fused = false;

        for (int index = 0; index < layer_count; ++index) {
            const auto [full_attention, bank] = qwen_text_model.layer_mapping(index);
            const auto& layer = layers[index];
            const auto& peer_layer = peer.layers[index];
            const bool first = index == 0;
            if (!input_fused) {
                primary.normalize(layer.input_norm, "Normalize layer input",
                                  first ? nullptr : &Workspace::hidden);
            }
            {
                decoder::DeviceScope peer_scope(peer.device);
                if (!input_fused) {
                    peer_batch.normalize(peer_layer.input_norm, "Normalize peer layer input",
                                         first ? nullptr : &Workspace::hidden);
                }
                if (full_attention) {
                    run_gqa(peer_layer.gqa, peer_batch, bank);
                } else {
                    run_gdn(peer_layer.gdn, peer_batch, bank);
                }
            }
            if (full_attention) {
                run_gqa(layer.gqa, primary, bank);
            } else {
                run_gdn(layer.gdn, primary, bank);
            }
            const bool attention_fused = reduce_norm(attention_reduce, layer.attention_norm,
                                                     peer_layer.attention_norm);
            if (!attention_fused) {
                primary.normalize(layer.attention_norm, "Normalize attention output",
                                  &Workspace::projected_attention);
            }
            {
                decoder::DeviceScope peer_scope(peer.device);
                if (!attention_fused) {
                    peer_batch.normalize(peer_layer.attention_norm,
                                         "Normalize peer attention output",
                                         &Workspace::projected_attention);
                }
                run_mlp(peer_layer, peer_batch);
            }
            run_mlp(layer, primary);
            const bool last = index + 1 == layer_count;
            input_fused = reduce_norm(hidden_reduce, last ? nullptr : layers[index + 1].input_norm,
                                      last ? nullptr : peer.layers[index + 1].input_norm);
        }
    };
    const bool replicated = tp_run_reductions(group, primary, peer_batch, run);

    tp_scatter(primary, count, stream, primary_partials.hidden);
    decoder::DeviceScope peer_scope(peer.device);
    tp_scatter(peer_batch, count, peer_stream, peer_partials.hidden);
    peer_scope.restore();
    return replicated;
}

template bool Decoder::Model::forward_layers_tp<decoder::ForwardLane>(
    const Model&, const decoder::ForwardLane*, const decoder::ForwardLane*, int, cudaStream_t,
    cudaStream_t, decoder::TensorParallel&, const decoder::TpPartials&,
    const decoder::TpPartials&, std::uint8_t*, std::uint8_t*, bool) const;
template bool Decoder::Model::forward_layers_tp<decoder::VerificationRow>(
    const Model&, const decoder::VerificationRow*, const decoder::VerificationRow*, int,
    cudaStream_t, cudaStream_t, decoder::TensorParallel&, const decoder::TpPartials&,
    const decoder::TpPartials&, std::uint8_t*, std::uint8_t*, bool) const;

}
