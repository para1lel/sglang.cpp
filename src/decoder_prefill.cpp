#include "sglang/runtime/execution/decoder_prefill_plan.h"

#include "sglang/decoder_prefill.h"
#include "sglang/decoder_batch.h"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/runtime/backend/tensor_parallel.h"
#include "sglang/prefill_fp8.h"
#include "sglang/runtime/execution/decoder_prefill_tiles.h"
#include "sglang/runtime/execution/decoder_prefill_staged_gqa.h"
#include "sglang/text/decoder_logprobs.h"
#include "sglang/text/gdn_local.h"
#include "sglang/text/prefill_attn_sm100.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace sglang {
namespace {

using namespace decoder;
using namespace decoder::prefill_detail;

template <typename Run>
void visit_layers(const std::array<Layer, layer_count>& layers, Run run) {
    for (int index = 0; index < layer_count; ++index) {
        const auto route = qwen_text_model.layer_mapping(index);
        const auto input = [first = index == 0](const auto& work) {
            return first ? LayerResidual{work.residual, nullptr}
                         : LayerResidual{work.hidden, work.residual};
        };
        const auto output = [](const auto& work) {
            return LayerResidual{work.projected_attention, work.residual};
        };
        run(layers[index], index, route, input, output);
    }
}

template <typename Normalize, typename Gqa, typename Gdn, typename Mlp>
void run_layers(const std::array<Layer, layer_count>& layers, Normalize normalize,
                Gqa gqa, Gdn gdn, Mlp mlp) {
    visit_layers(layers, [&](const Layer& layer, int, auto route, auto input, auto output) {
        normalize(layer.input_norm, "Normalize layer input", input);
        if (route.full_attention) gqa(layer.gqa, route.bank);
        else gdn(layer.gdn, route.bank);
        normalize(layer.attention_norm, "Normalize post-attention residual", output);
        mlp(layer);
    });
}

template <typename Normalize, typename Attention, typename Mlp>
void run_tp_layers(const std::array<Layer, layer_count>& primary_layers,
                   const std::array<Layer, layer_count>& peer_layers, Normalize normalize,
                   Attention attention, Mlp mlp) {
    visit_layers(primary_layers,
        [&](const Layer& primary, int index, auto route, auto input, auto output) {
            const auto& peer = peer_layers[index];
            normalize(primary.input_norm, peer.input_norm, input, !route.full_attention);
            if (route.full_attention) attention(primary.gqa, peer.gqa, route.bank);
            else attention(primary.gdn, peer.gdn, route.bank);
            normalize(primary.attention_norm, peer.attention_norm, output, false);
            mlp(primary, peer);
        });
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

template <typename Group, typename Native>
bool schedule_partial_projection_groups(const PrefillGrant* grants, int count, cudaStream_t stream,
                                       DecoderBatchStepStats& stats, Group group, Native native,
                                       int min_rows = min_partial_prefill_rows,
                                       int staged_visible = 0) {
    using Callbacks = std::pair<Group, Native>;
    const Callbacks callbacks{std::move(group), std::move(native)};
    const PartialProjectionSchedule schedule{
        &callbacks,
        [](const void* state, int source, int offset, const PrefillGrant* group_grants,
           PrefillGroup rows, PrefillWorkspace& workspace, PrefillAttentionMode mode,
           int capacity) {
            static_cast<const Callbacks*>(state)->first(source, offset, group_grants, rows,
                                                        workspace, mode, capacity);
        },
        [](const void* state, int source, const PrefillGrant* grant, PrefillWorkspace& workspace,
           int capacity, int offset) {
            static_cast<const Callbacks*>(state)->second(source, grant, workspace, capacity,
                                                         offset);
        },
    };
    return execute_partial_projection_groups(grants, count, stream, stats, schedule, min_rows,
                                             staged_visible);
}
struct PrefillGroupPlan {
    bool grouped = false;
    bool causal = false;
    bool full_causal = false;
    bool serial_attention = false;
    bool wide_partial = false;
    bool staged_group = false;
    bool packed_preparation = false;
    std::optional<PrefillPartners> partners;
    RmsQuantizeLaunch rms_launch = RmsQuantizeLaunch::ordinary;
};

PrefillGroupPlan plan_prefill_group(const PrefillGrant* grants, PrefillGroup group,
                                    PrefillAttentionMode attention_mode) {
    PrefillGroupPlan plan;
    plan.grouped = group.rows > max_layerwise_rows;
    plan.causal = attention_mode == PrefillAttentionMode::causal_segments;
    plan.full_causal = plan.causal && group.rows == grouped_prefill_max_rows;
    plan.serial_attention = plan.causal ||
        attention_mode == PrefillAttentionMode::serial_segments;
    plan.wide_partial = !plan.causal && plan.serial_attention &&
        is_full_partial_prefill_group(grants, group);
    const bool full_group = attention_mode == PrefillAttentionMode::full_group;
    plan.staged_group = plan.grouped && !plan.serial_attention &&
        can_batch_staged_gqa(grants, group.grants);
    plan.partners = full_group && !plan.staged_group
        ? mixed_prefill_partners(grants, group.grants) : std::nullopt;
    plan.packed_preparation = full_group || plan.wide_partial;
    plan.rms_launch = plan.packed_preparation ? RmsQuantizeLaunch::full_group_prefill
                                              : RmsQuantizeLaunch::ordinary;
    return plan;
}

void run_group_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                   const PrefillGrant* grants, int count, const PrefillGroupPlan& plan,
                   int serial_capacity) {
    if (plan.serial_attention) {
        prefill_serial_gqa(chunk, layer, bank, grants, count, serial_capacity, plan.causal,
                           !plan.wide_partial && !plan.full_causal);
    } else if (plan.staged_group) {
        prefill_group_gqa(chunk, layer, bank, grants, count, 0, true);
    } else if (plan.partners) {
        prefill_mixed_gqa(chunk, layer, bank, grants, *plan.partners);
    } else {
        each_attention_group(grants, count, [&](const PrefillGrant* first, int lanes, int offset) {
            prefill_group_gqa(chunk, layer, bank, first, lanes, offset);
        });
    }
}

void run_group_gdn(const Chunk& chunk, const GdnLayer& layer, int bank,
                   const PrefillGrant* grants, int count, const PrefillGroupPlan& plan) {
    if (plan.serial_attention) {
        prefill_partial_gdn(chunk, layer, bank, grants, count, plan.wide_partial);
    } else if (plan.grouped) {
        prefill_group_gdn(chunk, layer, bank, grants, count, 0);
    } else {
        each_attention_group(grants, count, [&](const PrefillGrant* first, int lanes, int offset) {
            prefill_group_gdn(chunk, layer, bank, first, lanes, offset);
        });
    }
}

void gate_group_gqa(const Chunk& chunk, const PrefillGrant* grants, int count,
                    const PrefillGroupPlan& plan) {
    if (plan.wide_partial) {
        each_prefill_segment(grants, count, 0, [&](const PrefillGrant& grant, int row) {
            prefill_attention_segment(chunk, grant, row).gqa_gate();
        });
    } else if (!plan.serial_attention || plan.full_causal) {
        chunk.gqa_gate();
    }
}

void finish_prefill_group(const DecoderModel& model, const Chunk& chunk,
                          const PrefillGrant* grants, int count, const TileBuffers& buffers,
                          cudaStream_t stream) {
    int offset = 0;
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
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
        capture_prefill_checkpoints(grant, &buffers, offset, stream);
        publish_prefill_checkpoints(grant);
        offset += grant.rows;
    }
}
template <typename Enqueue>
auto on_device(int device, Enqueue&& enqueue) {
    const DeviceScope scope(device);
    return enqueue();
}

struct PrefillGroupTp {
    const PrefillGrant* grants;
    const PrefillGrant* peer_grants;
    PrefillGroup group;
    const PrefillGroupPlan& plan;
    int serial_capacity;
    const Chunk& chunk;
    const Chunk& peer_chunk;
    cudaStream_t stream;
    cudaStream_t peer_stream;
    TensorParallel* comm;
    int peer_device;
    bool scaled_output;
    bool reuse_all_gqa;
    const PrefillPeerNorm& peer_norm;
    template <typename Enqueue>
    auto on_peer(Enqueue&& enqueue) const {
        const DeviceScope scope(peer_device);
        return enqueue();
    }

    void reduce(Bf16* primary, Bf16* mirror) const {
        Bf16* const reduce_buffers[] = {primary, mirror};
        const cudaStream_t reduce_streams[] = {stream, peer_stream};
        comm->allreduce_bf16(reduce_buffers, reduce_streams,
                             std::size_t(group.rows) * hidden_size);
    }

    template <typename Residuals>
    void normalize(const Bf16* weight, const Bf16* peer_weight,
                   const Residuals& residuals, bool save_normalized) const {
        const auto [branch, residual] = residuals(chunk.work);
        const auto peer_state = residuals(peer_chunk.work);
        if (residual) {
            const PrefillPeerNormRow root{branch, residual, weight,
                save_normalized ? chunk.work.normalized : nullptr, chunk.activation};
            const PrefillPeerNormRow mirror{peer_state.branch, peer_state.residual, peer_weight,
                save_normalized ? peer_chunk.work.normalized : nullptr, peer_chunk.activation};
            if (peer_norm.enqueue({root, mirror})) return;
            reduce(const_cast<Bf16*>(branch), const_cast<Bf16*>(peer_state.branch));
        }
        chunk.normalize_tp(branch, residual, weight, plan.rms_launch, save_normalized);
        on_peer([&] {
            peer_chunk.normalize_tp(peer_state.branch, peer_state.residual, peer_weight,
                                    plan.rms_launch, save_normalized);
        });
    }

    void attention(const GqaLayer& layer, const GqaLayer& peer_layer, int bank);
    void attention(const GdnLayer& layer, const GdnLayer& peer_layer, int bank);

    void mlp(const Layer& layer, const Layer& peer_layer, bool last) const {
        chunk.mlp(layer);
        on_peer([&] { peer_chunk.mlp(peer_layer); });
        if (last) reduce(chunk.work.hidden, peer_chunk.work.hidden);
    }
};

void PrefillGroupTp::attention(const GqaLayer& layer, const GqaLayer& peer_layer, int bank) {
    const auto& buffers = chunk.work;
    const auto& peer_buffers = peer_chunk.work;
    const auto& dims = buffers.dims;
    const auto& peer_dims = peer_buffers.dims;
    chunk.gqa_input(layer, reuse_all_gqa);
    on_peer([&] { peer_chunk.gqa_input(peer_layer, reuse_all_gqa); });
    run_group_gqa(chunk, layer, bank, grants, group.grants, plan, serial_capacity);
    on_peer([&] {
        run_group_gqa(peer_chunk, peer_layer, bank, peer_grants, group.grants, plan,
                      serial_capacity);
    });
    if (plan.packed_preparation || plan.full_causal) {
        const auto primary = try_prefill_gqa_postgate_quantize(
            group.rows, buffers.q_gate, buffers.attention, buffers.activation, stream,
            dims.gqa_geometry());
        const auto mirror = on_peer([&] {
            return try_prefill_gqa_postgate_quantize(
                group.rows, peer_buffers.q_gate, peer_buffers.attention,
                peer_buffers.activation, peer_stream, peer_dims.gqa_geometry());
        });
        if (primary && mirror) {
            check(*primary, "Gate and quantize prefill grouped-query attention");
            check(*mirror, "Gate and quantize prefill grouped-query attention");
            chunk.project_scaled(layer.output, {buffers.activation.values,
                buffers.activation.scales}, buffers.projected_attention, scaled_output);
            on_peer([&] {
                peer_chunk.project_scaled(peer_layer.output,
                    {peer_buffers.activation.values, peer_buffers.activation.scales},
                    peer_buffers.projected_attention, scaled_output);
            });
            return;
        }
        if (primary) check(*primary, "Gate and quantize prefill grouped-query attention");
        if (mirror) check(*mirror, "Gate and quantize prefill grouped-query attention");
    }
    if (plan.wide_partial || !plan.serial_attention || plan.full_causal) {
        gate_group_gqa(chunk, grants, group.grants, plan);
        on_peer([&] { gate_group_gqa(peer_chunk, peer_grants, group.grants, plan); });
    }
    chunk.project_scaled(layer.output, chunk.quantize(buffers.attention, dims.q_width),
                         buffers.projected_attention, scaled_output);
    on_peer([&] {
        peer_chunk.project_scaled(peer_layer.output,
                                  peer_chunk.quantize(peer_buffers.attention, peer_dims.q_width),
                                  peer_buffers.projected_attention, scaled_output);
    });
}

void PrefillGroupTp::attention(const GdnLayer& layer, const GdnLayer& peer_layer, int bank) {
    const auto& buffers = chunk.work;
    const auto& peer_buffers = peer_chunk.work;
    chunk.gdn_input(layer);
    on_peer([&] { peer_chunk.gdn_input(peer_layer); });
    run_group_gdn(chunk, layer, bank, grants, group.grants, plan);
    on_peer([&] {
        const auto& recurrent_layer = plan.serial_attention || plan.grouped ? peer_layer : layer;
        run_group_gdn(peer_chunk, recurrent_layer, bank, peer_grants, group.grants, plan);
    });
    chunk.project_scaled(layer.output, {buffers.activation.values,
        buffers.activation.scales}, buffers.projected_attention, scaled_output);
    on_peer([&] {
        peer_chunk.project_scaled(peer_layer.output, {peer_buffers.activation.values,
            peer_buffers.activation.scales}, peer_buffers.projected_attention, scaled_output);
    });
}

void finish_prefill_group_tp(const DecoderModel& model, const Chunk& chunk,
                             const Chunk& peer_chunk, const PrefillGrant* grants,
                             const PrefillGrant* peer_grants, int count, ScratchLayout& layout,
                             const PrefillWorkspace& workspace, cudaStream_t stream,
                             cudaStream_t peer_stream, int peer_device) {
    int seed_tile_rows = 0;
    int seed_kv_rows = 0;
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        if (grant.draft && grant.prompt_remaining > 1) {
            seed_tile_rows = std::max(seed_tile_rows,
                                      std::min({grant.rows, grant.prompt_remaining - 1,
                                                prefill_chunk_rows}));
            seed_kv_rows = std::max(seed_kv_rows,
                                    std::min(grant.rows, grant.prompt_remaining - 1));
        }
    }
    Bf16* seed_memory = nullptr;
    if (seed_tile_rows) {
        const std::size_t available = workspace.layout_bytes() - layout.bytes;
        if (seed_scratch_bytes(seed_tile_rows, seed_kv_rows) > available) {
            throw std::runtime_error("Prefill workspace too small for the MTP draft seed scratch");
        }
        seed_memory =
            layout.take_bf16(seed_scratch_bytes(seed_tile_rows, seed_kv_rows) / sizeof(Bf16));
    }
    int offset = 0;
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        const auto& peer_grant = peer_grants[index];
        if (grant.draft && grant.prompt_remaining > 1) {
            auto seed = chunk;
            seed.position = grant.position;
            seed.work.hidden += offset * hidden_size;
            seed.work.residual += offset * hidden_size;
            const bool wmma = model.backend.draft.wmma_seed_fc &&
                grant.position + grant.prompt_remaining >= wmma_seed_min_prompt_tokens;
            const auto scratch = make_seed_scratch(seed_memory, seed_tile_rows, seed_kv_rows);
            seed.seed_draft(*grant.draft, grant.tokens + 1,
                            std::min(grant.rows, grant.prompt_remaining - 1),
                            model.final_norm, wmma, grant.media, &scratch);
        }
        capture_prefill_scores(model, grant, &chunk.work, offset, stream);
        stage_prefill_row(*grant.request, chunk.work, offset + grant.rows - 1, stream);
        capture_prefill_checkpoints(grant, &chunk.work, offset, stream);
        on_device(peer_device, [&] {
            stage_prefill_row(*peer_grant.request, peer_chunk.work,
                              offset + grant.rows - 1, peer_stream);
            capture_prefill_checkpoints(peer_grant, &peer_chunk.work, offset, peer_stream);
        });
        publish_prefill_checkpoints(grant);
        offset += grant.rows;
    }
}

PairedPrefillScratch paired_long_staging(const PrefillWorkspace& local,
                                        const PrefillWorkspace& peer) {
    const auto scratch = local.long_gqa_staging();
    return scratch.keys && peer.long_gqa_staging().keys ? scratch : PairedPrefillScratch{};
}

void prefill_layerwise_tp(const DecoderModel& model, const DecoderModel& peer,
                          const PrefillGrant* grants, const PrefillGrant* peer_grants,
                          PrefillGroup group, const PrefillGroupPlan& plan, int serial_capacity,
                          const Chunk& chunk, ScratchLayout& layout, PrefillWorkspace& workspace,
                          PrefillWorkspace& peer_workspace, bool scaled_output,
                          bool reuse_all_gqa, cudaStream_t stream, cudaStream_t peer_stream,
                          TensorParallel& comm) {
    const auto peer_dims = PrefillDims::from_layers(peer.layers);
    ScratchLayout peer_layout{static_cast<std::byte*>(peer_workspace.data())};
    const TileBuffers peer_buffers(peer_layout, peer_workspace.capacity(), peer_dims);
    const Chunk peer_chunk{{peer_buffers.activation, group.rows, peer_stream,
                            chunk.tensor_core_fp8, plan.grouped || plan.serial_attention},
                           peer_buffers, 0, nullptr, 0,
                           paired_long_staging(peer_workspace, workspace)};
    PrefillPeerNorm peer_norm;
    if (plan.full_causal && chunk.tensor_core_fp8 && comm.rank_count() == 2 &&
        comm.device(0) == model.device && comm.device(1) == peer.device) {
        peer_norm.prepare(workspace, peer_workspace, model.device, peer.device);
    }
    PrefillGroupTp run{grants, peer_grants, group, plan, serial_capacity, chunk, peer_chunk,
                       stream, peer_stream, &comm, peer.device, scaled_output,
                       reuse_all_gqa, peer_norm};
    each_prefill_segment(grants, group.grants, 0, [&](const PrefillGrant& grant, int offset) {
        auto* residual = chunk.work.residual + offset * hidden_size;
        gather_embeddings(model.embedding, grant.tokens, grant.rows, residual, stream);
    });
    on_device(peer.device, [&] {
        each_prefill_segment(peer_grants, group.grants, 0,
                             [&](const PrefillGrant& grant, int offset) {
            auto* residual = peer_buffers.residual + offset * hidden_size;
            gather_embeddings(peer.embedding, grant.tokens, grant.rows, residual, peer_stream);
        });
    });
    run_tp_layers(model.layers, peer.layers,
        [&](const Bf16* weight, const Bf16* peer_weight, const auto& residuals,
            bool save_normalized) {
            run.normalize(weight, peer_weight, residuals, save_normalized);
        },
        [&](const auto& layer, const auto& peer_layer, int bank) {
            run.attention(layer, peer_layer, bank);
        },
        [&](const Layer& layer, const Layer& peer_layer) {
            run.mlp(layer, peer_layer, &layer == &model.layers.back());
        });
    finish_prefill_group_tp(model, chunk, peer_chunk, grants, peer_grants, group.grants, layout,
                            workspace, stream, peer_stream, peer.device);
}

void prefill_layerwise(const DecoderModel& model, const PrefillGrant* grants, PrefillGroup group,
                       PrefillWorkspace& workspace, PrefillAttentionMode attention_mode,
                       int serial_capacity = 0, const DecoderModel* peer = nullptr,
                       const PrefillGrant* peer_grants = nullptr,
                       PrefillWorkspace* peer_workspace = nullptr,
                       cudaStream_t peer_stream = nullptr, TensorParallel* comm = nullptr) {
    const auto stream = workspace.stream();
    ScratchLayout layout{static_cast<std::byte*>(workspace.data())};
    const auto dims = PrefillDims::from_layers(model.layers);
    const TileBuffers buffers(layout, workspace.capacity(), dims);
    const auto plan = plan_prefill_group(grants, group, attention_mode);
    const bool tensor_core = peer
        ? model.backend.tensor_core_prefill && peer->backend.tensor_core_prefill
        : true;
    const Chunk chunk{{buffers.activation, group.rows, stream, tensor_core,
                       plan.grouped || plan.serial_attention}, buffers, 0, nullptr, 0,
                      peer ? paired_long_staging(workspace, *peer_workspace)
                           : PairedPrefillScratch{}};
    const bool scaled_output = group.rows == grouped_prefill_max_rows;
    const bool reuse_all_gqa = std::all_of(grants, grants + group.grants,
        [](const PrefillGrant& grant) {
            return grant.position + grant.rows <= grant.request->cached_gqa_tokens;
        });
    if (peer) {
        prefill_layerwise_tp(model, *peer, grants, peer_grants, group, plan, serial_capacity,
                             chunk, layout, workspace, *peer_workspace, scaled_output,
                             reuse_all_gqa, stream, peer_stream, *comm);
        return;
    }
    each_prefill_segment(grants, group.grants, 0, [&](const PrefillGrant& grant, int offset) {
        auto* residual = buffers.residual + offset * hidden_size;
        gather_embeddings(model.embedding, grant.tokens, grant.rows, residual, stream);
        if (grant.media) grant.media->substitute(residual, grant.position, grant.rows, stream);
    });
    run_layers(model.layers,
        [&](const Bf16* weight, const char*, const auto& residuals) {
            const auto [branch, residual] = residuals(buffers);
            chunk.normalize(branch, residual, weight, plan.rms_launch);
        },
        [&](const GqaLayer& layer, int bank) {
            chunk.gqa_input(layer, reuse_all_gqa);
            run_group_gqa(chunk, layer, bank, grants, group.grants, plan, serial_capacity);
            if (plan.packed_preparation || plan.full_causal) {
                if (const auto status = try_prefill_gqa_postgate_quantize(
                        group.rows, buffers.q_gate, buffers.attention, buffers.activation,
                        stream)) {
                    check(*status, "Gate and quantize prefill grouped-query attention");
                    chunk.project_scaled(layer.output, {buffers.activation.values,
                        buffers.activation.scales}, buffers.projected_attention, scaled_output);
                    return;
                }
            }
            gate_group_gqa(chunk, grants, group.grants, plan);
            chunk.project_scaled(layer.output, chunk.quantize(buffers.attention, dims.q_width),
                                 buffers.projected_attention, scaled_output);
        },
        [&](const GdnLayer& layer, int bank) {
            chunk.gdn_input(layer);
            run_group_gdn(chunk, layer, bank, grants, group.grants, plan);
            chunk.project_scaled(layer.output, {buffers.activation.values,
                buffers.activation.scales}, buffers.projected_attention, scaled_output);
        },
        [&](const Layer& layer) { chunk.mlp(layer); });
    finish_prefill_group(model, chunk, grants, group.grants, buffers, stream);
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
            record_projection_group(stats, rows);
        }
    }
    workspace.close();
    return true;
}

void enqueue_native_prefill_grants(const DecoderModel& model, const PrefillGrant* grants,
                                   int count, PrefillWorkspace& workspace, int capacity,
                                   int first_offset = 0);

}

bool decoder::try_prefill_projection_groups(
    const DecoderModel& model, const PrefillGrant* grants, int count, cudaStream_t stream,
    DecoderBatchStepStats& stats) {
    constexpr int full_group_lanes = grouped_prefill_max_rows / batch_prefill_tile_rows;
    constexpr int max_paired_grant_rows = 8192;
    if (model.backend.paired_prefill && count % 2 == 0 && full_tile_prefill_grants(grants, count)) {
        if (count == full_group_lanes &&
            prefill_projection_tiles(model, grants, count, full_group_lanes, stream, stats)) {
            return true;
        }
        if (grants[0].rows <= max_paired_grant_rows / count) {
            return prefill_projection_tiles(model, grants, count, 2, stream, stats);
        }
    }
    if (!model.backend.tensor_core_prefill) return false;
    return schedule_partial_projection_groups(grants, count, stream, stats,
        [&](int, int, const PrefillGrant* group, PrefillGroup rows,
            PrefillWorkspace& workspace, PrefillAttentionMode attention, int capacity) {
            prefill_layerwise(model, group, rows, workspace, attention, capacity);
        },
        [&](int, const PrefillGrant* grant, PrefillWorkspace& workspace, int capacity,
            int offset) {
            enqueue_native_prefill_grants(model, grant, 1, workspace, capacity, offset);
        });
}

void Decoder::Model::prefill_chunk(Request& request, const int* tokens, int rows, int position,
                                   PrefillWorkspace& workspace, MtpDraftRequest* draft,
                                   int prompt_remaining, const MediaPrefill* media,
                                   const PromptLogprobCapture* scores, int logical_capacity,
                                   const PrefillGrant* checkpoint_grant) const {
    const auto stream = workspace.stream();
    const int capacity = logical_capacity ? logical_capacity : workspace.capacity();
    PrefillGrant grant{&request, tokens, position, rows, rows, prompt_remaining, draft, media};
    grant.scores = scores;
    if (checkpoint_grant != nullptr) {
        grant.checkpoints = checkpoint_grant->checkpoints;
        grant.checkpoint = checkpoint_grant->checkpoint;
    }
    validate_prefill_tile(grant, workspace, capacity);
    if (rows == 1 || !workspace.data()) {
        if (!std::all_of(tokens, tokens + rows,
                         [](int token) { return token >= 0 && token < vocab_size; })) {
            throw std::invalid_argument("Invalid decoder prefill token");
        }
        int pending = 0;
        const int first = scores ? scores->start : position;
        const int end = scores ? scores->score_end(position, rows) : position;
        auto checkpoints = prefill_checkpoints(grant);
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
            if (checkpoints.size != 0 && checkpoints.data[0].end == token_position + 1) {
                auto current = grant;
                current.position = token_position;
                current.rows = 1;
                current.checkpoints = {checkpoints.data, 1};
                capture_prefill_checkpoints(current, nullptr, 0, stream);
                publish_prefill_checkpoints(current);
                ++checkpoints.data;
                --checkpoints.size;
            }
        }
        if (scores) scores->collect(pending, stream);
        return;
    }
    ScratchLayout layout{static_cast<std::byte*>(workspace.data())};
    const auto dims = PrefillDims::from_layers(layers);
    const TileBuffers buffers(layout, capacity, dims);
    const bool tensor_core = backend.tensor_core_prefill;
    const Chunk chunk{{buffers.activation, rows, stream, tensor_core}, buffers, position,
        media ? media->positions + position : nullptr, request.cached_gqa_tokens};
    const auto& work = chunk.work;
    gather_embeddings(embedding, tokens, rows, work.residual, stream);
    if (media) {
        media->substitute(work.residual, position, rows, stream);
    }
    const auto normalize = [&](const Bf16* weight, const char*, const auto& residuals) {
        const auto [branch, residual] = residuals(work);
        chunk.normalize(branch, residual, weight);
    };
    run_layers(
        layers, normalize,
        [&](const GqaLayer& layer, int bank) {
            chunk.gqa_input(layer);
            chunk.gqa_attention(layer, request.caches[bank], request.pages, work.gqa_scratch());
            chunk.project(layer.output, chunk.quantize(work.attention, dims.q_width),
                          work.projected_attention);
        },
        [&](const GdnLayer& layer, int bank) {
            chunk.gdn_input(layer);
            const GdnPrefillCheckpoints checkpoints(grant, bank);
            chunk.gdn_attention(layer, request.states[bank], bank, checkpoints.view());
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
    capture_prefill_checkpoints(grant, &work, 0, stream);
    publish_prefill_checkpoints(grant);
}

namespace {

void prefill_chunk_tp(const DecoderModel& model, const DecoderModel& peer,
                      const PrefillGrant& grant, const PrefillGrant& peer_grant,
                      PrefillWorkspace& workspace, PrefillWorkspace& peer_workspace,
                      int capacity, cudaStream_t stream, cudaStream_t peer_stream,
                      TensorParallel& group) {
    auto& request = *grant.request;
    auto& peer_request = *peer_grant.request;
    const int rows = grant.rows;
    if (!workspace.data() || !peer_workspace.data()) {
        throw std::runtime_error("Prefill workspace unavailable under tensor parallelism");
    }
    ScratchLayout layout{static_cast<std::byte*>(workspace.data())};
    ScratchLayout peer_layout{static_cast<std::byte*>(peer_workspace.data())};
    const auto dims = PrefillDims::from_layers(model.layers);
    const auto peer_dims = PrefillDims::from_layers(peer.layers);
    const TileBuffers buffers(layout, capacity, dims);
    const TileBuffers peer_buffers(peer_layout, peer_workspace.capacity(), peer_dims);
    const bool tensor_core = model.backend.tensor_core_prefill &&
        peer.backend.tensor_core_prefill;
    const Chunk chunk{{buffers.activation, rows, stream, tensor_core}, buffers, grant.position,
                      nullptr, request.cached_gqa_tokens,
                      paired_long_staging(workspace, peer_workspace)};
    const Chunk peer_chunk{{peer_buffers.activation, rows, peer_stream, tensor_core},
                           peer_buffers, grant.position, nullptr, peer_request.cached_gqa_tokens,
                           paired_long_staging(peer_workspace, workspace)};
    const auto& work = chunk.work;
    const auto& peer_work = peer_chunk.work;
    gather_embeddings(model.embedding, grant.tokens, rows, work.residual, stream);
    on_device(peer.device, [&] {
        gather_embeddings(peer.embedding, peer_grant.tokens, rows, peer_work.residual,
                          peer_stream);
    });
    const auto reduce = [&](Bf16* primary, Bf16* mirror) {
        Bf16* const reduce_buffers[] = {primary, mirror};
        const cudaStream_t reduce_streams[] = {stream, peer_stream};
        group.allreduce_bf16(reduce_buffers, reduce_streams, std::size_t(rows) * hidden_size);
    };
    run_tp_layers(model.layers, peer.layers,
        [&](const Bf16* weight, const Bf16* peer_weight, const auto& residuals, bool) {
            const auto [branch, residual] = residuals(buffers);
            chunk.normalize(branch, residual, weight);
            const auto peer_state = residuals(peer_buffers);
            on_device(peer.device, [&] {
                peer_chunk.normalize(peer_state.branch, peer_state.residual, peer_weight);
            });
        },
        [&](const auto& layer, const auto& peer_layer, int bank) {
            if constexpr (std::is_same_v<std::decay_t<decltype(layer)>, GqaLayer>) {
                chunk.gqa_input(layer);
                on_device(peer.device, [&] { peer_chunk.gqa_input(peer_layer); });
                chunk.gqa_attention(layer, request.caches[bank], request.pages,
                                    work.gqa_scratch());
                on_device(peer.device, [&] {
                    peer_chunk.gqa_attention(peer_layer, peer_request.caches[bank],
                                             peer_request.pages, peer_work.gqa_scratch());
                });
                chunk.project(layer.output, chunk.quantize(work.attention, dims.q_width),
                              work.projected_attention);
                on_device(peer.device, [&] {
                    peer_chunk.project(peer_layer.output,
                                       peer_chunk.quantize(peer_work.attention,
                                                           peer_dims.q_width),
                                       peer_work.projected_attention);
                });
            } else {
                chunk.gdn_input(layer);
                on_device(peer.device, [&] { peer_chunk.gdn_input(peer_layer); });
                const GdnPrefillCheckpoints checkpoints(grant, bank);
                chunk.gdn_attention(layer, request.states[bank], bank, checkpoints.view());
                on_device(peer.device, [&] {
                    const GdnPrefillCheckpoints peer_checkpoints(peer_grant, bank);
                    peer_chunk.gdn_attention(peer_layer, peer_request.states[bank], bank,
                                             peer_checkpoints.view());
                });
            }
            reduce(work.projected_attention, peer_work.projected_attention);
        },
        [&](const Layer& layer, const Layer& peer_layer) {
            chunk.mlp(layer);
            on_device(peer.device, [&] { peer_chunk.mlp(peer_layer); });
            reduce(work.hidden, peer_work.hidden);
        });
    seed_ranked_prefill_draft(model, grant, chunk, layout, workspace.layout_bytes());
    capture_prefill_scores(model, grant, &work, 0, stream);
    stage_prefill_row(request, work, rows - 1, stream);
    capture_prefill_checkpoints(grant, &work, 0, stream);
    on_device(peer.device, [&] {
        stage_prefill_row(peer_request, peer_work, rows - 1, peer_stream);
        capture_prefill_checkpoints(peer_grant, &peer_work, 0, peer_stream);
    });
    publish_prefill_checkpoints(grant);
}
void enqueue_native_prefill_grants(const DecoderModel& model, const PrefillGrant* grants,
                                   int count, PrefillWorkspace& workspace, int capacity,
                                   int first_offset) {
    const auto stream = workspace.stream();
    for (int index = 0; index < count; ++index, first_offset = 0) {
        const auto& grant = grants[index];
        if (grant.rows == 1) {
            if (grant.draft && grant.prompt_remaining > 1) {
                model.seed_draft(grant.request->work, *grant.draft, grant.position,
                                 grant.tokens[1], stream, grant.media);
            }
            capture_prefill_scores(model, grant, nullptr, 0, stream);
            capture_prefill_checkpoints(grant, nullptr, 0, stream);
            publish_prefill_checkpoints(grant);
        } else {
            for (int offset = first_offset; offset < grant.rows;) {
                const int rows = std::min({grant.tile_rows, capacity, grant.rows - offset});
                model.prefill_chunk(*grant.request, grant.tokens + offset, rows,
                                    grant.position + offset, workspace, grant.draft,
                                    grant.prompt_remaining - offset, grant.media, grant.scores,
                                    capacity, &grant);
                offset += rows;
            }
        }
    }
}


void enqueue_native_prefill_grants_tp(const DecoderModel& model, const DecoderModel& peer,
                                      const PrefillGrant* grants, const PrefillGrant* peer_grants,
                                      int count, PrefillWorkspace& workspace,
                                      PrefillWorkspace& peer_workspace, int capacity,
                                      int first_offset, cudaStream_t stream,
                                      cudaStream_t peer_stream, TensorParallel& group) {
    for (int index = 0; index < count; ++index, first_offset = 0) {
        const auto& grant = grants[index];
        const auto& peer_grant = peer_grants[index];
        if (grant.rows == 1) {
            if (grant.draft && grant.prompt_remaining > 1) {
                model.seed_draft(grant.request->work, *grant.draft, grant.position,
                                 grant.tokens[1], stream, grant.media);
            }
            capture_prefill_scores(model, grant, nullptr, 0, stream);
            capture_prefill_checkpoints(grant, nullptr, 0, stream);
            if (peer_grant.checkpoints.size != 0) {
                on_device(peer.device, [&] {
                    capture_prefill_checkpoints(peer_grant, nullptr, 0, peer_stream);
                });
            }
            publish_prefill_checkpoints(grant);
        } else {

            for (int offset = first_offset; offset < grant.rows;) {
                const int rows = std::min({grant.tile_rows, capacity, grant.rows - offset});
                PrefillGrant tile = grant;
                PrefillGrant peer_tile = peer_grant;
                tile.tokens += offset;
                tile.position += offset;
                tile.rows = rows;
                tile.tile_rows = rows;
                tile.prompt_remaining = grant.prompt_remaining - offset;
                peer_tile.tokens += offset;
                peer_tile.position += offset;
                peer_tile.rows = rows;
                peer_tile.tile_rows = rows;
                peer_tile.prompt_remaining = peer_grant.prompt_remaining - offset;
                validate_prefill_tile(tile, workspace, capacity);
                prefill_chunk_tp(model, peer, tile, peer_tile, workspace, peer_workspace, capacity,
                                 stream, peer_stream, group);
                offset += rows;
            }
        }
    }
}

bool try_prefill_paired_tiles_tp(const DecoderModel& model, const DecoderModel& peer,
                                const PrefillGrant* grants, const PrefillGrant* peer_grants,
                                int count, cudaStream_t stream, cudaStream_t peer_stream,
                                TensorParallel& group, DecoderBatchStepStats& stats) {
    if (!can_pack_paired_tiles(grants, count) || !can_pack_paired_tiles(peer_grants, count)) {
        return false;
    }
    constexpr auto retention = PrefillWorkspace::Retention::transient;
    PrefillWorkspace workspace(stream, paired_tile_group.rows, retention);
    if (!workspace.data()) return false;
    auto peer_workspace = on_device(peer.device, [&] {
        return PrefillWorkspace(peer_stream, paired_tile_group.rows, retention);
    });
    if (!peer_workspace.data()) {
        workspace.close();
        check(cudaStreamSynchronize(stream), "Release provisional TP prefill workspace");
        return false;
    }
    const auto primary = paired_tile_segments(grants, workspace);
    const auto mirror = paired_tile_segments(peer_grants, peer_workspace);
    prefill_layerwise(model, primary.data(), paired_tile_group, workspace,
                      PrefillAttentionMode::serial_segments, batch_prefill_tile_rows, &peer,
                      mirror.data(), &peer_workspace, peer_stream, &group);
    stats.prefill_workspace_bytes = std::max({stats.prefill_workspace_bytes, workspace.bytes(),
                                              peer_workspace.bytes()});
    record_projection_group(stats, paired_tile_group.rows);
    workspace.close();
    on_device(peer.device, [&] { peer_workspace.close(); });
    return true;
}

bool try_prefill_projection_groups_tp(const DecoderModel& model, const DecoderModel& peer,
                                      const PrefillGrant* grants, const PrefillGrant* peer_grants,
                                      int count, cudaStream_t stream, cudaStream_t peer_stream,
                                      TensorParallel& group, DecoderBatchStepStats& stats) {
    if (!model.backend.tensor_core_prefill || !peer.backend.tensor_core_prefill) return false;
    if (try_prefill_paired_tiles_tp(model, peer, grants, peer_grants, count, stream, peer_stream,
                                    group, stats)) return true;
    const int staged_visible = long_serial_visible(grants, count, true);
    std::optional<PrefillWorkspace> peer_workspace;
    const auto ensure_peer = [&](int capacity) {
        if (peer_workspace) return;
        const DeviceScope scope(peer.device);
        peer_workspace.emplace(peer_stream, capacity, staged_visible);
        if (!peer_workspace->data()) {
            throw std::runtime_error("Prefill workspace unavailable under tensor parallelism");
        }
        stats.prefill_workspace_bytes =
            std::max(stats.prefill_workspace_bytes, peer_workspace->bytes());
    };
    const auto scheduled = schedule_partial_projection_groups(
        grants, count, stream, stats,
        [&](int source, int offset, const PrefillGrant* group_grants, PrefillGroup rows,
            PrefillWorkspace& workspace, PrefillAttentionMode attention, int capacity) {
            ensure_peer(workspace.capacity());
            std::array<PrefillGrant, causal_full_group.grants> peer_segments;
            const PrefillGrant* peer_group = peer_grants + source;
            if (attention == PrefillAttentionMode::causal_segments) {
                peer_causal_segments(peer_grants, source, offset, peer_segments.data(),
                                     rows.grants);
                peer_group = peer_segments.data();
            }
            prefill_layerwise(model, group_grants, rows, workspace, attention, capacity, &peer,
                              peer_group, &*peer_workspace, peer_stream, &group);
        },
        [&](int source, const PrefillGrant* grant, PrefillWorkspace& workspace, int capacity,
            int offset) {
            ensure_peer(workspace.capacity());
            enqueue_native_prefill_grants_tp(model, peer, grant, peer_grants + source,
                                             1, workspace, *peer_workspace, capacity, offset,
                                             stream, peer_stream, group);
        },
        prefill_chunk_rows, staged_visible);
    if (peer_workspace) {
        const DeviceScope scope(peer.device);
        peer_workspace->close();
    }
    return scheduled;
}

}

void Decoder::Model::prefill_grants(const PrefillGrant* grants, int count, cudaStream_t stream,
                                   DecoderBatchStepStats& stats) const {
    const DeviceScope local_scope(device);
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

void Decoder::Model::prefill_grants_tp(const Model& peer, const PrefillGrant* grants,
                                       const PrefillGrant* peer_grants, int count,
                                       cudaStream_t stream, cudaStream_t peer_stream,
                                       TensorParallel& group,
                                       DecoderBatchStepStats& stats) const {
    const DeviceScope local_scope(device);
    if (count == 0) return;
    validate_ranked_prefill_grants({grants, peer_grants}, count);
    if (try_prefill_projection_groups_tp(*this, peer, grants, peer_grants, count, stream,
                                         peer_stream, group, stats)) {
        return;
    }
    int capacity = 1;
    for (int index = 0; index < count; ++index) {
        capacity = std::max(capacity, grants[index].tile_rows);
    }
    const int staged_visible = backend.tensor_core_prefill && peer.backend.tensor_core_prefill
        ? long_serial_visible(grants, count, false) : 0;
    std::optional<PrefillWorkspace> workspace(std::in_place, stream, capacity, staged_visible);
    if (!workspace->data() && capacity > prefill_chunk_rows) {
        workspace.emplace(stream, prefill_chunk_rows, staged_visible);
    }
    stats.prefill_workspace_bytes = std::max(stats.prefill_workspace_bytes, workspace->bytes());
    std::optional<PrefillWorkspace> peer_workspace;
    {
        const DeviceScope scope(peer.device);
        peer_workspace.emplace(peer_stream, workspace->capacity(), staged_visible);
    }
    if (workspace->data() && !peer_workspace->data()) {
        throw std::runtime_error("Prefill workspace unavailable under tensor parallelism");
    }
    stats.prefill_workspace_bytes = std::max(stats.prefill_workspace_bytes,
                                             peer_workspace->bytes());
    enqueue_native_prefill_grants_tp(*this, peer, grants, peer_grants, count, *workspace,
                                     *peer_workspace, workspace->capacity(), 0, stream,
                                     peer_stream, group);
    workspace->close();
    {
        const DeviceScope scope(peer.device);
        peer_workspace->close();
    }
}

}
