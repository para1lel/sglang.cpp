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
#include <optional>
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

template <typename Group, typename Native, typename Prepare>
bool schedule_projection_groups(const PrefillGrant* grants, int count, cudaStream_t stream,
                                DecoderBatchStepStats& stats, Group group, Native native,
                                Prepare prepare, int min_rows, int staged_visible,
                                GqaGeometry geometry, bool resident_staging, bool whole_tiles,
                                bool ranked = false) {
    struct Callbacks {
        Group group;
        Native native;
        Prepare prepare;
    };
    const Callbacks callbacks{std::move(group), std::move(native), std::move(prepare)};
    const PartialProjectionSchedule schedule{
        &callbacks,
        [](const void* state, const PrefillSegment* segments, const PrefillGrant* group_grants,
           PrefillGroup rows, PrefillWorkspace& workspace, PrefillAttentionMode mode,
           int capacity) {
            static_cast<const Callbacks*>(state)->group(segments, group_grants, rows, workspace,
                                                        mode, capacity);
        },
        [](const void* state, int source, const PrefillGrant* grant, PrefillWorkspace& workspace,
           int capacity, int offset) {
            static_cast<const Callbacks*>(state)->native(source, grant, workspace, capacity,
                                                         offset);
        },
        [](const void* state, int capacity) {
            return static_cast<const Callbacks*>(state)->prepare(capacity);
        },
    };
    return execute_partial_projection_groups(grants, count, stream, stats, schedule, min_rows,
                                             geometry, staged_visible, resident_staging,
                                             whole_tiles, ranked);
}
struct PrefillGroupPlan {
    bool native = false;
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
    plan.native = attention_mode == PrefillAttentionMode::native;
    plan.causal = attention_mode == PrefillAttentionMode::causal_segments;
    plan.full_causal = plan.causal && group.rows == grouped_prefill_max_rows;
    plan.serial_attention =
        plan.native || plan.causal || attention_mode == PrefillAttentionMode::serial_segments;
    plan.wide_partial = attention_mode == PrefillAttentionMode::serial_segments &&
                        is_full_partial_prefill_group(grants, group);
    const bool full_group = attention_mode == PrefillAttentionMode::full_group;
    plan.staged_group = !plan.serial_attention && can_batch_staged_gqa(grants, group.grants);
    plan.partners = !plan.serial_attention && !plan.staged_group
                        ? mixed_prefill_partners(grants, group.grants)
                        : std::nullopt;
    plan.packed_preparation = full_group || plan.wide_partial;
    plan.rms_launch = plan.packed_preparation ? RmsQuantizeLaunch::full_group_prefill
                                              : RmsQuantizeLaunch::ordinary;
    return plan;
}

void run_group_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                   const PrefillGrant* grants, int count, const PrefillGroupPlan& plan,
                   int serial_capacity) {
    if (plan.native) {
        const auto& grant = grants[0];
        const auto segment = prefill_attention_segment(chunk, grant, 0);
        segment.gqa_attention(layer, grant.request->caches[bank], grant.request->pages,
                              chunk.work.gqa_scratch());
    } else if (plan.serial_attention) {
        prefill_serial_gqa(chunk, layer, bank, grants, count, serial_capacity, plan.causal,
                           !plan.wide_partial && !plan.full_causal);
    } else if (plan.staged_group) {
        prefill_group_gqa(chunk, layer, bank, grants, count, 0, true);
    } else if (plan.partners) {
        prefill_mixed_gqa(chunk, layer, bank, grants, count, *plan.partners);
    } else {
        prefill_group_gqa(chunk, layer, bank, grants, count, 0);
    }
}

void run_group_gdn(const Chunk& chunk, const GdnLayer& layer, int bank, const PrefillGrant* grants,
                   int count, const PrefillGroupPlan& plan, bool scaled_output) {
    if (plan.native) {
        auto segment = chunk;
        segment.position = grants[0].position;
        const GdnPrefillCheckpoints captures(grants[0], bank);
        segment.gdn_attention(layer, grants[0].request->states[bank], bank, captures.view());
        return;
    }
    if (plan.serial_attention) {
        prefill_partial_gdn(chunk, layer, bank, grants, count, plan.wide_partial);
    } else {
        prefill_group_gdn(chunk, layer, bank, grants, count, 0);
    }
    chunk.project_scaled(layer.output, {chunk.activation.values, chunk.activation.scales},
                         chunk.work.projected_attention, scaled_output);
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

void project_group_gqa(const Chunk& chunk, const GqaLayer& layer, const PrefillGrant* grants,
                       int count, const PrefillGroupPlan& plan, bool scaled_output) {
    const auto& work = chunk.work;
    if (plan.packed_preparation || plan.full_causal) {
        if (const auto status = try_prefill_gqa_postgate_quantize(
                chunk.rows, work.q_gate, work.attention, work.activation, chunk.stream,
                work.dims.gqa_geometry())) {
            check(*status, "Gate and quantize prefill grouped-query attention");
            chunk.project_scaled(layer.output, {work.activation.values, work.activation.scales},
                                 work.projected_attention, scaled_output);
            return;
        }
    }
    gate_group_gqa(chunk, grants, count, plan);
    chunk.project_scaled(layer.output, chunk.quantize(work.attention, work.dims.q_width),
                         work.projected_attention, scaled_output);
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
            const bool wmma =
                chunk.tensor_core_fp8 && model.backend.draft.wmma_seed_fc &&
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
        if (plan.native)
            chunk.normalize(branch, residual, weight, plan.rms_launch);
        else
            chunk.normalize_tp(branch, residual, weight, plan.rms_launch, save_normalized);
        on_peer([&] {
            if (plan.native) {
                peer_chunk.normalize(peer_state.branch, peer_state.residual, peer_weight,
                                     plan.rms_launch);
            } else {
                peer_chunk.normalize_tp(peer_state.branch, peer_state.residual, peer_weight,
                                        plan.rms_launch, save_normalized);
            }
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
    chunk.gqa_input(layer, reuse_all_gqa);
    on_peer([&] { peer_chunk.gqa_input(peer_layer, reuse_all_gqa); });
    run_group_gqa(chunk, layer, bank, grants, group.grants, plan, serial_capacity);
    on_peer([&] {
        run_group_gqa(peer_chunk, peer_layer, bank, peer_grants, group.grants, plan,
                      serial_capacity);
    });
    project_group_gqa(chunk, layer, grants, group.grants, plan, scaled_output);
    on_peer([&] {
        project_group_gqa(peer_chunk, peer_layer, peer_grants, group.grants, plan, scaled_output);
    });
}

void PrefillGroupTp::attention(const GdnLayer& layer, const GdnLayer& peer_layer, int bank) {
    chunk.gdn_input(layer);
    on_peer([&] { peer_chunk.gdn_input(peer_layer); });
    run_group_gdn(chunk, layer, bank, grants, group.grants, plan, scaled_output);
    on_peer([&] {
        run_group_gdn(peer_chunk, peer_layer, bank, peer_grants, group.grants, plan, scaled_output);
    });
}

void finish_prefill_group_tp(const DecoderModel& model, const DecoderModel& peer,
                             const Chunk& chunk, const Chunk& peer_chunk,
                             const std::array<const PrefillGrant*, 2>& ranked_grants, int count,
                             TensorParallel& comm) {
    int offset = 0;
    for (int index = 0; index < count; ++index) {
        const auto& grant = ranked_grants[0][index];
        const auto& peer_grant = ranked_grants[1][index];
        seed_ranked_prefill_draft({&model, &peer}, {&grant, &peer_grant}, {&chunk, &peer_chunk},
                                  offset, comm);
        capture_prefill_scores(model, grant, &chunk.work, offset, chunk.stream);
        stage_prefill_row(*grant.request, chunk.work, offset + grant.rows - 1, chunk.stream);
        capture_prefill_checkpoints(grant, &chunk.work, offset, chunk.stream);
        on_device(peer.device, [&] {
            stage_prefill_row(*peer_grant.request, peer_chunk.work, offset + grant.rows - 1,
                              peer_chunk.stream);
            capture_prefill_checkpoints(peer_grant, &peer_chunk.work, offset, peer_chunk.stream);
        });
        publish_prefill_checkpoints(grant);
        offset += grant.rows;
    }
}

void prefill_layerwise_tp(const DecoderModel& model, const DecoderModel& peer,
                          const PrefillGrant* grants, const PrefillGrant* peer_grants,
                          PrefillGroup group, const PrefillGroupPlan& plan, int serial_capacity,
                          const Chunk& chunk, PrefillWorkspace& workspace,
                          PrefillWorkspace& peer_workspace, bool scaled_output, bool reuse_all_gqa,
                          cudaStream_t stream, cudaStream_t peer_stream, TensorParallel& comm) {
    const auto peer_dims = PrefillDims::from_layers(peer.layers);
    ScratchLayout peer_layout{static_cast<std::byte*>(peer_workspace.data())};
    const TileBuffers peer_buffers(peer_layout, peer_workspace.capacity(), peer_dims);
    const PrefillProjections projections{
        peer_buffers.activation, group.rows, peer_stream, chunk.tensor_core_fp8, !plan.native,
    };
    const Chunk peer_chunk(projections, peer_buffers, 0, nullptr, 0,
                           select_gqa_staging(peer_workspace, &workspace));
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
    finish_prefill_group_tp(model, peer, chunk, peer_chunk, {grants, peer_grants}, group.grants,
                            comm);
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
    const TileBuffers buffers(layout,
                              attention_mode == PrefillAttentionMode::native ? serial_capacity
                                                                             : workspace.capacity(),
                              dims);
    const auto plan = plan_prefill_group(grants, group, attention_mode);
    const bool tensor_core =
        model.backend.tensor_core_prefill && (peer == nullptr || peer->backend.tensor_core_prefill);
    const PrefillProjections projections{
        buffers.activation, group.rows, stream, tensor_core, !plan.native,
    };
    const Chunk chunk(projections, buffers, 0, nullptr, 0,
                      select_gqa_staging(workspace, peer_workspace));
    const bool scaled_output = !plan.native && group.rows == grouped_prefill_max_rows;
    const bool reuse_all_gqa = std::all_of(grants, grants + group.grants,
        [](const PrefillGrant& grant) {
            return grant.position + grant.rows <= grant.request->cached_gqa_tokens;
        });
    if (peer) {
        prefill_layerwise_tp(model, *peer, grants, peer_grants, group, plan, serial_capacity, chunk,
                             workspace, *peer_workspace, scaled_output, reuse_all_gqa, stream,
                             peer_stream, *comm);
        return;
    }
    each_prefill_segment(grants, group.grants, 0, [&](const PrefillGrant& grant, int offset) {
        auto* residual = buffers.residual + offset * hidden_size;
        gather_embeddings(model.embedding, grant.tokens, grant.rows, residual, stream);
        if (grant.media) grant.media->substitute(residual, grant.position, grant.rows, stream);
    });
    run_layers(
        model.layers,
        [&](const Bf16* weight, const char*, const auto& residuals) {
            const auto [branch, residual] = residuals(buffers);
            chunk.normalize(branch, residual, weight, plan.rms_launch);
        },
        [&](const GqaLayer& layer, int bank) {
            chunk.gqa_input(layer, reuse_all_gqa);
            run_group_gqa(chunk, layer, bank, grants, group.grants, plan, serial_capacity);
            project_group_gqa(chunk, layer, grants, group.grants, plan, scaled_output);
        },
        [&](const GdnLayer& layer, int bank) {
            chunk.gdn_input(layer);
            run_group_gdn(chunk, layer, bank, grants, group.grants, plan, scaled_output);
        },
        [&](const Layer& layer) { chunk.mlp(layer); });
    finish_prefill_group(model, chunk, grants, group.grants, buffers, stream);
}

void enqueue_native_prefill_grants(const DecoderModel& model, const PrefillGrant* grants,
                                   int count, PrefillWorkspace& workspace, int capacity,
                                   int first_offset = 0);

}

bool decoder::try_prefill_projection_groups(
    const DecoderModel& model, const PrefillGrant* grants, int count, cudaStream_t stream,
    DecoderBatchStepStats& stats) {
    if (count == 0)
        return false;
    if (!model.backend.tensor_core_prefill) return false;
    const int resident_visible = resident_gqa_staging_visible(grants, count);
    return schedule_projection_groups(
        grants, count, stream, stats,
        [&](const PrefillSegment*, const PrefillGrant* group, PrefillGroup rows,
            PrefillWorkspace& workspace, PrefillAttentionMode attention, int capacity) {
            prefill_layerwise(model, group, rows, workspace, attention, capacity);
        },
        [&](int, const PrefillGrant* grant, PrefillWorkspace& workspace, int capacity, int offset) {
            enqueue_native_prefill_grants(model, grant, 1, workspace, capacity, offset);
        },
        [](int) { return true; }, min_partial_prefill_rows, resident_visible, gqa_full_geometry,
        resident_visible != 0, model.backend.paired_prefill);
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
    prefill_layerwise(*this, &grant, {1, rows}, workspace, PrefillAttentionMode::native, capacity);
}

namespace {

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
            seed_scalar_prefill_draft_tp(model, peer, grant, peer_grant, stream, peer_stream,
                                         group);
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
                const PrefillTileSpan span{.offset = offset, .rows = rows};
                const auto tile = prefill_tile(grant, span);
                const auto peer_tile = prefill_tile(peer_grant, span);
                validate_prefill_tile(tile, workspace, capacity);
                validate_prefill_tile(peer_tile, peer_workspace, capacity);
                if (workspace.data() == nullptr || peer_workspace.data() == nullptr) {
                    throw std::runtime_error("Tensor-parallel prefill workspace unavailable");
                }
                prefill_layerwise(model, &tile, {1, rows}, workspace, PrefillAttentionMode::native,
                                  capacity, &peer, &peer_tile, &peer_workspace, peer_stream,
                                  &group);
                offset += rows;
            }
        }
    }
}

bool try_prefill_projection_groups_tp(const DecoderModel& model, const DecoderModel& peer,
                                      const PrefillGrant* grants, const PrefillGrant* peer_grants,
                                      int count, cudaStream_t stream, cudaStream_t peer_stream,
                                      TensorParallel& group, DecoderBatchStepStats& stats) {
    if (!model.backend.tensor_core_prefill || !peer.backend.tensor_core_prefill) return false;
    const int resident_visible = resident_gqa_staging_visible(grants, count);
    const int staged_visible = std::max(resident_visible, long_serial_visible(grants, count, true));
    std::optional<PrefillWorkspace> peer_workspace;
    const auto scheduled = schedule_projection_groups(
        grants, count, stream, stats,
        [&](const PrefillSegment* segments, const PrefillGrant* group_grants, PrefillGroup rows,
            PrefillWorkspace& workspace, PrefillAttentionMode attention, int capacity) {
            std::array<PrefillGrant, decoder_batch_max_lanes> peer_segments;
            for (int lane = 0; lane < rows.grants; ++lane) {
                const auto [owner, begin] = segments[lane];
                peer_segments[lane] =
                    prefill_tile(peer_grants[owner], {
                                                         .offset = begin,
                                                         .rows = group_grants[lane].rows,
                                                     });
                validate_prefill_tile(peer_segments[lane], *peer_workspace, capacity);
            }
            prefill_layerwise(model, group_grants, rows, workspace, attention, capacity, &peer,
                              peer_segments.data(), &*peer_workspace, peer_stream, &group);
        },
        [&](int source, const PrefillGrant* grant, PrefillWorkspace& workspace, int capacity,
            int offset) {
            enqueue_native_prefill_grants_tp(model, peer, grant, peer_grants + source,
                                             1, workspace, *peer_workspace, capacity, offset,
                                             stream, peer_stream, group);
        },
        [&](int capacity) {
            const DeviceScope scope(peer.device);
            peer_workspace.emplace(peer_stream, capacity, gqa_rank_geometry, staged_visible,
                                   resident_visible != 0);
            stats.prefill_workspace_bytes =
                std::max(stats.prefill_workspace_bytes, peer_workspace->bytes());
            return peer_workspace->data() != nullptr;
        },
        prefill_chunk_rows, staged_visible, gqa_rank_geometry, resident_visible != 0,
        model.backend.paired_prefill && peer.backend.paired_prefill, true);
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
    const int staged_visible =
        backend.tensor_core_prefill ? resident_gqa_staging_visible(grants, count) : 0;
    std::optional<PrefillWorkspace> workspace(std::in_place, stream, capacity, gqa_full_geometry,
                                              staged_visible, staged_visible != 0);
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
    const bool tensor_core = backend.tensor_core_prefill && peer.backend.tensor_core_prefill;
    const int resident_visible = tensor_core ? resident_gqa_staging_visible(grants, count) : 0;
    const int legacy_visible = tensor_core ? long_serial_visible(grants, count, false) : 0;
    const int staged_visible = std::max(resident_visible, legacy_visible);
    std::optional<PrefillWorkspace> workspace;
    std::optional<PrefillWorkspace> peer_workspace;
    for (;;) {
        workspace.emplace(stream, capacity, gqa_rank_geometry, staged_visible,
                          resident_visible != 0);
        if (workspace->data() != nullptr || capacity == 1) {
            const DeviceScope scope(peer.device);
            peer_workspace.emplace(peer_stream, capacity, gqa_rank_geometry, staged_visible,
                                   resident_visible != 0);
            if (capacity == 1 || peer_workspace->data() != nullptr)
                break;
        }
        if (capacity <= prefill_chunk_rows) {
            throw std::runtime_error("Prefill workspace unavailable under tensor parallelism");
        }
        if (workspace->data() != nullptr) {
            workspace->close();
            check(cudaStreamSynchronize(stream), "Release provisional TP prefill workspace");
        }
        capacity = prefill_chunk_rows;
    }
    stats.prefill_workspace_bytes = std::max(stats.prefill_workspace_bytes, workspace->bytes());
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
