#include "sglang/runtime/execution/decoder_forward_layers.h"
#include <algorithm>
#include <type_traits>
#include <utility>

namespace sglang::decoder::forward_detail {

namespace {

template <typename Lane, typename Function>
void each_lane(const LayerBatch<Lane>& batch, Function function) {
    for (int index = 0; index < batch.count; ++index) {
        function(batch.lanes[index], workspace(batch.lanes[index]));
    }
}

constexpr int gdn_packet_rows = verification_packet_rows;

void project_gdn_gates(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch,
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
        bool dedicated = (batch.count >= 2 && batch.count <= text_fp8_max_batch) ||
                         batch.count % gdn_packet_rows == 0;
        for (int first = 0; dedicated && first < batch.count; first += gdn_packet_rows) {
            const int rows = std::min(gdn_packet_rows, batch.count - first);
            if (const auto status = fill(first, rows)) {
                check(*status, "Run wide verification GDN gate projections");
            } else {
                dedicated = false;
            }
        }
        if (dedicated) {
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

void run_gdn_verification(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch,
                          int state_index) {
    const auto& first = batch.lanes[0];
    if (first.binding != nullptr) {
        const auto& scratch = *first.graph_workspace;
        const gdn_replay_ssm::Lane job{
            {},          scratch.gdn, first.work->gdn_z, scratch.observable, first.work->attention,
            batch.count,
        };
        check(gdn_replay_ssm::launch(batch.tensor_parallel, job, layer.recurrent, batch.stream,
                                     &first.binding->gdn_replay[state_index]),
              "Advance bound verification GDN history");
        return;
    }
    if (tp_verify_graph::run_gdn(batch.stream, state_index, layer.recurrent))
        return;
    if (batch.count == first.verification->pending_rows()) {
        check(gdn_replay_ssm::launch(batch.tensor_parallel,
                                     first.verification->gdn_lane(state_index), layer.recurrent,
                                     batch.stream),
              "Advance verification GDN history");
        return;
    }

    gdn_replay_ssm::Lanes jobs{};
    int count = 0;
    int rows = 0;
    const auto submit = [&] {
        check(gdn_replay_ssm::launch(batch.tensor_parallel, jobs, count, rows, layer.recurrent,
                                     batch.stream),
              "Advance packed verification GDN history");
        count = 0;
        rows = 0;
    };
    for (int index = 0; index < batch.count; ++index) {
        const auto& row = batch.lanes[index];
        if (!row.first)
            continue;
        const auto job = row.verification->gdn_lane(state_index);
        jobs.lanes[count++] = job;
        rows = std::max(rows, job.rows);
        if (count == gdn_replay_ssm::max_packed_lanes)
            submit();
    }
    if (count != 0)
        submit();
}

template <typename Position>
cudaError_t read_gqa(Workspace& work, Mxfp8PageTable pages, Mxfp8KvCache cache,
                     Position position, TextGdnMode mode, bool narrow_gqa,
                     ExactGqaWorkspace singleton_scores, bool vector, cudaStream_t stream) {
    const int capacity = pages.page_count * mxfp8_page_size;
    const auto scores = work.gqa_scores(capacity);

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

void gate_gqa(Workspace& work, cudaStream_t stream, GqaGeometry geometry) {
    check(text_gqa_postgate(1, work.q_gate, work.attention, work.attention, stream, geometry),
          "Gate grouped-query attention");
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
    each_lane(batch, [&](const BatchGraphLane&, Workspace& work) {
        gate_gqa(work, batch.stream, batch.gqa_geometry());
    });
}

void attend_verify_gqa(const GqaLayer& layer, const LayerBatch<VerificationRow>& batch,
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
        each_lane(batch, [&](const VerificationRow&, Workspace& work) {
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
    each_lane(batch, [&](const Lane& lane, Workspace& work) {
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

void attend_eager_verification(const GqaLayer& layer,
                                      const LayerBatch<VerificationRow>& batch, int cache_index) {
    if (auto* live = tp_verify_graph::binding(batch.stream)) {
        tp_verify_graph::attend_gqa(*live, layer, batch.lanes, cache_index, batch.gqa_geometry());
        return;
    }
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
    const auto remainder = [&](const GqaVerifySelection& selected) {
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
    const GqaVerifyRemainder pending{
        &remainder,
        [](const void* context, const GqaVerifySelection& selected) {
            (*static_cast<const decltype(remainder)*>(context))(selected);
        },
    };
    if (try_gqa_verify_long(layer, batch.lanes, batch.count, cache_index, batch.vector_attention,
                            batch.target_split_end, batch.stream, pending)) return;
    if (!batch.packed_activation) {
        attend(batch);
        return;
    }
    for (int first = 0; first < batch.count; first += width) {
        attend(batch.verification_packet(first));
    }
}

}

template <typename Lane>
void run_gdn(const GdnLayer& layer, const LayerBatch<Lane>& batch, int state_index) {
    batch.quantize(&Workspace::normalized, hidden_size);
    batch.project_group({{layer.qkv, &Workspace::gdn_qkv}, {layer.z, &Workspace::gdn_z}});
    each_lane(batch, [&](const Lane& lane, Workspace& work) {
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
        if constexpr (!std::is_same_v<Lane, BatchGraphLane>) {
            if (lane.mode == TextGdnMode::prefill) {
                check(batch.tensor_parallel
                          ? text_gdn_forward_local(
                                1, lane.mode, work.gdn_qkv, work.gdn_z, work.gdn_b, work.gdn_a,
                                layer.recurrent, lane.request->states[state_index],
                                work.gdn_scratch, text_gdn_local_workspace_bytes(1), work.attention,
                                stream)
                          : text_gdn_forward(1, lane.mode, work.gdn_qkv, work.gdn_z, work.gdn_b,
                                             work.gdn_a, layer.recurrent,
                                             lane.request->states[state_index], work.gdn_scratch,
                                             text_gdn_workspace_bytes(1), work.attention, stream),
                      "Advance Gated DeltaNet state");
                return;
            }
        }
        const int qkv_width = batch.tensor_parallel ? GdnLocal::qkv_width : GdnFull::qkv_width;
        const auto [state, bound] =
            [&]() -> std::pair<gdn_replay_ssm::State, const gdn_replay_ssm::State*> {
            if constexpr (std::is_same_v<Lane, BatchGraphLane>) {
                return {{}, &lane.binding->gdn_replay[state_index]};
            } else {
                return {lane.request->gdn_replay[state_index], nullptr};
            }
        }();
        const gdn_replay_ssm::Lane job{
            state,          {work.gdn_qkv, work.gdn_scratch, work.gdn_b, work.gdn_a},
            work.gdn_z,     work.gdn_scratch + qkv_width,
            work.attention, 1,
            true,
        };
        check(gdn_replay_ssm::launch(batch.tensor_parallel, job, layer.recurrent, stream, bound),
              "Advance decode GDN history");
    });
    batch.quantize(&Workspace::attention, batch.gdn_attention_width());
    batch.project(layer.output, &Workspace::projected_attention);
}

void run_gdn(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch,
                    int state_index) {
    if (!batch.fuse_verify_activation() && !batch.fuse_wide_verify_norm()) {
        batch.quantize(&Workspace::normalized, hidden_size);
    }
    batch.project_group({{layer.qkv, &Workspace::gdn_qkv}, {layer.z, &Workspace::gdn_z}});
    project_gdn_gates(layer, batch, batch.packed_activation != nullptr);
    run_gdn_verification(layer, batch, state_index);
    batch.quantize(&Workspace::attention, batch.gdn_attention_width());
    batch.project(layer.output, &Workspace::projected_attention);
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

template void run_gqa(const GqaLayer&, const LayerBatch<ForwardLane>&, int);
template void run_gdn(const GdnLayer&, const LayerBatch<ForwardLane>&, int);
template void run_gqa(const GqaLayer&, const LayerBatch<GraphLane>&, int);
template void run_gdn(const GdnLayer&, const LayerBatch<GraphLane>&, int);
template void run_gqa(const GqaLayer&, const LayerBatch<VerificationRow>&, int);
template void run_gqa(const GqaLayer&, const LayerBatch<BatchGraphLane>&, int);
template void run_gdn(const GdnLayer&, const LayerBatch<BatchGraphLane>&, int);

}
