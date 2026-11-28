#include "sglang/runtime/execution/decoder_forward_layers.h"
#include "sglang/runtime/execution/tp_verify_graph.h"
#include <algorithm>
#include <bitset>
#include <optional>
#include <type_traits>
#include <utility>

namespace sglang::decoder::tp_verify_graph {

bool run_gdn(cudaStream_t stream, int layer, TextGdnWeights weights) {
    auto* live = binding(stream);
    if (live == nullptr)
        return false;
    ++live->gdn_calls;
    check(gdn_replay_ssm::launch(true, live->device->gdn[layer], live->lanes, live->depth, weights,
                                 stream),
          "Capture bound GDN history");
    return true;
}

void attend_gqa(HostBinding& live, const GqaLayer& layer, const VerificationRow* packet,
                int cache_index, GqaGeometry geometry) {
    GqaVerifyRows<GqaVerifyPreprocessRow, max_wide_mtp_rows> prep{};
    GqaVerifyRows<GqaVerifyAppendRow, max_wide_mtp_rows> append_rows{};
    GqaVerifyRows<GqaVerifyGateRow, max_wide_mtp_rows> gate{};
    for (int row = 0; row < live.rows; ++row) {
        const auto& work = *packet[row].work;
        prep.rows[row] = {work.q_gate, work.keys, work.rope_queries, work.rope_keys, 0};
        append_rows.rows[row].keys = work.rope_keys;
        append_rows.rows[row].values = work.values;
        gate.rows[row] = {work.q_gate, work.attention};
    }
    const auto* cache = &live.device->gqa[cache_index];
    ++live.gqa_calls;
    check(gqa_verify_preprocess_bound(prep, live.rows, live.device->rotary, geometry,
                                      layer.query_norm, layer.key_norm, live.stream),
          "Capture bound rotary");
    check(gqa_verify_append_bound(append_rows, cache->reads, live.rows, geometry.kv_heads,
                                  live.stream),
          "Capture bound KV append");
    if (live.short_count != 0) {
        check(read(cache, live.short_rows, live.short_count, geometry.query_heads, live.stream),
              "Capture bound short verifier attention");
    }
    for (int group = 0; group < live.split_groups; ++group) {
        const auto& split = live.splits[group];
        std::array<BoundPagedDecodeLane, verification_packet_rows> reads{};
        for (int index = 0; index < split.count; ++index) {
            const int row = live.split_rows[split.first + index];
            const auto& state = cache->reads[row];
            reads[index] = {
                &state.cache,
                &state.pages,
                &state.position,
                nullptr,
                paged_gqa_buffers(*packet[row].work),
            };
        }
        check(target_split_gqa_graph(reads.data(), split.count, split.variant, geometry.kv_heads,
                                     live.stream),
              "Capture bound split attention");
    }
    if (live.long_count != 0) {
        gqa_decode::BoundLongBatch batch{};
        for (int lane = 0; lane < live.long_count; ++lane) {
            batch.requests[lane] = &cache->long_reads[lane];
        }
        const auto status = gqa_decode::launch_long_decode(
            batch, live.long_count, geometry.kv_heads, live.long_shape, live.stream);
        check(status.value_or(cudaErrorNotSupported), "Capture bound long verifier attention");
    }
    check(gqa_verify_postgate(gate, live.rows, live.stream, geometry),
          "Capture bound attention gate");
}

}

namespace sglang::decoder::forward_detail {

namespace {

template <typename Lane, typename Function>
void each_lane(const LayerBatch<Lane>& batch, Function function) {
    for (int index = 0; index < batch.count; ++index) {
        function(batch.lanes[index], workspace(batch.lanes[index]));
    }
}

void project_gdn_gates(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch,
                       bool batched) {
    if (!batch.target_tc || batch.count < 2 || layer.b.rows != batch.gdn_gate_heads() ||
        layer.a.rows != batch.gdn_gate_heads() || layer.b.columns != hidden_size ||
        layer.a.columns != hidden_size) {
        project_verification_bf16(layer.b, batch.lanes, batch.count, &Workspace::gdn_b,
                                  batch.stream);
        project_verification_bf16(layer.a, batch.lanes, batch.count, &Workspace::gdn_a,
                                  batch.stream);
        return;
    }
    std::array<VerifyGdnGateRow, max_wide_mtp_rows> gates{};
    const auto fill = [&](int first, int rows) {
        for (int index = 0; index < rows; ++index) {
            const auto& work = workspace(batch.lanes[first + index]);
            gates[index] = {work.normalized, work.gdn_b, work.gdn_a};
        }
        const auto gates_kernel =
            batch.tensor_parallel ? try_verify_bf16_gdn_gates_local : try_verify_bf16_gdn_gates;
        return gates_kernel(layer.b.values, layer.a.values, gates.data(), rows, batch.stream);
    };
    if (batched && batch.count <= max_wide_mtp_rows &&
        (batch.count <= text_fp8_max_batch || batch.count % verification_packet_rows == 0)) {
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

void run_gdn_verification(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch,
                          int state_index) {
    const auto& first = batch.lanes[0];
    if (first.binding != nullptr) {
        int lanes = 0;
        int longest = 0;
        for (int index = 0; index < batch.count; ++index) {
            const auto& row = batch.lanes[index];
            lanes += static_cast<int>(row.first);
            longest = std::max(longest, row.binding_row + 1);
        }
        check(gdn_replay_ssm::launch(batch.tensor_parallel,
                                     first.binding->gdn[state_index] + first.binding_lane, lanes,
                                     longest, layer.recurrent, batch.stream),
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
        if (mode == TextGdnMode::prefill && vector) {
            const auto status = try_chunked_ordered_gqa(cache, pages, position, work.rope_queries,
                                                        work.attention, scores, stream);
            if (status)
                return *status;
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

cudaError_t read_bound_gqa(const BoundPagedDecodeLane& read, bool vector, int narrow_bytes,
                           cudaStream_t stream) {
    if (vector) {
        return vector_decode_gqa_attention(read.cache, read.pages, read.position,
                                           read.buffers.queries, read.buffers.output, read.scores,
                                           stream);
    }
    return exact_decode_gqa_attention(read.cache, read.pages, read.position, read.buffers.queries,
                                      read.buffers.output, read.scores, stream, narrow_bytes);
}

void attend_bound_gqa(const GqaLayer& layer, const LayerBatch<BatchGraphLane>& batch,
                             int cache_index) {
    const auto geometry = batch.gqa_geometry();
    auto* tp = tp_verify_graph::binding(batch.stream);
    if (tp != nullptr)
        ++tp->gqa_calls;
    const auto* live = batch.lanes[0].batch;
    GqaVerifyRows<GqaVerifyPreprocessRow, max_wide_mtp_rows> preprocess{};
    GqaVerifyRows<GqaVerifyAppendRow, max_wide_mtp_rows> append{};
    GqaVerifyRows<GqaVerifyGateRow, max_wide_mtp_rows> gate{};
    std::array<BoundPagedDecodeLane, decoder_batch_max_lanes> reads{};
    for (int index = 0; index < batch.count; ++index) {
        const auto& lane = batch.lanes[index];
        const auto* binding = lane.binding;
        const auto* cache = &binding->caches[cache_index];
        const auto* position = &binding->position;
        auto& work = workspace(lane);
        if (live != nullptr) {
            preprocess.rows[index] = {work.q_gate, work.keys, work.rope_queries, work.rope_keys, 0};
            append.rows[index].keys = work.rope_keys;
            append.rows[index].values = work.values;
        } else {
            check(decode_gqa_preprocess(&binding->rotary_position, work.q_gate, work.keys,
                                        layer.query_norm, layer.key_norm, work.rope_queries,
                                        work.rope_keys, batch.stream, geometry),
                  "Normalize and rotate bound Q/K");
            check(decode_kv_append(cache, &binding->pages, position, work.rope_keys, work.values,
                                   batch.stream),
                  "Append bound decode MXFP8 KV");
        }
        reads[index] = {
            cache, &binding->pages, position, &binding->scores, paged_gqa_buffers(work),
        };
        gate.rows[index] = {work.q_gate, work.attention};
    }
    if (live != nullptr) {
        check(gqa_verify_preprocess_bound(preprocess, batch.count, live->rotary, geometry,
                                          layer.query_norm, layer.key_norm, batch.stream),
              "Normalize and rotate bound decode Q/K");
        check(gqa_verify_append_bound(append, live->reads[cache_index], batch.count,
                                      geometry.kv_heads, batch.stream),
              "Append packed decode MXFP8 KV");
    }

    std::array<BoundPagedDecodeLane, target_split_eager_max_rows> split{};
    std::array<BoundPagedDecodeLane, paged_decode_lanes> paged{};
    int split_count = 0;
    int paged_count = 0;
    const auto flush_split = [&] {
        if (split_count) {
            check(target_split_gqa_graph(split.data(), split_count,
                                         gqa_decode::SplitVariant::multi_stage, geometry.kv_heads,
                                         batch.stream),
                  "Read bound split decode attention");
            split_count = 0;
        }
    };
    for (int index = 0; index < batch.count; ++index) {
        auto attention = batch.lanes[index].attention;
        if (tp != nullptr) {
            const auto mask = std::uint32_t{1} << static_cast<unsigned int>(index);
            attention = (tp->decode_split & mask) != 0 ? BatchGraphAttention::split
                                                       : BatchGraphAttention::direct;
        }
        const auto& read = reads[index];
        switch (attention) {
        case BatchGraphAttention::split:
            split[split_count++] = read;
            if (split_count == target_split_eager_max_rows)
                flush_split();
            break;
        case BatchGraphAttention::paged:
            paged[paged_count++] = read;
            break;
        case BatchGraphAttention::direct:
            check(read_bound_gqa(read, batch.vector_attention,
                                 batch.single_decode() && !batch.ordered_attention
                                     ? vocab_size * sizeof(Bf16)
                                     : 0,
                                 batch.stream),
                  "Read bound direct decode attention");
            break;
        }
    }
    flush_split();
    if (paged_count != 0) {
        check(paged_decode_gqa(paged.data(), paged_count, batch.stream),
              "Read bound paged decode attention");
    }
    check(gqa_verify_postgate(gate, batch.count, batch.stream, geometry),
          "Gate bound decode attention");
}

using ShortVerifyRows = std::bitset<verification_packet_rows>;

ShortVerifyRows shared_short_rows(const VerificationRow* rows, int count, bool bound) {
    const auto eligible = [bound](const VerificationRow& row) {
        return bound ? row.graph_attention == VerificationAttention::direct
                     : row.position < target_split_first_position && !row.constrained;
    };
    ShortVerifyRows selected;
    for (int index = 0; index < count; ++index) {
        const auto& row = rows[index];
        if (!eligible(row))
            continue;
        for (int previous = 0; previous < index; ++previous) {
            const auto& other = rows[previous];
            if (eligible(other) && other.request == row.request && other.position < row.position) {
                selected.set(static_cast<std::size_t>(previous));
                selected.set(static_cast<std::size_t>(index));
            }
        }
    }
    return selected;
}

void attend_verify_gqa(const GqaLayer& layer, const LayerBatch<VerificationRow>& batch,
                              int cache_index) {
    const auto& first = batch.lanes[0];
    const auto* live = first.binding->reads[cache_index] + first.binding_index;
    const auto geometry = batch.gqa_geometry();
    GqaVerifyRows<GqaVerifyPreprocessRow, max_wide_mtp_rows> preprocess{};
    GqaVerifyRows<GqaVerifyAppendRow, max_wide_mtp_rows> append{};
    GqaVerifyRows<GqaVerifyGateRow, max_wide_mtp_rows> gate{};
    for (int index = 0; index < batch.count; ++index) {
        auto& work = workspace(batch.lanes[index]);
        preprocess.rows[index] = {work.q_gate, work.keys, work.rope_queries, work.rope_keys, 0};
        append.rows[index].keys = work.rope_keys;
        append.rows[index].values = work.values;
        gate.rows[index] = {work.q_gate, work.attention};
    }
    check(gqa_verify_preprocess_bound(preprocess, batch.count,
                                      first.binding->rotary + first.binding_index, geometry,
                                      layer.query_norm, layer.key_norm, batch.stream),
          "Normalize and rotate bound verification Q/K");
    check(gqa_verify_append_bound(append, live, batch.count, geometry.kv_heads, batch.stream),
          "Append bound verification MXFP8 KV");

    gqa_decode::BoundLongBatch long_reads{};
    int long_count = 0;
    std::array<bool, max_wide_mtp_lanes> long_lanes{};
    const int width = batch.packed_activation != nullptr ? verification_packet_rows : batch.count;
    for (int offset = 0; offset < batch.count; offset += width) {
        std::array<BoundPagedDecodeLane, verification_packet_rows> split{};
        int split_count = 0;
        std::array<int, verification_packet_rows> short_indices{};
        std::array<const void*, verification_packet_rows> short_requests{};
        int short_count = 0;
        const int end = std::min(offset + width, batch.count);
        const bool share_short = !batch.tensor_parallel && batch.ordered_attention &&
                                 batch.vector_attention &&
                                 batch.target_split_end == text_attention_max_tokens;
        const auto shared = share_short
                                ? shared_short_rows(batch.lanes + offset, end - offset, true)
                                : ShortVerifyRows{};
        for (int index = offset; index < end; ++index) {
            const auto& row = batch.lanes[index];
            if (row.graph_attention == VerificationAttention::long_context) {
                long_lanes[row.binding_lane] = true;
                continue;
            }
            if (shared.test(static_cast<std::size_t>(index - offset))) {
                short_indices[short_count] = index;
                short_requests[short_count++] = row.request;
                continue;
            }
            const BoundPagedDecodeLane read{
                &live[index].cache,           &live[index].pages,
                &live[index].position,        &row.binding->scores[row.binding_index],
                paged_gqa_buffers(*row.work),
            };
            if (row.graph_attention == VerificationAttention::split) {
                split[split_count++] = read;
            } else {
                check(read_bound_gqa(read, batch.vector_attention, 0, batch.stream),
                      "Read bound verification MXFP8 attention");
            }
        }
        if (short_count != 0) {
            check(short_verify_gqa_shared(live, short_indices.data(), short_requests.data(),
                                          short_count, geometry, batch.stream),
                  "Read shared bound target MXFP8 attention");
        }
        if (split_count != 0) {
            const auto variant = batch.lanes[offset].graph_split;
            const int group = variant == gqa_decode::SplitVariant::multi_stage
                                  ? target_verify_max_rows
                                  : split_count;
            for (int i = 0; i < split_count; i += group) {
                check(target_split_gqa_graph(split.data() + i, std::min(group, split_count - i),
                                             variant, geometry.kv_heads, batch.stream),
                      "Read bound split attention");
            }
        }
    }
    for (int lane = 0; lane < static_cast<int>(long_lanes.size()); ++lane) {
        if (long_lanes[lane])
            long_reads.requests[long_count++] = &first.binding->history[lane][cache_index];
    }
    if (long_count != 0) {
        const auto status = gqa_decode::launch_long_decode(
            long_reads, long_count, geometry.kv_heads, first.graph_long_shape, batch.stream);
        check(status.value_or(cudaErrorNotSupported), "Read bound long verification attention");
    }
    check(gqa_verify_postgate(gate, batch.count, batch.stream, geometry),
          "Gate bound grouped-query attention");
}

template <typename Lane>
void prepare_eager_gqa(const GqaLayer& layer, const LayerBatch<Lane>& batch, const Lane& lane,
                       Workspace& work, Mxfp8KvCache cache) {
    const auto& request = *lane.request;
    const auto geometry = batch.gqa_geometry();
    const MropePosition* mrope = nullptr;
    if constexpr (std::is_same_v<Lane, ForwardLane>) {
        mrope = lane.mrope;
        if (lane.mode == TextGdnMode::prefill && lane.position < request.cached_gqa_tokens) {
            const int position = lane.rotary_position < 0 ? lane.position : lane.rotary_position;
            check(prefill_gqa_queries(position, mrope, 1, work.q_gate, layer.query_norm,
                                      work.rope_queries, batch.stream, geometry),
                  "Normalize cached prefill Q");
            return;
        }
    }
    append_gqa(layer, work, request.pages, cache, lane.position, batch.stream, mrope,
               lane.rotary_position, geometry);
}

template <typename Lane>
void read_direct_eager_gqa(const LayerBatch<Lane>& batch, const Lane& lane, Workspace& work,
                           Mxfp8KvCache cache) {
    const bool narrow_gqa = [&] {
        if constexpr (std::is_same_v<Lane, VerificationRow>) {
            return false;
        } else {
            return lane.narrow_gqa && !batch.ordered_attention;
        }
    }();
    const auto singleton_scores = [&]() -> ExactGqaWorkspace {
        if constexpr (std::is_same_v<Lane, ForwardLane>) {
            return lane.singleton_scores;
        } else {
            return {};
        }
    }();
    check(read_gqa(work, lane.request->pages, cache, lane.position, lane.mode, narrow_gqa,
                   singleton_scores, batch.vector_attention, batch.stream),
          "Read causal MXFP8 attention");
    gate_gqa(work, batch.stream, batch.gqa_geometry());
}

template <typename Lane>
std::optional<PagedDecodeLane> eager_paged_read(const LayerBatch<Lane>& batch, const Lane& lane,
                                                Workspace& work, Mxfp8KvCache cache) {
    if constexpr (std::is_same_v<Lane, ForwardLane>) {
        if (!batch.ordered_attention && lane.mode == TextGdnMode::decode && batch.count > 1 &&
            batch.count <= paged_decode_lanes && paged_decode_position(lane.position)) {
            return PagedDecodeLane{
                cache,
                lane.request->pages,
                lane.position,
                paged_gqa_buffers(work),
            };
        }
    }
    return std::nullopt;
}

template <typename Lane>
std::optional<PagedDecodeLane> eager_split_read(const LayerBatch<Lane>& batch, const Lane& lane,
                                                Workspace& work, Mxfp8KvCache cache) {
    if constexpr (std::is_same_v<decltype(lane.position), int>) {
        if (batch.target_split_end != 0 && batch.vector_attention &&
            lane.mode == TextGdnMode::decode &&
            target_split_position(lane.position, batch.target_split_end)) {
            return PagedDecodeLane{
                cache,
                lane.request->pages,
                lane.position,
                paged_gqa_buffers(work),
            };
        }
    }
    return std::nullopt;
}

template <typename Lane>
void attend_eager_gqa(const GqaLayer& layer, const LayerBatch<Lane>& batch, int cache_index,
                      const ShortVerifyRows& skip = ShortVerifyRows{}) {
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
    for (int row = 0; row < batch.count; ++row) {
        if constexpr (std::is_same_v<Lane, VerificationRow>) {
            if (skip.test(static_cast<std::size_t>(row)))
                continue;
        }
        const auto& lane = batch.lanes[row];
        auto& work = workspace(lane);
        const auto& request = *lane.request;
        const auto cache = request.caches[cache_index];
        prepare_eager_gqa(layer, batch, lane, work, cache);
        if (const auto read = eager_paged_read(batch, lane, work, cache)) {
            reads[count] = *read;
            pending[count++] = &work;
            continue;
        }
        if (const auto read = eager_split_read(batch, lane, work, cache)) {
            if (grouped) {
                split_reads[split_count] = *read;
                split_pending[split_count++] = &work;
                if (split_count == target_split_eager_max_rows) {
                    flush_split();
                }
                continue;
            }
            check(target_split_gqa(*read, batch.stream, batch.target_split_end),
                  "Read split target MXFP8 attention");
            gate_gqa(work, batch.stream, batch.gqa_geometry());
            continue;
        }
        read_direct_eager_gqa(batch, lane, work, cache);
    }
    flush_split();
    if (count > 0) {
        const auto status = paged_decode_gqa(reads.data(), count, batch.stream);
        check(status, "Read paged MXFP8 attention");
        for (int index = 0; index < count; ++index) {
            gate_gqa(*pending[index], batch.stream, batch.gqa_geometry());
        }
    }
}

ShortVerifyRows attend_shared_short(const GqaLayer& layer,
                                    const LayerBatch<VerificationRow>& packet, int cache_index) {
    if (packet.tensor_parallel || !packet.ordered_attention || !packet.vector_attention)
        return {};
    const auto selected = shared_short_rows(packet.lanes, packet.count, false);
    if (selected.none())
        return {};
    std::array<VerificationRow, verification_packet_rows> rows;
    int count = 0;
    for (int index = 0; index < packet.count; ++index) {
        if (selected.test(static_cast<std::size_t>(index)))
            rows[count++] = packet.lanes[index];
    }
    return try_gqa_verify_short(layer, rows.data(), count, cache_index, true, packet.stream, true)
               ? selected
               : ShortVerifyRows{};
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
            attend_eager_gqa(layer, packet, cache_index,
                             attend_shared_short(layer, packet, cache_index));
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
    if constexpr (std::is_same_v<Lane, BatchGraphLane>) {
        if (auto* live = tp_verify_graph::binding(batch.stream))
            ++live->gdn_calls;
    }
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
