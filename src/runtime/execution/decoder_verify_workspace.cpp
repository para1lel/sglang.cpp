#include "sglang/decoder_verify.h"
#include "sglang/runtime/execution/verification_row_copy.h"
#include "sglang/runtime/cache/kv_residency_policy.h"
#include "sglang/text/gdn_local.h"
#include "sglang/text/gqa_verify_batch.h"

#include <algorithm>
#include <tuple>

namespace sglang::decoder {

bool VerificationGraphLaneShape::operator==(const VerificationGraphLaneShape& other) const {
    return rows == other.rows && split_begin == other.split_begin && long_begin == other.long_begin;
}

bool VerificationGraphShape::operator==(const VerificationGraphShape& other) const {
    return lanes == other.lanes && packet_rows == other.packet_rows && split == other.split &&
           long_decode == other.long_decode && packed == other.packed;
}

VerificationWorkspace::VerificationWorkspace(Allocations& memory, int rows, bool rank_local,
                                             const VerificationWorkspace* shared_head)
    : bf16_head_packed(
          shared_head != nullptr
              ? shared_head->bf16_head_packed
              : memory.allocate<Bf16>(static_cast<std::size_t>(verification_packet_rows) *
                                      (hidden_size + vocab_size))),
      bf16_head_packed_logits(bf16_head_packed +
                              (static_cast<std::size_t>(verification_packet_rows) * hidden_size)) {
    const int qkv_width = rank_local ? GdnLocal::qkv_width : GdnFull::qkv_width;
    const int gate_heads = rank_local ? GdnLocal::value_heads : GdnFull::value_heads;
    const int value_width = rank_local ? GdnLocal::value_width : GdnFull::value_width;
    const auto row_count = static_cast<std::size_t>(rows);
    const auto qkv = row_count * qkv_width;
    const auto gates = row_count * gate_heads;
    const auto values = row_count * value_width;
    auto* storage = memory.allocate<Bf16>(2 * (qkv + gates));
    auto* gate_storage = storage + (2 * qkv);
    gdn = {storage, storage + qkv, gate_storage, gate_storage + gates};
    auto* z = memory.allocate<Bf16>(values);
    auto* attention = memory.allocate<Bf16>(values);
    observable = memory.allocate<Bf16>(values);
    winners = memory.allocate<TextLogitCandidate>(rows);
    work.reserve(row_count);
    for (std::size_t index = 0; index < row_count; ++index) {
        const auto value_offset = index * value_width;
        const auto gate_offset = index * gate_heads;
        auto& row = work.emplace_back(memory, attention + value_offset, winners + index);
        row.gdn_z = z + value_offset;
        row.gdn_qkv = gdn.qkv + (index * qkv_width);
        row.gdn_b = gdn.b + gate_offset;
        row.gdn_a = gdn.a + gate_offset;
    }
}

VerificationGraph::VerificationGraph(Allocations& owner) : execution(owner) {}

namespace {

auto tensor_map_identity(const Mxfp8KvCache& cache) {
    return std::tie(cache.key_payload, cache.value_payload, cache.key_scales, cache.value_scales,
                    cache.page_count, cache.kv_heads);
}

VerificationGraphShape graph_shape(const VerificationRow* rows, int total, bool packed,
                                   int split_end) {
    VerificationGraphShape shape{};
    shape.packed = packed;
    for (int first = 0; first < total;) {
        const int packet = rows[first].binding_lane / decoder_batch_max_mtp_lanes;
        int end = first + 1;
        while (end < total && rows[end].binding_lane / decoder_batch_max_mtp_lanes == packet)
            ++end;
        shape.packet_rows[packet] = end - first;
        first = end;
    }
    GqaVerifySelection history{};
    const auto plan_long = [&](int packet, const VerificationRow* start, int count) {
        if (split_end != gqa_decode::LongBatch::end)
            return;
        GqaVerifySelection selected{};
        gqa_decode::LongBatch requests;
        const int lanes = plan_gqa_verify_long(start, count, 0, requests, selected);
        if (lanes != 0) {
            shape.long_decode[packet] = gqa_decode::long_decode_shape(requests, lanes);
            std::copy_n(selected.begin(), count, history.begin() + (start - rows));
        }
    };
    if (packed)
        plan_long(0, rows, total);
    int first = 0;
    for (int packet = 0; packet < VerificationGraphShape::packets; ++packet) {
        const int width = shape.packet_rows[packet];
        if (width == 0)
            break;
        if (!packed)
            plan_long(packet, rows + first, width);
        std::array<PagedDecodeLane, verification_packet_rows> split;
        int split_count = 0;
        for (int index = first; index < first + width; ++index) {
            const auto& row = rows[index];
            auto& lane = shape.lanes[row.binding_lane];
            ++lane.rows;
            if (history[index])
                continue;
            ++lane.long_begin;
            if (target_split_position(row.position, split_end)) {
                split[split_count++] = {
                    row.request->caches.front(),
                    row.request->pages,
                    row.position,
                    paged_gqa_buffers(*row.work),
                };
            } else {
                ++lane.split_begin;
            }
        }
        shape.split[packet] = gqa_decode::split_variant(split.data(), split_count, split_end);
        first += width;
    }
    return shape;
}

bool bind_verification_gqa(VerificationGraphBinding& host, const VerificationSequence& sequence,
                           const VerificationRow& row, int index, int begin, Workspace& work) {
    const auto& request = *row.request;
    for (int layer = 0; layer < full_attention_layers; ++layer) {
        const auto& cache = request.caches[layer];
        auto& read = host.reads[layer][index];
        read = {cache, request.pages, row.position, paged_gqa_buffers(work)};
        if (!gqa_decode::readable(read) || !gqa_geometry_matches(gqa_full_geometry, cache)) {
            return false;
        }
        if (row.binding_row < begin)
            continue;
        if (!gqa_decode::tensor_aligned(read))
            return false;
        auto& history = host.history[row.binding_lane][layer];
        if (tensor_map_identity(cache) != tensor_map_identity(history.request.cache)) {
            history.request.cache = cache;
            const auto status = gqa_decode::prepare_long_decode_binding(history);
            if (!status || *status != cudaSuccess)
                history.request.cache = {};
            if (!status)
                return false;
            check(*status, "Encode verifier graph tensor maps");
        }
        history.request.pages = request.pages;
        history.request.first = sequence.position + begin;
        history.request.count = sequence.count - begin;
        history.request.rows[row.binding_row - begin] = read.buffers;
    }
    return true;
}

VerificationWorkspace& active_workspace(std::optional<VerificationWorkspace>& workspace) {
    if (!workspace.has_value())
        throw std::bad_optional_access();
    return *workspace;
}

VerificationRow bind_verification_row(VerificationRow row, const VerificationGraphShape& shape,
                                      VerificationGraphScratch& scratch) {
    const auto& lane = shape.lanes[row.binding_lane];
    row.work = &active_workspace(scratch.workspaces[row.binding_lane]).work[row.binding_row];
    row.binding = scratch.device_binding;
    row.graph_attention = VerificationAttention::direct;
    if (row.binding_row >= lane.long_begin) {
        row.graph_attention = VerificationAttention::long_context;
    } else if (row.binding_row >= lane.split_begin) {
        row.graph_attention = VerificationAttention::split;
    }
    const int packet = row.binding_lane / decoder_batch_max_mtp_lanes;
    row.graph_split = shape.split[packet];
    row.graph_long_shape = shape.long_decode[shape.packed ? 0 : packet];
    return row;
}

}

void VerificationGraphScratch::prepare(const DecoderModel& model, int lanes, bool packed) {
    static_assert(max_wide_mtp_lanes <= gdn_replay_ssm::max_device_lanes);
    if (lanes > prepared_lanes) {
        auto added = std::make_unique<Allocations>(model.device);
        try {
            for (int lane = prepared_lanes; lane < lanes; ++lane) {
                const auto* head = lane == 0 ? nullptr : &active_workspace(workspaces.front());
                workspaces[lane].emplace(*added, target_verify_max_rows, false, head);
            }
            if (available_device_bytes() < decoder_shared_reserve)
                check(cudaErrorMemoryAllocation, "Reserve device headroom for target verification");
        } catch (...) {
            for (int lane = prepared_lanes; lane < lanes; ++lane)
                workspaces[lane].reset();
            throw;
        }
        workspace_allocations[prepared_lanes] = std::move(added);
        prepared_lanes = lanes;
    }
    if (packed && packed_activation == nullptr) {
        constexpr auto bytes = static_cast<std::size_t>(max_wide_mtp_rows) * intermediate_size;
        if (available_device_bytes() < decoder_shared_reserve + bytes)
            check(cudaErrorMemoryAllocation,
                  "Reserve device headroom for packed target verification");
        packed_activation = memory.allocate<std::uint8_t>(bytes);
    }
}

bool VerificationGraphScratch::upload(const VerificationSequence* sequences, int count,
                                      const VerificationRow* rows, int total,
                                      const VerificationGraphShape& shape, cudaStream_t stream) {
    auto& host = *host_binding;
    for (int lane = 0; lane < count; ++lane) {
        const auto& sequence = sequences[lane];
        const auto& workspace = active_workspace(workspaces[lane]);
        const auto& work = workspace.work.front();
        for (int layer = 0; layer < gdn_layers; ++layer) {
            host.gdn[layer][lane] = {
                sequence.verification->gdn_lane(layer).state,
                workspace.gdn,
                work.gdn_z,
                workspace.observable,
                work.attention,
                sequence.count,
            };
        }
    }
    for (int index = 0; index < total; ++index) {
        const auto& row = rows[index];
        const auto& sequence = sequences[row.binding_lane];
        const auto& request = *row.request;
        auto& work = active_workspace(workspaces[row.binding_lane]).work[row.binding_row];
        host.tokens[index] = sequence.tokens[row.binding_row];
        host.scores[index] = work.gqa_scores(request.pages.page_count * mxfp8_page_size);
        host.rotary[index] = row.rotary_position;
        const int begin = shape.lanes[row.binding_lane].long_begin;
        if (!bind_verification_gqa(host, sequence, row, index, begin, work))
            return false;
    }
    auto* device = device_binding;
    copy_cuda_async(device->tokens, host.tokens, total * sizeof(host.tokens[0]),
                    cudaMemcpyHostToDevice, stream, "Upload verifier graph token IDs");
    copy_cuda_async(device->scores, host.scores, total * sizeof(host.scores[0]),
                    cudaMemcpyHostToDevice, stream, "Upload verifier graph score scratch");
    copy_cuda_async(device->rotary, host.rotary, total * sizeof(host.rotary[0]),
                    cudaMemcpyHostToDevice, stream, "Upload verifier graph positions");
    check(cudaMemcpy2DAsync(device->gdn, sizeof(host.gdn[0]), host.gdn, sizeof(host.gdn[0]),
                            count * sizeof(host.gdn[0][0]), gdn_layers, cudaMemcpyHostToDevice,
                            stream),
          "Upload verifier graph GDN state");
    check(cudaMemcpy2DAsync(device->reads, sizeof(host.reads[0]), host.reads, sizeof(host.reads[0]),
                            total * sizeof(host.reads[0][0]), full_attention_layers,
                            cudaMemcpyHostToDevice, stream),
          "Upload verifier graph cache state");
    for (int lane = 0; lane < count; ++lane) {
        if (shape.lanes[lane].long_begin != sequences[lane].count) {
            copy_cuda_async(device->history[lane], host.history[lane], sizeof(host.history[lane]),
                            cudaMemcpyHostToDevice, stream, "Upload verifier graph tensor maps");
        }
    }
    return true;
}

bool VerificationGraph::replay(const DecoderModel& model, const VerificationSequence* sequences,
                               int count, VerificationRow* rows, int total, bool packed,
                               cudaStream_t stream) {
    const int split_end = model.backend.target_split_end;
    if (!std::all_of(sequences, sequences + count, [&](const VerificationSequence& sequence) {
            return sequence.verification->graph_eligible(sequence, split_end != 0);
        }))
        return false;

    const auto shape = graph_shape(rows, total, packed, split_end);
    auto* slot = std::find_if(slots.begin(), slots.end(),
                              [&](const Slot& item) { return item.used && item.shape == shape; });
    if (slot == slots.end()) {
        slot = std::min_element(slots.begin(), slots.end(),
                                [](const auto& a, const auto& b) { return a.used < b.used; });
        if (slot->used != 0)
            execution.reset_verification_graph(static_cast<int>(slot - slots.begin()));
        *slot = {shape, ++clock};
        return false;
    }
    slot->used = ++clock;
    const auto bind_row = [&](VerificationRow row) {
        return bind_verification_row(row, shape, *scratch);
    };
    const auto slot_index = static_cast<int>(slot - slots.begin());
    const bool replayed = execution.replay_verification_graph(
        slot_index, total,
        [&] {
            if (!scratch)
                scratch = std::make_unique<VerificationGraphScratch>(model.device);
            scratch->prepare(model, count, packed);
        },
        [&] {
            std::array<VerificationRow, max_wide_mtp_rows> captured;
            VerificationCopyRows inputs{};
            for (int index = 0; index < total; ++index) {
                const auto& row = captured[index] = bind_row(rows[index]);
                inputs.destination[index] = row.work->residual;
            }
            check(gather_verification_embeddings(model.embedding, scratch->device_binding->tokens,
                                                 inputs, total, stream),
                  "Look up bound verification inputs");
            if (packed) {
                model.forward_verification_layers(captured.data(), total, stream,
                                                  scratch->packed_activation);
                return;
            }
            int first = 0;
            for (const int width : shape.packet_rows) {
                if (!width)
                    break;
                model.forward_verification_layers(captured.data() + first, width, stream, nullptr);
                first += width;
            }
        },
        [&] { return scratch->upload(sequences, count, rows, total, shape, stream); });
    if (!replayed)
        return false;
    for (int lane = 0; lane < count; ++lane)
        sequences[lane].verification->use_graph_workspace(
            active_workspace(scratch->workspaces[lane]));
    for (int index = 0; index < total; ++index)
        rows[index] = bind_row(rows[index]);
    return true;
}

const ConstraintCandidates& MtpConstraintRound::candidates(int row) const { return rows_[row]; }
int MtpConstraintRound::rows() const { return prepared_rows_; }

bool TargetVerification::needs_logits() const { return host_logits_ != nullptr; }
bool TargetVerification::needs_generated_logprobs() const {
    return host_generated_logprobs_ != nullptr;
}
int TargetVerification::pending_rows() const { return pending_rows_; }

void TargetVerification::commit(int count) {
    if (!ready_ || count < 0 || count > pending_rows_) {
        throw std::logic_error("Invalid target verification commit");
    }
    check(gdn_replay_ssm::commit(request_.gdn_replay.front().cursor, gdn_layers, count, stream_),
          "Commit accepted target GDN history");
    valid_prefix_ += count;
    ready_ = false;
    pending_rows_ = 0;
    active_workspace_ = nullptr;
}
bool TargetVerification::rank_local_gdn() const { return rank_local_gdn_; }
cudaStream_t TargetVerification::stream() const { return stream_; }

Bf16* TargetVerification::bf16_head_staging() const {
    return active_workspace_ ? active_workspace_->bf16_head_packed : nullptr;
}

Bf16* TargetVerification::bf16_head_logits_staging() const {
    return active_workspace_ ? active_workspace_->bf16_head_packed_logits : nullptr;
}

VerificationWorkspace& TargetVerification::workspace() { return *active_workspace_; }

}
