#include "sglang/decoder_verify.h"
#include "sglang/text/gdn_local.h"

namespace sglang::decoder {

VerificationWorkspace::VerificationWorkspace(Allocations& memory, int rows, bool rank_local)
    : bf16_head_packed(memory.allocate<Bf16>(static_cast<std::size_t>(verification_packet_rows) *
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
