#include "sglang/decoder_verify.h"

namespace sglang::decoder {

VerificationWorkspace::VerificationWorkspace(Allocations& memory, int rows) {
    const int layer_elements = 2 * rows * (text_gdn_qkv_width + text_gdn_value_heads);
    auto* storage = memory.allocate<Bf16>(gdn_layers * layer_elements);
    for (auto& layer : replay) {
        const int qkv = rows * text_gdn_qkv_width;
        const int gates = rows * text_gdn_value_heads;
        layer.inputs = {storage, storage + qkv, storage + 2 * qkv,
                        storage + 2 * qkv + gates};
        storage += layer_elements;
    }
    auto* z = memory.allocate<Bf16>(rows * text_gdn_value_width);
    auto* attention = memory.allocate<Bf16>(rows * text_gdn_value_width);
    observable = memory.allocate<Bf16>(rows * text_gdn_value_width);
    winners = memory.allocate<TextLogitCandidate>(rows);
    bf16_head_packed = memory.allocate<Bf16>(
        static_cast<std::size_t>(verification_packet_rows) * (hidden_size + vocab_size));
    bf16_head_packed_logits =
        bf16_head_packed + static_cast<std::size_t>(verification_packet_rows) * hidden_size;
    work.reserve(rows);
    for (int index = 0; index < rows; ++index) {
        auto& row = work.emplace_back(memory, attention + index * text_gdn_value_width,
                                      winners + index);
        row.gdn_z = z + index * text_gdn_value_width;
    }
}

void VerificationWorkspace::bind_gdn(int layer, int rows) {
    const auto inputs = replay[layer].inputs;
    for (int index = 0; index < rows; ++index) {
        work[index].gdn_qkv = inputs.qkv + index * text_gdn_qkv_width;
        work[index].gdn_b = inputs.b + index * text_gdn_value_heads;
        work[index].gdn_a = inputs.a + index * text_gdn_value_heads;
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
