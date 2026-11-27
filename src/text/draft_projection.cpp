#include "sglang/text/mtp/draft_projection.h"
#include "sglang/text/decode_proj.h"
#include "sglang/text/mtp/draft_batch_ops.h"
#include "sglang/text/mtp/draft_state.h"
#include "sglang/text/mtp/draft_stream.h"
#include "sglang/text/target_fp8.h"

#include <algorithm>
#include <array>

namespace sglang::decoder {

void check_draft_submission(cudaError_t direct, const char* operation) {
    if (direct == cudaSuccess) return;
    const auto recorded = cudaGetLastError();
    const auto status = !CudaError::terminal(direct) && CudaError::terminal(recorded)
        ? recorded : direct;
    check(status, operation);
}

bool use_streaming_projection(bool supported, cudaStream_t stream) {
    if (!supported) return false;
    cudaStreamCaptureStatus status;
    check(cudaStreamIsCapturing(stream, &status), "Check draft projection capture status");
    return status == cudaStreamCaptureStatusNone;
}

template <typename State, typename Work>
DraftProjectionBatch<State, Work>::DraftProjectionBatch(
    State* const* states, int count, Bf16* packed, cudaStream_t stream, bool exact, bool streaming)
    : states_(states), count_(count), packed_(packed), stream_(stream), exact_(exact),
      streaming_(streaming) {}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::prepare_fc(
    const MtpDraftInput* input, const Bf16* embedding, const Bf16* embedding_norm,
    const Bf16* hidden_norm, Bf16Weight fc) {
    if (packed_) {
        DraftRmsRows embeddings;
        DraftRmsRows hidden;
        for (int lane = 0; lane < count_; ++lane) {
            auto* concatenated = packed_ + lane * 2 * hidden_size;
            const auto token = static_cast<std::size_t>(input[lane].token);
            embeddings.rows[lane] = {embedding + token * hidden_size, nullptr, concatenated};
            hidden.rows[lane] = {input[lane].hidden, nullptr, concatenated + hidden_size};
        }
        check(cudaGetLastError(), "Check grouped MTP FC stream status");
        check_draft_submission(normalize_draft_inputs(embeddings, hidden, embedding_norm,
                               hidden_norm, count_, stream_),
                               "Normalize grouped MTP FC inputs");
        if (!exact_) {
            auto* output = packed_ + draft_fc_input_values;
            if (const auto status = try_text_bf16_decode_sm100(fc.values, packed_, fc.rows,
                                    fc.columns, count_, output, stream_)) {
                check_draft_submission(*status, "Run dense BF16 MTP FC");
                dense_residual_ = output;
                return;
            }
            check(cudaGetLastError(), "Check declined dense MTP FC status");
        }
    }
    std::array<TextBf16Projection, draft_projection_rows> projections;
    for (int lane = 0; lane < count_; ++lane) {
        auto& state = *states_[lane];
        if (!packed_) state.embed(input[lane].token, input[lane].hidden, stream_);
        const auto* concatenated = packed_ ? packed_ + lane * 2 * hidden_size
                                           : state.work.concatenated;
        projections[lane] = {concatenated, state.work.residual};
    }
    if (streaming_) {
        check_draft_submission(draft_stream::project_fc(fc.values, projections.data(), 2,
                                                       stream_),
                               "Run streaming batched BF16 MTP FC");
        return;
    }
    const int width = exact_ ? count_ : draft_wave_rows;
    for (int first = 0; first < count_; first += width) {
        const int size = std::min(count_ - first, width);
        check_draft_submission(text_bf16_gemm(fc.values, fc.rows, fc.columns,
            projections.data() + first, size, stream_), "Run batched BF16 MTP FC");
    }
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::project(std::initializer_list<Projection> stage) const {
    std::array<TextFp8DecodeTarget, text_decode_max_targets> targets;
    std::array<TextFp8Projection, text_decode_max_targets * draft_projection_rows> rows;
    int target = 0;
    for (const auto& [weight, output] : stage) {
        targets[target] = {weight.values, weight.scales, weight.rows, nullptr};
        for (int index = 0; index < count_; ++index) {
            const auto& work = states_[index]->work;
            rows[target * count_ + index] = {
                {work.activation.values, work.activation.scales}, work.*output};
        }
        ++target;
    }
    const int columns = stage.begin()->first.columns;
    if (packed_ && !exact_) {
        check(cudaGetLastError(), "Check draft projection stream status");
        const auto status = try_target_fp8_tc(targets.data(), target, columns, rows.data(),
                                             count_, packed_, stream_);
        if (status) {
            check_draft_submission(*status, "Run depth-major MTP FP8");
            return;
        }
    }
    if (streaming_ && (target == 2 || target == 3) &&
        stage.begin()[target - 2].second == &Work::keys &&
        stage.begin()[target - 1].second == &Work::values) {
        submit_joined_kv(targets.data(), rows.data(), target - 2, columns);
        return;
    }
    target = 0;
    const int width = exact_ ? count_ : draft_wave_rows;
    for (const auto& projection : stage) {
        const auto& weight = projection.first;
        for (int first = 0; first < count_; first += width) {
            const int size = std::min(count_ - first, width);
            check_draft_submission(text_fp8_gemm(weight.values, weight.scales, weight.rows,
                weight.columns, rows.data() + target * count_ + first, size, stream_),
                "Run batched MTP FP8");
        }
        ++target;
    }
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::rms_activation(
    Output input, const Bf16* weight, bool add_residual) const {
    WideVerifyRmsQuantizeRows batch{};
    for (int lane = 0; lane < count_; ++lane) {
        const auto& work = states_[lane]->work;
        const auto* value = input == &Work::residual ? residual(lane) : work.*input;
        batch.entries[lane] = {value, add_residual ? residual(lane) : nullptr,
                               work.normalized, work.activation};
    }
    check_draft_submission(verify_rms_quantize_wide(batch, count_, weight, stream_),
                           "Prepare depth-major MTP RMS and FP8");
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::quantize_activation(
    Output input, Output up, int width) const {
    WideVerifyQuantizeRows batch{};
    for (int lane = 0; lane < count_; ++lane) {
        const auto& work = states_[lane]->work;
        batch.entries[lane] = {work.*input, up ? work.*up : nullptr, work.activation};
    }
    check_draft_submission(verify_quantize_fp8_wide(batch, count_, width, stream_),
                           "Prepare depth-major MTP FP8 activation");
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::append(
    const MtpDraftInput* rows, const Bf16* q_norm, const Bf16* k_norm, bool score) const {
    GqaVerifyRows<GqaVerifyPreprocessRow> preprocess{};
    GqaVerifyRows<GqaVerifyAppendRow> append{};
    for (int lane = 0; lane < count_; ++lane) {
        const auto& row = rows[lane];
        const auto& request = *states_[lane];
        const auto& work = request.work;
        const int rotary = row.rotary_position < 0 ? row.position : row.rotary_position;
        preprocess.rows[lane] = {work.q_gate, work.keys, work.rope_queries, work.rope_keys,
                                 rotary};
        append.rows[lane] = {request.cache, request.pages, work.rope_keys, work.values,
                             row.position};
    }
    const auto status = score
        ? gqa_verify_preprocess(preprocess, count_, q_norm, k_norm, stream_)
        : draft_verify_preprocess_keys(preprocess, count_, k_norm, stream_);
    check_draft_submission(status, score ? "Rotate depth-major MTP Q/K"
                                         : "Rotate depth-major MTP keys");
    check_draft_submission(gqa_verify_append(append, count_, stream_),
                           "Append depth-major MTP MXFP8 KV");
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::read_attention(
    const MtpDraftInput* rows, bool online) const {
    if (online && count_ == draft_attention_batch_rows) {
        read_attention_packets<draft_attention_batch_rows>(rows, online);
        return;
    }
    read_attention_packets<draft_wave_rows>(rows, online);
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::gate_attention() const {
    GqaVerifyRows<GqaVerifyGateRow> gate{};
    for (int lane = 0; lane < count_; ++lane) {
        const auto& work = states_[lane]->work;
        gate.rows[lane] = {work.q_gate, work.attention};
    }
    check_draft_submission(gqa_verify_postgate(gate, count_, stream_),
                           "Gate depth-major MTP attention");
    quantize_activation(&Work::attention, nullptr, text_gqa_q_width);
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::normalize_output(const Bf16* weight) const {
    DraftRmsRows rows;
    for (int lane = 0; lane < count_; ++lane) {
        const auto& work = states_[lane]->work;
        rows.rows[lane] = {work.branch, residual(lane), work.normalized};
    }
    check_draft_submission(normalize_draft_output(rows, weight, count_, stream_),
                           "Normalize depth-major MTP output");
}

template <typename State, typename Work>
template <int Width>
void DraftProjectionBatch<State, Work>::read_attention_packets(
    const MtpDraftInput* rows, bool online) const {
    for (int first = 0; first < count_; first += Width) {
        const int size = std::min(count_ - first, Width);
        std::array<DraftAttentionRow, Width> attention;
        for (int index = 0; index < size; ++index) {
            const int lane = first + index;
            attention[index] = states_[lane]->attention_row(rows[lane].position);
        }
        check_draft_submission(mtp_gqa_attention_batch(attention.data(), size, online, stream_),
                               "Read batched MTP MXFP8 attention");
    }
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::submit_joined_kv(
    const TextFp8DecodeTarget* targets, const TextFp8Projection* rows,
    int first, int columns) const {
    if (first == 1) {
        const auto& q = targets[0];
        check_draft_submission(text_fp8_gemm(q.weight, q.weight_scales, q.out_width, columns,
            rows, count_, stream_), "Run batched MTP Q");
    }
    const auto& key = targets[first];
    const auto& value = targets[first + 1];
    check_draft_submission(draft_stream::project_kv(key.weight, key.weight_scales,
        value.weight, value.weight_scales, rows + first * count_, count_, stream_),
        "Run joined MTP KV");
}

template <typename State, typename Work>
Bf16* DraftProjectionBatch<State, Work>::residual(int lane) const {
    return dense_residual_ ? dense_residual_ + lane * hidden_size
                           : states_[lane]->work.residual;
}

template class DraftProjectionBatch<MtpDraftRequest::State, DraftWorkspace>;

template DraftProjectionBatch<RepairProjectionRow, RepairProjectionRow::Work>::DraftProjectionBatch(
    RepairProjectionRow* const*, int, Bf16*, cudaStream_t, bool, bool);
template void DraftProjectionBatch<RepairProjectionRow, RepairProjectionRow::Work>::prepare_fc(
    const MtpDraftInput*, const Bf16*, const Bf16*, const Bf16*, Bf16Weight);
template void DraftProjectionBatch<RepairProjectionRow, RepairProjectionRow::Work>::rms_activation(
    Output, const Bf16*, bool) const;
template void DraftProjectionBatch<RepairProjectionRow, RepairProjectionRow::Work>::project(
    std::initializer_list<Projection>) const;

}
