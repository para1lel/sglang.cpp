#include "sglang/text/mtp/draft_projection.h"
#include "sglang/text/decode_proj.h"
#include "sglang/text/mtp/draft_batch_ops.h"
#include "sglang/decoder_prefill.h"
#include "sglang/runtime/backend/tensor_parallel.h"
#include "sglang/text/mtp/draft_state.h"
#include "sglang/text/mtp/draft_stream.h"
#include "sglang/text/target_fp8.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

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

void MtpDraftRequest::State::append(int position, bool score, const MropePosition* mrope,
                                    int rotary_position, cudaStream_t stream) {
    const auto& model = *weights->state_;
    const auto& gqa = model.gqa;
    const int rotary = rotary_position < 0 ? position : rotary_position;
    if (score) {
        const auto status =
            mrope != nullptr ? mrope_gqa_preprocess(mrope, 1, work.q_gate, work.keys,
                                                    gqa.query_norm, gqa.key_norm, work.rope_queries,
                                                    work.rope_keys, stream, model.geometry)
                             : text_gqa_preprocess(rotary, 1, work.q_gate, work.keys,
                                                   gqa.query_norm, gqa.key_norm, work.rope_queries,
                                                   work.rope_keys, stream, model.geometry);
        check(status, "Rotate MTP Q/K");
    } else {
        const auto status = mrope != nullptr
                                ? mrope_gqa_preprocess_keys(mrope, 1, work.keys, gqa.key_norm,
                                                            work.rope_keys, stream, model.geometry)
                                : text_gqa_preprocess_keys(rotary, 1, work.keys, gqa.key_norm,
                                                           work.rope_keys, stream, model.geometry);
        check(status, "Rotate MTP key");
    }
    check(mxfp8_kv_append(cache, pages, position, 1, work.rope_keys, work.values, stream,
                          model.geometry),
          "Append MTP MXFP8 KV");
}

DraftAttentionRow MtpDraftRequest::State::attention_row(int position) {
    return {
        cache,          pages,
        position,       work.rope_queries,
        work.attention, work.attention_scores(weights->state_->geometry),
    };
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::prepare_fc(const MtpDraftInput* input,
                                                   const DraftFcWeights& weights, Bf16* output) {
    const auto fc = weights.projection;
    if (output != nullptr)
        dense_residual_ = output;
    if (packed_) {
        DraftRmsRows embeddings;
        DraftRmsRows hidden;
        for (int lane = 0; lane < count_; ++lane) {
            auto* concatenated = packed_ + lane * 2 * hidden_size;
            const auto token = static_cast<std::size_t>(input[lane].token);
            embeddings.rows[lane] = {
                weights.embedding + (token * hidden_size),
                nullptr,
                concatenated,
            };
            hidden.rows[lane] = {input[lane].hidden, nullptr, concatenated + hidden_size};
        }
        check(cudaGetLastError(), "Check grouped MTP FC stream status");
        check_draft_submission(normalize_draft_inputs(embeddings, hidden, weights.embedding_norm,
                                                      weights.hidden_norm, count_, stream_),
                               "Normalize grouped MTP FC inputs");
        if (!exact_) {
            auto* result = output != nullptr ? output : packed_ + draft_fc_input_values;
            if (const auto status = try_text_bf16_decode_sm100(
                    fc.values, packed_, fc.rows, fc.columns, count_, result, stream_)) {
                check_draft_submission(*status, "Run dense BF16 MTP FC");
                dense_residual_ = result;
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
        const auto offset = static_cast<std::size_t>(lane) * fc.rows;
        projections[lane] = {
            concatenated,
            output != nullptr ? output + offset : state.work.residual,
        };
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
void DraftProjectionBatch<State, Work>::bind_partials(TpPartials partials) {
    dense_residual_ = partials.attention;
    dense_branch_ = partials.hidden;
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::project(std::initializer_list<Projection> stage) const {
    std::array<TextFp8DecodeTarget, text_decode_max_targets> targets;
    std::array<TextFp8Projection, text_decode_max_targets * draft_projection_rows> rows;
    int target = 0;
    for (const auto& [weight, output] : stage) {
        targets[target] = {weight.values, weight.scales, weight.rows, this->output(output, 0)};
        for (int index = 0; index < count_; ++index) {
            const auto& work = states_[index]->work;
            rows[target * count_ + index] = {
                {work.activation.values, work.activation.scales},
                this->output(output, index),
            };
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
        batch.entries[lane] = {output(input, lane),
                               add_residual ? output(&Work::residual, lane) : nullptr,
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
void DraftProjectionBatch<State, Work>::append(const MtpDraftInput* rows, int count,
                                               const Bf16* q_norm, const Bf16* k_norm, bool score,
                                               GqaGeometry geometry) const {
    GqaVerifyRows<GqaVerifyPreprocessRow> preprocess{};
    GqaVerifyRows<GqaVerifyAppendRow> append{};
    for (int lane = 0; lane < count; ++lane) {
        const auto& row = rows[lane];
        const auto& request = *states_[lane];
        const auto& work = request.work;
        const int rotary = row.rotary_position < 0 ? row.position : row.rotary_position;
        preprocess.rows[lane] = {work.q_gate, work.keys, work.rope_queries, work.rope_keys,
                                 rotary};
        append.rows[lane] = {request.cache, request.pages, work.rope_keys, work.values,
                             row.position};
    }
    const auto status =
        score ? gqa_verify_preprocess(preprocess, count, q_norm, k_norm, stream_, geometry)
              : draft_verify_preprocess_keys(preprocess, count, k_norm, stream_, geometry);
    check_draft_submission(status, score ? "Rotate depth-major MTP Q/K"
                                         : "Rotate depth-major MTP keys");
    check_draft_submission(gqa_verify_append(append, count, stream_),
                           "Append depth-major MTP MXFP8 KV");
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::prepare_attention(const MtpDraftInput* rows, int count,
                                                          const GqaLayer& gqa, bool score,
                                                          GqaGeometry geometry,
                                                          bool grouped_rope) const {
    if (score) {
        project({
            {gqa.q_gate, &Work::q_gate},
            {gqa.key, &Work::keys},
            {gqa.value, &Work::values},
        });
    } else {
        project({{gqa.key, &Work::keys}, {gqa.value, &Work::values}});
    }
    if (grouped_rope) {
        append(rows, count, gqa.query_norm, gqa.key_norm, score, geometry);
    } else {
        for (int lane = 0; lane < count; ++lane) {
            const auto& input = rows[lane];
            states_[lane]->append(input.position, score, input.mrope, input.rotary_position,
                                  stream_);
        }
    }
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::read_attention(const MtpDraftInput* rows,
                                                       bool online) const {
    if (online && count_ == draft_attention_batch_rows) {
        read_attention_packets<draft_attention_batch_rows>(rows, online);
        return;
    }
    read_attention_packets<draft_wave_rows>(rows, online);
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::gate_attention(GqaGeometry geometry) const {
    GqaVerifyRows<GqaVerifyGateRow> gate{};
    for (int lane = 0; lane < count_; ++lane) {
        const auto& work = states_[lane]->work;
        gate.rows[lane] = {work.q_gate, work.attention};
    }
    check_draft_submission(gqa_verify_postgate(gate, count_, stream_, geometry),
                           "Gate depth-major MTP attention");
    quantize_activation(&Work::attention, nullptr, geometry.query_heads * mxfp8_head_dim);
}

template <typename State, typename Work>
void DraftProjectionBatch<State, Work>::normalize_output(const Bf16* weight) const {
    DraftRmsRows rows;
    for (int lane = 0; lane < count_; ++lane) {
        const auto& work = states_[lane]->work;
        rows.rows[lane] = {
            output(&Work::branch, lane),
            output(&Work::residual, lane),
            work.normalized,
        };
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
Bf16* DraftProjectionBatch<State, Work>::output(Output member, int lane) const {
    const auto offset = static_cast<std::size_t>(lane) * hidden_size;
    if (member == &Work::residual && dense_residual_ != nullptr)
        return dense_residual_ + offset;
    if (member == &Work::branch && dense_branch_ != nullptr)
        return dense_branch_ + offset;
    return states_[lane]->work.*member;
}

template class DraftProjectionBatch<MtpDraftRequest::State, DraftWorkspace>;

namespace {

void interleave_fc(const Bf16* gathered, Bf16* output, int rows, cudaStream_t stream) {
    constexpr auto bytes = hidden_size * sizeof(Bf16) / 2;
    for (int rank = 0; rank < 2; ++rank) {
        check(cudaMemcpy2DAsync(output + (rank * hidden_size / 2), 2 * bytes,
                                gathered + (rank * rows * hidden_size / 2), bytes, bytes, rows,
                                cudaMemcpyDeviceToDevice, stream),
              "Assemble tensor-parallel MTP FC rows");
    }
}

}

bool MtpDraftRequest::State::bind_tp_rows(MtpDraftInput* const (&inputs)[2], int rank,
                                          State** states, int count) {
    const auto* rows = inputs[rank];
    const auto* primary = inputs[0];
    if (rows[0].request == nullptr)
        throw std::invalid_argument("Missing MTP rank");
    const auto* weights = rows[0].request->state_->weights.get();
    const auto& shard = weights->state_->shard;
    if (shard.world_size != 2 || shard.rank != rank) {
        throw std::invalid_argument("Invalid tensor-parallel MTP weights");
    }
    bool grouped_rope = true;
    for (int lane = 0; lane < count; ++lane) {
        const auto& row = rows[lane];
        if (row.request == nullptr || row.position != primary[lane].position ||
            row.token != primary[lane].token) {
            throw std::invalid_argument("Mismatched tensor-parallel MTP input");
        }
        auto* state = row.request->state_.get();
        state->validate(row.position, row.token, row.hidden);
        if (state->weights.get() != weights ||
            std::find(states, states + lane, state) != states + lane) {
            throw std::invalid_argument("Invalid tensor-parallel MTP lane");
        }
        states[lane] = state;
        const int rotary = row.rotary_position < 0 ? row.position : row.rotary_position;
        grouped_rope &= row.mrope == nullptr && rotary < text_attention_max_tokens;
    }
    return grouped_rope;
}

void MtpDraftRequest::enqueue_tp(MtpDraftInput* const (&rows)[2], int count, bool score,
                                 const cudaStream_t (&streams)[2], const TpPartials (&partials)[2],
                                 std::uint8_t* const (&packed)[2], TensorParallel& group,
                                 const ConstraintWorkspace* scratch,
                                 const ConstraintCandidates* const* allowed) {
    if (count == 0)
        return;
    if (count < 0 || count > draft_projection_rows || rows[0] == nullptr || rows[1] == nullptr) {
        throw std::invalid_argument("Invalid tensor-parallel MTP wave");
    }
    State* states[2][draft_projection_rows];
    const MtpDraftWeights::State* models[2];
    bool grouped_rope[2];
    for (int rank = 0; rank < 2; ++rank) {
        grouped_rope[rank] = State::bind_tp_rows(rows, rank, states[rank], count);
        models[rank] = states[rank][0]->weights->state_.get();
    }
    using Batch = DraftProjectionBatch<State, DraftWorkspace>;
    Batch batch[2] = {
        {states[0], count, reinterpret_cast<Bf16*>(packed[0]), streams[0], false, false},
        {states[1], count, reinterpret_cast<Bf16*>(packed[1]), streams[1], false, false},
    };
    const Bf16* shards[] = {partials[0].attention, partials[1].attention};
    Bf16* reduced[] = {partials[0].hidden, partials[1].hidden};
    const auto each_rank = [&](const auto& submit) {
        for (int rank = 0; rank < 2; ++rank) {
            const DeviceScope scope(models[rank]->device);
            submit(rank, *models[rank], batch[rank], streams[rank]);
        }
    };
    const auto elements = static_cast<std::size_t>(count) * hidden_size;
    try {
        each_rank([&](int rank, const auto& model, Batch& work, cudaStream_t) {
            work.prepare_fc(rows[rank],
                            {
                                model.embedding,
                                model.embedding_norm,
                                model.hidden_norm,
                                model.fc,
                            },
                            partials[rank].attention);
        });
        group.allgather_bf16(shards, reduced, streams, elements / 2);
        each_rank([&](int rank, const auto& model, Batch& work, cudaStream_t stream) {
            interleave_fc(reduced[rank], partials[rank].attention, count, stream);
            work.bind_partials(partials[rank]);
            work.rms_activation(&DraftWorkspace::residual, model.input_norm, false);
            const auto& gqa = model.gqa;
            work.prepare_attention(rows[rank], count, gqa, score, model.geometry,
                                   grouped_rope[rank]);
            if (score) {
                work.read_attention(rows[rank], model.capabilities.online_attention);
                work.gate_attention(model.geometry);
                work.project({{gqa.output, &DraftWorkspace::branch}});
            }
        });
        if (score) {
            group.allreduce_bf16(reduced, streams, elements);
            each_rank([&](int, const auto& model, Batch& work, cudaStream_t) {
                work.rms_activation(&DraftWorkspace::branch, model.attention_norm, true);
                work.project({
                    {model.mlp_gate, &DraftWorkspace::gate},
                    {model.mlp_up, &DraftWorkspace::up},
                });
                work.quantize_activation(&DraftWorkspace::gate, &DraftWorkspace::up,
                                         model.shard.local_intermediate_size());
                work.project({{model.mlp_down, &DraftWorkspace::branch}});
            });
            group.allreduce_bf16(reduced, streams, elements);
            each_rank([](int, const auto& model, Batch& work, cudaStream_t) {
                work.normalize_output(model.final_norm);
            });
            State* const* const score_states[] = {states[0], states[1]};
            score_tp(score_states, count, streams, group, scratch, allowed);
        }
    } catch (...) {
        for (const auto& rank : states) {
            for (int lane = 0; lane < count; ++lane)
                rank[lane]->failed = true;
        }
        throw;
    }
    for (int rank = 0; rank < 2; ++rank) {
        for (int lane = 0; lane < count; ++lane) {
            auto& state = *states[rank][lane];
            ++state.prefix;
            if (!score)
                state.scored = false;
            rows[rank][lane].hidden = score ? state.work.normalized : nullptr;
        }
    }
}

void MtpDraftRequest::seed_tp(MtpDraftInput primary, MtpDraftInput peer, TensorParallel& group,
                              cudaStream_t stream, cudaStream_t peer_stream) {
    MtpDraftInput* const rows[] = {&primary, &peer};
    const cudaStream_t streams[] = {stream, peer_stream};
    const auto& root = primary.request->state_->work;
    const auto& other = peer.request->state_->work;
    const TpPartials partials[] = {{root.residual, root.branch}, {other.residual, other.branch}};
    std::uint8_t* const packed[] = {
        reinterpret_cast<std::uint8_t*>(root.concatenated),
        reinterpret_cast<std::uint8_t*>(other.concatenated),
    };
    enqueue_tp(rows, 1, false, streams, partials, packed, group);
}

void MtpDraftRequest::score_tp(State* const* const (&states)[2], int count,
                               const cudaStream_t (&streams)[2], TensorParallel& group,
                               const ConstraintWorkspace* scratch,
                               const ConstraintCandidates* const* allowed) {
    const auto& root = *states[0][0]->weights->state_;
    const auto& peer = *states[1][0]->weights->state_;
    const bool int8 = root.quantized_head != nullptr && peer.quantized_head != nullptr;
    for (int rank = 0; rank < 2; ++rank) {
        const auto& model = rank != 0 ? peer : root;
        const DeviceScope scope(model.device);
        State* unconstrained[draft_wave_rows];
        int size = 0;
        for (int first = 0; first < count; first += draft_wave_rows) {
            const int end = std::min(count, first + draft_wave_rows);
            for (int lane = first; lane < end; ++lane) {
                auto* state = states[rank][lane];
                state->retried_score = false;
                if (allowed != nullptr && allowed[lane] != nullptr) {
                    State::project_scores(&state, 1, false, streams[rank], model.vocab_begin);
                } else {
                    unconstrained[size++] = state;
                }
            }
            if (size != 0) {
                State::project_scores(unconstrained, size, int8, streams[rank]);
                State::select_tp_wave(unconstrained, size, int8, streams[rank]);
                size = 0;
            }
        }
    }
    for (int lane = 0; allowed != nullptr && lane < count; ++lane) {
        if (allowed[lane] == nullptr)
            continue;
        auto& primary = *states[0][lane];
        const auto& other = *states[1][lane];
        const Bf16* local[] = {primary.work.logits, other.work.logits + peer.vocab_begin};
        group.allgather_bf16(local, std::array{primary.work.logits, other.work.logits}.data(),
                             streams, root.lm_head.rows);
        const DeviceScope scope(root.device);
        primary.select_score(streams[0], scratch, allowed[lane]);
    }
}

bool MtpDraftRequest::retry_invalid_tp(MtpDraftRequest& primary, MtpDraftRequest& peer,
                                       const cudaStream_t (&streams)[2]) {
    auto& root = *primary.state_;
    auto& other = *peer.state_;
    if (!root.int8_score || (root.tp_winner().token_id >= 0 && other.tp_winner().token_id >= 0)) {
        return false;
    }
    static_cast<void>(std::fputs(
        "INT8 MTP scores are invalid; retrying draft vocabulary head in BF16\n", stderr));
    for (int rank = 0; rank < 2; ++rank) {
        auto* state = rank == 0 ? &root : &other;
        const DeviceScope scope(state->weights->state_->device);
        State::project_scores(&state, 1, false, streams[rank]);
        State::select_tp_wave(&state, 1, false, streams[rank]);
        state->retried_score = true;
    }
    return true;
}

int MtpDraftRequest::read_tp_token(MtpDraftRequest& primary, MtpDraftRequest& peer) {
    auto& root = *primary.state_;
    auto& other = *peer.state_;
    if (root.constrained_score)
        return primary.read_token();
    if (root.failed || other.failed || !root.scored || !other.scored) {
        throw std::logic_error("MTP draft ranks have no completed score to read");
    }
    const auto left = root.tp_winner();
    const auto right = other.tp_winner();
    const auto winner =
        left.token_id >= 0 && (right.token_id < 0 || left.score >= right.score) ? left : right;
    if (root.retried_score && (winner.token_id < 0 || !std::isfinite(winner.score)))
        return -1;
    if (winner.token_id < 0)
        throw std::runtime_error("MTP argmax did not return a valid token ID");
    return winner.token_id;
}

void MtpDraftRequest::seed_prefill_tp(MtpDraftRequest& primary, MtpDraftRequest& peer, int position,
                                      const int* shifted_tokens, const Bf16* (&hidden)[2],
                                      const Bf16* (&residual)[2], const Bf16* (&final_norm)[2],
                                      const PrefillProjections (&projections)[2],
                                      MtpPrefillBuffers (&buffers)[2], TensorParallel& group) {
    State* states[] = {primary.state_.get(), peer.state_.get()};
    const int total = projections[0].rows;
    if (position < 0 || total <= 0 || projections[1].rows != total || shifted_tokens == nullptr ||
        states[0]->prefix != states[1]->prefix) {
        throw std::invalid_argument("Invalid tensor-parallel MTP prefill interval");
    }
    for (int rank = 0; rank < 2; ++rank) {
        const auto& state = *states[rank];
        if (state.failed || position > state.prefix || total > state.capacity - position ||
            hidden[rank] == nullptr || residual[rank] == nullptr || final_norm[rank] == nullptr) {
            throw std::invalid_argument("Invalid MTP prefill rank state");
        }
    }
    const cudaStream_t streams[] = {projections[0].stream, projections[1].stream};
    const Bf16* local[] = {buffers[0].embedding, buffers[1].embedding};
    try {
        for (int offset = states[0]->prefix - position; offset < total;
             offset += prefill_chunk_rows) {
            const int rows = std::min(prefill_chunk_rows, total - offset);
            const auto elements = static_cast<std::size_t>(rows) * hidden_size;
            const auto hidden_offset = static_cast<std::size_t>(offset) * hidden_size;
            for (int rank = 0; rank < 2; ++rank) {
                const auto& model = *states[rank]->weights->state_;
                const DeviceScope scope(model.device);
                const auto& work = buffers[rank];
                auto tile = projections[rank];
                tile.rows = rows;
                auto* const stream = streams[rank];
                check(text_rms_norm_rows(hidden[rank] + hidden_offset,
                                         residual[rank] + hidden_offset, final_norm[rank],
                                         hidden_size, rows, work.normalized_hidden, nullptr,
                                         stream),
                      "Normalize target MTP rows");
                check(text_embedding_rows(model.embedding, vocab_size, hidden_size,
                                          shifted_tokens + offset, rows, work.embedding, stream),
                      "Gather shifted MTP tokens");
                check(text_rms_norm_pair_rows(work.embedding, work.normalized_hidden,
                                              model.embedding_norm, model.hidden_norm, hidden_size,
                                              rows, work.concatenated, stream),
                      "Normalize concatenated MTP rows");
                tile.project_seed_fc(model.fc, work.concatenated, work.embedding);
            }
            group.allgather_bf16(local,
                                 std::array{buffers[0].layer_input, buffers[1].layer_input}.data(),
                                 streams, elements / 2);
            for (int rank = 0; rank < 2; ++rank) {
                const auto& state = *states[rank];
                const auto& model = *state.weights->state_;
                const DeviceScope scope(model.device);
                const auto& work = buffers[rank];
                auto tile = projections[rank];
                tile.rows = rows;
                auto* const stream = streams[rank];
                auto* assembled = work.layer_input + elements;
                interleave_fc(work.layer_input, assembled, rows, stream);
                check(text_rms_norm_rows(assembled, nullptr, model.input_norm, hidden_size, rows,
                                         work.embedding, nullptr, stream),
                      "Normalize MTP layer input");
                const auto input = tile.quantize(work.embedding, hidden_size);
                tile.project(model.gqa.key, input, work.keys);
                tile.project(model.gqa.value, input, work.values);
                const int first = position + offset;
                check(text_gqa_preprocess_keys(first, rows, work.keys, model.gqa.key_norm,
                                               work.rope_keys, stream, model.geometry),
                      "Rotate tensor-parallel MTP prefill keys");
                check(mxfp8_kv_append(state.cache, state.pages, first, rows, work.rope_keys,
                                      work.values, stream, model.geometry),
                      "Append rank-local MTP prefill KV");
            }
            for (auto* state : states) {
                state->prefix += rows;
                state->scored = false;
            }
        }
    } catch (...) {
        for (auto* state : states)
            state->failed = true;
        throw;
    }
}
}
