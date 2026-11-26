#include "sglang/text/mtp/draft_state.h"

#include "sglang/decoder_verify.h"
#include "sglang/text/mtp/draft_projection.h"
#include "sglang/text/mtp/tc_vocab.h"

#include <algorithm>
#include <utility>

namespace sglang::decoder {

namespace {

constexpr int repair_lanes = draft_projection_rows;
constexpr int repair_depth = decoder_batch_max_mtp_steps;
constexpr int repair_seed_rows = repair_lanes * target_verify_max_rows;
constexpr int repair_kv_width = text_gqa_kv_width;
constexpr auto repair_kv_values = repair_depth * repair_lanes * 2 * repair_kv_width;
constexpr int repair_scale_stride = text_fp8_group_count(hidden_size);

int repair_seed_index(int lane, int depth) {
    return lane / draft_wave_rows * draft_wave_rows * target_verify_max_rows +
           depth * draft_wave_rows + lane % draft_wave_rows;
}

struct RepairEvent {
    explicit RepairEvent(int device) {
        DeviceScope scope(device);
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "Create peer repair event");
    }
    ~RepairEvent() {
        (void)cudaEventDestroy(event);
    }
    RepairEvent(const RepairEvent&) = delete;
    RepairEvent& operator=(const RepairEvent&) = delete;

    void record(cudaStream_t stream) const {
        check(cudaEventRecord(event, stream), "Record peer repair dependency");
    }
    void wait(cudaStream_t stream) const {
        check(cudaStreamWaitEvent(stream, event, 0), "Wait for peer repair dependency");
    }

    cudaEvent_t event{};
};

struct RepairProjectionRow {
    struct Work {
        Bf16* attention{};
        Bf16* residual{};
        Bf16* concatenated{};
        Bf16* normalized{};
        TextFp8Activation activation{};
        Bf16* keys{};
        Bf16* values{};
    } work;

    void embed(int, const Bf16*, cudaStream_t) {
        throw std::logic_error("Peer repair requires packed input");
    }
};

}

struct PeerDraftRepair::State {
    State(const MtpDraftWeights& weights, int peer)
        : weights(weights), primary(weights.state_->device), peer(peer),
          primary_memory(primary), peer_memory(peer), target_done(peer), returned(peer),
          consumed(primary) {
        {
            DeviceScope scope(primary);
            primary_kv = primary_memory.allocate<Bf16>(repair_kv_values);
        }
        DeviceScope scope(peer);
        peer_memory.open_stream();
        seeds = peer_memory.allocate<Bf16>(repair_seed_rows * hidden_size);
        projected = peer_memory.allocate<Bf16>(repair_kv_values);
        packed = peer_memory.allocate<Bf16>(draft_fc_input_values);
        auto* residual = peer_memory.allocate<Bf16>(repair_lanes * hidden_size);
        auto* normalized = peer_memory.allocate<Bf16>(repair_lanes * hidden_size);
        auto* activation = peer_memory.allocate<std::uint8_t>(repair_lanes * hidden_size);
        auto* scales = peer_memory.allocate<float>(repair_lanes * repair_scale_stride);
        for (int lane = 0; lane < repair_lanes; ++lane) {
            auto& work = projection[lane].work;
            work.residual = residual + lane * hidden_size;
            work.normalized = normalized + lane * hidden_size;
            work.activation = {activation + lane * hidden_size,
                               scales + lane * repair_scale_stride};
        }
        const auto& model = *weights.state_;
        embedding_gain = replicate(model.embedding_norm, hidden_size);
        hidden_gain = replicate(model.hidden_norm, hidden_size);
        input_gain = replicate(model.input_norm, hidden_size);
        fc = {replicate(model.fc.values, std::size_t(model.fc.rows) * model.fc.columns),
              model.fc.rows, model.fc.columns};
        keys = replicate(model.gqa.key);
        values = replicate(model.gqa.value);
        synchronize_cuda(peer_memory.stream(), "Complete peer draft repair weight preparation");
    }

    template <class T>
    T* replicate(const T* source, std::size_t count) {
        auto* destination = peer_memory.allocate<T>(count);
        check(cudaMemcpyPeerAsync(destination, peer, source, primary, count * sizeof(T),
                                 peer_memory.stream()), "Replicate peer draft repair weights");
        return destination;
    }

    Fp8Weight replicate(const Fp8Weight& weight) {
        return {replicate(weight.values, std::size_t(weight.rows) * weight.columns),
                replicate(weight.scales, std::size_t(text_fp8_group_count(weight.rows)) *
                                         text_fp8_group_count(weight.columns)),
                weight.rows, weight.columns};
    }

    const MtpDraftWeights& weights;
    int primary, peer;
    Allocations primary_memory, peer_memory;
    RepairEvent target_done, returned, consumed;
    const Bf16 *embedding_gain{}, *hidden_gain{}, *input_gain{};
    Bf16Weight fc{};
    Fp8Weight keys{}, values{};
    Bf16 *primary_kv{}, *seeds{}, *projected{}, *packed{};
    std::array<RepairProjectionRow, repair_lanes> projection;
    std::array<DraftRepairs, repair_lanes> expected;
    bool submitted = false;
    bool consume_recorded = false;
};

PeerDraftRepair::PeerDraftRepair(std::unique_ptr<State> state) : state_(std::move(state)) {}

PeerDraftRepair::~PeerDraftRepair() { drain(); }

bool PeerDraftRepair::eligible(const MtpRound* rounds, int count) {
    return count == repair_lanes && std::all_of(rounds, rounds + count, [](const MtpRound& row) {
        return row.draft && row.position <= text_attention_max_tokens - target_verify_max_rows &&
               row.proposal_limit == repair_depth && row.remaining > target_verify_max_rows &&
               !row.media && !row.constraint && !row.sampling;
    });
}

std::unique_ptr<PeerDraftRepair> PeerDraftRepair::prepare(const MtpDraftWeights& weights,
                                                        int peer) {
    const auto& model = *weights.state_;
    const int primary = model.device;
    try {
        int forward = 0, reverse = 0;
        check(cudaDeviceCanAccessPeer(&forward, primary, peer), "Check peer repair forward access");
        check(cudaDeviceCanAccessPeer(&reverse, peer, primary), "Check peer repair reverse access");
        if (!forward || !reverse) return nullptr;

        const auto fp8_bytes = [](const Fp8Weight& weight) {
            return std::size_t(weight.rows) * weight.columns + sizeof(float) *
                text_fp8_group_count(weight.rows) * text_fp8_group_count(weight.columns);
        };
        const std::size_t primary_bytes = repair_kv_values * sizeof(Bf16);
        const std::size_t peer_bytes = sizeof(Bf16) * (
            std::size_t(model.fc.rows) * model.fc.columns + 3 * hidden_size +
            repair_seed_rows * hidden_size + repair_kv_values + draft_fc_input_values +
            2 * repair_lanes * hidden_size) + repair_lanes * hidden_size +
            sizeof(float) * repair_lanes * repair_scale_stride + fp8_bytes(model.gqa.key) +
            fp8_bytes(model.gqa.value);
        {
            const DeviceScope scope(primary);
            if (available_device_bytes() < decoder_shared_reserve + primary_bytes) return nullptr;
        }
        {
            const DeviceScope scope(peer);
            if (available_device_bytes() < decoder_shared_reserve + peer_bytes) return nullptr;
        }
        auto state = std::make_unique<State>(weights, peer);
        return std::unique_ptr<PeerDraftRepair>(new PeerDraftRepair(std::move(state)));
    } catch (const CudaError& error) {
        if (error.status != cudaErrorMemoryAllocation && error.status != cudaErrorNotSupported &&
            error.status != cudaErrorPeerAccessUnsupported) throw;
        const auto pending = cudaGetLastError();
        if (pending != cudaSuccess && pending != error.status) {
            check(pending, "Check peer repair setup failure");
        }
        return nullptr;
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

void PeerDraftRepair::submit(const MtpRound* rounds, const VerificationRow* peer_rows,
                             const Bf16* peer_embedding, const Bf16* peer_norm,
                             cudaStream_t primary_stream, cudaStream_t peer_stream) {
    auto& state = *state_;
    if (state.submitted) throw std::logic_error("Peer draft repair is still pending");
    if (!use_streaming_projection(true, primary_stream)) return;
    DeviceScope scope(state.peer);
    if (!use_streaming_projection(true, peer_stream)) return;
    const auto stream = state.peer_memory.stream();
    state.submitted = true;
    state.target_done.record(peer_stream);
    state.target_done.wait(stream);
    if (state.consume_recorded) state.consumed.wait(stream);

    for (int first = 0; first < repair_seed_rows; first += tc_vocab_wide_rows) {
        TcVocabRmsRows batch{};
        for (int row = 0; row < tc_vocab_wide_rows; ++row) {
            const auto& work = *peer_rows[first + row].work;
            batch.rows[row] = {work.hidden, work.residual,
                               state.seeds + (first + row) * hidden_size};
        }
        check(pack_tc_vocab_rms(batch, peer_norm, state.seeds + first * hidden_size, stream),
              "Normalize peer target repair inputs");
    }

    std::array<RepairProjectionRow*, repair_lanes> lanes;
    std::array<MtpDraftInput, repair_lanes> inputs{};
    for (int lane = 0; lane < repair_lanes; ++lane) lanes[lane] = &state.projection[lane];
    for (int depth = 0; depth < repair_depth; ++depth) {
        for (int lane = 0; lane < repair_lanes; ++lane) {
            const auto& round = rounds[lane];
            const int token = round.tokens[depth + 1];
            const auto row = round.target->row(depth);
            state.expected[lane][depth] =
                round.draft_input(row.position, token, row.work->normalized);
            inputs[lane].token = token;
            inputs[lane].hidden = state.seeds + repair_seed_index(lane, depth) * hidden_size;
            auto* kv = state.projected + (depth * repair_lanes + lane) * 2 * repair_kv_width;
            lanes[lane]->work.keys = kv;
            lanes[lane]->work.values = kv + repair_kv_width;
        }
        using Work = RepairProjectionRow::Work;
        DraftProjectionBatch<RepairProjectionRow, Work> batch(
            lanes.data(), repair_lanes, state.packed, stream, true, false);
        batch.prepare_fc(inputs.data(), peer_embedding, state.embedding_gain, state.hidden_gain,
                         state.fc);
        batch.rms_activation(&Work::residual, state.input_gain, false);
        batch.project({{state.keys, &Work::keys}, {state.values, &Work::values}});
    }

    check(cudaMemcpyPeerAsync(state.primary_kv, state.primary, state.projected, state.peer,
                             repair_kv_values * sizeof(Bf16), stream),
          "Return peer draft repair KV");
    state.returned.record(stream);
}

bool PeerDraftRepair::apply(const DraftRepairs* repairs, cudaStream_t stream) {
    auto& state = *state_;
    if (!state.submitted) return false;
    for (int lane = 0; lane < repair_lanes; ++lane) {
        for (int depth = 0; depth < repair_depth; ++depth) {
            const auto& row = repairs[lane][depth];
            const auto& saved = state.expected[lane][depth];
            if (row.request != saved.request || row.token != saved.token ||
                row.position != saved.position || row.hidden != saved.hidden || row.mrope ||
                row.rotary_position != saved.rotary_position) {
                discard();
                return false;
            }
        }
        auto& request = *repairs[lane][0].request->state_;
        if (request.weights.get() != &state.weights ||
            repairs[lane][repair_depth - 1].position >= request.capacity) {
            discard();
            return false;
        }
        const auto& initial = repairs[lane][0];
        request.validate(initial.position, initial.token, initial.hidden);
    }

    state.returned.wait(stream);
    try {
        const auto* gain = state.weights.state_->gqa.key_norm;
        for (int depth = 0; depth < repair_depth; ++depth) {
            GqaVerifyRows<GqaVerifyPreprocessRow> preprocess{};
            GqaVerifyRows<GqaVerifyAppendRow> append{};
            for (int lane = 0; lane < repair_lanes; ++lane) {
                const auto& row = repairs[lane][depth];
                auto& request = *row.request->state_;
                const int offset = (depth * repair_lanes + lane) * 2 * repair_kv_width;
                const auto* kv = state.primary_kv + offset;
                preprocess.rows[lane] = {nullptr, kv, nullptr, request.work.rope_keys,
                                         row.position};
                append.rows[lane] = {request.cache, request.pages, request.work.rope_keys,
                                     kv + repair_kv_width, row.position};
            }
            check_draft_submission(
                draft_verify_preprocess_keys(preprocess, repair_lanes, gain, stream),
                "Rotate accepted peer draft repair keys");
            check_draft_submission(gqa_verify_append(append, repair_lanes, stream),
                                   "Append accepted peer draft repair KV");
        }
        state.consumed.record(stream);
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        for (int lane = 0; lane < repair_lanes; ++lane) {
            repairs[lane][0].request->state_->failed = true;
        }
        throw;
    }
    for (int lane = 0; lane < repair_lanes; ++lane) {
        repairs[lane][0].request->state_->prefix += repair_depth;
    }
    state.submitted = false;
    state.consume_recorded = true;
    return true;
}

void PeerDraftRepair::discard() {
    auto& state = *state_;
    if (!state.submitted) return;
    const DeviceScope scope(state.peer);
    synchronize_cuda(state.peer_memory.stream(), "Discard provisional peer draft repair");
    state.submitted = false;
}

void PeerDraftRepair::drain() noexcept {
    auto& state = *state_;
    if (state.submitted) (void)cudaStreamSynchronize(state.peer_memory.stream());
    if (state.consume_recorded) (void)cudaEventSynchronize(state.consumed.event);
    state.submitted = state.consume_recorded = false;
}

}
