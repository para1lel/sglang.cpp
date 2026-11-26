#include "sglang/runtime/execution/tp_verify_graph.h"
#include "sglang/runtime/backend/tensor_parallel.h"

#include <exception>
#include <stdexcept>

namespace sglang::decoder::tp_verify_graph {

thread_local HostBinding* active[2] = {};
namespace {
bool capture_rejected(cudaError_t status) {
    return status == cudaErrorNotSupported || status == cudaErrorMemoryAllocation ||
        status == cudaErrorStreamCaptureUnsupported ||
        status == cudaErrorStreamCaptureInvalidated ||
        status == cudaErrorStreamCaptureUnjoined;
}

void finish_capture(const std::array<int, 2>& devices,
                    const std::array<cudaStream_t, 2>& streams) {
    const auto pending = cudaGetLastError();
    if (!capture_rejected(pending)) check(pending, "Check TP graph capture error");
    for (int rank = 0; rank < 2; ++rank) {
        DeviceScope scope(devices[rank]);
        cudaStreamCaptureStatus status;
        check(cudaStreamIsCapturing(streams[rank], &status), "Check TP graph capture unwind");
        if (status != cudaStreamCaptureStatusNone) {
            throw std::runtime_error("TP verifier rank remains in capture");
        }
    }
}

bool streams_available(const std::array<int, 2>& devices,
                       const std::array<cudaStream_t, 2>& streams) {
    for (int rank = 0; rank < 2; ++rank) {
        DeviceScope scope(devices[rank]);
        cudaStreamCaptureStatus status;
        check(cudaStreamIsCapturing(streams[rank], &status), "Check TP verifier stream capture");
        if (status != cudaStreamCaptureStatusNone) return false;
    }
    return true;
}

struct Rank {
    Rank(int id, cudaStream_t lane_stream) : device(id), stream(lane_stream), memory(id) {
        DeviceScope owner(device);
        binding.stream = stream;
        binding.device = memory.allocate<DeviceBinding>(1);
        binding.replay = replay;
        auto* attention = memory.allocate<Bf16>(rows * GdnLocal::value_width);
        auto* z = memory.allocate<Bf16>(rows * GdnLocal::value_width);
        auto* observed = memory.allocate<Bf16>(rows * GdnLocal::value_width);
        for (int lane = 0; lane < lanes; ++lane) {
            for (int step = 0; step < depth; ++step) {
                const int index = step * lanes + lane;
                const int lane_row = lane * depth + step;
                work[index] = Workspace(memory, attention + lane_row * GdnLocal::value_width);
                work[index].gdn_z = z + lane_row * GdnLocal::value_width;
            }
            for (int layer = 0; layer < gdn_layers; ++layer) {
                constexpr int qkv = depth * GdnLocal::qkv_width;
                constexpr int gates = depth * GdnLocal::value_heads;
                auto* packed = memory.allocate<Bf16>(2 * (qkv + gates));
                replay[layer][lane] = {packed, packed + qkv, packed + 2 * qkv,
                                       packed + 2 * qkv + gates};
                gdn[layer][lane] = {replay[layer][lane], work[lane].gdn_z, {}, {},
                                    observed + lane * depth * GdnLocal::value_width,
                                    work[lane].attention};
            }
        }
        try {
            check(cudaMallocHost(&staging, sizeof(*staging)), "Allocate pinned graph binding");
            check(cudaEventCreateWithFlags(&edge, cudaEventDisableTiming), "Create TP graph edge");
        } catch (...) {
            if (staging) cudaFreeHost(staging);
            throw;
        }
    }

    ~Rank() {
        DeviceScope owner(device);
        if (edge) cudaEventDestroy(edge);
        if (staging) cudaFreeHost(staging);
    }

    int device;
    cudaStream_t stream;
    Allocations memory;
    HostBinding binding{};
    DeviceBinding* staging = nullptr;
    cudaEvent_t edge = nullptr;
    Workspace work[rows];
    TextGdnReplayRows replay[gdn_layers][lanes];
    GdnVerificationLane gdn[gdn_layers][lanes];
};

bool eligible(const Input& input, const std::array<cudaStream_t, 2>& streams) {
    for (int rank = 0; rank < 2; ++rank) {
        const auto* packet = input[rank];
        if (packet[0].verification == packet[1].verification) return false;
        for (int index = 0; index < rows; ++index) {
            const auto& row = packet[index];
            const auto& first = packet[index % lanes];
            if (row.constrained || row.binding || row.position < 0 ||
                row.position >= score_capacity || row.first != (index < lanes) ||
                !row.verification || row.verification != first.verification ||
                row.request != first.request || row.position != first.position + index / lanes) {
                return false;
            }
        }
        for (int lane = 0; lane < lanes; ++lane) {
            const auto& first = packet[lane];
            if (!first.verification->rank_local_gdn() ||
                first.verification->stream() != streams[rank] ||
                first.verification->pending_rows() != depth) return false;
            for (const auto& cache : first.request->caches) {
                if (cache.kv_heads != mxfp8_kv_heads / 2 ||
                    !cache_covers(cache, first.request->pages, first.position + depth)) {
                    return false;
                }
            }
        }
    }
    return true;
}
}

class Graph {
public:
    Graph(const std::array<int, 2>& devices, const std::array<cudaStream_t, 2>& streams) {
        for (int index = 0; index < 2; ++index) {
            ranks[index] = std::make_unique<Rank>(devices[index], streams[index]);
        }
    }

    ~Graph() {
        if (executable) {
            DeviceScope root(ranks[0]->device);
            cudaGraphExecDestroy(executable);
        }
    }

    bool matches(const std::array<int, 2>& devices,
                 const std::array<cudaStream_t, 2>& streams) const {
        for (int rank = 0; rank < 2; ++rank) {
            if (ranks[rank]->device != devices[rank] || ranks[rank]->stream != streams[rank]) {
                return false;
            }
        }
        return true;
    }

    bool capture(const Input& input, const Forward& forward) {
        auto& root = *ranks[0];
        auto& peer = *ranks[1];
        std::array<std::array<VerificationRow, rows>, 2> resident;
        for (int rank = 0; rank < 2; ++rank) {
            for (int index = 0; index < rows; ++index) {
                resident[rank][index] = input[rank][index];
                resident[rank][index].work = &ranks[rank]->work[index];
            }
        }
        DeviceScope root_scope(root.device);
        const auto status = cudaStreamBeginCapture(root.stream, cudaStreamCaptureModeThreadLocal);
        if (capture_rejected(status)) return false;
        check(status, "Begin fused TP verifier capture");
        bool joined = false;
        std::exception_ptr error;
        try {
            check(cudaEventRecord(root.edge, root.stream), "Fork capture from root");
            {
                DeviceScope device(peer.device);
                check(cudaStreamWaitEvent(peer.stream, root.edge), "Join peer capture");
                joined = true;
            }
            for (auto& rank : ranks) {
                DeviceScope device(rank->device);
                check(exchange_rows(rank->binding.device, false, rank->stream),
                      "Capture input copy");
            }
            active[0] = &root.binding;
            active[1] = &peer.binding;
            forward({resident[0].data(), resident[1].data()});
            for (auto& rank : ranks) {
                DeviceScope device(rank->device);
                check(exchange_rows(rank->binding.device, true, rank->stream),
                      "Capture output copy");
                check(exchange_replay(rank->binding.device, rank->stream), "Capture rollback copy");
            }
        } catch (...) {
            error = std::current_exception();
        }
        active[0] = active[1] = nullptr;
        if (joined) {
            cudaError_t peer_edge;
            {
                DeviceScope device(peer.device);
                peer_edge = cudaEventRecord(peer.edge, peer.stream);
            }
            if (peer_edge == cudaSuccess) cudaStreamWaitEvent(root.stream, peer.edge);
        }
        cudaGraph_t graph = nullptr;
        const auto ended = cudaStreamEndCapture(root.stream, &graph);
        if (ended != cudaSuccess || error) {
            if (graph) check(cudaGraphDestroy(graph), "Destroy failed TP source graph");
            if (error) std::rethrow_exception(error);
            if (capture_rejected(ended)) return false;
            check(ended, "End fused TP verifier capture");
        }
        bool complete = true;
        for (const auto& rank : ranks) {
            const auto& calls = rank->binding;
            complete &= calls.gqa_calls == full_attention_layers && calls.gdn_calls == gdn_layers &&
                        calls.fused_calls == 2 * (gdn_layers + full_attention_layers) - 1 &&
                        calls.regular_calls == 1;
        }
        if (!complete) {
            check(cudaGraphDestroy(graph), "Destroy incomplete TP source graph");
            return false;
        }
        const auto instantiated = cudaGraphInstantiateWithFlags(&executable, graph, 0);
        check(cudaGraphDestroy(graph), "Destroy TP source graph");
        if (capture_rejected(instantiated) || instantiated == cudaErrorInvalidValue) return false;
        check(instantiated, "Instantiate fused TP verifier graph");
        return true;
    }

    void prepare(const Input& input) {
        for (int rank = 0; rank < 2; ++rank) {
            auto& state = *ranks[rank];
            auto& table = *state.staging;
            for (int index = 0; index < rows; ++index) {
                const auto& row = input[rank][index];
                auto& fixed = state.work[index];
                table.request[index] = {row.work->hidden, row.work->residual};
                table.resident[index] = {fixed.hidden, fixed.residual};
                table.rotary[index] = row.rotary_position < 0 ? row.position : row.rotary_position;
                for (int layer = 0; layer < full_attention_layers; ++layer) {
                    table.gqa[layer].reads[index] = {row.request->caches[layer], row.request->pages,
                                                    row.position, paged_gqa_buffers(fixed)};
                }
            }
            for (int layer = 0; layer < gdn_layers; ++layer) {
                for (int lane = 0; lane < lanes; ++lane) {
                    const auto request = input[rank][lane].verification->gdn_lane(
                        layer, state.binding.weights[layer]);
                    auto job = state.gdn[layer][lane];
                    job.initial = request.initial;
                    job.tail = request.tail;
                    table.gdn[layer][lane] = job;
                    table.replay[layer][lane] = {job.inputs, request.inputs};
                }
            }
        }
    }

    void release_binding() noexcept {
        for (auto& rank : ranks) *rank->staging = {};
    }

    void submit() {
        auto& root = *ranks[0];
        auto& peer = *ranks[1];
        for (auto& rank : ranks) {
            DeviceScope device(rank->device);
            check(cudaMemcpyAsync(rank->binding.device, rank->staging, sizeof(DeviceBinding),
                                  cudaMemcpyHostToDevice, rank->stream), "Upload TP graph binding");
        }
        {
            DeviceScope device(peer.device);
            check(cudaEventRecord(peer.edge, peer.stream), "Publish peer graph binding");
        }
        {
            DeviceScope device(root.device);
            check(cudaStreamWaitEvent(root.stream, peer.edge), "Wait for peer graph binding");
            check(cudaGraphLaunch(executable, root.stream), "Replay fused verifier graph");
            check(cudaEventRecord(root.edge, root.stream), "Publish joined verifier completion");
        }
        {
            DeviceScope device(peer.device);
            check(cudaStreamWaitEvent(peer.stream, root.edge), "Order peer verifier completion");
        }
    }

private:
    std::array<std::unique_ptr<Rank>, 2> ranks;
    cudaGraphExec_t executable = nullptr;
};

bool replay(std::shared_ptr<Graph>& owner, bool& disabled, const Input& input,
            const std::array<int, 2>& devices, const std::array<cudaStream_t, 2>& streams,
            TensorParallel& group, const Forward& forward) {
    if (disabled || !group.fused_graph_eligible() || !eligible(input, streams)) return false;
    if (!streams_available(devices, streams)) return false;
    if (!owner) {
        check(cudaGetLastError(), "Check CUDA status before TP verifier capture");
        try {
            auto candidate = std::make_shared<Graph>(devices, streams);
            if (candidate->capture(input, forward)) owner = std::move(candidate);
        } catch (const CudaError& error) {
            if (!capture_rejected(error.status)) throw;
        }
        finish_capture(devices, streams);
        if (!owner) {
            disabled = true;
            return false;
        }
    }
    if (!owner->matches(devices, streams)) return false;
    owner->prepare(input);
    try {
        owner->submit();
        group.record_fused_graph_replay();
    } catch (...) {
        group.fail_fused_graph_replay();
        throw;
    }
    return true;
}

Submission::Submission(TensorParallel& transport, Graph* graph) noexcept
    : group(transport), owner(graph), uncaught(std::uncaught_exceptions()) {}

Submission::~Submission() {
    if (!owner) return;
    if (std::uncaught_exceptions() != uncaught) group.fail_fused_graph_replay();
    else owner->release_binding();
}

bool bind_gdn_inputs(cudaStream_t stream, const VerificationRow* packet, int layer) {
    const auto* live = binding(stream);
    if (!live) return false;
    for (int index = 0; index < rows; ++index) {
        const auto& input = live->replay[layer][index % lanes];
        auto& work = *packet[index].work;
        const int step = index / lanes;
        work.gdn_qkv = input.qkv + step * GdnLocal::qkv_width;
        work.gdn_b = input.b + step * GdnLocal::value_heads;
        work.gdn_a = input.a + step * GdnLocal::value_heads;
    }
    return true;
}

bool run_gdn(cudaStream_t stream, int layer, TextGdnWeights weights) {
    auto* live = binding(stream);
    if (!live) return false;
    live->weights[layer] = weights;
    ++live->gdn_calls;
    check(gdn(live->device->gdn[layer], weights, stream), "Capture bound GDN");
    return true;
}

void attend_gqa(HostBinding& live, const GqaLayer& layer, const VerificationRow* packet,
                int cache_index, GqaGeometry geometry) {
    GqaVerifyRows<GqaVerifyPreprocessRow> prep{};
    GqaVerifyRows<GqaVerifyAppendRow> append_rows{};
    GqaVerifyRows<GqaVerifyGateRow> gate{};
    for (int row = 0; row < rows; ++row) {
        const auto& work = *packet[row].work;
        prep.rows[row] = {work.q_gate, work.keys, work.rope_queries, work.rope_keys, 0};
        append_rows.rows[row] = {{}, {}, work.rope_keys, work.values, 0};
        gate.rows[row] = {work.q_gate, work.attention};
    }
    const auto* cache = &live.device->gqa[cache_index];
    ++live.gqa_calls;
    check(preprocess(prep, live.device->rotary, geometry, layer.query_norm, layer.key_norm,
                     live.stream), "Capture bound rotary");
    check(append(append_rows, cache, geometry.kv_heads, live.stream), "Capture bound KV append");
    check(read(cache, geometry.query_heads, live.stream), "Capture bound GQA");
    check(gqa_verify_postgate(gate, rows, live.stream, geometry), "Capture bound attention gate");
}

void record_fused(cudaStream_t stream) {
    if (binding(stream)) {
        for (auto* rank : active) ++rank->fused_calls;
    }
}

void record_regular(cudaStream_t stream) {
    if (binding(stream)) {
        for (auto* rank : active) ++rank->regular_calls;
    }
}
}

namespace sglang {
using namespace decoder;

tp_verify_graph::Graph* Decoder::Model::forward_verification_tp(
    const Model& peer, const VerificationRow* rows, const VerificationRow* peer_rows,
    int count, int total, cudaStream_t stream, cudaStream_t peer_stream, TensorParallel& group,
    const TpPartials& primary_partials, const TpPartials& peer_partials,
    std::uint8_t* primary_packed, std::uint8_t* peer_packed, bool greedy, bool capture,
    bool& eager_twoshot) const {
    const tp_verify_graph::Input input{rows, peer_rows};
    const auto forward = [&](const tp_verify_graph::Input& live) {
        return forward_layers_tp(peer, live[0], live[1], total, stream, peer_stream, group,
                                 primary_partials, peer_partials, primary_packed, peer_packed,
                                 count == 1 && greedy);
    };
    if (capture && tp && count == 2 && total == tp_verify_graph::rows && greedy &&
        !primary_packed && !peer_packed && backend.vector_decode_attention &&
        peer.backend.vector_decode_attention &&
        (!tp->verification_capture_stream || tp->verification_capture_stream == stream)) {
        tp->verification_capture_stream = stream;
        if (tp_verify_graph::replay(tp->verification_capture, tp->verification_capture_disabled,
                                   input, {device, peer.device}, {stream, peer_stream}, group,
                                   tp_verify_graph::Forward{&forward})) {
            return tp->verification_capture.get();
        }
    }
    eager_twoshot = forward(input);
    return nullptr;
}
}
