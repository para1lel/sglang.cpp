#include "sglang/decoder_internal.h"

#include "sglang/checkpoint.h"
#include "sglang/decoder_batch.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>

namespace sglang::decoder {
namespace {

struct CleanupError {
    cudaError_t status = cudaSuccess;
    const char* operation = "";

    explicit operator bool() const { return status != cudaSuccess; }
    void raise() const { check(status, operation); }
};

struct CudaError : std::runtime_error {
    CudaError(cudaError_t status, const char* operation)
        : std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status)),
          status(status) {}

    cudaError_t status;
};

bool capture_rejected(cudaError_t status) {
    return status == cudaErrorNotSupported || status == cudaErrorMemoryAllocation ||
        status == cudaErrorStreamCaptureUnsupported ||
        status == cudaErrorStreamCaptureInvalidated;
}

void report_cleanup(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "sglang decoder: %s: %s\n", operation, cudaGetErrorString(status));
    }
}

bool terminal_context_error(cudaError_t status) {
    switch (status) {
    case cudaErrorCudartUnloading:
    case cudaErrorContextIsDestroyed:
    case cudaErrorECCUncorrectable:
    case cudaErrorIllegalAddress:
    case cudaErrorLaunchTimeout:
    case cudaErrorAssert:
    case cudaErrorHardwareStackError:
    case cudaErrorIllegalInstruction:
    case cudaErrorMisalignedAddress:
    case cudaErrorInvalidAddressSpace:
    case cudaErrorInvalidPc:
    case cudaErrorLaunchFailure:
        return true;
    default:
        return false;
    }
}

template <typename Element>
void zero(Element* buffer, std::size_t count, cudaStream_t stream) {
    check(cudaMemsetAsync(buffer, 0, count * sizeof(Element), stream), "Reset request state");
}

void allocate_kv(Request& request, int capacity, cudaStream_t stream) {
    const int page_count = (capacity + mxfp8_page_size - 1) / mxfp8_page_size;
    auto* table = request.memory.allocate<std::int32_t>(page_count);
    const auto payload_per_layer = page_count * mxfp8_payload_bytes_per_page;
    const auto scales_per_layer = page_count * mxfp8_scale_bytes_per_page;
    const auto scale_bytes = full_attention_layers * scales_per_layer;
    auto* keys = request.memory.allocate<std::uint8_t>(full_attention_layers * payload_per_layer);
    auto* values = request.memory.allocate<std::uint8_t>(full_attention_layers * payload_per_layer);
    auto* key_scales = request.memory.allocate<std::uint8_t>(scale_bytes);
    auto* val_scales = request.memory.allocate<std::uint8_t>(scale_bytes);
    for (int index = 0; index < full_attention_layers; ++index) {
        request.caches[index] = {
            keys + index * payload_per_layer, values + index * payload_per_layer,
            key_scales + index * scales_per_layer, val_scales + index * scales_per_layer,
            page_count,
        };
    }
    auto& logical_pages = request.memory.transfers().logical_pages;
    logical_pages.resize(page_count);
    std::iota(logical_pages.begin(), logical_pages.end(), 0);
    check(cudaMemcpyAsync(table, logical_pages.data(), logical_pages.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice, stream),
          "Upload MXFP8 page table");
    request.pages = {table, page_count};
}

}

struct Allocations::State {
    struct BatchGraph {
        cudaGraph_t capture = nullptr;
        cudaGraphExec_t executable = nullptr;
        bool disabled = false;
    };

    struct Deferred {
        std::mutex mutex;
        std::unique_ptr<State> first;
    };

    static Deferred& deferred() {
        static auto& queue = *new Deferred;
        return queue;
    }

    static void reclaim(int device) {
        auto& queue = deferred();
        for (;;) {
            std::shared_ptr<const void> drained_owners;
            std::unique_ptr<State> finished;
            CleanupError error;
            {
                const std::lock_guard lock(queue.mutex);
                auto* pending = &queue.first;
                while (*pending && (*pending)->device != device) {
                    pending = &(*pending)->next;
                }
                if (!*pending) {
                    return;
                }
                auto& entry = **pending;
                error = entry.release(drained_owners);
                if (!entry.owns_cuda()) {
                    finished = std::move(*pending);
                    *pending = std::move(finished->next);
                }
            }
            drained_owners.reset();
            error.raise();
        }
    }

    static void retain(std::unique_ptr<State> entry) {
        auto& queue = deferred();
        const std::lock_guard lock(queue.mutex);
        entry->next = std::move(queue.first);
        queue.first = std::move(entry);
    }

    explicit State(int owner) : device(owner) { reclaim(owner); }

    bool owns_cuda() const { return stream != nullptr || !blocks.empty(); }

    CleanupError close_stream(std::shared_ptr<const void>& drained_owners);
    CleanupError release_selected(std::shared_ptr<const void>& drained_owners);
    CleanupError release(std::shared_ptr<const void>& drained_owners);
    CleanupError failure(CleanupError error);
    CleanupError destroy_capture(cudaGraph_t& handle);
    CleanupError destroy_executable(cudaGraphExec_t& handle);

    int device;
    cudaStream_t stream = nullptr;
    cudaGraph_t capture = nullptr;
    cudaGraphExec_t graph = nullptr;
    std::array<BatchGraph, decoder_batch_max_lanes - 1> batch_graphs{};
    bool uncertain_stream = false;
    bool uncertain_graph = false;
    std::vector<void*> blocks;
    std::size_t uncertain_blocks = 0;
    std::size_t bytes = 0;
    CleanupError uncertain;
    CleanupError terminal;
    TransferBuffers transfers;
    std::shared_ptr<const void> owners;
    std::unique_ptr<State> next;
};

CleanupError Allocations::State::failure(CleanupError error) {
    if (terminal_context_error(error.status)) {
        terminal = error;
    }
    return error;
}

CleanupError Allocations::State::destroy_capture(cudaGraph_t& handle) {
    if (handle) {
        const CleanupError error{cudaGraphDestroy(handle), "Destroy decode capture graph"};
        if (error) {
            uncertain_graph = true;
            uncertain = error;
            return failure(error);
        }
        handle = nullptr;
    }
    return {};
}

CleanupError Allocations::State::destroy_executable(cudaGraphExec_t& handle) {
    if (handle) {
        const CleanupError error{cudaGraphExecDestroy(handle), "Destroy decode executable graph"};
        if (error) {
            uncertain_graph = true;
            uncertain = error;
            return failure(error);
        }
        handle = nullptr;
    }
    return {};
}

CleanupError Allocations::State::close_stream(std::shared_ptr<const void>& drained_owners) {
    if (terminal || uncertain_stream || uncertain_graph || stream == nullptr) {
        return terminal ? terminal : uncertain;
    }
    const CleanupError sync{cudaStreamSynchronize(stream), "Synchronize decoder stream"};
    if (sync) {
        return failure(sync);
    }
    if (const auto error = destroy_executable(graph)) {
        return error;
    }
    if (const auto error = destroy_capture(capture)) {
        return error;
    }
    for (auto& batch : batch_graphs) {
        if (const auto error = destroy_executable(batch.executable)) {
            return error;
        }
        if (const auto error = destroy_capture(batch.capture)) {
            return error;
        }
    }
    transfers = {};
    drained_owners = std::move(owners);
    const CleanupError destroy{cudaStreamDestroy(stream), "Destroy decoder stream"};
    if (destroy) {
        uncertain_stream = true;
        uncertain = destroy;
        return failure(destroy);
    }
    stream = nullptr;
    return {};
}

CleanupError Allocations::State::release_selected(std::shared_ptr<const void>& drained_owners) {
    const auto error = close_stream(drained_owners);
    if (terminal || (stream && !uncertain_stream)) {
        return error;
    }
    while (blocks.size() > uncertain_blocks) {
        const CleanupError freed{cudaFree(blocks.back()), "Free decoder allocation"};
        if (!freed) {
            blocks.pop_back();
            continue;
        }
        failure(freed);
        if (terminal) {
            return terminal;
        }
        if (!uncertain) {
            uncertain = freed;
        }
        std::swap(blocks[uncertain_blocks++], blocks.back());
    }
    return uncertain;
}

CleanupError Allocations::State::release(std::shared_ptr<const void>& drained_owners) {
    if (terminal || !owns_cuda()) {
        return terminal;
    }
    int previous = 0;
    const CleanupError current{cudaGetDevice(&previous), "Get CUDA device for decoder cleanup"};
    if (current) {
        return failure(current);
    }
    if (previous == device) {
        return release_selected(drained_owners);
    }
    const CleanupError select{cudaSetDevice(device), "Select CUDA device for decoder cleanup"};
    if (select) {
        return failure(select);
    }
    const auto error = release_selected(drained_owners);
    const CleanupError restore{cudaSetDevice(previous), "Restore CUDA device after cleanup"};
    if (error) {
        report_cleanup(restore.status, restore.operation);
        return error;
    }
    return restore;
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw CudaError(status, operation);
    }
}

DeviceScope::DeviceScope(int device) {
    check(cudaGetDevice(&previous_), "Get current CUDA device");
    changed_ = device != previous_;
    if (changed_) {
        check(cudaSetDevice(device), "Select CUDA device");
    }
}

DeviceScope::~DeviceScope() {
    if (changed_) {
        report_cleanup(cudaSetDevice(previous_), "Restore CUDA device");
    }
}

void DeviceScope::restore() {
    if (changed_) {
        check(cudaSetDevice(previous_), "Restore CUDA device");
        changed_ = false;
    }
}

Allocations::Allocations(int device) : state_(std::make_unique<State>(device)) {}

Allocations::~Allocations() {
    std::shared_ptr<const void> drained_owners;
    const auto error = state_->release(drained_owners);
    report_cleanup(error.status, error.operation);
    if (state_->owns_cuda()) {
        State::retain(std::move(state_));
    }
}

std::size_t Allocations::bytes() const { return state_->bytes; }

TransferBuffers& Allocations::transfers() { return state_->transfers; }

cudaStream_t Allocations::stream() const { return state_->stream; }

cudaGraphExec_t Allocations::graph() const { return state_->graph; }

void Allocations::capture_graph(const std::function<void()>& enqueue) {
    const auto stream = state_->stream;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "Begin decode graph capture");
    try {
        enqueue();
    } catch (...) {
        report_cleanup(cudaStreamEndCapture(stream, &state_->capture),
                       "End interrupted decode graph capture");
        throw;
    }
    check(cudaStreamEndCapture(stream, &state_->capture), "End decode graph capture");
    check(cudaGraphInstantiate(&state_->graph, state_->capture, 0),
          "Instantiate decode graph");
}

bool Allocations::replay_batch_graph(int width, const std::function<void()>& prepare,
                                     const std::function<void()>& capture_nodes,
                                     const std::function<void()>& bind) {
    auto& batch = state_->batch_graphs.at(width - 2);
    if (batch.disabled) {
        return false;
    }
    const auto stream = state_->stream;
    const auto fallback = [&](cudaError_t rejected, const char* operation) {
        const auto previous = cudaGetLastError();
        if (previous != rejected && !capture_rejected(previous)) {
            check(previous, "Check CUDA error before eager batch fallback");
        }
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        check(cudaStreamIsCapturing(stream, &capture_status), "Check batch capture ended");
        if (capture_status != cudaStreamCaptureStatusNone) {
            check(rejected, operation);
        }
        check(cudaStreamSynchronize(stream), "Drain stream before eager batch fallback");
        state_->destroy_executable(batch.executable).raise();
        state_->destroy_capture(batch.capture).raise();
        batch.disabled = true;
        std::fprintf(stderr, "sglang decoder: %s at width %d: %s, using eager batch decode\n",
                     operation, width, cudaGetErrorString(rejected));
        return false;
    };
    if (!batch.executable) {
        check(cudaGetLastError(), "Check CUDA error before batch graph capture");
        try {
            prepare();
        } catch (const CudaError& error) {
            if (error.status == cudaErrorMemoryAllocation) {
                return fallback(error.status, "Allocate batch decode graph scratch");
            }
            throw;
        }
        const auto begun = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (capture_rejected(begun)) {
            return fallback(begun, "Begin batch decode graph capture");
        }
        check(begun, "Begin batch decode graph capture");
        cudaError_t enqueue_status = cudaSuccess;
        try {
            capture_nodes();
        } catch (const CudaError& error) {
            enqueue_status = error.status;
            if (!capture_rejected(enqueue_status)) {
                report_cleanup(cudaStreamEndCapture(stream, &batch.capture),
                               "End failed batch decode graph capture");
                throw;
            }
        } catch (...) {
            report_cleanup(cudaStreamEndCapture(stream, &batch.capture),
                           "End interrupted batch decode graph capture");
            throw;
        }
        const auto ended = cudaStreamEndCapture(stream, &batch.capture);
        if (capture_rejected(ended)) {
            return fallback(ended, "End batch decode graph capture");
        }
        check(ended, "End batch decode graph capture");
        if (enqueue_status != cudaSuccess) {
            return fallback(enqueue_status, "Record batch decode graph");
        }
        const auto built = cudaGraphInstantiate(&batch.executable, batch.capture, 0);
        if (capture_rejected(built) || built == cudaErrorInvalidValue) {
            return fallback(built, "Instantiate batch decode graph");
        }
        check(built, "Instantiate batch decode graph");
        state_->destroy_capture(batch.capture).raise();
    }
    bind();
    check(cudaGetLastError(), "Check CUDA error before batch graph launch");
    const auto launched = cudaGraphLaunch(batch.executable, stream);
    if (launched == cudaErrorInvalidValue || launched == cudaErrorInvalidResourceHandle ||
        launched == cudaErrorNotSupported) {
        return fallback(launched, "Launch batch decode graph");
    }
    check(launched, "Launch batch decode graph");
    return true;
}

void Allocations::reclaim_deferred() {
    State::reclaim(state_->device);
}

void Allocations::open_stream(std::shared_ptr<const void> owners) {
    check(cudaStreamCreateWithFlags(&state_->stream, cudaStreamNonBlocking),
          "Create decoder stream");
    state_->owners = std::move(owners);
}

void Allocations::close_stream() {
    std::shared_ptr<const void> drained_owners;
    state_->close_stream(drained_owners).raise();
}

void Allocations::close() {
    std::shared_ptr<const void> drained_owners;
    state_->release(drained_owners).raise();
}

void* Allocations::allocate_bytes(std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - state_->bytes) {
        throw std::overflow_error("CUDA allocation total overflow");
    }
    auto& blocks = state_->blocks;
    blocks.push_back(nullptr);
    const auto status = cudaMalloc(&blocks.back(), bytes);
    if (status != cudaSuccess) {
        blocks.pop_back();
        check(status, "Allocate CUDA memory");
    }
    state_->bytes += bytes;
    return blocks.back();
}

Workspace::Workspace(Allocations& allocations, Bf16* verification_attention,
                     TextLogitCandidate* verification_winner)
    : hidden(allocations.allocate<Bf16>(hidden_size)),
      normalized(allocations.allocate<Bf16>(hidden_size)),
      residual(allocations.allocate<Bf16>(hidden_size)),
      projected_attention(allocations.allocate<Bf16>(hidden_size)),
      attention(verification_attention ? verification_attention :
                allocations.allocate<Bf16>(text_gqa_q_width)),
      mlp_gate(allocations.allocate<Bf16>(intermediate_size)),
      mlp_up(allocations.allocate<Bf16>(intermediate_size)),
      q_gate(allocations.allocate<Bf16>(text_gqa_q_gate_width)),
      keys(allocations.allocate<Bf16>(text_gqa_kv_width)),
      values(allocations.allocate<Bf16>(text_gqa_kv_width)),
      rope_queries(allocations.allocate<Bf16>(text_gqa_q_width)),
      rope_keys(allocations.allocate<Bf16>(text_gqa_kv_width)),
      gdn_qkv(verification_attention ? nullptr : allocations.allocate<Bf16>(text_gdn_qkv_width)),
      gdn_z(verification_attention ? nullptr : allocations.allocate<Bf16>(text_gdn_value_width)),
      gdn_b(verification_attention ? nullptr : allocations.allocate<Bf16>(text_gdn_value_heads)),
      gdn_a(verification_attention ? nullptr : allocations.allocate<Bf16>(text_gdn_value_heads)),
      gdn_scratch(verification_attention ? nullptr : allocations.allocate<Bf16>(
          text_gdn_workspace_bytes(1) / sizeof(Bf16))),
      logits(reinterpret_cast<Bf16*>(allocations.allocate<std::byte>(vocab_size * sizeof(Bf16)))),
      activation{allocations.allocate<std::uint8_t>(intermediate_size),
                 allocations.allocate<float>(text_fp8_group_count(intermediate_size))},
      argmax_scratch(
          allocations.allocate<TextLogitCandidate>(text_argmax_scratch_count(vocab_size))),
      argmax(verification_winner ? verification_winner :
             allocations.allocate<TextLogitCandidate>(1)) {
    static_assert(text_gqa_q_width == text_gdn_value_width);
}

ExactGqaWorkspace Workspace::gqa_scores(int capacity) const {
    constexpr int bytes_per_page = mxfp8_page_size * mxfp8_query_heads * sizeof(float);
    constexpr int score_pages = vocab_size * sizeof(Bf16) / bytes_per_page;
    return {reinterpret_cast<float*>(logits), std::min(capacity, score_pages * mxfp8_page_size)};
}

void allocate_gdn(Allocations& memory, GdnBank& states) {
    auto* history = memory.allocate<Bf16>(gdn_layers * gdn_history_elements);
    auto* recurrent = memory.allocate<float>(gdn_layers * gdn_state_elements);
    for (int index = 0; index < gdn_layers; ++index) {
        states[index] = {history + index * gdn_history_elements,
                         recurrent + index * gdn_state_elements};
    }
}

Request::Request(int device, bool reusable)
    : memory(device), work(memory), decode_position(reusable ? memory.allocate<int>(1) : nullptr) {}

Request::Request(int device, int capacity, std::shared_ptr<const void> owners, bool reusable)
    : Request(device, reusable) {
    memory.open_stream(std::move(owners));
    initialize(capacity, memory.stream());
}

void Request::initialize(int capacity, cudaStream_t stream) {
    allocate_kv(*this, capacity, stream);
    allocate_gdn(memory, states);
}

void Request::reset(cudaStream_t stream) {
    zero(states.front().history, gdn_layers * gdn_history_elements, stream);
    zero(states.front().recurrent, gdn_layers * gdn_state_elements, stream);
}

void Request::reset() {
    reset(memory.stream());
}

}
