#include "sglang/decoder_internal.h"

#include "sglang/checkpoint.h"

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

void allocate_gdn(Request& request) {
    const std::size_t history_count = gdn_layers * gdn_history_elements;
    const std::size_t recurrent_count = gdn_layers * gdn_state_elements;
    auto* history = request.memory.allocate<Bf16>(history_count);
    auto* recurrent = request.memory.allocate<float>(recurrent_count);
    for (int index = 0; index < gdn_layers; ++index) {
        request.states[index] = {history + index * gdn_history_elements,
                                 recurrent + index * gdn_state_elements};
    }
}

void allocate_kv(Request& request, int capacity) {
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
                          cudaMemcpyHostToDevice, request.memory.stream()),
          "Upload MXFP8 page table");
    request.pages = {table, page_count};
}

}

struct Allocations::State {
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
            std::shared_ptr<const Allocations> drained_weights;
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
                error = entry.release(drained_weights);
                if (!entry.owns_cuda()) {
                    finished = std::move(*pending);
                    *pending = std::move(finished->next);
                }
            }
            drained_weights.reset();
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

    CleanupError close_stream(std::shared_ptr<const Allocations>& drained_weights);
    CleanupError release_selected(std::shared_ptr<const Allocations>& drained_weights);
    CleanupError release(std::shared_ptr<const Allocations>& drained_weights);
    CleanupError failure(CleanupError error);

    int device;
    cudaStream_t stream = nullptr;
    cudaGraph_t capture = nullptr;
    cudaGraphExec_t graph = nullptr;
    bool uncertain_stream = false;
    bool uncertain_graph = false;
    std::vector<void*> blocks;
    std::size_t uncertain_blocks = 0;
    std::size_t bytes = 0;
    CleanupError uncertain;
    CleanupError terminal;
    TransferBuffers transfers;
    std::shared_ptr<const Allocations> weights;
    std::unique_ptr<State> next;
};

CleanupError Allocations::State::failure(CleanupError error) {
    if (terminal_context_error(error.status)) {
        terminal = error;
    }
    return error;
}

CleanupError Allocations::State::close_stream(std::shared_ptr<const Allocations>& drained_weights) {
    if (terminal || uncertain_stream || uncertain_graph || stream == nullptr) {
        return terminal ? terminal : uncertain;
    }
    const CleanupError sync{cudaStreamSynchronize(stream), "Synchronize decoder stream"};
    if (sync) {
        return failure(sync);
    }
    if (graph) {
        const CleanupError destroy{cudaGraphExecDestroy(graph), "Destroy decode executable graph"};
        if (destroy) {
            uncertain_graph = true;
            uncertain = destroy;
            return failure(destroy);
        }
        graph = nullptr;
    }
    if (capture) {
        const CleanupError destroy{cudaGraphDestroy(capture), "Destroy decode capture graph"};
        if (destroy) {
            uncertain_graph = true;
            uncertain = destroy;
            return failure(destroy);
        }
        capture = nullptr;
    }
    transfers = {};
    drained_weights = std::move(weights);
    const CleanupError destroy{cudaStreamDestroy(stream), "Destroy decoder stream"};
    if (destroy) {
        uncertain_stream = true;
        uncertain = destroy;
        return failure(destroy);
    }
    stream = nullptr;
    return {};
}

CleanupError Allocations::State::release_selected(
    std::shared_ptr<const Allocations>& drained_weights) {
    const auto error = close_stream(drained_weights);
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

CleanupError Allocations::State::release(std::shared_ptr<const Allocations>& drained_weights) {
    if (terminal || !owns_cuda()) {
        return terminal;
    }
    int previous = 0;
    const CleanupError current{cudaGetDevice(&previous), "Get CUDA device for decoder cleanup"};
    if (current) {
        return failure(current);
    }
    if (previous == device) {
        return release_selected(drained_weights);
    }
    const CleanupError select{cudaSetDevice(device), "Select CUDA device for decoder cleanup"};
    if (select) {
        return failure(select);
    }
    const auto error = release_selected(drained_weights);
    const CleanupError restore{cudaSetDevice(previous), "Restore CUDA device after cleanup"};
    if (error) {
        report_cleanup(restore.status, restore.operation);
        return error;
    }
    return restore;
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
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
    std::shared_ptr<const Allocations> drained_weights;
    const auto error = state_->release(drained_weights);
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

void Allocations::reclaim_deferred() {
    State::reclaim(state_->device);
}

void Allocations::open_stream(std::shared_ptr<const Allocations> weights) {
    check(cudaStreamCreateWithFlags(&state_->stream, cudaStreamNonBlocking),
          "Create decoder stream");
    state_->weights = std::move(weights);
}

void Allocations::close_stream() {
    std::shared_ptr<const Allocations> drained_weights;
    state_->close_stream(drained_weights).raise();
}

void Allocations::close() {
    std::shared_ptr<const Allocations> drained_weights;
    state_->release(drained_weights).raise();
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

Workspace::Workspace(Allocations& allocations)
    : hidden(allocations.allocate<Bf16>(hidden_size)),
      normalized(allocations.allocate<Bf16>(hidden_size)),
      residual(allocations.allocate<Bf16>(hidden_size)),
      projected_attention(allocations.allocate<Bf16>(hidden_size)),
      attention(allocations.allocate<Bf16>(text_gqa_q_width)),
      mlp_gate(allocations.allocate<Bf16>(intermediate_size)),
      mlp_up(allocations.allocate<Bf16>(intermediate_size)),
      q_gate(allocations.allocate<Bf16>(text_gqa_q_gate_width)),
      keys(allocations.allocate<Bf16>(text_gqa_kv_width)),
      values(allocations.allocate<Bf16>(text_gqa_kv_width)),
      rope_queries(allocations.allocate<Bf16>(text_gqa_q_width)),
      rope_keys(allocations.allocate<Bf16>(text_gqa_kv_width)),
      gdn_qkv(allocations.allocate<Bf16>(text_gdn_qkv_width)),
      gdn_z(allocations.allocate<Bf16>(text_gdn_value_width)),
      gdn_b(allocations.allocate<Bf16>(text_gdn_value_heads)),
      gdn_a(allocations.allocate<Bf16>(text_gdn_value_heads)),
      gdn_scratch(allocations.allocate<Bf16>(text_gdn_workspace_bytes(1) / sizeof(Bf16))),
      logits(allocations.allocate<Bf16>(vocab_size)),
      activation{allocations.allocate<std::uint8_t>(intermediate_size),
                 allocations.allocate<float>(text_fp8_group_count(intermediate_size))},
      argmax_scratch(
          allocations.allocate<TextLogitCandidate>(text_argmax_scratch_count(vocab_size))),
      argmax(allocations.allocate<TextLogitCandidate>(1)) {
    static_assert(text_gqa_q_width == text_gdn_value_width);
}

Request::Request(int device, int capacity, std::shared_ptr<const Allocations> weights,
                 bool reusable)
    : memory(device), work(memory), decode_position(reusable ? memory.allocate<int>(1) : nullptr) {
    memory.open_stream(std::move(weights));
    allocate_kv(*this, capacity);
    allocate_gdn(*this);
}

void Request::reset() {
    zero(states.front().history, gdn_layers * gdn_history_elements, memory.stream());
    zero(states.front().recurrent, gdn_layers * gdn_state_elements, memory.stream());
}

}
