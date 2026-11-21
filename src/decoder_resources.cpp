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

int pages_for(int capacity) {
    return (capacity + mxfp8_page_size - 1) / mxfp8_page_size;
}

std::size_t kv_bytes(int pages) {
    return 2 * full_attention_layers * pages *
        (mxfp8_payload_bytes_per_page + mxfp8_scale_bytes_per_page);
}

std::array<Mxfp8KvCache, full_attention_layers> allocate_kv(Allocations& memory, int page_count) {
    const auto payload_per_layer = page_count * mxfp8_payload_bytes_per_page;
    const auto scales_per_layer = page_count * mxfp8_scale_bytes_per_page;
    const auto scale_bytes = full_attention_layers * scales_per_layer;
    const auto payload_bytes = full_attention_layers * payload_per_layer;
    auto* keys = memory.allocate<std::uint8_t>(kv_bytes(page_count));
    auto* values = keys + payload_bytes;
    auto* key_scales = values + payload_bytes;
    auto* val_scales = key_scales + scale_bytes;
    std::array<Mxfp8KvCache, full_attention_layers> caches;
    for (int index = 0; index < full_attention_layers; ++index) {
        caches[index] = {
            keys + index * payload_per_layer, values + index * payload_per_layer,
            key_scales + index * scales_per_layer, val_scales + index * scales_per_layer,
            page_count,
        };
    }
    return caches;
}

void copy_kv_plane(std::uint8_t* destination, const std::uint8_t* source, std::size_t bytes,
                   cudaStream_t stream) {
    check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream),
          "Copy MXFP8 cache during growth");
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

    bool owns_cuda() const { return stream || packed_winners || !blocks.empty(); }

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
    std::array<BatchGraph, decoder_batch_max_graph_lanes + 1 + packed_decode_buckets.size()>
        batch_graphs{};
    bool verification_graph_incomplete = false;
    bool uncertain_stream = false;
    bool uncertain_graph = false;
    std::vector<void*> blocks;
    std::size_t uncertain_blocks = 0;
    std::size_t bytes = 0;
    CleanupError uncertain;
    CleanupError terminal;
    TransferBuffers transfers;
    TextLogitCandidate* packed_winners = nullptr;
    bool packed_winners_unavailable = false;
    std::optional<ExactGqaWorkspace> singleton_scores;
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
    CleanupError pinned;
    if (packed_winners) {
        pinned = {cudaFreeHost(packed_winners), "Free packed winner storage"};
        if (pinned) {
            failure(pinned);
            if (terminal) {
                return terminal;
            }
        } else {
            packed_winners = nullptr;
        }
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
    return uncertain ? uncertain : pinned;
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

CudaError::CudaError(cudaError_t status, const char* operation)
    : std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status)),
      status(status) {}

std::size_t available_device_bytes() {
    std::size_t free = 0;
    std::size_t total = 0;
    check(cudaMemGetInfo(&free, &total), "Measure available CUDA memory");
    return free;
}

int reserved_verification_rows(int output, int mtp_steps) {
    return mtp_steps && output > 1 ? std::min(mtp_steps + 1, output - 1) : 0;
}

std::size_t request_resident_bytes(int capacity, int verification_rows, bool draft) {
    constexpr std::size_t gdn_bytes = gdn_layers *
        (gdn_history_elements * sizeof(Bf16) + gdn_state_elements * sizeof(float));
    constexpr std::size_t workspace_allowance = 16 * 1024 * 1024;
    const int pages = pages_for(capacity);
    const auto target = kv_bytes(pages) + pages * sizeof(std::int32_t) + gdn_bytes;
    const auto draft_kv = kv_bytes(pages) / full_attention_layers + pages * sizeof(std::int32_t);
    const auto verifier = verification_rows ? gdn_bytes + verification_rows * workspace_allowance
                                            : 0;
    return target + workspace_allowance + (draft ? draft_kv + workspace_allowance : 0) + verifier;
}

int limit_resident_output(int prompt, const DecoderOptions& options, std::size_t total_bytes,
                          std::size_t free_bytes, std::size_t extra_bytes) {
    const auto fits = [&](int output, std::size_t bytes) {
        const auto resident = request_resident_bytes(
            prompt + output - 1, reserved_verification_rows(output, options.mtp_steps),
            options.mtp_steps != 0);
        return resident + extra_bytes <= bytes;
    };
    int output = options.max_new_tokens;
    if (options.incremental_kv) {
        int low = 0;
        while (low < output) {
            const int middle = low + (output - low + 1) / 2;
            if (fits(middle, total_bytes)) {
                low = middle;
            } else {
                output = middle - 1;
            }
        }
    }
    if (output == 0 || !fits(output, total_bytes)) {
        throw DecoderCapacityError("Prompt and output budget exceed resident decoder capacity",
                                   false);
    }
    if (!fits(output, free_bytes)) {
        throw DecoderCapacityError("Insufficient free resident decoder memory. Retry later.", true);
    }
    return output;
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

TextLogitCandidate* Allocations::prepare_packed_winners() {
    if (!state_->packed_winners && !state_->packed_winners_unavailable) {
        check(cudaPeekAtLastError(), "Check CUDA error before allocating packed winner storage");
        void* storage = nullptr;
        const auto bytes = packed_decode_max_rows * sizeof(TextLogitCandidate);
        const auto status = cudaMallocHost(&storage, bytes);
        if (status == cudaErrorMemoryAllocation) {
            const auto pending = cudaGetLastError();
            if (pending != status) {
                check(pending, "Check CUDA error after allocating packed winner storage");
            }
            state_->packed_winners_unavailable = true;
        } else {
            check(status, "Allocate packed winner storage");
            state_->packed_winners = static_cast<TextLogitCandidate*>(storage);
        }
    }
    return state_->packed_winners;
}

ExactGqaWorkspace Allocations::prepare_singleton_scores() {
    if (!state_->singleton_scores) {
        check(cudaPeekAtLastError(), "Check CUDA error before allocating singleton scores");
        constexpr auto count = static_cast<std::size_t>(singleton_score_max_visible) *
            mxfp8_query_heads;
        try {
            state_->singleton_scores = {allocate<float>(count), singleton_score_max_visible};
        } catch (const CudaError& error) {
            if (error.status != cudaErrorMemoryAllocation) {
                throw;
            }
            const auto pending = cudaGetLastError();
            if (pending != error.status) {
                check(pending, "Check CUDA error before singleton score fallback");
            }
            state_->singleton_scores = ExactGqaWorkspace{};
            std::fprintf(stderr, "sglang decoder: Allocate singleton scores: %s, "
                         "using ordered eager decode for long singletons\n",
                         cudaGetErrorString(error.status));
        }
    }
    return *state_->singleton_scores;
}

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
                                     const std::function<bool()>& bind, bool target_verify) {
    const int index = target_verify ? decoder_batch_max_graph_lanes : width - 1;
    return replay_graph_slot(index, width, prepare, capture_nodes, bind, target_verify);
}

bool Allocations::replay_packed_graph(int bucket, int rows, const std::function<void()>& prepare,
                                      const std::function<void()>& capture_nodes,
                                      const std::function<bool()>& bind) {
    if (bucket < 0 || bucket >= static_cast<int>(packed_decode_buckets.size())) {
        throw std::out_of_range("Invalid packed decode graph bucket");
    }
    return replay_graph_slot(decoder_batch_max_graph_lanes + 1 + bucket, rows, prepare,
                             capture_nodes, bind, false);
}

bool Allocations::replay_graph_slot(int index, int width, const std::function<void()>& prepare,
                                    const std::function<void()>& capture_nodes,
                                    const std::function<bool()>& bind, bool target_verify) {
    auto& batch = state_->batch_graphs.at(index);
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
        if (target_verify) {
            state_->verification_graph_incomplete = false;
        }
        std::fprintf(stderr, "sglang decoder: %s at width %d: %s, using eager %s\n",
                     operation, width, cudaGetErrorString(rejected),
                     target_verify ? "K1 target verification" : "batch decode");
        return false;
    };
    if (!batch.executable) {
        check(cudaGetLastError(), "Check CUDA error before batch graph capture");
        try {
            prepare();
        } catch (const CudaError& error) {
            if (error.status == cudaErrorMemoryAllocation) {
                return fallback(error.status, "Allocate batch graph scratch");
            }
            throw;
        }
        if (target_verify) {
            state_->verification_graph_incomplete = true;
        }
        const auto begun = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (capture_rejected(begun)) {
            return fallback(begun, "Begin batch graph capture");
        }
        check(begun, "Begin batch graph capture");
        cudaError_t enqueue_status = cudaSuccess;
        try {
            capture_nodes();
        } catch (const CudaError& error) {
            enqueue_status = error.status;
            if (!capture_rejected(enqueue_status)) {
                report_cleanup(cudaStreamEndCapture(stream, &batch.capture),
                               "End failed batch graph capture");
                throw;
            }
        } catch (...) {
            report_cleanup(cudaStreamEndCapture(stream, &batch.capture),
                           "End interrupted batch graph capture");
            throw;
        }
        const auto ended = cudaStreamEndCapture(stream, &batch.capture);
        if (capture_rejected(ended)) {
            return fallback(ended, "End batch graph capture");
        }
        check(ended, "End batch graph capture");
        if (enqueue_status != cudaSuccess) {
            return fallback(enqueue_status, "Record batch graph");
        }
        const auto built = cudaGraphInstantiate(&batch.executable, batch.capture, 0);
        if (capture_rejected(built) || built == cudaErrorInvalidValue) {
            return fallback(built, "Instantiate batch graph");
        }
        check(built, "Instantiate batch graph");
        state_->destroy_capture(batch.capture).raise();
        if (target_verify) {
            state_->verification_graph_incomplete = false;
        }
    }
    if (!bind()) {
        return false;
    }
    check(cudaGetLastError(), "Check CUDA error before batch graph launch");
    const auto launched = cudaGraphLaunch(batch.executable, stream);
    if (target_verify && launched != cudaSuccess) {
        state_->verification_graph_incomplete = true;
    } else if (launched == cudaErrorInvalidValue || launched == cudaErrorInvalidResourceHandle ||
               launched == cudaErrorNotSupported) {
        return fallback(launched, "Launch batch graph");
    }
    check(launched, "Launch batch graph");
    return true;
}

bool Allocations::verification_graph_failed() const {
    return state_->verification_graph_incomplete || state_->uncertain_graph;
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

void Allocations::discard_graph() {
    state_->destroy_executable(state_->graph).raise();
    state_->destroy_capture(state_->capture).raise();
}

void Allocations::free_block(void* pointer, std::size_t bytes) {
    auto& blocks = state_->blocks;
    const auto block = std::find(blocks.begin() + state_->uncertain_blocks, blocks.end(), pointer);
    if (block == blocks.end()) {
        throw std::logic_error("Unknown decoder allocation");
    }
    const CleanupError freed{cudaFree(pointer), "Free replaced decoder allocation"};
    if (freed) {
        state_->uncertain = freed;
        std::iter_swap(blocks.begin() + state_->uncertain_blocks++, block);
        state_->failure(freed).raise();
    }
    blocks.erase(block);
    state_->bytes -= bytes;
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

BatchGraphScratch::BatchGraphScratch(int device)
    : memory(device), device_bindings(memory.allocate<BatchDecodeBinding>(work.size())) {
    for (auto& workspace : work) {
        workspace.emplace(memory);
    }
}

ExactGqaWorkspace Request::batch_graph_scores() {
    if (long_graph_scores) {
        return *long_graph_scores;
    }
    if (pages.page_count <= 0 ||
        pages.page_count > text_attention_max_tokens / mxfp8_page_size) {
        throw std::invalid_argument("Invalid batch graph score capacity");
    }
    const int capacity = pages.page_count * mxfp8_page_size;
    const auto count = static_cast<std::size_t>(capacity) * mxfp8_query_heads;
    try {
        long_graph_scores = {memory.allocate<float>(count), capacity};
    } catch (const CudaError& error) {
        if (error.status != cudaErrorMemoryAllocation) {
            throw;
        }
        const auto previous = cudaGetLastError();
        if (previous != cudaErrorMemoryAllocation) {
            check(previous, "Check CUDA error before eager batch score fallback");
        }
        std::fprintf(stderr, "sglang decoder: Allocate batch scores for %d positions: %s, "
                     "using eager decode for long positions\n", capacity,
                     cudaGetErrorString(error.status));
        long_graph_scores = ExactGqaWorkspace{};
    }
    return *long_graph_scores;
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

Request::Request(int device, int capacity, std::shared_ptr<const void> owners, bool reusable,
                 int maximum_kv)
    : Request(device, reusable) {
    memory.open_stream(std::move(owners));
    initialize(capacity, memory.stream(), maximum_kv);
}

void Request::initialize(int capacity, cudaStream_t stream, int maximum_kv) {
    kv_limit = maximum_kv;
    const int page_count = pages_for(capacity);
    const int table_pages = std::max(page_count, pages_for(maximum_kv));
    auto* table = memory.allocate<std::int32_t>(table_pages);
    auto& logical_pages = memory.transfers().logical_pages;
    logical_pages.resize(table_pages);
    std::iota(logical_pages.begin(), logical_pages.end(), 0);
    check(cudaMemcpyAsync(table, logical_pages.data(), table_pages * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice, stream), "Upload MXFP8 page table");
    pages = {table, page_count};
    caches = allocate_kv(memory, page_count);
    allocate_gdn(memory, states);
}

void Request::grow_kv(int positions, int written, cudaStream_t stream) {
    const int needed = pages_for(positions);
    const int previous = pages.page_count;
    if (needed <= previous) {
        return;
    }
    if (positions > kv_limit || written > previous * mxfp8_page_size) {
        throw std::logic_error("Invalid decoder KV cache growth");
    }
    const int next = std::min(pages_for(kv_limit), std::max(needed, 2 * previous));
    const auto replacement = allocate_kv(memory, next);
    if (written) {
        const auto payload = pages_for(written) * mxfp8_payload_bytes_per_page;
        const auto scales = pages_for(written) * mxfp8_scale_bytes_per_page;
        for (int layer = 0; layer < full_attention_layers; ++layer) {
            const auto& old = caches[layer];
            const auto& fresh = replacement[layer];
            copy_kv_plane(fresh.key_payload, old.key_payload, payload, stream);
            copy_kv_plane(fresh.value_payload, old.value_payload, payload, stream);
            copy_kv_plane(fresh.key_scales, old.key_scales, scales, stream);
            copy_kv_plane(fresh.value_scales, old.value_scales, scales, stream);
        }
    }
    check(cudaStreamSynchronize(stream), "Complete MXFP8 cache growth");
    memory.discard_graph();
    memory.free_block(caches.front().key_payload, kv_bytes(previous));
    if (long_graph_scores && long_graph_scores->scores) {
        memory.free_block(long_graph_scores->scores,
                          long_graph_scores->capacity * mxfp8_query_heads * sizeof(float));
    }
    long_graph_scores.reset();
    caches = replacement;
    pages.page_count = next;
}

void Request::reset(cudaStream_t stream) {
    zero(states.front().history, gdn_layers * gdn_history_elements, stream);
    zero(states.front().recurrent, gdn_layers * gdn_state_elements, stream);
}

void Request::reset() {
    reset(memory.stream());
}

}
