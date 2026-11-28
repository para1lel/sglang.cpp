#include "sglang/runtime/backend/cuda_resources.h"

#include "sglang/decoder_batch.h"
#include "sglang/runtime/backend/device_arena.h"
#include "sglang/runtime/model/qwen_request.h"
#include "sglang/text/decoder_logprobs.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace sglang::decoder {
namespace {

struct CleanupError {
    cudaError_t status = cudaSuccess;
    const char* operation = "";
    bool restore_failed = false;

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

void finish_retired_cleanup(CleanupError error) {
    if (!error) return;
    if (error.restore_failed || CudaError::terminal(error.status)) error.raise();
    const auto pending = cudaGetLastError();
    report_cleanup(error.status, error.operation);
    if (CudaError::terminal(pending)) check(pending, "Check CUDA error after retirement");
    if (pending != error.status) report_cleanup(pending, "Check CUDA error after retirement");
}

}

struct DetachedDeviceBytes {
    std::mutex mutex;
    std::size_t value = 0;
};

struct ExportedDeviceBlock {
    ExportedDeviceBlock(std::shared_ptr<DeviceArena> arena,
                        std::shared_ptr<DetachedDeviceBytes> bytes)
        : arena(std::move(arena)), bytes(std::move(bytes)) {
    }

    ~ExportedDeviceBlock() {
        if (!block.pointer) return;
        arena->release(block.pointer);
        if (bytes) {
            const std::scoped_lock lock(bytes->mutex);
            bytes->value -= block.bytes;
        }
    }

    void detach(void* pointer, std::size_t size) {
        block = {pointer, size};
        if (bytes) {
            const std::lock_guard lock(bytes->mutex);
            bytes->value += size;
        }
    }

    std::shared_ptr<DeviceArena> arena;
    std::shared_ptr<DetachedDeviceBytes> bytes;
    DeviceBlock block{};
};

namespace {

bool detach_exported_block(DeviceBlock& block) {
    if (block.export_owner.use_count() <= 1) {
        block.export_owner.reset();
        return false;
    }
    block.export_owner->detach(block.pointer, block.bytes);
    return true;
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
                while (*pending && ((*pending)->device != device || (*pending)->quarantined)) {
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

    State(int owner, RequestBlockCache* cache = nullptr, bool retired = false)
        : device(owner), cache(cache), arena(DeviceArena::current(owner)) {
        if (cache && !cache->accepts(owner)) {
            throw std::invalid_argument("Decoder request cache device mismatch");
        }
        if (!retired) reclaim(owner);
    }

    bool owns_cuda() const {
        return stream || generated_logprobs || !blocks.empty();
    }

    CleanupError close_stream(std::shared_ptr<const void>& drained_owners);
    CleanupError release_selected(std::shared_ptr<const void>& drained_owners);
    CleanupError release(std::shared_ptr<const void>& drained_owners);
    CleanupError failure(CleanupError error);
    CleanupError destroy_capture(cudaGraph_t& handle);
    CleanupError destroy_executable(cudaGraphExec_t& handle);

    template <typename Element>
    CleanupError release_pinned(Element*& pointer, const char* operation) {
        if (!pointer) return {};
        const CleanupError error{cudaFreeHost(pointer), operation};
        if (error) return failure(error);
        pointer = nullptr;
        return {};
    }

    int device;
    RequestBlockCache* cache;
    std::shared_ptr<DeviceArena> arena;
    cudaStream_t stream = nullptr;
    cudaGraph_t capture = nullptr;
    cudaGraphExec_t graph = nullptr;
    std::array<BatchGraph, batch_graph_slots + verification_graph_slots> batch_graphs{};
    bool verification_graph_incomplete = false;
    bool uncertain_stream = false;
    bool uncertain_graph = false;
    std::vector<DeviceBlock> blocks;
    std::size_t bytes = 0;
    bool quarantined = false;
    CleanupError uncertain;
    CleanupError terminal;
    TransferBuffers transfers;
    GeneratedLogprobReadback* generated_logprobs = nullptr;
    std::optional<ExactGqaWorkspace> singleton_scores;
    std::shared_ptr<const void> owners;
    std::unique_ptr<State> next;
};

CleanupError Allocations::State::failure(CleanupError error) {
    if (CudaError::terminal(error.status)) {
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
    const auto logprobs = release_pinned(generated_logprobs, "Free generated logprob readback");
    if (terminal) return terminal;
    while (!blocks.empty()) {
        auto& block = blocks.back();
        if (!detach_exported_block(block))
            arena->release(block.pointer);
        bytes -= block.bytes;
        blocks.pop_back();
    }
    if (uncertain) return uncertain;
    return logprobs;
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
    const CleanupError restore{cudaSetDevice(previous), "Restore CUDA device after cleanup", true};
    if (restore) {
        report_cleanup(error.status, error.operation);
        return failure(restore);
    }
    return error;
}

std::size_t available_device_bytes() {
    int device = 0;
    check(cudaGetDevice(&device), "Read current CUDA arena device");
    return DeviceArena::current(device)->available_bytes();
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

RequestBlockCache::RequestBlockCache(int device, bool resident, bool detached_reclamation)
    : state_(std::make_unique<Allocations::State>(device, nullptr, true)),
      detached_bytes_(std::make_shared<DetachedDeviceBytes>()), resident_(resident),
      reusable_(resident) {
    if (resident || detached_reclamation)
        enable_detached_reclamation();
    if (resident)
        state_->blocks.reserve(resident_blocks_limit());
}

RequestBlockCache::~RequestBlockCache() {
    std::shared_ptr<const void> drained_owners;
    try {
        clear();
    } catch (const CudaError& error) {
        report_cleanup(error.status, "Drain retired decoder allocations during shutdown");
    }
    const auto error = state_->release(drained_owners);
    report_cleanup(error.status, error.operation);
    if (state_->owns_cuda()) {
        state_->quarantined = true;
        Allocations::State::retain(std::move(state_));
    }
}

bool RequestBlockCache::accepts(int device) const { return state_->device == device; }

std::size_t RequestBlockCache::bytes() const {
    return reusable_ ? state_->bytes : 0;
}

std::size_t RequestBlockCache::unavailable_bytes() const {
    const std::lock_guard lock(detached_bytes_->mutex);
    return quarantined_bytes_ + (state_->bytes - bytes()) + detached_bytes_->value +
           pending_bytes();
}

std::size_t RequestBlockCache::pending_bytes() const {
    return reclaimer_ ? reclaimer_->pending_bytes() : 0;
}

std::size_t RequestBlockCache::failed_release_bytes() const {
    return quarantined_bytes_;
}

void RequestBlockCache::enable_detached_reclamation() {
    if (!reclamation_enabled_) {
        reclaimer_ = RetiredBlockReclaimer::create(state_->device);
        reclamation_enabled_ = true;
    }
}

void RequestBlockCache::check_releases() const {
    state_->arena->reclaim();
}

void* RequestBlockCache::take(std::size_t bytes, bool cache_storage) {
    if (!this->bytes()) return nullptr;
    auto& blocks = state_->blocks;
    const auto block = std::find_if(blocks.begin(), blocks.end(), [=](const DeviceBlock& block) {
        return block.bytes == bytes && block.cache_storage == cache_storage;
    });
    if (block == blocks.end()) return nullptr;
    const auto pointer = block->pointer;
    *block = blocks.back();
    blocks.pop_back();
    state_->bytes -= bytes;
    return pointer;
}

bool RequestBlockCache::retain(DeviceBlock block) {
    auto& state = *state_;
    if (!block.pointer || !reusable_ || state.blocks.size() == resident_blocks_limit() ||
        block.bytes > resident_bytes_limit() - state.bytes) {
        return false;
    }
    state.blocks.push_back(block);
    state.bytes += block.bytes;
    return true;
}

std::size_t RequestBlockCache::resident_bytes_limit() const {
    return reclaimer_ ? RetiredBlockReclaimer::byte_limit : max_bytes;
}

std::size_t RequestBlockCache::resident_blocks_limit() const {
    return reclaimer_ ? RetiredBlockReclaimer::block_limit : max_blocks;
}

void RequestBlockCache::release_blocks(Allocations::State& state, bool keep) {
    for (std::size_t count = state.blocks.size(); count > 0; --count) {
        auto& block = state.blocks[count - 1];
        if (detach_exported_block(block) || (keep && retain(block)) ||
            (resident_ && reclaimer_ && reclaimer_->submit(block))) {
            const auto bytes = block.bytes;
            if (count != state.blocks.size()) block = std::move(state.blocks.back());
            state.blocks.pop_back();
            state.bytes -= bytes;
        }
    }
}

void RequestBlockCache::quarantine(Allocations::State& state) {
    quarantined_bytes_ += state.bytes;
    state.quarantined = true;
    state.cache = nullptr;
}

void RequestBlockCache::clear() {
    if (!state_->uncertain && !state_->terminal) release_blocks(*state_, false);
    if (reclaimer_) reclaimer_->drain();
    check_releases();
    if (!state_->terminal && state_->blocks.empty())
        return;
    std::shared_ptr<const void> drained_owners;
    const auto error = state_->release(drained_owners);
    if (error) reusable_ = false;
    finish_retired_cleanup(error);
}

Allocations::Allocations(int device, RequestBlockCache* cache)
    : state_(std::make_unique<State>(device, cache)) {}

Allocations::~Allocations() {
    std::shared_ptr<const void> drained_owners;
    if (!state_->quarantined) {
        const auto error = state_->release(drained_owners);
        report_cleanup(error.status, error.operation);
        if (error && state_->cache && !state_->stream && state_->owns_cuda()) {
            state_->transfers = {};
            state_->cache->quarantine(*state_);
        }
    }
    if (state_->owns_cuda()) {
        State::retain(std::move(state_));
    }
}

std::size_t Allocations::bytes() const { return state_->bytes; }

TransferBuffers& Allocations::transfers() { return state_->transfers; }

GeneratedLogprobReadback* Allocations::prepare_generated_logprobs() {
    auto& host = state_->generated_logprobs;
    if (!host) {
        void* memory = nullptr;
        check(cudaMallocHost(&memory, decoder_batch_max_lanes * sizeof(GeneratedLogprobReadback)),
              "Allocate generated logprob readback");
        host = static_cast<GeneratedLogprobReadback*>(memory);
    }
    return host;
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
                         "using direct ordered eager attention\n",
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
                                     const std::function<bool()>& bind, int slot) {
    return replay_graph_slot(slot < 0 ? width - 1 : slot, width, prepare, capture_nodes, bind,
                             false);
}

void Allocations::reset_batch_graph(int slot) {
    reset_graph_slot(slot);
}

bool Allocations::replay_verification_graph(int slot, int rows,
                                            const std::function<void()>& prepare,
                                            const std::function<void()>& capture_nodes,
                                            const std::function<bool()>& bind) {
    return replay_graph_slot(batch_graph_slots + slot, rows, prepare, capture_nodes, bind, true);
}

void Allocations::reset_verification_graph(int slot) {
    reset_graph_slot(batch_graph_slots + slot);
}

void Allocations::reset_graph_slot(int index) {
    auto& batch = state_->batch_graphs.at(index);
    state_->destroy_executable(batch.executable).raise();
    state_->destroy_capture(batch.capture).raise();
    batch.disabled = false;
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
        std::cerr << "sglang decoder: " << operation << " at width " << width << ": "
                  << cudaGetErrorString(rejected) << ", using eager "
                  << (target_verify ? "target verification" : "batch decode") << '\n';
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
        if (target_verify && !bind())
            return false;
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
    } else if (target_verify && !bind()) {
        return false;
    }
    if (!target_verify && !bind()) {
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

void Allocations::retain_stream_owner(std::shared_ptr<const void> owner) {
    state_->owners = std::move(owner);
}

void Allocations::close_stream() {
    std::shared_ptr<const void> drained_owners;
    state_->close_stream(drained_owners).raise();
}

void Allocations::close() {
    std::shared_ptr<const void> drained_owners;
    state_->release(drained_owners).raise();
}

void Allocations::recycle_after_completion() {
    auto& state = *state_;
    if (!state.cache || state.stream) {
        close();
        return;
    }
    if (!state.uncertain && !state.terminal) {
        state.cache->release_blocks(state, true);
    }
    std::shared_ptr<const void> drained_owners;
    const auto error = state.release(drained_owners);
    if (error && state.owns_cuda()) {
        state.transfers = {};
        state.cache->quarantine(state);
    }
    finish_retired_cleanup(error);
}

void Allocations::discard_graph() {
    state_->destroy_executable(state_->graph).raise();
    state_->destroy_capture(state_->capture).raise();
}

std::shared_ptr<const void> Allocations::retain_block(const void* pointer) const {
    auto& blocks = state_->blocks;
    const auto block =
        std::find_if(blocks.begin(), blocks.end(),
                     [pointer](const DeviceBlock& item) { return item.pointer == pointer; });
    if (block == blocks.end()) throw std::logic_error("Unknown decoder export allocation");
    if (!block->export_owner) {
        const auto* cache = state_->cache;
        block->export_owner = std::make_shared<ExportedDeviceBlock>(
            state_->arena, cache != nullptr ? cache->detached_bytes_ : nullptr);
    }
    return block->export_owner;
}

void Allocations::free_block(void* pointer, std::size_t bytes) {
    auto& blocks = state_->blocks;
    const auto block =
        std::find_if(blocks.begin(), blocks.end(),
                     [pointer](const DeviceBlock& block) { return block.pointer == pointer; });
    if (block == blocks.end() || block->bytes != bytes) {
        throw std::logic_error("Unknown decoder allocation");
    }
    if (!detach_exported_block(*block))
        state_->arena->release(pointer);
    blocks.erase(block);
    state_->bytes -= bytes;
}

void* Allocations::allocate_bytes(std::size_t bytes, bool cache_storage) {
    if (bytes == 0)
        return nullptr;
    if (bytes > std::numeric_limits<std::size_t>::max() - state_->bytes) {
        throw std::overflow_error("CUDA allocation total overflow");
    }
    auto& blocks = state_->blocks;
    blocks.push_back({nullptr, bytes, {}, cache_storage});
    auto& pointer = blocks.back().pointer;
    auto* cache = state_->cache;
    if (cache != nullptr)
        pointer = cache->take(bytes, cache_storage);
    const auto allocate = [&] {
        return cache_storage ? state_->arena->allocate_cache(bytes)
                             : state_->arena->allocate(bytes);
    };
    try {
        if (pointer == nullptr) {
            try {
                pointer = allocate();
            } catch (const CudaError& error) {
                if (error.status != cudaErrorMemoryAllocation || cache == nullptr ||
                    (cache->bytes() == 0 && cache->pending_bytes() == 0)) {
                    throw;
                }
                cache->clear();
                pointer = allocate();
            }
        }
    } catch (...) {
        blocks.pop_back();
        throw;
    }
    state_->bytes += bytes;
    return pointer;
}
}

namespace sglang::decoder {

void copy_cuda_async(void* destination, const void* source, std::size_t bytes,
                            cudaMemcpyKind kind, cudaStream_t stream, const char* operation) {
    check(cudaMemcpyAsync(destination, source, bytes, kind, stream), operation);
}

void zero_cuda_async(void* destination, std::size_t bytes, cudaStream_t stream,
                            const char* operation) {
    check(cudaMemsetAsync(destination, 0, bytes, stream), operation);
}

void synchronize_cuda(cudaStream_t stream, const char* operation) {
    check(cudaStreamSynchronize(stream), operation);
}

void* Allocations::allocate_elements(std::size_t count, std::size_t element_size) {
    if (element_size == 0 || count > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("CUDA allocation size overflow");
    }
    return allocate_bytes(count * element_size, false);
}

void* Allocations::allocate_cache_elements(std::size_t count, std::size_t element_size) {
    if (element_size == 0 || count > std::numeric_limits<std::size_t>::max() / element_size) {
        throw std::overflow_error("CUDA cache allocation size overflow");
    }
    return allocate_bytes(count * element_size, true);
}

template <typename Element>
Element* Allocations::allocate(std::size_t count) {
    return static_cast<Element*>(allocate_elements(count, sizeof(Element)));
}

template Bf16* Allocations::allocate<Bf16>(std::size_t);
template float* Allocations::allocate<float>(std::size_t);
template int* Allocations::allocate<int>(std::size_t);
template std::uint8_t* Allocations::allocate<std::uint8_t>(std::size_t);
template std::uint32_t* Allocations::allocate<std::uint32_t>(std::size_t);
template std::byte* Allocations::allocate<std::byte>(std::size_t);
template TextLogitCandidate* Allocations::allocate<TextLogitCandidate>(std::size_t);
template DecoderLogitBias* Allocations::allocate<DecoderLogitBias>(std::size_t);
template MropePosition* Allocations::allocate<MropePosition>(std::size_t);

template <typename Element> Element* Allocations::allocate_cache(std::size_t count) {
    return static_cast<Element*>(allocate_cache_elements(count, sizeof(Element)));
}

template Bf16* Allocations::allocate_cache<Bf16>(std::size_t);
template float* Allocations::allocate_cache<float>(std::size_t);
template std::uint8_t* Allocations::allocate_cache<std::uint8_t>(std::size_t);
template std::byte* Allocations::allocate_cache<std::byte>(std::size_t);
}
