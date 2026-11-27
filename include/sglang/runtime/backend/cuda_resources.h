#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>

namespace sglang::decoder {

struct ExactGqaWorkspace;
struct GeneratedLogprobReadback;
struct TransferBuffers;

void check(cudaError_t status, const char* operation);
struct CudaError : std::runtime_error {
    CudaError(cudaError_t status, const char* operation);
    static cudaError_t from_driver(CUresult status) noexcept;
    static bool terminal(cudaError_t error);
    cudaError_t status;
};

std::size_t available_device_bytes();

cudaError_t release_cuda_block(void* pointer, cudaStream_t stream,
                                      bool wait_for_release = false) noexcept;

void copy_cuda_async(void* destination, const void* source, std::size_t bytes,
                            cudaMemcpyKind kind, cudaStream_t stream, const char* operation);

void zero_cuda_async(void* destination, std::size_t bytes, cudaStream_t stream,
                            const char* operation);

void synchronize_cuda(cudaStream_t stream, const char* operation);

class DeviceScope {
public:
    explicit DeviceScope(int device);
    ~DeviceScope();
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

    void restore();

private:
    int previous_ = 0;
    bool changed_ = false;
};

struct ExportedDeviceBlock;
struct DetachedDeviceBytes;

struct DeviceBlock {
    void* pointer;
    std::size_t bytes;
    std::shared_ptr<ExportedDeviceBlock> export_owner{};
};

class RequestBlockCache;
class RetiredBlockReclaimer;

class Allocations {
public:
    explicit Allocations(int device, RequestBlockCache* cache = nullptr);
    ~Allocations();
    Allocations(const Allocations&) = delete;
    Allocations& operator=(const Allocations&) = delete;

    template <typename Element>
    Element* allocate(std::size_t count);

    void* allocate_elements(std::size_t count, std::size_t element_size);
    std::size_t bytes() const;
    TransferBuffers& transfers();
    GeneratedLogprobReadback* prepare_generated_logprobs();
    ExactGqaWorkspace prepare_singleton_scores();
    cudaStream_t stream() const;
    cudaGraphExec_t graph() const;
    void capture_graph(const std::function<void()>& enqueue);
    void discard_graph();
    void free_block(void* pointer, std::size_t bytes);
    std::shared_ptr<const void> retain_block(const void* pointer) const;
    bool replay_batch_graph(int width, const std::function<void()>& prepare,
                            const std::function<void()>& capture,
                            const std::function<bool()>& bind, bool target_verify = false);
    bool verification_graph_failed() const;
    void reclaim_deferred();
    void open_stream(std::shared_ptr<const void> owners = nullptr);
    void retain_stream_owner(std::shared_ptr<const void> owner);
    void close_stream();
    void close();
    void recycle_after_completion();

private:
    friend class RequestBlockCache;
    struct State;
    void* allocate_bytes(std::size_t bytes);
    bool replay_graph_slot(int index, int width, const std::function<void()>& prepare,
                           const std::function<void()>& capture,
                           const std::function<bool()>& bind, bool target_verify);

    std::unique_ptr<State> state_;
};

class RequestBlockCache {
public:
    explicit RequestBlockCache(int device, bool resident = false);
    ~RequestBlockCache();
    RequestBlockCache(const RequestBlockCache&) = delete;
    RequestBlockCache& operator=(const RequestBlockCache&) = delete;

    std::size_t bytes() const;
    std::size_t unavailable_bytes() const;
    std::size_t pending_bytes() const;
    void clear();

private:
    friend class Allocations;
    bool accepts(int device) const;
    void* take(std::size_t bytes);
    bool retain(DeviceBlock block);
    void quarantine(Allocations::State& state);

    void release_blocks(Allocations::State& state, bool keep);
    std::size_t resident_bytes_limit() const;
    std::size_t resident_blocks_limit() const;
    static constexpr std::size_t max_bytes = 768 * 1024 * 1024;
    static constexpr std::size_t max_blocks = 384;
    std::unique_ptr<Allocations::State> state_;
    std::shared_ptr<RetiredBlockReclaimer> reclaimer_;
    std::shared_ptr<DetachedDeviceBytes> detached_bytes_;
    std::size_t quarantined_bytes_ = 0;
    bool reusable_ = true;
};

class RetiredBlockReclaimer {
public:
    static constexpr std::size_t byte_limit = 3ULL * 1024 * 1024 * 1024;
    static constexpr std::size_t block_limit = 1024;

    static std::unique_ptr<RetiredBlockReclaimer> create(int device);
    ~RetiredBlockReclaimer();
    bool submit(DeviceBlock block, bool detached = false);
    std::size_t unavailable_bytes() const;
    std::size_t pending_bytes() const;
    void drain();

private:
    struct State;
    RetiredBlockReclaimer(int device, cudaStream_t stream);
    std::unique_ptr<State> state_;
};

}
