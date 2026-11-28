#include "sglang/decoder_prefill.h"

#include "sglang/prefill_fp8.h"
#include "sglang/runtime/backend/device_arena.h"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace sglang {
namespace {

using namespace decoder;

void check_prefill_pool_result(cudaError_t status, bool can_decline, const char* operation) {
    if (status == cudaSuccess) return;
    const auto recorded = cudaGetLastError();
    if (recorded != status && !CudaError::terminal(status)) check(recorded, operation);
    if (!can_decline) check(status, operation);
}

bool prefill_allocation_unavailable(cudaError_t status) {
    return status == cudaErrorMemoryAllocation || status == cudaErrorNotSupported;
}

bool prefill_peer_setup(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    const bool unavailable = prefill_allocation_unavailable(status) ||
        status == cudaErrorInvalidValue;
    check_prefill_pool_result(status, unavailable, operation);
    return false;
}

cudaError_t prefill_peer_status(cudaError_t status) {
    const auto recorded = cudaGetLastError();
    return !CudaError::terminal(status) && CudaError::terminal(recorded) ? recorded : status;
}

void check_prefill_peer_submission(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) check(prefill_peer_status(status), operation);
}

bool fast_prefill_peer(int device, int remote) {
    int major = 0, minor = 0, access = 0, performance = 0, atomics = 0;
    check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
          "Check peer prefill device architecture");
    check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device),
          "Check peer prefill device architecture");
    if (major != 10 || minor != 0) return false;
    check(cudaDeviceGetP2PAttribute(&access, cudaDevP2PAttrAccessSupported, device, remote),
          "Check grouped prefill peer access");
    if (!access) return false;
    check(cudaDeviceGetP2PAttribute(&performance, cudaDevP2PAttrPerformanceRank, device, remote),
          "Check grouped prefill peer bandwidth class");
    check(cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, device, remote),
          "Check grouped prefill native peer fabric");
    return performance == 0 && atomics;
}

}

void decoder::reclaim_prefill_workspaces(int device) {
    DeviceArena::current(device)->reclaim();
}

PrefillWorkspace::PrefillWorkspace(cudaStream_t stream, int capacity, GqaGeometry staged_geometry,
                                   int staged_visible, bool resident_staging)
    : stream_(stream), capacity_(capacity) {
    const int staged_kv_heads = staged_geometry.kv_heads;
    if (stream == nullptr || capacity < 1 || capacity > grouped_prefill_max_rows ||
        (staged_kv_heads != gqa_rank_geometry.kv_heads &&
         staged_kv_heads != gqa_full_geometry.kv_heads)) {
        throw std::invalid_argument("Invalid decoder prefill workspace");
    }
    if (capacity == 1) {
        return;
    }
    check(cudaGetLastError(), "Check CUDA state before allocating prefill workspace");
    int device = 0;
    check(cudaStreamGetDevice(stream, &device), "Get owner of prefill workspace");
    arena_ = DeviceArena::current(device);
    const auto base = prefill_workspace_bytes(capacity);
    constexpr std::size_t alignment = 128;
    const auto offset = (base + alignment - 1) / alignment * alignment;
    const auto visible = std::clamp(staged_visible, 0, text_attention_max_tokens);
    const auto pages = (visible + mxfp8_page_size - 1) / mxfp8_page_size;
    const auto page_heads = static_cast<std::size_t>(pages) * staged_kv_heads;
    constexpr std::size_t bytes_per_page_head =
        static_cast<std::size_t>(mxfp8_page_size) * mxfp8_head_dim * sizeof(__nv_bfloat16);
    static_assert(bytes_per_page_head % alignment == 0);
    const auto plane = page_heads * bytes_per_page_head;
    const auto values = plane + page_heads * sizeof(int);
    auto bytes = plane ? offset + plane + values : base;
    void* data = nullptr;
    try {
        data = arena_->allocate(bytes);
    } catch (const CudaError& error) {
        if (error.status != cudaErrorMemoryAllocation)
            throw;
        if (plane == 0)
            return;
        bytes = base;
        try {
            data = arena_->allocate(bytes);
        } catch (const CudaError& fallback) {
            if (fallback.status != cudaErrorMemoryAllocation)
                throw;
            return;
        }
    }
    data_ = data;
    bytes_ = bytes;
    if (bytes != base) {
        auto* keys = static_cast<std::byte*>(data) + offset;
        staging_ = {keys, plane, keys + plane, values};
        resident_staging_ = resident_staging;
    }
}

std::size_t PrefillWorkspace::bytes() const { return bytes_; }
cudaStream_t PrefillWorkspace::stream() const { return stream_; }
int PrefillWorkspace::capacity() const { return capacity_; }
void* PrefillWorkspace::data() const { return data_; }
PairedPrefillScratch PrefillWorkspace::long_gqa_staging() const { return staging_; }
bool PrefillWorkspace::resident_gqa_staging() const {
    return resident_staging_ && staging_.keys != nullptr;
}

bool PrefillWorkspace::allow_peer_access(int peer_device) {
    return arena_ && arena_->allows_peer_access(peer_device);
}

PrefillPeerNorm::PrefillPeerNorm() noexcept = default;

PrefillPeerNorm::~PrefillPeerNorm() {
    for (const auto& rank : ranks_) {
        if (rank.produced) (void)cudaEventDestroy(rank.produced);
        if (rank.consumed) (void)cudaEventDestroy(rank.consumed);
    }
}

void PrefillPeerNorm::prepare(PrefillWorkspace& root, PrefillWorkspace& peer,
                             int root_device, int peer_device) {
    if (root_device == peer_device || !fast_prefill_peer(root_device, peer_device) ||
        !fast_prefill_peer(peer_device, root_device)) return;
    ranks_ = {{{root_device, root.stream()}, {peer_device, peer.stream()}}};
    for (const auto& rank : ranks_) {
        cudaStreamCaptureStatus capture{};
        check(cudaStreamIsCapturing(rank.stream, &capture), "Check prefill peer stream capture");
        if (capture != cudaStreamCaptureStatusNone) return;
    }
    if (!root.allow_peer_access(peer_device)) return;
    {
        const DeviceScope scope(peer_device);
        if (!peer.allow_peer_access(root_device)) return;
    }
    for (auto& rank : ranks_) {
        const DeviceScope scope(rank.device);
        if (!prefill_peer_setup(cudaEventCreateWithFlags(&rank.produced, cudaEventDisableTiming),
                                "Create grouped prefill producer event") ||
            !prefill_peer_setup(cudaEventCreateWithFlags(&rank.consumed, cudaEventDisableTiming),
                                "Create grouped prefill consumer event")) return;
    }
    ready_ = true;
}

bool PrefillPeerNorm::enqueue(const std::array<PrefillPeerNormRow, 2>& rows) const {
    if (!ready_ || !aligned_prefill_peer_norm(rows[0]) || !aligned_prefill_peer_norm(rows[1])) {
        return false;
    }
    try {
        for (const auto& rank : ranks_) {
            const DeviceScope scope(rank.device);
            check_prefill_peer_submission(cudaEventRecord(rank.produced, rank.stream),
                                          "Publish prefill projection");
        }
        for (std::size_t i = 0; i < ranks_.size(); ++i) {
            const auto& rank = ranks_[i];
            const auto& peer = ranks_[1 - i];
            const DeviceScope scope(rank.device);
            check_prefill_peer_submission(cudaStreamWaitEvent(rank.stream, peer.produced, 0),
                                          "Wait for peer projection");
            check(prefill_peer_rms_quantize(rows[i], rows[1 - i].branch, rank.stream),
                  "Reduce, normalize and quantize paired prefill projection");
            check_prefill_peer_submission(cudaEventRecord(rank.consumed, rank.stream),
                                          "Publish prefill peer read end");
        }
        for (std::size_t i = 0; i < ranks_.size(); ++i) {
            check_prefill_peer_submission(
                cudaStreamWaitEvent(ranks_[i].stream, ranks_[1 - i].consumed, 0),
                "Fence reuse of grouped prefill projections");
        }
    } catch (...) {
        cudaError_t terminal = cudaSuccess;
        for (const auto& rank : ranks_) {
            const auto status = prefill_peer_status(cudaStreamSynchronize(rank.stream));
            if (terminal == cudaSuccess && CudaError::terminal(status)) terminal = status;
        }
        check(terminal, "Drain grouped prefill after failed peer submission");
        throw;
    }
    return true;
}

PrefillWorkspace::~PrefillWorkspace() {
    if (release_status_ == cudaSuccess) {
        const auto status = release();
        if (status != cudaSuccess) {
            std::fprintf(stderr, "Release prefill workspace: %s\n", cudaGetErrorString(status));
        }
    }
}

cudaError_t PrefillWorkspace::release() noexcept {
    if (data_ && release_status_ == cudaSuccess) {
        release_status_ = arena_->release_after(data_, stream_);
        data_ = nullptr;
        staging_ = {};
    }
    stream_ = nullptr;
    return release_status_;
}

void PrefillWorkspace::close() {
    check(release(), "Release prefill workspace");
}

}
