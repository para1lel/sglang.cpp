#include "sglang/runtime/backend/tensor_parallel.h"

#include <nccl.h>

#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/execution/tp_verify_graph.h"

#if defined(SGLANG_MNNVL_AR) && SGLANG_MNNVL_AR
#include "sglang/runtime/backend/mnnvl_rms_quantize.h"
#include "sglang/runtime/backend/mnnvl_ar.h"
#define SGLANG_USE_MNNVL_AR 1
#else
#define SGLANG_USE_MNNVL_AR 0
#endif

namespace sglang::decoder {
namespace {

void check_nccl(ncclResult_t status, const char* operation) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
    }
}

class CallerDevice {
public:
    CallerDevice() { check(cudaGetDevice(&previous_), "Read the current CUDA device"); }
    ~CallerDevice() { (void)cudaSetDevice(previous_); }
    CallerDevice(const CallerDevice&) = delete;
    CallerDevice& operator=(const CallerDevice&) = delete;

    void select(int device, const char* operation) { check(cudaSetDevice(device), operation); }

private:
    int previous_ = 0;
};

struct CollectiveStreams {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    bool same_capture = true;
};

CollectiveStreams validate_collective_streams(const std::vector<int>& devices,
                                              const cudaStream_t* streams, std::size_t ranks) {
    CollectiveStreams state;
    unsigned long long capture_id = 0;
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        int device = -1;
        if (cudaStreamGetDevice(streams[rank], &device) != cudaSuccess ||
            device != devices[rank]) {
            (void)cudaGetLastError();
            throw std::invalid_argument(
                "TensorParallel collectives require each rank's stream on that rank's device.");
        }
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        unsigned long long id = 0;
        if (cudaStreamGetCaptureInfo(streams[rank], &status, &id) != cudaSuccess ||
            (status != cudaStreamCaptureStatusNone &&
             status != cudaStreamCaptureStatusActive)) {
            (void)cudaGetLastError();
            throw std::invalid_argument(
                "TensorParallel collectives require a usable stream capture state.");
        }
        if (rank == 0) {
            state.status = status;
            capture_id = id;
            if (status == cudaStreamCaptureStatusActive && id == 0) {
                state.same_capture = false;
            }
        } else {
            if (status != state.status) {
                throw std::invalid_argument(
                    "TensorParallel collectives require one capture state across ranks.");
            }
            if (status == cudaStreamCaptureStatusActive &&
                (id == 0 || id != capture_id)) {
                state.same_capture = false;
            }
        }
    }
    return state;
}

#if defined(NCCL_MAJOR) && defined(NCCL_MINOR)
constexpr bool nccl_supports_cuda_graphs = NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >= 9);
#else
constexpr bool nccl_supports_cuda_graphs = false;
#endif

enum class ArFallback {
    Disabled, SetupUnavailable, Oversize, CaptureMismatch, Unaligned, AfterError
};

constexpr std::size_t mnnvl_ar_hidden_width = 5120;

}   

struct TensorParallel::State {
    std::vector<int> devices;
    std::vector<ncclComm_t> comms;
    bool failed = false;

#if SGLANG_USE_MNNVL_AR
    std::unique_ptr<MnnvlAllReduce> mnnvl;
    bool mnnvl_sticky = false;
#endif

    bool mnnvl_active_mirror = false;

    std::mutex counter_mutex;
    std::vector<TpArRankCounters> ar_counters;
    struct ArWidthEntry {
        std::vector<std::uint64_t> mnnvl_calls;
        std::vector<std::uint64_t> nccl_calls;
        std::vector<std::uint64_t> payload_bytes;
    };
    std::map<std::size_t, ArWidthEntry> ar_widths;
    std::set<std::size_t> logged_widths;

    void enqueue_nccl_allreduce(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                                std::size_t count);
    void bump_transport_width(std::size_t count, std::size_t payload, bool vendor,
                              std::size_t calls = 1);
    void record_nccl_success(std::size_t count, std::size_t payload, ArFallback reason);
#if SGLANG_USE_MNNVL_AR
    ArFallback vendor_fallback_reason(std::size_t count,
                                      const CollectiveStreams& capture_state, bool two_shot) const;
    bool vendor_allreduce(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                          std::size_t count, const CollectiveStreams& capture_state, bool two_shot,
                          const MnnvlRmsQuantizeRows* rms_rows = nullptr,
                          const __nv_bfloat16* const* gains = nullptr);
    void note_vendor_failure();
    void record_vendor_success(std::size_t count, std::size_t payload, bool two_shot,
                               std::size_t calls = 1);
#endif

    ~State() {
        int restore = -1;
        if (cudaGetDevice(&restore) != cudaSuccess) restore = -1;
        for (std::size_t rank = 0; rank < comms.size(); ++rank) {
            if (comms[rank] == nullptr) continue;
            (void)cudaSetDevice(devices[rank]);
            ncclResult_t asynchronous = ncclSuccess;
            const ncclResult_t polled = ncclCommGetAsyncError(comms[rank], &asynchronous);
            if (failed || polled != ncclSuccess || asynchronous != ncclSuccess) {
                (void)ncclCommAbort(comms[rank]);
            } else {
                (void)ncclCommDestroy(comms[rank]);
            }
        }
        if (restore >= 0) (void)cudaSetDevice(restore);
    }
};

void TensorParallel::State::enqueue_nccl_allreduce(__nv_bfloat16* const buffers[],
                                                   const cudaStream_t streams[],
                                                   std::size_t count) {
    check_nccl(ncclGroupStart(), "Open the NCCL operation group");
    try {
        CallerDevice device_scope;
        for (std::size_t rank = 0; rank < devices.size(); ++rank) {
            device_scope.select(devices[rank], "Select a tensor-parallel device");
            const ncclResult_t status =
                ncclAllReduce(buffers[rank], buffers[rank], count, ncclBfloat16, ncclSum,
                              comms[rank], streams[rank]);
            if (status != ncclSuccess) {
                throw std::runtime_error(std::string("Enqueue the NCCL BF16 allreduce: ") +
                                         ncclGetErrorString(status));
            }
        }
    } catch (...) {
        (void)ncclGroupEnd();
        failed = true;
        throw;
    }
    const ncclResult_t grouped = ncclGroupEnd();
    if (grouped != ncclSuccess) {
        failed = true;
        throw std::runtime_error(std::string("Close the NCCL operation group: ") +
                                 ncclGetErrorString(grouped));
    }
}

void TensorParallel::State::bump_transport_width(std::size_t count, std::size_t payload,
                                                 bool vendor, std::size_t calls) {
    ArWidthEntry& width = ar_widths[count];
    if (width.mnnvl_calls.empty()) {
        width.mnnvl_calls.assign(devices.size(), 0);
        width.nccl_calls.assign(devices.size(), 0);
        width.payload_bytes.assign(devices.size(), 0);
    }
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        if (vendor) {
            width.mnnvl_calls[rank] += calls;
        } else {
            width.nccl_calls[rank] += calls;
        }
        width.payload_bytes[rank] += calls * payload;
    }
}

void TensorParallel::State::record_nccl_success(std::size_t count, std::size_t payload,
                                                ArFallback reason) {
    std::uint64_t TpArRankCounters::* reason_counter = nullptr;
    switch (reason) {
        case ArFallback::Disabled:
            reason_counter = &TpArRankCounters::ar_fallback_disabled_total;
            break;
        case ArFallback::SetupUnavailable:
            reason_counter = &TpArRankCounters::ar_fallback_setup_unavailable_total;
            break;
        case ArFallback::Oversize:
            reason_counter = &TpArRankCounters::ar_fallback_oversize_total;
            break;
        case ArFallback::CaptureMismatch:
            reason_counter = &TpArRankCounters::ar_fallback_capture_mismatch_total;
            break;
        case ArFallback::Unaligned:
            reason_counter = &TpArRankCounters::ar_fallback_unaligned_total;
            break;
        case ArFallback::AfterError:
            reason_counter = &TpArRankCounters::ar_fallback_after_error_total;
            break;
    }
    const std::lock_guard<std::mutex> lock(counter_mutex);
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        TpArRankCounters& counter = ar_counters[rank];
        ++counter.ar_calls_total;
        ++counter.ar_nccl_total;
        ++(counter.*reason_counter);
        counter.ar_bytes_payload_total += payload;
        counter.ar_bytes_aggregated_total += payload * devices.size();
    }
    bump_transport_width(count, payload, false);
#if SGLANG_USE_MNNVL_AR
    mnnvl_active_mirror = mnnvl != nullptr && !mnnvl_sticky && mnnvl->healthy();
#endif
    if (logged_widths.insert(count).second) {
        std::cerr << "tp2_transport width=" << count << " transport=nccl" << std::endl;
    }
}

#if SGLANG_USE_MNNVL_AR
ArFallback TensorParallel::State::vendor_fallback_reason(std::size_t count,
                                                         const CollectiveStreams& capture_state,
                                                         bool two_shot) const {
    if (mnnvl == nullptr) {
        return ArFallback::SetupUnavailable;
    }
    if (mnnvl_sticky || !mnnvl->healthy()) {
        return ArFallback::AfterError;
    }
    if (!two_shot && (count > mnnvl->max_count() || count % mnnvl_ar_hidden_width != 0)) {
        return ArFallback::Oversize;
    }
    if (capture_state.status == cudaStreamCaptureStatusActive && !capture_state.same_capture) {
        return ArFallback::CaptureMismatch;
    }
    return ArFallback::Unaligned;
}

bool TensorParallel::State::vendor_allreduce(
    __nv_bfloat16* const buffers[], const cudaStream_t streams[], std::size_t count,
    const CollectiveStreams& capture_state, bool two_shot, const MnnvlRmsQuantizeRows* rms_rows,
    const __nv_bfloat16* const* gains) {
    if (mnnvl == nullptr || mnnvl_sticky || !mnnvl->healthy() ||
        (!two_shot && (count > mnnvl->max_count() || count % mnnvl_ar_hidden_width != 0))) {
        return false;
    }
    if (capture_state.status == cudaStreamCaptureStatusActive && !capture_state.same_capture) {
        return false;
    }
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        if (reinterpret_cast<std::uintptr_t>(buffers[rank]) % 16 != 0) {
            return false;
        }
    }
    try {
        if (rms_rows) {
            return mnnvl->try_allreduce_rms_quantize(buffers, streams, rms_rows, gains);
        }
        mnnvl->allreduce_bf16(buffers, streams, count, two_shot);
    } catch (const std::invalid_argument&) {
        throw;
    } catch (const MnnvlArError& error) {
        if (error.peer_enqueued) {
            failed = true;
        } else {
            mnnvl_sticky = true;
        }
        note_vendor_failure();
        throw;
    } catch (...) {
        failed = true;
        note_vendor_failure();
        throw;
    }
    return true;
}

void TensorParallel::State::note_vendor_failure() {
    const std::lock_guard<std::mutex> lock(counter_mutex);
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        ++ar_counters[rank].ar_vendor_failed_total;
    }
    mnnvl_active_mirror = false;
}

void TensorParallel::State::record_vendor_success(std::size_t count, std::size_t payload,
                                                  bool two_shot, std::size_t calls) {
    const auto transport = two_shot ? &TpArRankCounters::ar_mnnvl_twoshot_total
                                     : &TpArRankCounters::ar_mnnvl_oneshot_total;
    const std::lock_guard<std::mutex> lock(counter_mutex);
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        TpArRankCounters& counter = ar_counters[rank];
        counter.ar_calls_total += calls;
        (counter.*transport) += calls;
        counter.ar_bytes_payload_total += calls * payload;
        counter.ar_bytes_aggregated_total += calls * payload * devices.size();
    }
    bump_transport_width(count, payload, true, calls);
    mnnvl_active_mirror = true;
    if (logged_widths.insert(count).second) {
        std::cerr << "tp2_transport width=" << count << " transport=mnnvl" << std::endl;
    }
}
#endif

std::unique_ptr<TensorParallel> TensorParallel::create(std::vector<int> devices) {
    return std::unique_ptr<TensorParallel>(new TensorParallel(std::move(devices)));
}

TensorParallel::TensorParallel(std::vector<int> devices) : state_(new State) {
    state_->devices = std::move(devices);
    const std::size_t ranks = state_->devices.size();
    if (ranks == 0) {
        throw std::invalid_argument("TensorParallel requires at least one device.");
    }
    for (std::size_t left = 0; left < ranks; ++left) {
        for (std::size_t right = left + 1; right < ranks; ++right) {
            if (state_->devices[left] == state_->devices[right]) {
                throw std::invalid_argument("TensorParallel requires distinct devices.");
            }
        }
    }

    {
        CallerDevice validation_scope;
        for (const int device : state_->devices) {
            validation_scope.select(device, "Select a tensor-parallel device");
        }
    }

    ncclUniqueId id{};
    check_nccl(ncclGetUniqueId(&id), "Create an NCCL communicator id");
    state_->comms.assign(ranks, nullptr);

    check_nccl(ncclGroupStart(), "Open the NCCL communicator group");
    ncclResult_t first = ncclSuccess;
    try {
        CallerDevice device_scope;
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            device_scope.select(state_->devices[rank], "Select a tensor-parallel device");
            const ncclResult_t status =
                ncclCommInitRank(&state_->comms[rank], static_cast<int>(ranks), id,
                                 static_cast<int>(rank));
            if (status != ncclSuccess && first == ncclSuccess) first = status;
        }
    } catch (...) {
        (void)ncclGroupEnd();
        state_->failed = true;
        throw;
    }
    const ncclResult_t grouped = ncclGroupEnd();
    if (first != ncclSuccess || grouped != ncclSuccess) {
        state_->failed = true;
        throw std::runtime_error(std::string("Initialize NCCL tensor-parallel communicators: ") +
                                 ncclGetErrorString(first != ncclSuccess ? first : grouped));
    }

    state_->ar_counters.assign(ranks, TpArRankCounters{});

#if SGLANG_USE_MNNVL_AR
    if (ranks == 2) {
        state_->mnnvl = MnnvlAllReduce::create(state_->devices);
        const std::lock_guard<std::mutex> lock(state_->counter_mutex);
        state_->mnnvl_active_mirror = state_->mnnvl != nullptr && state_->mnnvl->healthy();
    }
#endif
}

TensorParallel::~TensorParallel() = default;

int TensorParallel::rank_count() const noexcept { return static_cast<int>(state_->devices.size()); }

int TensorParallel::device(int rank) const {
    if (rank < 0 || static_cast<std::size_t>(rank) >= state_->devices.size()) {
        throw std::invalid_argument("TensorParallel::device requires a rank of the group.");
    }
    return state_->devices[static_cast<std::size_t>(rank)];
}

void TensorParallel::ensure_healthy() {
    if (state_->failed) {
        throw std::runtime_error("A tensor-parallel collective failed earlier.");
    }
    for (ncclComm_t comm : state_->comms) {
        ncclResult_t asynchronous = ncclSuccess;
        const ncclResult_t polled = ncclCommGetAsyncError(comm, &asynchronous);
        if (polled != ncclSuccess || asynchronous != ncclSuccess) {
            state_->failed = true;
            throw std::runtime_error(
                std::string("A tensor-parallel collective failed asynchronously: ") +
                ncclGetErrorString(polled != ncclSuccess ? polled : asynchronous));
        }
    }
}

void TensorParallel::allreduce_bf16(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                                    std::size_t count) {
    const std::size_t ranks = state_->devices.size();
    if (count == 0) {
        throw std::invalid_argument(
            "TensorParallel::allreduce_bf16 requires a positive element count.");
    }
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        if (buffers[rank] == nullptr || streams[rank] == nullptr) {
            throw std::invalid_argument(
                "TensorParallel::allreduce_bf16 requires a buffer and a stream for every rank.");
        }
    }
    ensure_healthy();
    [[maybe_unused]] const CollectiveStreams capture_state =
        validate_collective_streams(state_->devices, streams, ranks);
    const std::size_t payload = count * sizeof(__nv_bfloat16);
#if SGLANG_USE_MNNVL_AR
    const bool two_shot = capture_state.status == cudaStreamCaptureStatusNone &&
                          state_->mnnvl != nullptr && state_->mnnvl->supports_twoshot(count);
    if (state_->vendor_allreduce(buffers, streams, count, capture_state, two_shot)) {
        if (tp_verify_graph::binding(streams[0])) tp_verify_graph::record_regular(streams[0]);
        else state_->record_vendor_success(count, payload, two_shot);
        return;
    }
    if (tp_verify_graph::binding(streams[0])) {
        throw CudaError(cudaErrorNotSupported, "Capture TP verification allreduce");
    }
    const ArFallback reason = state_->vendor_fallback_reason(count, capture_state, two_shot);
#else
    const ArFallback reason = ArFallback::Disabled;
#endif
    state_->enqueue_nccl_allreduce(buffers, streams, count);
    state_->record_nccl_success(count, payload, reason);
}

bool TensorParallel::try_allreduce_rms_quantize(
    [[maybe_unused]] __nv_bfloat16* const buffers[], [[maybe_unused]] const cudaStream_t streams[],
    [[maybe_unused]] const MnnvlRmsQuantizeRows rows[],
    [[maybe_unused]] const __nv_bfloat16* const gains[]) {
#if SGLANG_USE_MNNVL_AR
    ensure_healthy();
    if (!state_->mnnvl || state_->mnnvl_sticky || !state_->mnnvl->healthy() ||
        !streams[0] || !streams[1]) return false;
    const auto width = rows[0].count;
    if (!MnnvlRmsQuantizeRows::supports(width) || rows[1].count != width) return false;
    const std::size_t count = width * mnnvl_ar_hidden_width;
    const auto capture = validate_collective_streams(state_->devices, streams, 2);
    const bool owned_capture = capture.status == cudaStreamCaptureStatusActive &&
        capture.same_capture && width == tp_verify_graph::rows &&
        tp_verify_graph::binding(streams[0]) && tp_verify_graph::binding(streams[1]);
    if ((capture.status != cudaStreamCaptureStatusNone && !owned_capture) ||
        !state_->vendor_allreduce(buffers, streams, count, capture, false, rows, gains)) {
        return false;
    }
    if (owned_capture) tp_verify_graph::record_fused(streams[0]);
    else state_->record_vendor_success(count, count * sizeof(__nv_bfloat16), false);
    return true;
#else
    return false;
#endif
}

void TensorParallel::snapshot_transport(TpArCounters& out) const {
    out = TpArCounters{};
    out.ranks = static_cast<int>(state_->devices.size());
    out.mnnvl_built = SGLANG_USE_MNNVL_AR != 0;
    const std::lock_guard<std::mutex> lock(state_->counter_mutex);
    out.mnnvl_active = state_->mnnvl_active_mirror;
    out.per_rank = state_->ar_counters;
    out.widths.clear();
    out.widths.reserve(state_->ar_widths.size());
    for (const auto& [width, entry] : state_->ar_widths) {
        TpArCounters::TpArWidthCounters line;
        line.width = width;
        line.mnnvl_calls_total = entry.mnnvl_calls;
        line.nccl_calls_total = entry.nccl_calls;
        line.bytes_payload_total = entry.payload_bytes;
        out.widths.push_back(std::move(line));
    }
}

void TensorParallel::broadcast_int32(std::int32_t* const buffers[], const cudaStream_t streams[],
                                     std::size_t count, int root) {
    const std::size_t ranks = state_->devices.size();
    if (count == 0) {
        throw std::invalid_argument(
            "TensorParallel::broadcast_int32 requires a positive element count.");
    }
    if (root < 0 || static_cast<std::size_t>(root) >= ranks) {
        throw std::invalid_argument(
            "TensorParallel::broadcast_int32 requires a rank of the group as root.");
    }
    for (std::size_t rank = 0; rank < ranks; ++rank) {
        if (buffers[rank] == nullptr || streams[rank] == nullptr) {
            throw std::invalid_argument(
                "TensorParallel::broadcast_int32 requires a buffer and a stream for every rank.");
        }
    }
    ensure_healthy();
    validate_collective_streams(state_->devices, streams, ranks);

    check_nccl(ncclGroupStart(), "Open the NCCL operation group");
    try {
        CallerDevice device_scope;
        for (std::size_t rank = 0; rank < ranks; ++rank) {
            device_scope.select(state_->devices[rank], "Select a tensor-parallel device");
            const ncclResult_t status =
                ncclBroadcast(buffers[rank], buffers[rank], count, ncclInt32, root,
                              state_->comms[rank], streams[rank]);
            if (status != ncclSuccess) {
                throw std::runtime_error(std::string("Enqueue the NCCL int32 broadcast: ") +
                                         ncclGetErrorString(status));
            }
        }
    } catch (...) {
        (void)ncclGroupEnd();
        state_->failed = true;
        throw;
    }
    const ncclResult_t grouped = ncclGroupEnd();
    if (grouped != ncclSuccess) {
        state_->failed = true;
        throw std::runtime_error(std::string("Close the NCCL operation group: ") +
                                 ncclGetErrorString(grouped));
    }
}

bool TensorParallel::graph_eligible() const noexcept { return nccl_supports_cuda_graphs; }

bool TensorParallel::fused_graph_eligible() const noexcept {
#if SGLANG_USE_MNNVL_AR
    return !state_->failed && !state_->mnnvl_sticky && state_->mnnvl && state_->mnnvl->healthy();
#else
    return false;
#endif
}

void TensorParallel::record_fused_graph_replay() {
#if SGLANG_USE_MNNVL_AR
    constexpr std::size_t count = tp_verify_graph::rows * mnnvl_ar_hidden_width;
    state_->record_vendor_success(count, count * sizeof(__nv_bfloat16), false, 128);
#endif
}

void TensorParallel::fail_fused_graph_replay() noexcept { state_->failed = true; }

}   
