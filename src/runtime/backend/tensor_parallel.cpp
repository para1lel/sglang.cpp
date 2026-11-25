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

#if defined(SGLANG_MNNVL_AR) && SGLANG_MNNVL_AR
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

// Preserves the caller's current CUDA device across the scope. Selection
// failures throw; restoration is best effort because the scope may unwind.
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

// A collective is only well-formed when every rank's stream belongs to that
// rank's device and all ranks share one capture state. NCCL notices some of
// these violations only after enqueueing (or never), and a mismatch can
// leave one rank spinning inside a kernel its peer will never enter, so
// every path — vendor or NCCL — validates this before launching anything.
// Under an active capture the query also returns the capture session id,
// which distinguishes one joined capture session from two independent
// captures that merely happen to be in progress at the same time.
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
            // Two ranks reporting Active may still belong to two independent
            // capture sessions; only an equal capture id proves one session,
            // and an unidentifiable capture proves nothing.
            if (status == cudaStreamCaptureStatusActive &&
                (id == 0 || id != capture_id)) {
                state.same_capture = false;
            }
        }
    }
    return state;
}

// NCCL 2.9 added CUDA-graph capture support for collectives.
#if defined(NCCL_MAJOR) && defined(NCCL_MINOR)
constexpr bool nccl_supports_cuda_graphs = NCCL_MAJOR > 2 || (NCCL_MAJOR == 2 && NCCL_MINOR >= 9);
#else
constexpr bool nccl_supports_cuda_graphs = false;
#endif

// Reason a given allreduce ran through NCCL instead of the oneshot
// transport. Exactly one is recorded per NCCL-served call. `Oversize` covers
// every count outside the vendor one-shot envelope: above max_count() or not
// a whole multiple of the model hidden width. `CaptureMismatch` covers calls
// whose ranks sit in two independent capture sessions: one-shot replay of a
// single side would strand the other rank's Lamport wait. `Unaligned` covers
// the one in-envelope geometry NCCL accepts but the vendor transport does
// not: a buffer that is not 16-byte aligned. The vendor transport's
// remaining preflight rejections (wrong-device stream, mismatched capture
// state) are invalid for NCCL as well and propagate to the caller unchanged.
enum class ArFallback {
    Disabled, SetupUnavailable, Oversize, CaptureMismatch, Unaligned, AfterError
};

// The vendor one-shot envelope accepts whole rows of the model hidden width
// (mnnvl_ar.h contract); other counts must ride NCCL.
constexpr std::size_t mnnvl_ar_hidden_width = 5120;

}  // namespace

struct TensorParallel::State {
    std::vector<int> devices;
    std::vector<ncclComm_t> comms;
    // A host-side enqueue failed or a communicator reported an asynchronous
    // error: the communicators must abort, not destroy, on shutdown.
    bool failed = false;

#if SGLANG_USE_MNNVL_AR
    // Optional MNNVL oneshot transport for small allreduces. Null after a
    // setup failure; `mnnvl_sticky` records a vendor enqueue failure, after
    // which every allreduce runs through NCCL.
    std::unique_ptr<MnnvlAllReduce> mnnvl;
    bool mnnvl_sticky = false;
#endif

    // Mirror of the dispatch-thread transport decision, written only under
    // counter_mutex (together with the counters) so the metrics thread never
    // reads vendor or sticky state while the dispatch thread mutates it.
    bool mnnvl_active_mirror = false;

    // Cumulative transport counters, one entry per rank. Guarded by
    // counter_mutex: the metrics endpoint snapshots from another thread.
    std::mutex counter_mutex;
    // One struct per rank (never per-field vectors) so every counter is
    // sized together when ranks are assigned.
    std::vector<TpArRankCounters> ar_counters;
    // Completed enqueues per allreduce width (element count), split by
    // transport, one entry per rank. std::map keeps entries ascending by
    // width for snapshots.
    struct ArWidthEntry {
        std::vector<std::uint64_t> mnnvl_calls;
        std::vector<std::uint64_t> nccl_calls;
        std::vector<std::uint64_t> payload_bytes;
    };
    std::map<std::size_t, ArWidthEntry> ar_widths;
    // Widths whose first use has been reported on stderr.
    std::set<std::size_t> logged_widths;

    // Allreduce dispatch helpers, defined below: the vendor attempt (with
    // its failure semantics), the grouped NCCL enqueue, and the counter
    // recorders.
    void enqueue_nccl_allreduce(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                                std::size_t count);
    void bump_transport_width(std::size_t count, std::size_t payload, bool vendor);
    void record_nccl_success(std::size_t count, std::size_t payload, ArFallback reason);
#if SGLANG_USE_MNNVL_AR
    ArFallback vendor_fallback_reason(std::size_t count,
                                      const CollectiveStreams& capture_state) const;
    bool vendor_allreduce(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                          std::size_t count, const CollectiveStreams& capture_state);
    void note_vendor_failure();
    void record_vendor_success(std::size_t count, std::size_t payload);
#endif

    // Best effort and never throws: this runs while the owning thread may be
    // unwinding. The caller has already fenced all enqueued collectives.
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

// Group the per-rank enqueues: collectives on multiple communicators from
// one host thread must be grouped or the thread can block inside an enqueue
// waiting for the other communicator's progress.
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

// Bump the per-width histogram; counter_mutex must already be held.
void TensorParallel::State::bump_transport_width(std::size_t count, std::size_t payload,
                                                 bool vendor) {
    ArWidthEntry& width = ar_widths[count];
    if (width.mnnvl_calls.empty()) {
        width.mnnvl_calls.assign(devices.size(), 0);
        width.nccl_calls.assign(devices.size(), 0);
        width.payload_bytes.assign(devices.size(), 0);
    }
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        if (vendor) {
            ++width.mnnvl_calls[rank];
        } else {
            ++width.nccl_calls[rank];
        }
        width.payload_bytes[rank] += payload;
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
    // Refresh the mirror from the same decision inputs the dispatch just
    // used, so a transport that went unhealthy without throwing (or a
    // recovered width mix) reports the truth at the next snapshot.
    mnnvl_active_mirror = mnnvl != nullptr && !mnnvl_sticky && mnnvl->healthy();
#endif
    if (logged_widths.insert(count).second) {
        std::cerr << "tp2_transport width=" << count << " transport=nccl" << std::endl;
    }
}

#if SGLANG_USE_MNNVL_AR
ArFallback TensorParallel::State::vendor_fallback_reason(std::size_t count,
                                                         const CollectiveStreams& capture_state)
    const {
    if (mnnvl == nullptr) {
        return ArFallback::SetupUnavailable;
    }
    if (mnnvl_sticky || !mnnvl->healthy()) {
        return ArFallback::AfterError;
    }
    if (count > mnnvl->max_count() || count % mnnvl_ar_hidden_width != 0) {
        return ArFallback::Oversize;
    }
    if (capture_state.status == cudaStreamCaptureStatusActive && !capture_state.same_capture) {
        return ArFallback::CaptureMismatch;
    }
    return ArFallback::Unaligned;
}

bool TensorParallel::State::vendor_allreduce(__nv_bfloat16* const buffers[],
                                             const cudaStream_t streams[], std::size_t count,
                                             const CollectiveStreams& capture_state) {
    if (mnnvl == nullptr || mnnvl_sticky || !mnnvl->healthy() ||
        count > mnnvl->max_count() || count % mnnvl_ar_hidden_width != 0) {
        return false;
    }
    if (capture_state.status == cudaStreamCaptureStatusActive && !capture_state.same_capture) {
        return false;
    }
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        // The oneshot kernel performs 16-byte vectorized accesses. A buffer
        // starting at an odd BF16 element is legal for NCCL but not for the
        // vendor transport; such a call rides NCCL with the group intact.
        if (reinterpret_cast<std::uintptr_t>(buffers[rank]) % 16 != 0) {
            return false;
        }
    }
    try {
        mnnvl->allreduce_bf16(buffers, streams, count);
    } catch (const std::invalid_argument&) {
        // Wrong-device stream or mismatched capture state: invalid for
        // NCCL too — one rank may already be spinning while its peer is
        // captured. Surface the rejection unchanged: no counter, no
        // poison, no fallback.
        throw;
    } catch (const MnnvlArError& error) {
        if (error.peer_enqueued) {
            // A peer already entered this boundary; the other rank can
            // never match it on any transport. Later collectives would
            // hang, so the group must abort.
            failed = true;
        } else {
            // Nothing was enqueued on any rank: fall back to NCCL for
            // later calls.
            mnnvl_sticky = true;
        }
        note_vendor_failure();
        throw;
    } catch (...) {
        // Unrecognized vendor failure: assume the worst, one rank may
        // already be inside the boundary.
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

void TensorParallel::State::record_vendor_success(std::size_t count, std::size_t payload) {
    const std::lock_guard<std::mutex> lock(counter_mutex);
    for (std::size_t rank = 0; rank < devices.size(); ++rank) {
        TpArRankCounters& counter = ar_counters[rank];
        ++counter.ar_calls_total;
        ++counter.ar_mnnvl_oneshot_total;
        counter.ar_bytes_payload_total += payload;
        counter.ar_bytes_aggregated_total += payload * devices.size();
    }
    bump_transport_width(count, payload, true);
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

    // Validate every ordinal before entering the NCCL rendezvous: a rank
    // rejected inside the group would leave the other ranks waiting for it in
    // ncclGroupEnd.
    {
        CallerDevice validation_scope;
        for (const int device : state_->devices) {
            validation_scope.select(device, "Select a tensor-parallel device");
        }
    }

    ncclUniqueId id{};
    check_nccl(ncclGetUniqueId(&id), "Create an NCCL communicator id");
    state_->comms.assign(ranks, nullptr);

    // All ranks live in this process, so every communicator must be created
    // inside one NCCL group; a grouped init from a single host thread is the
    // documented rendezvous for multiple per-device communicators.
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

    // One struct per rank sizes every counter together.
    state_->ar_counters.assign(ranks, TpArRankCounters{});

#if SGLANG_USE_MNNVL_AR
    // Best effort: the transport reports its own setup outcome (nullptr on
    // any failure) and every allreduce then runs through NCCL.
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
    if (state_->vendor_allreduce(buffers, streams, count, capture_state)) {
        state_->record_vendor_success(count, payload);
        return;
    }
    const ArFallback reason = state_->vendor_fallback_reason(count, capture_state);
#else
    const ArFallback reason = ArFallback::Disabled;
#endif
    state_->enqueue_nccl_allreduce(buffers, streams, count);
    state_->record_nccl_success(count, payload, reason);
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

    // Same grouping rule as allreduce_bf16: multiple communicators, one thread.
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

}  // namespace sglang::decoder
