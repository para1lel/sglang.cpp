#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sglang::decoder {

/// Cumulative allreduce transport counters, one entry per rank. The values are
/// maintained by the single host thread that enqueues collectives and are
/// meaningful for eager (ungraphed) traffic: under graph capture they count
/// the captured enqueues, not graph replays.
/// Per-rank allreduce transport counters. One struct per rank (never a
/// per-field vector) so no counter can be left unsized when ranks are
/// initialized.
struct TpArRankCounters {
    std::uint64_t ar_calls_total = 0;
    std::uint64_t ar_mnnvl_oneshot_total = 0;
    std::uint64_t ar_nccl_total = 0;
    std::uint64_t ar_fallback_disabled_total = 0;
    std::uint64_t ar_fallback_setup_unavailable_total = 0;
    std::uint64_t ar_fallback_oversize_total = 0;
    std::uint64_t ar_fallback_capture_mismatch_total = 0;
    std::uint64_t ar_fallback_unaligned_total = 0;
    std::uint64_t ar_fallback_after_error_total = 0;
    std::uint64_t ar_vendor_failed_total = 0;
    /// Logical payload bytes this rank supplied (count * 2).
    std::uint64_t ar_bytes_payload_total = 0;
    /// Logical bytes aggregated (payload * ranks).
    std::uint64_t ar_bytes_aggregated_total = 0;
};

struct TpArCounters {
    int ranks = 0;
    /// Built with the MNNVL oneshot transport compiled in.
    bool mnnvl_built = false;
    /// The transport initialized and has not failed since.
    bool mnnvl_active = false;
    /// One entry per rank, always sized to `ranks`.
    std::vector<TpArRankCounters> per_rank;

    /// Completed enqueues for one allreduce width (element count), split by
    /// transport; each vector holds one entry per rank.
    struct TpArWidthCounters {
        std::uint64_t width = 0;
        std::vector<std::uint64_t> mnnvl_calls_total;
        std::vector<std::uint64_t> nccl_calls_total;
        std::vector<std::uint64_t> bytes_payload_total;
    };
    /// One entry per distinct width served so far, ascending by width.
    std::vector<TpArWidthCounters> widths;
};

/// In-process tensor-parallel group: every rank is a CUDA device of this process.
///
/// Rank r is bound to devices[r] for the group's lifetime. All collectives are
/// enqueued from one host thread; each call carries every rank's buffers and
/// streams and enqueues all rank halves in rank order, so a single thread drives
/// the whole group without deadlocking. Calls preserve the caller's current CUDA
/// device. Serialize calls: one host thread, one outstanding collective at a
/// time. The caller fences all enqueued collectives (for example by
/// synchronizing the streams passed to them) before destroying the group.
class TensorParallel {
public:
    /// `devices` must list distinct CUDA device ordinals. Throws
    /// std::invalid_argument, CudaError, or std::runtime_error (NCCL) on
    /// failure. On failure no communicators survive and the caller's current
    /// CUDA device is preserved.
    static std::unique_ptr<TensorParallel> create(std::vector<int> devices);
    ~TensorParallel();

    TensorParallel(const TensorParallel&) = delete;
    TensorParallel& operator=(const TensorParallel&) = delete;

    int rank_count() const noexcept;
    int device(int rank) const;

    /// Enqueue a checked asynchronous BF16 elementwise-sum allreduce across the
    /// group.
    ///
    /// buffers[r] must be a device(r) allocation holding `count` elements and
    /// streams[r] a device(r) stream. Every rank half is enqueued in rank
    /// order; once the streams complete, buffers[r] holds the sum of all ranks'
    /// inputs, bitwise identical on every rank. All ranks reduce the same
    /// `count`. Host-side enqueue failures throw and mark the group unusable;
    /// asynchronous NCCL failures surface from the next call or from
    /// destruction.
    void allreduce_bf16(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                        std::size_t count);

    /// Enqueue a checked asynchronous broadcast of `count` 32-bit integers
    /// from rank `root` to every rank of the group.
    ///
    /// buffers[r] must be a device(r) allocation holding `count` integers and
    /// streams[r] a device(r) stream. Every rank half is enqueued in rank
    /// order; once the streams complete, buffers[r] holds the root's input,
    /// bitwise identical on every rank. All ranks broadcast the same `count`
    /// and `root` within one step. Host-side enqueue failures throw and mark
    /// the group unusable; asynchronous NCCL failures surface from the next
    /// call or from destruction.
    void broadcast_int32(std::int32_t* const buffers[], const cudaStream_t streams[],
                         std::size_t count, int root);

    /// Copy the cumulative allreduce transport counters (see TpArCounters).
    /// Thread-safe against the enqueue thread.
    void snapshot_transport(TpArCounters& out) const;

    /// Whether this group's collectives can be captured into CUDA graphs. When
    /// true, a capture must enqueue each collective on every rank stream in the
    /// same rank order as eager execution, and the captured graphs must be
    /// launched on all ranks together. Fixed for the group's lifetime.
    bool graph_eligible() const noexcept;

private:
    // Throws and marks the group failed if a collective failed earlier or a
    // communicator reports an asynchronous error.
    void ensure_healthy();

    explicit TensorParallel(std::vector<int> devices);

    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace sglang::decoder
