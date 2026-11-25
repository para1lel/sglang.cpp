#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

namespace sglang::decoder {

/// Thrown by allreduce_bf16 when a vendor enqueue fails (a std::runtime_error
/// subclass). `peer_enqueued` is true iff an earlier rank's enqueue for the
/// same call already succeeded: the caller must treat that case as a hard
/// group abort (a spinning vendor kernel on one stream would hang any later
/// collective enqueued there), while a rank-0-only failure permits a sticky
/// fallback to NCCL for later calls.
struct MnnvlArError : std::runtime_error {
    MnnvlArError(const std::string& what, bool peer_enqueued_arg);

    bool peer_enqueued;
};

/// In-process two-rank BF16 allreduce over an NVLink multicast object,
/// embedding the FlashInfer TRT-LLM MNNVL one-shot kernel (Apache-2.0,
/// flashinfer 0.7.4, trtllm_mnnvl_allreduce.cuh, compiled from source — no
/// Python or JIT at build or run time).
///
/// Rank r is bound to devices[r] (any distinct CUDA device ordinals, any
/// order) for the group's lifetime. Like TensorParallel, all collectives are
/// enqueued from one host thread; each call carries every rank's buffers and
/// streams and enqueues all rank halves in rank order. Calls preserve the
/// caller's current CUDA device. Serialize calls: one host thread, one
/// outstanding collective at a time.
///
/// The transport covers only the one-shot envelope (see max_count()); larger
/// counts are the dispatch layer's business and must use NCCL. Any setup
/// failure releases every partially created resource and yields a null group.
class MnnvlAllReduce {
public:
    /// `devices` must list exactly two distinct CUDA device ordinals on one
    /// NVLink fabric. Returns nullptr when the platform, driver, or fabric
    /// cannot provide the multicast workspace (every partial allocation is
    /// released first, and the caller's current CUDA device is preserved).
    static std::unique_ptr<MnnvlAllReduce> create(std::vector<int> devices);
    ~MnnvlAllReduce();

    MnnvlAllReduce(const MnnvlAllReduce&) = delete;
    MnnvlAllReduce& operator=(const MnnvlAllReduce&) = delete;

    int rank_count() const noexcept;
    int device(int rank) const;

    /// Enqueue a checked asynchronous in-place BF16 elementwise-sum allreduce
    /// across the group using the vendor one-shot kernel.
    ///
    /// buffers[r] must be a device(devices[r]) allocation holding `count`
    /// elements and streams[r] a device(devices[r]) stream; the sum is written
    /// back into buffers[r] itself (input == output). Every rank half is
    /// enqueued in rank order; once the streams complete, buffers[r] holds the
    /// sum of all ranks' inputs, bitwise identical on every rank. All ranks
    /// reduce the same `count`. `count` must be a positive multiple of the
    /// model hidden width (5120) no greater than max_count(); violating either
    /// throws std::invalid_argument. Host-side enqueue failures throw MnnvlArError
    /// (peer_enqueued marks whether an earlier rank's half already launched) and
    /// mark the group unusable (poisoned); later calls
    /// rethrow without enqueueing anything. Collectives are CUDA-graph
    /// capturable: capture must enqueue each collective on every rank stream
    /// in the same rank order as eager execution, and captured graphs must be
    /// launched on all ranks together.
    ///
    /// Preflight (before any enqueue): each stream must belong to its rank's
    /// device and carry no pending error, and each buffer must be 16-byte
    /// aligned (the vendor kernel performs vectorized accesses). A violation
    /// throws std::invalid_argument (or MnnvlArError with peer_enqueued=false
    /// for a pre-existing stream error) without enqueueing anything — the
    /// vendor kernel spins on Lamport flags, so a partially enqueued call
    /// would otherwise hang the earlier rank.
    void allreduce_bf16(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                        std::size_t count);

    /// Largest element count this group's one-shot transport accepts. Derived
    /// from the upstream MNNVL one-shot threshold (1,048,576 aggregated
    /// payload bytes = count * 2 bytes * ranks); the workspace is sized to
    /// exactly that bound, so any count up to max_count() is dispatchable
    /// without reallocation. The dispatch layer must route larger counts to
    /// NCCL.
    std::size_t max_count() const noexcept;

    /// Whether the group still accepts collectives. False after an enqueue
    /// failure poisoned it.
    bool healthy() const noexcept;

private:
    struct State;

    explicit MnnvlAllReduce(std::unique_ptr<State> state);
    void ensure_healthy() const;

    // Setup and preflight stages (definitions in mnnvl_ar.cu); static
    // members so each stage can touch State directly. probe_devices rejects
    // non-SM100a devices; size_workspace rounds the allocation to every
    // granularity and returns the multicast granularity needed when mapping;
    // create_physical allocates the per-rank buffers and the multicast
    // object; map_workspace maps the unicast and multicast VA ranges;
    // init_lamport sizes the Lamport buffers and primes the per-rank flag
    // state; preflight runs the per-call stream and buffer checks before any
    // enqueue.
    static void probe_devices(State& state);
    static std::uint64_t size_workspace(State& state, std::uint64_t requested);
    static void create_physical(State& state);
    static void map_workspace(State& state, std::uint64_t mc_granularity);
    static void init_lamport(State& state);
    static void preflight(State& state, __nv_bfloat16* const buffers[],
                          const cudaStream_t streams[]);

    std::unique_ptr<State> state_;
};

}  // namespace sglang::decoder
