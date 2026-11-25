#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sglang::decoder {

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
