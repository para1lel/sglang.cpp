#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sglang::decoder {

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
    std::uint64_t ar_bytes_payload_total = 0;
    std::uint64_t ar_bytes_aggregated_total = 0;
};

struct TpArCounters {
    int ranks = 0;
    bool mnnvl_built = false;
    bool mnnvl_active = false;
    std::vector<TpArRankCounters> per_rank;

    struct TpArWidthCounters {
        std::uint64_t width = 0;
        std::vector<std::uint64_t> mnnvl_calls_total;
        std::vector<std::uint64_t> nccl_calls_total;
        std::vector<std::uint64_t> bytes_payload_total;
    };
    std::vector<TpArWidthCounters> widths;
};

class TensorParallel {
public:
    static std::unique_ptr<TensorParallel> create(std::vector<int> devices);
    ~TensorParallel();

    TensorParallel(const TensorParallel&) = delete;
    TensorParallel& operator=(const TensorParallel&) = delete;

    int rank_count() const noexcept;
    int device(int rank) const;

    void allreduce_bf16(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                        std::size_t count);

    void broadcast_int32(std::int32_t* const buffers[], const cudaStream_t streams[],
                         std::size_t count, int root);

    void snapshot_transport(TpArCounters& out) const;

    bool graph_eligible() const noexcept;

private:
    void ensure_healthy();

    explicit TensorParallel(std::vector<int> devices);

    struct State;
    std::unique_ptr<State> state_;
};

}
