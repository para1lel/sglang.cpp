#pragma once

#include "sglang/runtime/backend/tp_ar_counters.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sglang::decoder {

struct MnnvlRmsQuantizeRows;
struct MnnvlRmsQuantizeRows40;

enum class TpAllReduceRoute { nccl, mnnvl_one_shot, mnnvl_two_shot };

class TensorParallel {
public:
    static std::unique_ptr<TensorParallel> create(std::vector<int> devices);
    ~TensorParallel();

    TensorParallel(const TensorParallel&) = delete;
    TensorParallel& operator=(const TensorParallel&) = delete;

    int rank_count() const noexcept;
    int device(int rank) const;

    TpAllReduceRoute allreduce_bf16(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                                   std::size_t count);
    bool supports_mnnvl_oneshot(std::size_t count) const noexcept;
    bool supports_mnnvl_twoshot(std::size_t count) const noexcept;
    bool try_allreduce_rms_quantize(
        __nv_bfloat16* const buffers[], const cudaStream_t streams[],
        const MnnvlRmsQuantizeRows rows[], const __nv_bfloat16* const gains[]);
    bool try_allreduce_rms_quantize_m40(
        __nv_bfloat16* const buffers[], const cudaStream_t streams[],
        const MnnvlRmsQuantizeRows40 rows[], const __nv_bfloat16* const gains[]);

    void broadcast_int32(std::int32_t* const buffers[], const cudaStream_t streams[],
                         std::size_t count, int root);

    void snapshot_transport(TpArCounters& out) const;

    bool graph_eligible() const noexcept;
    bool fused_graph_eligible() const noexcept;
    void record_fused_graph_replay();
    void fail_fused_graph_replay() noexcept;

private:
    void ensure_healthy();

    explicit TensorParallel(std::vector<int> devices);

    struct State;
    std::unique_ptr<State> state_;
};

}
