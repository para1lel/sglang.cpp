#include "sglang/runtime/execution/decoder_execution_policy.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/runtime/backend/cuda_backend_policy.h"

namespace sglang::decoder {

DecoderExecutionPolicy DecoderExecutionPolicy::for_backend(const CudaBackendPolicy& backend,
                                                           const Limits& limits) {
    return {
        backend.ordered_attention,
        !backend.ordered_attention,
        backend.tensor_core_prefill,
        limits.tensor_prefill_rows,
        limits.fallback_prefill_rows,
        backend.target_split_end,
    };
}

bool DecoderExecutionPolicy::regular_decode_graph(int count) const {
    return count > 0 && count <= decoder_batch_max_lanes;
}

int DecoderExecutionPolicy::prefill_limit() const {
    return tensor_core_prefill ? tensor_prefill_rows : fallback_prefill_rows;
}

bool DecoderExecutionPolicy::split_decode(int prompt_count, int limit) const {
    return target_split_end != 0 && limit > target_split_first_position &&
           prompt_count < target_split_end;
}

}
