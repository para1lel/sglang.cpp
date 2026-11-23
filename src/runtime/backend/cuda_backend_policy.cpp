#include "sglang/runtime/backend/cuda_backend_policy.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_prefill.h"

#include <stdexcept>
#include <string_view>

namespace sglang::decoder {

CudaBackendPolicy CudaBackendPolicy::for_device(const cudaDeviceProp& properties) {
    if (properties.major != 10 || properties.minor != 0) {
        throw std::invalid_argument("The decoder requires an sm_100 CUDA device");
    }
    const bool b200 = std::string_view(properties.name) == "NVIDIA B200";
    CudaBackendPolicy backend{};
    backend.uses_b200_operators = b200;
    backend.ordered_attention = b200;
    backend.tensor_core_decode = b200;
    backend.vector_decode_attention = b200;
    backend.grouped_target_attention = b200;
    backend.paired_prefill = b200;
    backend.tensor_core_prefill = b200;
    backend.target_split_end = b200 ? text_attention_max_tokens : 0;
    backend.prefill_tile_rows = b200 ? batch_prefill_tile_rows : text_embedding_max_rows;
    backend.draft.int8_head = b200;
    backend.draft.online_attention = b200;
    backend.draft.grouped_argmax = b200;
    backend.draft.paired_seed_kv = b200;
    backend.draft.wmma_seed_fc = b200;
    return backend;
}

}
