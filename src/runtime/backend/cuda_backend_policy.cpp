#include "sglang/runtime/backend/cuda_backend_policy.h"

#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_prefill.h"

#include <string_view>

namespace sglang::decoder {

CudaBackendPolicy CudaBackendPolicy::for_device(const cudaDeviceProp& properties) {
    const bool b200 = properties.major == 10 && properties.minor == 0 &&
        std::string_view(properties.name) == "NVIDIA B200";
    return from_capabilities({b200, b200 ? text_attention_max_tokens : 0});
}

CudaBackendPolicy CudaBackendPolicy::from_capabilities(
    const CudaBackendCapabilities& capabilities) {
    const bool b200 = capabilities.tensor_core_prefill &&
        capabilities.target_split_end == text_attention_max_tokens;
    return {
        b200,
        b200,
        b200,
        b200,
        capabilities.target_split_end,
        b200 ? batch_prefill_tile_rows : text_embedding_max_rows,
    };
}

}
