#pragma once

typedef struct cudaDeviceProp cudaDeviceProp;

namespace sglang::decoder {

struct CudaBackendCapabilities {
    bool tensor_core_prefill;
    int target_split_end;
};

struct CudaBackendPolicy {
    static CudaBackendPolicy for_device(const cudaDeviceProp& properties);
    static CudaBackendPolicy from_capabilities(const CudaBackendCapabilities& capabilities);

    bool uses_b200_operators;
    bool ordered_attention;
    bool tensor_core_prefill;
    bool int8_draft_head;
    int target_split_end;
    int prefill_tile_rows;
};

}
