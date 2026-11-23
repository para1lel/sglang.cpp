#pragma once

typedef struct cudaDeviceProp cudaDeviceProp;

namespace sglang::decoder {

inline constexpr int wide_mtp_lanes = 8;

struct DraftOperatorCapabilities {
    bool int8_head;
    bool online_attention;
    bool grouped_argmax;
    bool paired_seed_kv;
    bool wmma_seed_fc;
};

struct CudaBackendPolicy {
    static CudaBackendPolicy for_device(const cudaDeviceProp& properties);

    bool uses_b200_operators;
    bool ordered_attention;
    bool tensor_core_decode;
    bool vector_decode_attention;
    bool grouped_target_attention;
    bool paired_prefill;
    bool tensor_core_prefill;
    int mtp_verification_lanes;
    int target_split_end;
    int prefill_tile_rows;
    DraftOperatorCapabilities draft;
};

}
