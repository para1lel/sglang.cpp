#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>

#include "sglang/text/gdn_local.h"

namespace sglang::decoder {

std::string gdn_vendor_prefill_identity_tag(const int* devices, int count);

}

#if defined(SGLANG_GDN_VENDOR) && SGLANG_GDN_VENDOR

namespace sglang::decoder {

bool gdn_vendor_prefill_enabled();

bool gdn_vendor_prefill_eligible_rows(int rows);

std::size_t gdn_vendor_prefill_workspace_bytes(int rows);

cudaError_t gdn_vendor_prefill_local(int rows, const __nv_bfloat16* qkv,
                                     const __nv_bfloat16* z, const __nv_bfloat16* b,
                                     const __nv_bfloat16* a, TextGdnWeights weights,
                                     TextGdnState state, __nv_bfloat16* workspace,
                                     std::size_t workspace_bytes, float2* gates,
                                     __nv_bfloat16* output, cudaStream_t stream);

cudaError_t gdn_vendor_prefill_local_fp8(int rows, const __nv_bfloat16* qkv,
                                         const __nv_bfloat16* z, const __nv_bfloat16* b,
                                         const __nv_bfloat16* a, TextGdnWeights weights,
                                         TextGdnState state, __nv_bfloat16* workspace,
                                         std::size_t workspace_bytes, float2* gates,
                                         TextFp8Activation output, cudaStream_t stream);

}
#else

namespace sglang::decoder {

inline bool gdn_vendor_prefill_enabled() { return false; }
inline bool gdn_vendor_prefill_eligible_rows(int) { return false; }

inline std::size_t gdn_vendor_prefill_workspace_bytes(int rows) {
    return sglang::text_gdn_local_workspace_bytes(rows);
}

inline cudaError_t gdn_vendor_prefill_local(int, const __nv_bfloat16*, const __nv_bfloat16*,
                                            const __nv_bfloat16*, const __nv_bfloat16*,
                                            TextGdnWeights, TextGdnState, __nv_bfloat16*,
                                            std::size_t, float2*, __nv_bfloat16*,
                                            cudaStream_t) {
    return cudaErrorNotSupported;
}

inline cudaError_t gdn_vendor_prefill_local_fp8(int, const __nv_bfloat16*, const __nv_bfloat16*,
                                                const __nv_bfloat16*, const __nv_bfloat16*,
                                                TextGdnWeights, TextGdnState, __nv_bfloat16*,
                                                std::size_t, float2*, TextFp8Activation,
                                                cudaStream_t) {
    return cudaErrorNotSupported;
}

}

#endif
