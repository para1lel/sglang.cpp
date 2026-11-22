#pragma once

#include "sglang/decoder.h"

#include <stdexcept>
#include <string_view>

namespace sglang::cli {

inline Fp8PrefillPolicy parse_fp8_prefill(std::string_view value) {
    if (value == "exact") {
        return Fp8PrefillPolicy::exact;
    }
    if (value == "tensor-core") {
        return Fp8PrefillPolicy::tensor_core;
    }
    if (value == "tensor-core-mtp") {
        return Fp8PrefillPolicy::tensor_core_mtp;
    }
    if (value == "tensor-core-target") {
        return Fp8PrefillPolicy::tensor_core_target;
    }
    if (value == "tensor-core-mtp-target") {
        return Fp8PrefillPolicy::tensor_core_mtp_target;
    }
    throw std::invalid_argument("Expected --fp8-prefill exact, tensor-core, tensor-core-mtp, "
                                "tensor-core-target, or tensor-core-mtp-target");
}

inline constexpr std::string_view fp8_prefill_name(Fp8PrefillPolicy policy) {
    switch (policy) {
    case Fp8PrefillPolicy::exact: return "exact";
    case Fp8PrefillPolicy::tensor_core: return "tensor-core";
    case Fp8PrefillPolicy::tensor_core_mtp: return "tensor-core-mtp";
    case Fp8PrefillPolicy::tensor_core_target: return "tensor-core-target";
    case Fp8PrefillPolicy::tensor_core_mtp_target: return "tensor-core-mtp-target";
    }
    return "unknown";
}

inline constexpr char fp8_prefill_usage[] =
    "FP8 prefill defaults to exact. Tensor Core prefill uses a different FP32 summation order.\n"
    "Model scores and output token IDs can differ, even when the top scores are far apart.\n"
    "tensor-core-mtp enables Tensor Core MTP prefill, tensor-core-target enables Tensor Core "
    "target decode, and tensor-core-mtp-target enables both. These modes require cache off.\n";

}
