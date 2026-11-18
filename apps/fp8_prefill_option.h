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
    throw std::invalid_argument("Expected --fp8-prefill exact or tensor-core");
}

inline constexpr char fp8_prefill_usage[] =
    "FP8 prefill defaults to exact. Tensor Core prefill uses a different FP32 summation order.\n"
    "Model scores and output token IDs can differ, even when the top scores are far apart.\n"
    "MTP prompt seeding uses row-wise execution with either policy.\n";

}
