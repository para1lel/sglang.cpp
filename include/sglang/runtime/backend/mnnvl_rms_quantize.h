#pragma once

#include "sglang/text_ops.h"

namespace sglang::decoder {

struct MnnvlRmsQuantizeRows {
    static constexpr int size = 10;

    struct Row {
        __nv_bfloat16* residual;
        __nv_bfloat16* normalized;
        TextFp8Activation activation;
    };

    Row entries[size];
};

}
