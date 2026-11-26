#pragma once

#include "sglang/text_ops.h"

namespace sglang::decoder {

struct MnnvlRmsQuantizeRows {
    static constexpr int size = 10;

    static constexpr bool supports(int count) {
        return count == size || (count >= 3 && count <= 5);
    }

    struct Row {
        __nv_bfloat16* residual;
        __nv_bfloat16* normalized;
        TextFp8Activation activation;
    };

    int count;
    Row entries[size];
};

}
