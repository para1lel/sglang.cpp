#pragma once

#include "sglang/text_ops.h"

namespace sglang::decoder {

struct MnnvlRmsQuantizeRows {
    static constexpr int size = 10;

    static bool supports(int count);

    struct Row {
        __nv_bfloat16* residual;
        __nv_bfloat16* normalized;
        TextFp8Activation activation;
    };

    int count;
    Row entries[size];
};

struct MnnvlRmsQuantizeRows40 {
    static constexpr int size = 40;

    MnnvlRmsQuantizeRows::Row entries[size];
};
}
