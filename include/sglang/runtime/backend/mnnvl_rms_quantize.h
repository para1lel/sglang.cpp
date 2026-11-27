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

template <int Width> struct MnnvlWideRmsQuantizeRows {
    static_assert(Width == 40 || Width == 80);
    static constexpr int size = Width;

    MnnvlRmsQuantizeRows::Row entries[size];
};
}
