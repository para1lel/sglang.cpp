#pragma once

#include "sglang/multimodal_prompt.h"

#include <cstddef>
#include <memory>

#include <cuda_bf16.h>

namespace sglang {

inline constexpr int decoder_media_feature_width = 5120;

struct DecoderMedia {
    DecoderMediaLayout layout;
    /// Completed, immutable storage on the decoder's CUDA device, held by shared ownership.
    std::shared_ptr<const __nv_bfloat16> features;
    /// Element count of the contiguous [merged rows, 5120] view beginning at features.get().
    std::size_t feature_elements;
};

}
