#pragma once

#include "sglang/multimodal_prompt.h"

#include <cstddef>
#include <memory>
#include <string>

#include <cuda_bf16.h>

namespace sglang {

namespace server {
class VisionCompletion;
}

struct DecoderMedia;

class DecoderMediaCacheIdentity {
public:
    bool matches(const DecoderMedia& media) const;
    bool same_content(const DecoderMediaCacheIdentity& other) const;

private:
    friend class server::VisionCompletion;
    DecoderMediaCacheIdentity(const DecoderMedia& media, std::shared_ptr<const void> encoder,
                              std::string patches_digest);

    DecoderMediaLayout layout_;
    std::weak_ptr<const __nv_bfloat16> features_;
    const __nv_bfloat16* address_;
    std::size_t elements_;
    std::shared_ptr<const void> encoder_;
    std::string patches_digest_;
};

inline constexpr int decoder_media_feature_width = 5120;

struct DecoderMedia {
    DecoderMediaLayout layout;
    /// Completed, immutable storage on the decoder's CUDA device, held by shared ownership.
    std::shared_ptr<const __nv_bfloat16> features;
    /// Element count of the contiguous [merged rows, 5120] view beginning at features.get().
    std::size_t feature_elements;
    std::shared_ptr<const DecoderMediaCacheIdentity> cache_identity = {};
};

}
