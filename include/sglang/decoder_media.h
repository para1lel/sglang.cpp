#pragma once

#include "sglang/multimodal_prompt.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

struct __nv_bfloat16;

namespace sglang {

namespace server {
class VisionCompletion;
}

struct DecoderMedia;

class DecoderMediaCacheIdentity {
public:
    bool matches(const DecoderMedia& media) const;
    std::string_view patches_digest() const;

private:
    friend class server::VisionCompletion;
    DecoderMediaCacheIdentity(const DecoderMedia& media, std::string patches_digest);

    DecoderMediaLayout layout_;
    std::weak_ptr<const __nv_bfloat16> features_;
    const __nv_bfloat16* address_;
    std::size_t elements_;
    std::string patches_digest_;
};

inline constexpr int decoder_media_feature_width = 5120;

struct DecoderMedia {
    DecoderMediaLayout layout;
    std::shared_ptr<const __nv_bfloat16> features;
    std::size_t feature_elements;
    std::shared_ptr<const DecoderMediaCacheIdentity> cache_identity = {};
};

}
