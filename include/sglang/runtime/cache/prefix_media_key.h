#pragma once

#include "sglang/multimodal_prompt.h"

#include <memory>
#include <optional>
#include <vector>

namespace sglang {

class DecoderMediaCacheIdentity;

namespace decoder {

struct MultimodalInput;

class PrefixMediaKey {
public:
    static std::optional<PrefixMediaKey> capture(const MultimodalInput* media, int tokens);
    bool matches(const MultimodalInput* media, int tokens) const;

private:
    struct Span {
        int token_begin;
        int rows;
        int feature_begin;
        std::shared_ptr<const DecoderMediaCacheIdentity> identity;
    };

    std::vector<MropePosition> positions_;
    std::vector<Span> spans_;
};

}
}
