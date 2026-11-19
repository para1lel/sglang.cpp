#pragma once

#include "sglang/server/text_completion.h"

namespace sglang {
class VisionEncoder;
}

namespace sglang::server {

class VisionCompletion {
public:
    VisionCompletion(const std::filesystem::path& checkpoint, int device);
    ~VisionCompletion();

    DecoderResult generate(const Decoder& decoder, const MediaPrompt& prompt,
                           const DecoderOptions& options, const DecoderCallbacks& callbacks);

private:
    std::filesystem::path checkpoint_;
    int device_;
    std::unique_ptr<VisionEncoder> encoder_;
};

}
