#pragma once

#include "sglang/decoder_media.h"
#include "sglang/server/text_completion.h"

namespace sglang {
class VisionEncoder;
}

namespace sglang::server {

class VisionCompletion {
public:
    VisionCompletion(const std::filesystem::path& checkpoint, int device);
    ~VisionCompletion();

    DecoderResult generate(BatchService& service, int slot, const MediaPrompt& prompt,
                           const DecoderOptions& options, const DecoderCallbacks& callbacks);
    std::optional<std::vector<DecoderMedia>> encode(const MediaPrompt& prompt,
                                                   const DecoderCallbacks& callbacks);

private:
    std::filesystem::path checkpoint_;
    int device_;
    std::once_flag loaded_;
    std::unique_ptr<VisionEncoder> encoder_;
};

}
