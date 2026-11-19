#include "sglang/server/vision_completion.h"

#include "sglang/checkpoint.h"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_media.h"
#include "sglang/vision.h"

namespace sglang::server {
namespace {

using decoder::Allocations;
using decoder::check;
using decoder::DeviceScope;

DecoderResult cancelled() {
    DecoderResult result;
    result.finish_reason = DecoderFinishReason::cancelled;
    return result;
}

}

VisionCompletion::VisionCompletion(const std::filesystem::path& checkpoint, int device)
    : checkpoint_(checkpoint), device_(device) {}

VisionCompletion::~VisionCompletion() = default;

DecoderResult VisionCompletion::generate(const Decoder& decoder, const MediaPrompt& prompt,
                                        const DecoderOptions& options,
                                        const DecoderCallbacks& callbacks) {
    if (callbacks.is_cancelled && callbacks.is_cancelled()) {
        return cancelled();
    }
    if (options.max_new_tokens == 0) {
        return {};
    }
    const DeviceScope device(device_);
    if (!encoder_) {
        encoder_ = std::make_unique<VisionEncoder>(Checkpoint(checkpoint_));
    }
    const auto storage = std::make_shared<Allocations>(device_);
    std::vector<DecoderMedia> media;
    media.reserve(prompt.parts.size());
    for (const auto& part : prompt.parts) {
        if (callbacks.is_cancelled && callbacks.is_cancelled()) {
            return cancelled();
        }
        const VisionGrid grid{part.layout.temporal_pairs, part.layout.patch_height,
                              part.layout.patch_width};
        static_assert(vision_feature_width == decoder_media_feature_width);
        const auto elements = vision_feature_rows(grid) * vision_feature_width;
        auto* features = storage->allocate<__nv_bfloat16>(elements);
        Allocations work(device_);
        work.open_stream();
        auto* pixels = work.allocate<float>(part.patches.size());
        check(cudaMemcpyAsync(pixels, part.patches.data(), part.patches.size() * sizeof(float),
                              cudaMemcpyHostToDevice, work.stream()), "Upload visual patches");
        encoder_->encode(grid, pixels, part.patches.size(), features, elements, work.stream());
        media.push_back({part.layout, {storage, features}, elements});
    }
    return decoder.generate_multimodal(prompt.template_ids, media, options, callbacks);
}

}
