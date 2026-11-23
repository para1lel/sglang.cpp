#include "sglang/server/vision_completion.h"

#include "sglang/checkpoint.h"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_media.h"
#include "sglang/server/batch_service.h"
#include "sglang/tokenizer/digest.h"
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

std::optional<std::vector<DecoderMedia>> VisionCompletion::encode(
    const MediaPrompt& prompt, const DecoderCallbacks& callbacks) {
    if (callbacks.is_cancelled && callbacks.is_cancelled()) {
        return std::nullopt;
    }
    const DeviceScope device(device_);
    std::call_once(loaded_, [&] {
        encoder_ = std::make_unique<VisionEncoder>(Checkpoint(checkpoint_));
    });
    const auto storage = std::make_shared<Allocations>(device_);
    std::vector<DecoderMedia> media;
    media.reserve(prompt.parts.size());
    for (const auto& part : prompt.parts) {
        if (callbacks.is_cancelled && callbacks.is_cancelled()) {
            return std::nullopt;
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
        auto& item = media.emplace_back(DecoderMedia{part.layout, {storage, features}, elements});
        const auto patch_bytes = std::string_view(
            reinterpret_cast<const char*>(part.patches.data()),
            part.patches.size() * sizeof(float));
        item.cache_identity.reset(new DecoderMediaCacheIdentity(
            item, encoder_identity_, tokenizer::sha256(patch_bytes)));
    }
    return media;
}

DecoderResult VisionCompletion::generate(BatchService& service, int slot, const MediaPrompt& prompt,
                                        const DecoderOptions& options,
                                        const DecoderCallbacks& callbacks) {
    if (options.max_new_tokens == 0) {
        return callbacks.is_cancelled && callbacks.is_cancelled() ? cancelled() : DecoderResult{};
    }
    auto media = encode(prompt, callbacks);
    return media ? service.generate(slot, prompt.template_ids, options, callbacks,
                                    std::move(*media), prompt.expanded_tokens) : cancelled();
}

}
