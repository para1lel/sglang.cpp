#pragma once

#include "sglang/decoder_internal.h"
#include "sglang/decoder_media.h"

namespace sglang::decoder {

static_assert(hidden_size == decoder_media_feature_width);

struct MultimodalInput {
    std::shared_ptr<const Allocations> model;
    MultimodalPrompt prompt;
    std::vector<std::shared_ptr<const Bf16>> features;
};

struct MediaPrefill {
    const MultimodalInput* input = nullptr;
    const MropePosition* positions = nullptr;

    const Bf16* feature(int token_position) const;
    void substitute(Bf16* embeddings, int position, int rows, cudaStream_t stream) const;
};

std::shared_ptr<MultimodalInput> prepare_input(const std::vector<int>& template_ids,
                                               const std::vector<DecoderMedia>& media,
                                               std::shared_ptr<const Allocations> model);
void validate_feature_device(const MultimodalInput& input, int device);
MediaPrefill upload_media_positions(Request& request, const MultimodalInput& input,
                                     cudaStream_t stream);

}
