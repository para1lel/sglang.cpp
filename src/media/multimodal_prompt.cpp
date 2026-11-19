#include "sglang/multimodal_prompt.h"

#include <algorithm>
#include <stdexcept>

namespace sglang {
namespace {

constexpr int vision_start = 248053;
constexpr int vision_end = 248054;
constexpr int image_pad = 248056;
constexpr int video_pad = 248057;
constexpr int vocab_size = 248320;
constexpr int context_tokens = 262144;

bool valid_token(int id) {
    return id >= 0 && id < vocab_size;
}

bool timestamp_token(int id) {
    return valid_token(id) && id != vision_start && id != vision_end &&
        id != image_pad && id != video_pad;
}

void require_room(const MultimodalPrompt& prompt, std::size_t rows) {
    if (rows > context_tokens - prompt.token_ids.size()) {
        throw std::invalid_argument("Expanded multimodal prompt exceeds the checkpoint context");
    }
}

void append_text(MultimodalPrompt& prompt, int id) {
    require_room(prompt, 1);
    const int position = prompt.next_position++;
    prompt.token_ids.push_back(id);
    prompt.positions.push_back({position, position, position});
}

void append_plane(MultimodalPrompt& prompt, int id, int height, int width,
                  std::size_t media_index, int feature_begin) {
    const int rows = height * width;
    require_room(prompt, rows);
    prompt.spans.push_back({
        static_cast<int>(prompt.token_ids.size()), rows, media_index, feature_begin});
    prompt.token_ids.insert(prompt.token_ids.end(), rows, id);
    const int base = prompt.next_position;
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            prompt.positions.push_back({base, base + row, base + column});
        }
    }
    prompt.next_position += std::max(height, width);
}

void append_media(MultimodalPrompt& prompt, int id, const DecoderMediaLayout& media,
                  std::size_t media_index) {
    const int rows = decoder_media_feature_rows(media);
    require_room(prompt, rows);
    const int height = media.patch_height / 2;
    const int width = media.patch_width / 2;
    if (id == image_pad && media.kind == DecoderMediaKind::image &&
        media.temporal_pairs == 1 && media.video_timestamp_tokens.empty()) {
        append_plane(prompt, id, height, width, media_index, 0);
        return;
    }
    if (id != video_pad || media.kind != DecoderMediaKind::video ||
        media.video_timestamp_tokens.size() != std::size_t(media.temporal_pairs)) {
        throw std::invalid_argument("Media kind, grid, or timestamps disagree with placeholder");
    }
    for (int plane = 0; plane < media.temporal_pairs; ++plane) {
        const auto& timestamp = media.video_timestamp_tokens[plane];
        if (timestamp.empty() ||
            !std::all_of(timestamp.begin(), timestamp.end(), timestamp_token)) {
            throw std::invalid_argument("Invalid tokenized video timestamp");
        }
        for (const int token : timestamp) {
            append_text(prompt, token);
        }
        append_text(prompt, vision_start);
        append_plane(prompt, id, height, width, media_index, plane * height * width);
        append_text(prompt, vision_end);
    }
}

}

int decoder_media_feature_rows(const DecoderMediaLayout& media) {
    if (media.temporal_pairs < 1 || media.patch_height < 2 || media.patch_width < 2 ||
        media.patch_height % 2 || media.patch_width % 2) {
        throw std::invalid_argument("Media grid requires positive pairs and even spatial axes");
    }
    const int height = media.patch_height / 2;
    const int width = media.patch_width / 2;
    if (height > context_tokens / width ||
        media.temporal_pairs > context_tokens / (height * width)) {
        throw std::invalid_argument("Media grid exceeds the checkpoint context");
    }
    return media.temporal_pairs * height * width;
}

MultimodalPrompt prepare_multimodal_prompt(const std::vector<int>& template_ids,
                                          const std::vector<DecoderMediaLayout>& media) {
    if (template_ids.empty() || template_ids.size() > context_tokens || media.empty() ||
        media.size() > template_ids.size()) {
        throw std::invalid_argument("Invalid multimodal prompt or media count");
    }
    MultimodalPrompt prompt;
    prompt.token_ids.reserve(template_ids.size());
    prompt.positions.reserve(template_ids.size());
    std::size_t media_index = 0;
    for (std::size_t index = 0; index < template_ids.size(); ++index) {
        const int id = template_ids[index];
        if (!valid_token(id)) {
            throw std::invalid_argument("Invalid token in multimodal template");
        }
        if (id != image_pad && id != video_pad) {
            append_text(prompt, id);
            continue;
        }
        if (index == 0 || index + 1 == template_ids.size() ||
            template_ids[index - 1] != vision_start || template_ids[index + 1] != vision_end ||
            media_index == media.size()) {
            throw std::invalid_argument("Media placeholder must match one wrapped media item");
        }
        append_media(prompt, id, media[media_index], media_index);
        ++media_index;
    }
    if (media_index != media.size()) {
        throw std::invalid_argument("Media item has no corresponding placeholder");
    }
    return prompt;
}

}
