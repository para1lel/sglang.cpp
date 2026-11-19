#pragma once

#include "sglang/json.h"
#include "sglang/media_preprocess.h"
#include "sglang/multimodal_prompt.h"

namespace sglang {
class Tokenizer;
}

namespace sglang::server {

struct MediaPart {
    DecoderMediaKind kind;
    std::string_view encoded;
    int height;
    int width;
    VideoMetadata video;
    std::vector<std::uint8_t> rgb;
};

struct ChatMedia {
    json::OrderedObject chat;
    std::vector<MediaPart> parts;
};

struct PreparedMedia {
    DecoderMediaLayout layout;
    std::vector<float> patches;
};

struct MediaPrompt {
    std::vector<int> template_ids;
    std::vector<PreparedMedia> parts;
    std::size_t expanded_tokens = 0;
};

std::optional<ChatMedia> parse_chat_media(const json::OrderedObject& chat, bool enabled);
std::vector<int> expand_chat_media(MediaPrompt& prompt, const std::vector<MediaPart>& parts,
                                   const Tokenizer& tokenizer);

}
