#include "sglang/server/chat_media.h"

#include "sglang/tokenizer/tokenizer.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <stdexcept>

namespace sglang::server {
namespace {

using json::get;
using json::OrderedObject;
using json::Value;

constexpr std::string_view rgb_url = "data:application/x-sglang-rgb8;base64,";
constexpr int max_dimension = 1024;
constexpr int min_dimension = 16;
constexpr std::size_t max_media = 4;
constexpr std::size_t max_frames = 32;
constexpr int max_source_frames = 1'000'000'000;
constexpr int max_grid_patches = 16384;
constexpr int max_frame_patches = 4096;
constexpr int resize_alignment = 32;
constexpr int patch_edge = 16;
constexpr int vision_start = 248053;
constexpr int vision_end = 248054;
constexpr int image_pad = 248056;
constexpr int video_pad = 248057;

const OrderedObject* object(const OrderedObject& parent, std::string_view key) {
    const auto* value = get(parent, key);
    return value ? value->ordered_object() : nullptr;
}

const json::Array* array(const OrderedObject& parent, std::string_view key) {
    const auto* value = get(parent, key);
    return value ? value->array() : nullptr;
}

std::optional<std::string_view> string(const OrderedObject& parent, std::string_view key) {
    const auto* value = get(parent, key);
    return value ? value->string() : std::nullopt;
}

int integer(const OrderedObject& parent, std::string_view key, int minimum, int maximum) {
    const auto* value = get(parent, key);
    const auto number = value ? value->unsigned_integer() : std::nullopt;
    if (!number || *number < std::uint64_t(minimum) || *number > std::uint64_t(maximum)) {
        throw std::invalid_argument("Missing or out-of-range decoded media field: " +
                                    std::string(key));
    }
    return static_cast<int>(*number);
}

VideoMetadata parse_video(const OrderedObject& payload) {
    VideoMetadata video;
    video.source_frame_count = integer(payload, "source_frame_count", 2, max_source_frames);
    const auto* fps = get(payload, "source_fps");
    video.source_fps = fps ? fps->number().value_or(0) : 0;
    if (!std::isfinite(video.source_fps) || video.source_fps < 0.001 ||
        video.source_fps > 1'000'000) {
        throw std::invalid_argument("source_fps must be finite and in [0.001, 1000000]");
    }
    const auto* indices = array(payload, "frame_indices");
    if (!indices || indices->size() < 2 || indices->size() > max_frames) {
        throw std::invalid_argument("Expected 2 to 32 selected video frame indices");
    }
    for (const auto& index : *indices) {
        const auto number = index.unsigned_integer();
        if (!number || *number >= std::uint64_t(video.source_frame_count) ||
            (!video.frame_indices.empty() &&
             *number < std::uint64_t(video.frame_indices.back()))) {
            throw std::invalid_argument("Frame indices must be nondecreasing source ordinals");
        }
        video.frame_indices.push_back(static_cast<int>(*number));
    }
    return video;
}

std::size_t frame_count(const MediaPart& part) {
    return part.kind == DecoderMediaKind::image ? 1 : part.video.frame_indices.size();
}

std::size_t rgb_bytes(const MediaPart& part) {
    return frame_count(part) * part.height * part.width * 3;
}

int bound_grid(const MediaPart& part) {
    int height = part.height;
    int width = part.width;
    const auto frames = frame_count(part);
    const bool image = part.kind == DecoderMediaKind::image;
    if (!image && std::min(height, width) < resize_alignment) {
        const double scale = double(resize_alignment) / std::min(height, width);
        height = static_cast<int>(height * scale);
        width = static_cast<int>(width * scale);
    }
    const double min_pixels = image ? 65536 : 4096;
    const double scale = std::max(1.0, std::sqrt(min_pixels / (frames * height * width)));
    const auto patch_axis = [scale](int extent) {
        return static_cast<int>(std::ceil(extent * scale / resize_alignment)) *
            (resize_alignment / patch_edge);
    };
    const int frame_patches = patch_axis(height) * patch_axis(width);
    if (frame_patches > max_frame_patches) {
        throw std::invalid_argument("Decoded media exceeds the per-frame vision patch budget");
    }
    return frame_patches * static_cast<int>((frames + 1) / 2);
}

int base64_digit(char character) {
    if (character >= 'A' && character <= 'Z') {
        return character - 'A';
    }
    if (character >= 'a' && character <= 'z') {
        return character - 'a' + 26;
    }
    if (character >= '0' && character <= '9') {
        return character - '0' + 52;
    }
    return character == '+' ? 62 : character == '/' ? 63 : -1;
}

void validate_base64(std::string_view encoded, std::size_t bytes) {
    if (encoded.size() != bytes / 3 * 4) {
        throw std::invalid_argument("Decoded RGB8 byte count disagrees with media dimensions");
    }
    if (!std::all_of(encoded.begin(), encoded.end(), [](char digit) {
            return base64_digit(digit) >= 0;
        })) {
        throw std::invalid_argument("Expected canonical base64 RGB8");
    }
}

std::vector<std::uint8_t> decode_rgb(const MediaPart& part) {
    std::vector<std::uint8_t> rgb;
    rgb.reserve(rgb_bytes(part));
    for (std::size_t offset = 0; offset < part.encoded.size(); offset += 4) {
        const auto group = part.encoded.substr(offset, 4);
        const auto first = base64_digit(group[0]);
        const auto second = base64_digit(group[1]);
        const auto third = base64_digit(group[2]);
        rgb.push_back(std::uint8_t((first << 2) | (second >> 4)));
        rgb.push_back(std::uint8_t((second << 4) | (third >> 2)));
        rgb.push_back(std::uint8_t((third << 6) | base64_digit(group[3])));
    }
    return rgb;
}

MediaPart parse_part(const OrderedObject& part, std::string_view type) {
    const bool image = type == "image_url";
    const auto* payload = object(part, type);
    if (part.size() != 2 || !payload || payload->size() != (image ? 3 : 6)) {
        throw std::invalid_argument("Expected a decoded image_url or video_url object");
    }
    const auto url = string(*payload, "url").value_or("");
    if (url.substr(0, rgb_url.size()) != rgb_url) {
        throw std::invalid_argument(
            "Media URL must be an inline application/x-sglang-rgb8 data URL");
    }
    MediaPart result{image ? DecoderMediaKind::image : DecoderMediaKind::video,
                     url.substr(rgb_url.size()),
                     integer(*payload, "height", min_dimension, max_dimension),
                     integer(*payload, "width", min_dimension, max_dimension), {}};
    if (!image) {
        result.video = parse_video(*payload);
    }
    validate_base64(result.encoded, rgb_bytes(result));
    return result;
}

std::optional<std::string_view> media_type(const OrderedObject& part) {
    const auto type = string(part, "type");
    if (type == "image_url" || type == "video_url") {
        const auto* payload = get(part, *type);
        if (payload && !payload->is_null()) {
            return type;
        }
    }
    return std::nullopt;
}

bool has_media(const Value& value) {
    const auto* turn = value.ordered_object();
    const auto* parts = turn ? array(*turn, "content") : nullptr;
    return parts && std::any_of(parts->begin(), parts->end(), [](const Value& value) {
        const auto* part = value.ordered_object();
        return part && media_type(*part).has_value();
    });
}

std::optional<Value> normalize_turn(const OrderedObject& turn, std::vector<MediaPart>& media,
                                    bool enabled) {
    const auto* parts = array(turn, "content");
    if (!parts) {
        return std::nullopt;
    }
    json::Array normalized;
    for (const auto& value : *parts) {
        const auto* part = value.ordered_object();
        const auto type = part ? media_type(*part) : std::nullopt;
        if (!type) {
            normalized.push_back(value);
            continue;
        }
        if (!enabled || string(turn, "role") != "user" || media.size() == max_media) {
            throw std::invalid_argument(
                "Decoded media requires serial chat, user content and at most four attachments");
        }
        media.push_back(parse_part(*part, *type));
        normalized.emplace_back(OrderedObject{{"type", *type == "image_url" ? "image" : "video"}});
    }
    OrderedObject result;
    for (const auto& [key, value] : turn) {
        result.emplace_back(key, key == "content" ? Value(std::move(normalized)) : value);
    }
    return Value(std::move(result));
}

std::string timestamp_text(double seconds) {
    char buffer[48];
    const auto formatted = std::to_chars(buffer, buffer + sizeof(buffer), seconds,
                                         std::chars_format::fixed, 1);
    if (formatted.ec != std::errc{}) {
        throw std::invalid_argument("Video timestamp cannot be represented");
    }
    return "<" + std::string(buffer, formatted.ptr) + " seconds>";
}

PreparedMedia prepare(const MediaPart& part, const Tokenizer& tokenizer, int& remaining_patches) {
    const auto rgb = decode_rgb(part);
    const DecodedRgb input{rgb.data(), rgb.size(), part.height, part.width,
                           static_cast<int>(frame_count(part))};
    auto patches = part.kind == DecoderMediaKind::image
        ? preprocess_image(input)
        : preprocess_video(input, part.video, VideoPreprocess::Checkpoint);
    const auto [pairs, height, width] = patches.grid_thw;
    if (height * width > max_frame_patches || pairs * height * width > remaining_patches) {
        throw std::invalid_argument("Decoded media exceeds the vision patch budget");
    }
    remaining_patches -= pairs * height * width;
    PreparedMedia result{{part.kind, pairs, height, width, {}}, std::move(patches.pixels)};
    for (const auto seconds : patches.timestamps_seconds) {
        const auto ids = tokenizer.encode(timestamp_text(seconds));
        result.layout.video_timestamp_tokens.emplace_back(ids.begin(), ids.end());
    }
    return result;
}

void validate_markers(const std::vector<int>& ids, const std::vector<MediaPart>& parts) {
    std::vector<int> expected;
    for (const auto& part : parts) {
        expected.insert(expected.end(), {vision_start,
            part.kind == DecoderMediaKind::image ? image_pad : video_pad, vision_end});
    }
    std::size_t offset = 0;
    for (const auto id : ids) {
        if (id == vision_start || id == vision_end || id == image_pad || id == video_pad) {
            if (offset == expected.size() || id != expected[offset++]) {
                throw std::invalid_argument("Unexpected visual marker in chat text");
            }
        }
    }
    if (offset != expected.size()) {
        throw std::invalid_argument("Missing visual marker in rendered chat");
    }
}

}

std::optional<ChatMedia> parse_chat_media(const OrderedObject& chat, bool enabled) {
    const auto* messages = array(chat, "messages");
    if (!messages || !std::any_of(messages->begin(), messages->end(), has_media)) {
        return std::nullopt;
    }
    ChatMedia result;
    json::Array normalized;
    for (const auto& value : *messages) {
        const auto* turn = value.ordered_object();
        auto replacement = turn ? normalize_turn(*turn, result.parts, enabled) : std::nullopt;
        normalized.push_back(replacement ? std::move(*replacement) : value);
    }
    int remaining = max_grid_patches;
    for (const auto& part : result.parts) {
        remaining -= bound_grid(part);
        if (remaining < 0) {
            throw std::invalid_argument("Decoded media exceeds the total vision patch budget");
        }
    }
    for (const auto& [key, value] : chat) {
        result.chat.emplace_back(key, key == "messages" ? Value(std::move(normalized)) : value);
    }
    return result;
}

std::vector<int> expand_chat_media(MediaPrompt& prompt, const std::vector<MediaPart>& parts,
                                   const Tokenizer& tokenizer) {
    validate_markers(prompt.template_ids, parts);
    std::vector<DecoderMediaLayout> layouts;
    int remaining_patches = max_grid_patches;
    for (const auto& part : parts) {
        auto prepared = prepare(part, tokenizer, remaining_patches);
        layouts.push_back(prepared.layout);
        prompt.parts.push_back(std::move(prepared));
    }
    return prepare_multimodal_prompt(prompt.template_ids, layouts).token_ids;
}

}
