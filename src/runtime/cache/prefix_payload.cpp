#include "sglang/runtime/cache/prefix_payload.h"

#include "sglang/runtime/cache/mxfp8_cache_geometry.h"
#include "sglang/runtime/cache/prefix_cache_layout.h"
#include "sglang/runtime/model/qwen_text_model.h"
#include "sglang/tokenizer/digest.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace {

void encode_integer(char*& destination, int value) {
    const auto bits = static_cast<std::uint32_t>(value);
    for (int offset = 0; offset < 32; offset += 8) {
        *destination++ = static_cast<char>(bits >> offset);
    }
}

}

namespace sglang {

DecoderMediaCacheIdentity::DecoderMediaCacheIdentity(const DecoderMedia& media,
                                                     std::string patches_digest)
    : layout_(media.layout), features_(media.features), address_(media.features.get()),
      elements_(media.feature_elements), patches_digest_(std::move(patches_digest)) {}

bool DecoderMediaCacheIdentity::matches(const DecoderMedia& media) const {
    const auto& other = media.layout;
    return address_ == media.features.get() && elements_ == media.feature_elements &&
        !features_.owner_before(media.features) && !media.features.owner_before(features_) &&
        layout_.kind == other.kind && layout_.temporal_pairs == other.temporal_pairs &&
        layout_.patch_height == other.patch_height && layout_.patch_width == other.patch_width &&
        layout_.video_timestamp_tokens == other.video_timestamp_tokens;
}

std::string_view DecoderMediaCacheIdentity::patches_digest() const { return patches_digest_; }

namespace decoder {
namespace {

constexpr int model_rows = QwenTextModel::max_position_embeddings;
constexpr auto kv_row_bytes = 2 * mxfp8_kv_heads * (mxfp8_head_dim + mxfp8_groups);

std::string media_digest(const DecoderMedia& media) {
    if (!media.cache_identity || !media.cache_identity->matches(media)) return {};
    const auto& layout = media.layout;
    std::string material(16, '\0');
    char* out = material.data();
    encode_integer(out, static_cast<int>(layout.kind));
    encode_integer(out, layout.temporal_pairs);
    encode_integer(out, layout.patch_height);
    encode_integer(out, layout.patch_width);
    material.append(media.cache_identity->patches_digest());
    return tokenizer::sha256(material);
}

}

std::string prefix_cache_domain(const PrefixCacheLayout& layout) {
    if (layout.world_size == 1) return "sglang/qwen3_5/split-prefix-mxfp8-v2/";
    std::string domain = "sglang/qwen3_5/split-prefix-mxfp8-v3/tp";
    domain += std::to_string(layout.world_size);
    domain += "-r";
    domain += std::to_string(layout.rank);
    domain += "/kv";
    domain += std::to_string(layout.kv_heads);
    domain += "/gdn";
    domain += std::to_string(layout.gdn_key_heads);
    domain += 'x';
    domain += std::to_string(layout.gdn_value_heads);
    domain += layout.draft_lane ? "/draft/" : "/target/";
    return domain;
}

std::size_t prefix_payload_bytes(PrefixPayloadKind kind, int begin, int end) {
    return prefix_payload_bytes(kind, begin, end, tp1_prefix_cache_layout);
}

std::size_t prefix_payload_bytes(PrefixPayloadKind kind, int begin, int end,
                                 const PrefixCacheLayout& layout) {
    if (begin < 0 || end <= begin || end > model_rows) return 0;
    const bool kv_chunk = begin % prefix_gqa_chunk_tokens == 0 &&
        end - begin <= prefix_gqa_chunk_tokens;
    switch (kind) {
    case PrefixPayloadKind::gqa: {
        const auto row = 2 * static_cast<std::size_t>(layout.kv_heads) *
            (mxfp8_head_dim + mxfp8_groups);
        return kv_chunk
            ? std::size_t(end - begin) * row * qwen_text_model.full_attention_layers()
            : 0;
    }
    case PrefixPayloadKind::gdn: {
        if (begin) return 0;
        std::size_t total = 0;
        for (const auto bytes : prefix_gdn_state_bytes(layout)) total += bytes;
        return total;
    }
    case PrefixPayloadKind::draft:
        return kv_chunk ? std::size_t(end - begin) * kv_row_bytes : 0;
    }
    return 0;
}

PrefixCacheKeys::PrefixCacheKeys(const std::vector<int>& prompt, std::string_view salt,
                                  const std::vector<DecoderMedia>* media,
                                  const PrefixCacheLayout* layout) {
    const auto& geometry = layout != nullptr ? *layout : tp1_prefix_cache_layout;
    world_size_ = geometry.world_size;
    rank_ = geometry.rank;
    std::string domain = prefix_cache_domain(geometry);
    domain.append(salt);
    if (!media || media->empty()) {
        keys_.reserve(prompt.size() + 1);
        keys_.push_back(tokenizer::sha256_digest(domain));
        for (int token : prompt) append(token);
        return;
    }
    std::vector<DecoderMediaLayout> layouts;
    std::vector<std::string> identities;
    layouts.reserve(media->size());
    identities.reserve(media->size());
    for (const auto& item : *media) {
        layouts.push_back(item.layout);
        identities.push_back(media_digest(item));
    }
    const auto expanded = prepare_multimodal_prompt(prompt, layouts);
    tokens_ = static_cast<int>(expanded.token_ids.size());
    next_position_ = expanded.next_position;
    keys_.reserve(expanded.token_ids.size() + 1);
    keys_.push_back(tokenizer::sha256_digest(domain));
    auto span = expanded.spans.begin();
    for (int row = 0; row < tokens_; ++row) {
        if (span != expanded.spans.end() && row == span->token_begin + span->rows) ++span;
        if (span != expanded.spans.end() && row >= span->token_begin) {
            const auto& identity = identities[span->media_index];
            if (identity.empty()) break;
            append_row(expanded.token_ids[row], expanded.positions[row], identity,
                       span->feature_begin + row - span->token_begin);
        } else {
            append_row(expanded.token_ids[row], expanded.positions[row], {}, -1);
        }
    }
}

bool PrefixCacheKeys::matches_rank(int world_size, int rank) const {
    return world_size_ == world_size && rank_ == rank;
}

std::string PrefixCacheKeys::key(int end) const {
    if (end < 0 || end > cacheable_tokens()) return {};
    return tokenizer::format_sha256(keys_[end]);
}

void PrefixCacheKeys::append(int token) {
    if (tokens_ == cacheable_tokens()) {
        append_row(token, {next_position_, next_position_, next_position_}, {}, -1);
    }
    ++tokens_;
    ++next_position_;
}

void PrefixCacheKeys::append_row(int token, MropePosition position, std::string_view media,
                                  int feature_row) {
    constexpr std::size_t fixed = tokenizer::Sha256Digest{}.size() + 5 * 4;
    std::array<char, fixed + 64> input;
    char* out = std::copy(keys_.back().begin(), keys_.back().end(), input.begin());
    encode_integer(out, token);
    encode_integer(out, position.temporal);
    encode_integer(out, position.height);
    encode_integer(out, position.width);
    encode_integer(out, feature_row);
    std::copy(media.begin(), media.end(), out);
    keys_.push_back(tokenizer::sha256_digest({input.data(), fixed + media.size()}));
}

}
}

namespace sglang::decoder {

int PrefixCacheKeys::tokens() const { return tokens_; }

int PrefixCacheKeys::cacheable_tokens() const { return static_cast<int>(keys_.size()) - 1; }

}
