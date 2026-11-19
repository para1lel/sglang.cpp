#pragma once

#include <cstddef>
#include <vector>

namespace sglang {

enum class DecoderMediaKind { image, video };

struct DecoderMediaLayout {
    DecoderMediaKind kind;
    int temporal_pairs;
    int patch_height;
    int patch_width;
    std::vector<std::vector<int>> video_timestamp_tokens;
};

struct MropePosition {
    int temporal;
    int height;
    int width;
};

struct DecoderMediaSpan {
    int token_begin;
    int rows;
    std::size_t media_index;
    int feature_begin;
};

struct MultimodalPrompt {
    std::vector<int> token_ids;
    std::vector<MropePosition> positions;
    std::vector<DecoderMediaSpan> spans;
    int next_position = 0;
};

int decoder_media_feature_rows(const DecoderMediaLayout& media);
MultimodalPrompt prepare_multimodal_prompt(const std::vector<int>& template_ids,
                                          const std::vector<DecoderMediaLayout>& media);

}
