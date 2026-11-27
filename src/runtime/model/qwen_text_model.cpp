#include "sglang/runtime/model/qwen_text_model.h"

namespace sglang::decoder {

bool QwenTextModel::is_full_attention(int layer_index) const {
    return (layer_index + 1) % full_attention_interval == 0;
}

int QwenTextModel::full_attention_layers() const { return layer_count / full_attention_interval; }

int QwenTextModel::gdn_layers() const { return layer_count - full_attention_layers(); }

QwenTextModel::LayerMapping QwenTextModel::layer_mapping(int layer_index) const {
    const int full_attention_before = layer_index / full_attention_interval;
    const bool full_attention = is_full_attention(layer_index);
    return {full_attention, full_attention ? full_attention_before
                                          : layer_index - full_attention_before};
}

}
