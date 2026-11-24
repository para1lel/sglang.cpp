#pragma once

#include "sglang/json.h"
#include "sglang/runtime/model/qwen_text_model.h"

#include <filesystem>
#include <string_view>

namespace sglang::decoder {

inline constexpr std::string_view qwen_vision_tensor_prefix = "model.visual.";
inline constexpr int qwen_vision_tensor_count = 333;

const QwenTextModel& validate_qwen_text_config(const json::Object& config,
                                               const std::filesystem::path& path);
void validate_qwen_vision_config(const json::Object& config, const std::filesystem::path& path,
                                 const QwenTextModel& model);

}
