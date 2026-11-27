#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace sglang::server {

std::string cache_identity_source(const std::filesystem::path& directory,
                                  std::string_view recurrence_tag);

std::string cache_identity(const std::filesystem::path& directory,
                           std::string_view recurrence_tag = std::string_view{});

}
