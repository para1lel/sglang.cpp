#pragma once

#include "sglang/tokenizer/digest.h"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <vector>

namespace sglang::server {

inline std::string cache_identity_source(const std::filesystem::path& directory,
                                         std::string_view recurrence_tag) {
    std::vector<std::filesystem::path> files{directory / "config.json",
                                             directory / "model.safetensors.index.json"};
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".safetensors") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    std::string identity = "sglang-qwen3.8-prefix-v1\n";
    for (const auto& path : files) {
        struct stat info{};
        if (stat(path.c_str(), &info) != 0) {
            throw std::runtime_error("Cannot identify checkpoint file " + path.string());
        }
        const auto name = path.filename().string();
        identity += std::to_string(name.size()) + ':' + name;
        for (const auto field : {static_cast<std::int64_t>(info.st_dev),
                                 static_cast<std::int64_t>(info.st_ino), info.st_size,
                                 info.st_mtim.tv_sec, info.st_mtim.tv_nsec,
                                 info.st_ctim.tv_sec, info.st_ctim.tv_nsec}) {
            identity += ':' + std::to_string(field);
        }
        identity += '\n';
    }
    identity.append(recurrence_tag);
    return identity;
}

inline std::string cache_identity(const std::filesystem::path& directory,
                                  std::string_view recurrence_tag = std::string_view{}) {
    return tokenizer::sha256(cache_identity_source(directory, recurrence_tag));
}

}
