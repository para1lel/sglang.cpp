#include "sglang/runtime/server/cache_identity.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string historical_identity_source(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> files{directory / "config.json",
                                             directory / "model.safetensors.index.json"};
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".safetensors") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    std::string identity = "sglang-qwen3.8-prefix-v1\n";
    for (const auto& path : files) {
        struct stat info {};
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
    return identity;
}

int failures = 0;

void expect(bool condition, std::string_view what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: cache_identity_test FIXTURE_DIR\n";
        return 2;
    }
    const std::filesystem::path directory(argv[1]);
    const std::string historical = sglang::tokenizer::sha256(
        historical_identity_source(directory));
    const std::string native = sglang::server::cache_identity(directory);
    expect(native == historical, "empty recurrence tag preserves the historical identity");
    expect(sglang::server::cache_identity_source(directory, "") ==
               historical_identity_source(directory),
           "empty recurrence tag preserves the identity source byte-for-byte");

    const std::string vendor_tag = "\ngdn-recurrence-vendor-v1 mask=11\n";
    const std::string vendor = sglang::server::cache_identity(directory, vendor_tag);
    expect(vendor != native, "vendor recurrence tag changes the identity");
    expect(vendor == sglang::server::cache_identity(directory, vendor_tag),
           "vendor identity is stable across calls");
    const std::string partial = sglang::server::cache_identity(
        directory, "\ngdn-recurrence-vendor-v1 mask=01\n");
    expect(partial != vendor && partial != native,
           "partial rank support is an explicit distinct namespace");

    std::cout << "historical=" << historical << '\n';
    std::cout << "vendor_mask_11=" << vendor << '\n';
    std::cout << "vendor_mask_01=" << partial << '\n';
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "cache_identity_test ok\n";
    return 0;
}
