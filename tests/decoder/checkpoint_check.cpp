#include "sglang/checkpoint.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

namespace fs = std::filesystem;

constexpr std::string_view index_json =
    R"({"weight_map":{"sample.bf16":"weights.safetensors","sample.fp8":"weights.safetensors"}})";
constexpr std::string_view valid_header =
    R"({"sample.bf16":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},)"
    R"("sample.fp8":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[4,6]}})";
constexpr std::string_view malformed_header =
    R"({"sample.bf16":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},)"
    R"("sample.fp8":{"dtype":"F8_E4M3","shape":[3],"data_offsets":[4,7]}})";
constexpr std::string_view duplicate_index =
    R"({"weight_map":{"sample.bf16":"weights.safetensors","sample.bf16":"weights.safetensors"}})";
constexpr std::array<unsigned char, 6> payload{0x80, 0x3f, 0x00, 0x40, 0x38, 0xc0};

void check(bool condition, const char* detail) {
    if (!condition) {
        throw std::runtime_error(detail);
    }
}

void create_fixture(const fs::path& directory, const fs::path& config, std::string_view index,
                    std::string_view header) {
    check(fs::create_directory(directory), "Fixture destination already exists");
    fs::copy_file(config, directory / "config.json");
    {
        std::ofstream output(directory / "model.safetensors.index.json", std::ios::binary);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output.write(index.data(), index.size());
        output.close();
    }
    std::ofstream output(directory / "weights.safetensors", std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    const std::uint64_t header_size = header.size();
    for (unsigned byte = 0; byte < 8; ++byte) {
        output.put(static_cast<char>((header_size >> (8 * byte)) & 0xff));
    }
    output.write(header.data(), header.size());
    for (const auto byte : payload) {
        output.put(static_cast<char>(byte));
    }
    output.close();
}

void expect_malformed(const fs::path& fixture, std::string_view detail) {
    try {
        const sglang::Checkpoint ignored(fixture);
    } catch (const std::exception& error) {
        std::cout << error.what() << '\n';
        check(std::string_view(error.what()).find(detail) != std::string_view::npos,
              "Fixture failed at the wrong loader boundary");
        return;
    }
    throw std::runtime_error("Malformed fixture loaded successfully");
}

void verify_bytes(const sglang::HostTensorView& tensor, unsigned start, unsigned count) {
    check(tensor.size_bytes == count, "Wrong tensor byte extent");
    for (unsigned index = 0; index < count; ++index) {
        check(std::to_integer<unsigned>(tensor.data[index]) == payload[start + index],
              "Wrong mapped tensor data");
    }
}

void run(const fs::path& directory, const fs::path& config) {
    check(fs::create_directory(directory), "Fixture root already exists");
    const auto valid = directory / "valid";
    const auto malformed = directory / "out-of-bounds";
    const auto duplicate = directory / "duplicate-key";
    create_fixture(valid, config, index_json, valid_header);
    create_fixture(malformed, config, index_json, malformed_header);
    create_fixture(duplicate, config, duplicate_index, valid_header);
    const sglang::Checkpoint checkpoint(valid);
    verify_bytes(checkpoint.tensor("sample.bf16", sglang::TensorDtype::BF16, {2}), 0, 4);
    verify_bytes(checkpoint.tensor("sample.fp8", sglang::TensorDtype::F8_E4M3, {2}), 4, 2);
    expect_malformed(malformed, "data_offsets out of bounds");
    expect_malformed(duplicate, "duplicate JSON object key");
    std::cout << "Checkpoint input checks passed\n";
}

}

int main(int argc, const char** argv) {
    try {
        check(argc == 3, "Usage: checkpoint_check CONFIG_JSON NEW_FIXTURE_DIRECTORY");
        run(argv[2], argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Checkpoint check failed: " << error.what() << '\n';
        return 1;
    }
}
