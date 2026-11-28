#pragma once

#include "sglang/runtime/model/qwen_text_model.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <string_view>
#include <vector>

namespace sglang {

enum class TensorDtype { BF16, F8_E4M3 };

struct HostTensorView {
    TensorDtype dtype;
    std::vector<std::uint64_t> shape;
    const std::byte* data;
    std::size_t size_bytes;
};

class Checkpoint {
public:
    explicit Checkpoint(const std::filesystem::path& directory);
    ~Checkpoint();

    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;
    Checkpoint(Checkpoint&&) = delete;
    Checkpoint& operator=(Checkpoint&&) = delete;

    const decoder::QwenTextModel& text_model() const;

    const HostTensorView& tensor(std::string_view name) const;
    const HostTensorView& tensor(std::string_view name, TensorDtype dtype,
                                 std::initializer_list<std::uint64_t> shape) const;

    void validate_vision() const;

private:
    struct Storage;
    std::unique_ptr<Storage> storage_;
};

inline constexpr std::string_view checkpoint_model_type = decoder::QwenTextModel::model_type;
inline constexpr std::string_view checkpoint_architecture = decoder::QwenTextModel::architecture;

}
