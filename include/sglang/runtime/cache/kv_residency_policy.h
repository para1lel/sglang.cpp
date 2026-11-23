#pragma once

#include <cstddef>
#include <optional>
#include <utility>

namespace sglang {
struct DecoderBatchOptions;

inline constexpr std::size_t decoder_shared_reserve = std::size_t{2} * 1024 * 1024 * 1024;
}

namespace sglang::decoder {

class KvResidencyPolicy {
public:
    struct Options {
        std::optional<std::size_t> resident_memory_limit;
    };

    explicit KvResidencyPolicy(const sglang::DecoderBatchOptions& options);
    explicit KvResidencyPolicy(const Options& options);

    bool resident() const;
    std::size_t capacity() const;
    void measure_available(std::size_t free_bytes);
    std::pair<int, int> request_storage(int prompt, int capacity, bool incremental) const;

private:
    bool resident_;
    std::optional<std::size_t> requested_capacity_;
    std::optional<std::size_t> capacity_;
};

}
