#include "sglang/decoder.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" cudaError_t __real_cudaMallocFromPoolAsync(void**, std::size_t, cudaMemPool_t,
                                                      cudaStream_t);

namespace {

using Attempt = std::pair<std::size_t, cudaError_t>;
constexpr std::size_t bytes_128 = 31944704;
constexpr std::size_t bytes_129 = bytes_128 / 128 * 129;
constexpr std::size_t bytes_512 = 127778816;
constexpr std::size_t bytes_1024 = 255557632;
constexpr int vocabulary = 248320;
std::vector<Attempt> attempts;
std::size_t reject_at = 0;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<int> prompt(int length) {
    std::vector<int> result;
    std::uint32_t state = 0x365f7b21;
    for (int index = 0; index < length; ++index) {
        state = 1664525U * state + 1013904223U;
        result.push_back(32 + (state >> 8U) % 19000U);
    }
    return result;
}

sglang::DecoderResult generate(const sglang::Decoder& decoder, int length,
                               std::size_t reject, const char* label,
                               const std::vector<Attempt>& expected) {
    attempts.clear();
    reject_at = reject;
    const auto result = decoder.generate(prompt(length), {.top_logits = vocabulary});
    reject_at = 0;
    std::cout << label << " prompt=" << length << " allocations=";
    for (const auto& [bytes, status] : attempts) {
        std::cout << bytes << ':' << static_cast<int>(status) << ' ';
    }
    std::cout << std::endl;
    require(attempts == expected, "Unexpected prefill workspace allocation sequence");
    require(result.finish_reason == sglang::DecoderFinishReason::length &&
            result.output_token_ids.size() == 1 && result.step_top_logits.size() == 1 &&
            result.step_top_logits.front().size() == vocabulary, "Incomplete generated scores");
    return result;
}

void compare(const sglang::DecoderResult& control, const sglang::DecoderResult& selected) {
    require(control.output_token_ids == selected.output_token_ids &&
            std::equal(control.step_top_logits.front().begin(),
                       control.step_top_logits.front().end(),
                       selected.step_top_logits.front().begin(),
                       [](const auto& left, const auto& right) {
                           return left.token_id == right.token_id && std::isfinite(left.logit) &&
                                  left.logit == right.logit;
                       }), "Generated token or complete score row changed");
}

}

extern "C" cudaError_t __wrap_cudaMallocFromPoolAsync(void** storage, std::size_t bytes,
                                                      cudaMemPool_t pool, cudaStream_t stream) {
    const auto status = reject_at && bytes >= reject_at
        ? cudaErrorMemoryAllocation : __real_cudaMallocFromPoolAsync(storage, bytes, pool, stream);
    attempts.emplace_back(bytes, status);
    return status;
}

int main(int argc, char** argv) try {
    require(argc == 2, "Usage: automatic-rows CHECKPOINT (reserved NVIDIA B200 visible)");
    const sglang::Decoder decoder(argv[1]);
    generate(decoder, 128, 0, "128 rows", {{bytes_128, cudaSuccess}});
    generate(decoder, 129, 0, "129 rows", {{bytes_129, cudaSuccess}});
    generate(decoder, 1025, 0, "1024 rows and tail", {{bytes_1024, cudaSuccess}});
    const auto control = generate(decoder, 512, 0, "512 rows", {{bytes_512, cudaSuccess}});
    compare(control, generate(decoder, 512, bytes_512, "retry-small",
                              {{bytes_512, cudaErrorMemoryAllocation},
                               {bytes_128, cudaSuccess}}));
    const std::vector<Attempt> rowwise_failures{
        {bytes_512, cudaErrorMemoryAllocation}, {bytes_128, cudaErrorMemoryAllocation}};
    const auto rowwise = generate(decoder, 512, bytes_128, "retry-row-wise", rowwise_failures);
    compare(rowwise, generate(decoder, 512, bytes_128, "repeat-row-wise", rowwise_failures));
    compare(control, generate(decoder, 512, 0, "after-allocation-failure",
                              {{bytes_512, cudaSuccess}}));
    std::cout << "Automatic serial prefill selection passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
