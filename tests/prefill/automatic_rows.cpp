#include "sglang/decoder.h"
#include "sglang/runtime/backend/cuda_resources.h"
#include "sglang/runtime/backend/device_arena.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using sglang::decoder::CudaError;
using sglang::decoder::DeviceArena;

extern "C" void*
real_arena_lease(DeviceArena* arena,
                 std::size_t bytes) asm("__real__ZN6sglang7decoder11DeviceArena8allocateEm");
extern "C" void*
wrap_arena_lease(DeviceArena* arena,
                 std::size_t bytes) asm("__wrap__ZN6sglang7decoder11DeviceArena8allocateEm");

namespace {

using Attempt = std::pair<std::size_t, cudaError_t>;
constexpr std::size_t bytes_128 = 31944704;
constexpr std::size_t bytes_129 = bytes_128 / 128 * 129;
constexpr std::size_t bytes_512 = 127778816;
constexpr std::size_t bytes_1024 = 255557632;
constexpr int vocabulary = 248320;
std::vector<Attempt> attempts;
std::size_t reject_at = 0;

bool& observe() {
    static bool active = false;
    return active;
}

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

sglang::DecoderResult generate(const sglang::Decoder& decoder, int length, const char* label,
                               std::size_t reject, const std::vector<Attempt>& expected,
                               int rows = 0) {
    attempts.clear();
    reject_at = reject;
    observe() = true;
    sglang::DecoderOptions options;
    options.top_logits = vocabulary;
    options.max_num_batched_tokens = rows;
    const auto result = decoder.generate(prompt(length), options);
    observe() = false;
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

void* wrap_arena_lease(DeviceArena* arena, std::size_t bytes) {
    constexpr auto row_bytes = bytes_128 / 128;
    if (!observe() || bytes < bytes_128 || bytes > bytes_1024 || bytes % row_bytes != 0)
        return real_arena_lease(arena, bytes);
    if (reject_at != 0 && bytes >= reject_at) {
        attempts.emplace_back(bytes, cudaErrorMemoryAllocation);
        throw CudaError(cudaErrorMemoryAllocation, "Test prefill workspace arena exhaustion");
    }
    auto* storage = real_arena_lease(arena, bytes);
    attempts.emplace_back(bytes, cudaSuccess);
    return storage;
}

int main(int argc, char** argv) try {
    require(argc == 2, "Usage: automatic-rows CHECKPOINT (reserved NVIDIA B200 visible)");
    const sglang::Decoder decoder(argv[1]);
    generate(decoder, 128, "128 rows", 0, {{bytes_128, cudaSuccess}});
    generate(decoder, 129, "129 rows", 0, {{bytes_129, cudaSuccess}});
    generate(decoder, 1025, "1024 rows and tail", 0, {{bytes_1024, cudaSuccess}});
    const auto control = generate(decoder, 512, "512 rows", 0, {{bytes_512, cudaSuccess}});
    const auto tiled = generate(decoder, 512, "128-row tiles", 0, {{bytes_128, cudaSuccess}}, 128);
    require(tiled.output_token_ids == control.output_token_ids, "128-row tiles changed the token");
    compare(tiled, generate(decoder, 512, "retry-small", bytes_512,
                            {{bytes_512, cudaErrorMemoryAllocation}, {bytes_128, cudaSuccess}}));
    const std::vector<Attempt> rowwise_failures{
        {bytes_512, cudaErrorMemoryAllocation}, {bytes_128, cudaErrorMemoryAllocation}};
    const auto rowwise = generate(decoder, 512, "retry-row-wise", bytes_128, rowwise_failures);
    compare(rowwise, generate(decoder, 512, "direct-row-wise", 0, {}, 1));
    compare(control,
            generate(decoder, 512, "after-allocation-failure", 0, {{bytes_512, cudaSuccess}}));
    std::cout << "Automatic serial prefill selection passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
