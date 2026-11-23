#include "sglang/decoder_prefix_cache.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace {

int copy_budget = -1;
cudaError_t recorded_copy_error = cudaSuccess;
cudaError_t rejected_copy_status = cudaSuccess;
std::size_t failed_allocation_bytes = 0;
std::size_t free_bytes_override = 0;

}

extern "C" cudaError_t __real_cudaMemcpy2DAsync(
    void*, std::size_t, const void*, std::size_t, std::size_t, std::size_t, cudaMemcpyKind,
    cudaStream_t);

extern "C" cudaError_t __wrap_cudaMemcpy2DAsync(
    void* dst, std::size_t dst_stride, const void* src, std::size_t src_stride,
    std::size_t width, std::size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    if (copy_budget == 0) {
        copy_budget = -1;
        rejected_copy_status = __real_cudaMemcpy2DAsync(
            dst, width - 1, src, src_stride, width, height, kind, stream);
        recorded_copy_error = cudaPeekAtLastError();
        return rejected_copy_status;
    }
    if (copy_budget > 0) {
        --copy_budget;
    }
    return __real_cudaMemcpy2DAsync(dst, dst_stride, src, src_stride, width, height, kind, stream);
}

extern "C" cudaError_t __real_cudaMemGetInfo(std::size_t*, std::size_t*);

extern "C" cudaError_t __wrap_cudaMemGetInfo(std::size_t* free, std::size_t* total) {
    const auto status = __real_cudaMemGetInfo(free, total);
    if (status == cudaSuccess && free_bytes_override) *free = free_bytes_override;
    return status;
}

extern "C" cudaError_t __real_cudaMalloc(void**, std::size_t);

extern "C" cudaError_t __wrap_cudaMalloc(void** pointer, std::size_t bytes) {
    if (bytes == failed_allocation_bytes && bytes != 0) {
        failed_allocation_bytes = 0;
        return cudaErrorMemoryAllocation;
    }
    return __real_cudaMalloc(pointer, bytes);
}

namespace {

using namespace sglang;
using namespace sglang::decoder;

constexpr std::uint8_t guard = 0xcd;
constexpr int page = mxfp8_page_size;
constexpr int full_pages = decoder_batch_prefix_checkpoint_interval / page;
constexpr int fixture_page_stride = full_pages + 5;

constexpr std::size_t target_kv_bytes(int pages) {
    return 2 * full_attention_layers * pages *
        (mxfp8_payload_bytes_per_page + mxfp8_scale_bytes_per_page);
}

constexpr std::size_t target_snapshot_bytes(int pages) {
    return target_kv_bytes(pages) + gdn_layers * (gdn_history_elements * sizeof(Bf16) +
        gdn_state_elements * sizeof(float)) + 2 * hidden_size * sizeof(Bf16);
}

static_assert(full_pages == 64);
static_assert(decoder_batch_prefix_max_tokens == 262144);
static_assert(decoder_batch_prefix_max_tokens / page == 2048);

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint8_t pattern(std::size_t index, unsigned tag) {
    return (index * 31 + (index >> 9) * 17 + tag * 23 + (index >> 6) * tag) % 199;
}

void fill(void* target, std::size_t bytes, unsigned tag) {
    std::vector<std::uint8_t> host(bytes);
    for (std::size_t index = 0; index < bytes; ++index) {
        host[index] = pattern(index, tag);
    }
    check(cudaMemcpy(target, host.data(), bytes, cudaMemcpyHostToDevice), "Fill test state");
}

void compare(const void* source, std::size_t bytes, unsigned tag, const char* message) {
    std::vector<std::uint8_t> host(bytes);
    check(cudaMemcpy(host.data(), source, bytes, cudaMemcpyDeviceToHost), "Read test state");
    for (std::size_t index = 0; index < bytes; ++index) {
        require(host[index] == (tag ? pattern(index, tag) : guard), message);
    }
}

void compare_page(const std::uint8_t* source, std::size_t bytes, unsigned tag, int rows,
                  bool scales, const char* message) {
    std::vector<std::uint8_t> data(bytes);
    check(cudaMemcpy(data.data(), source, bytes, cudaMemcpyDeviceToHost), "Read test KV page");
    constexpr int token_groups = page / mxfp8_group_size;
    for (std::size_t index = 0; index < bytes; ++index) {
        const auto scale_row = index / mxfp8_groups;
        const auto token = scales ? scale_row % token_groups * mxfp8_group_size +
            scale_row / token_groups % mxfp8_group_size : index / (bytes / page);
        require(data[index] == (token < std::size_t(rows) ? pattern(index, tag) : guard), message);
    }
}

void kv_page(Mxfp8KvCache bank, int physical, int logical, int layer, bool write,
             int partial_rows = 0) {
    const auto page = static_cast<std::size_t>(physical);
    const auto payload = mxfp8_payload_bytes_per_page;
    const auto scales = mxfp8_scale_bytes_per_page;
    const auto tag = logical < 0 ? 0u : 1u + (logical + layer * fixture_page_stride) * 4u;
    if (write) {
        fill(bank.key_payload + page * payload, payload, tag);
        fill(bank.value_payload + page * payload, payload, tag + 1);
        fill(bank.key_scales + page * scales, scales, tag + 2);
        fill(bank.value_scales + page * scales, scales, tag + 3);
        return;
    }
    if (partial_rows) {
        compare_page(bank.key_payload + page * payload, payload, tag, partial_rows, false,
                     "Key payload tail mismatch");
        compare_page(bank.value_payload + page * payload, payload, tag + 1, partial_rows, false,
                     "Value payload tail mismatch");
        compare_page(bank.key_scales + page * scales, scales, tag + 2, partial_rows, true,
                     "Key scale tail mismatch");
        compare_page(bank.value_scales + page * scales, scales, tag + 3, partial_rows, true,
                     "Value scale tail mismatch");
        return;
    }
    compare(bank.key_payload + page * payload, payload, tag, "Key payload mismatch");
    compare(bank.value_payload + page * payload, payload, tag ? tag + 1 : 0,
            "Value payload mismatch");
    compare(bank.key_scales + page * scales, scales, tag ? tag + 2 : 0, "Key scales mismatch");
    compare(bank.value_scales + page * scales, scales, tag ? tag + 3 : 0,
            "Value scales mismatch");
}

void guard_kv(Request& request) {
    for (const auto& cache : request.caches) {
        for (const auto plane : {cache.key_payload, cache.value_payload}) {
            check(cudaMemset(plane, guard, cache.page_count * mxfp8_payload_bytes_per_page),
                  "Set payload guard");
        }
        for (const auto plane : {cache.key_scales, cache.value_scales}) {
            check(cudaMemset(plane, guard, cache.page_count * mxfp8_scale_bytes_per_page),
                  "Set scale guard");
        }
    }
}

void install_map(const Request& request, const std::vector<std::int32_t>& mapping) {
    require(mapping.size() == static_cast<std::size_t>(request.pages.page_count),
            "Test mapping size changed");
    check(cudaMemcpy(const_cast<std::int32_t*>(request.pages.device_pages), mapping.data(),
                     mapping.size() * sizeof(mapping.front()), cudaMemcpyHostToDevice),
          "Install logical page map");
    check(cudaStreamSynchronize(nullptr), "Complete test page map");
}

std::vector<std::int32_t> page_map(Request& request, int shift, bool reverse) {
    std::vector<std::int32_t> result(request.pages.page_count);
    check(cudaStreamSynchronize(request.memory.stream()), "Complete request allocation");
    std::iota(result.begin(), result.end(), 0);
    if (reverse) {
        std::reverse(result.begin(), result.end());
    }
    std::rotate(result.begin(), result.begin() + shift, result.end());
    install_map(request, result);
    return result;
}

void check_map(const Request& request, const std::vector<std::int32_t>& expected) {
    std::vector<std::int32_t> observed(expected.size());
    check(cudaMemcpy(observed.data(), request.pages.device_pages, observed.size() *
                     sizeof(observed.front()), cudaMemcpyDeviceToHost), "Read logical page map");
    require(observed == expected, "Request page map changed");
}

void fill_kv(Request& request, const std::vector<std::int32_t>& mapping) {
    for (int layer = 0; layer < full_attention_layers; ++layer) {
        for (int logical = 0; logical < static_cast<int>(mapping.size()); ++logical) {
            kv_page(request.caches[layer], mapping[logical], logical, layer, true);
        }
    }
}

void check_kv(const Request& request, const std::vector<std::int32_t>& mapping, int matched,
              int tail = 0) {
    for (int layer = 0; layer < full_attention_layers; ++layer) {
        for (int logical = 0; logical < static_cast<int>(mapping.size()); ++logical) {
            const int partial = logical == matched ? tail : 0;
            kv_page(request.caches[layer], mapping[logical],
                    logical < matched || partial ? logical : -1, layer, false, partial);
        }
    }
}

void check_state(const Request& request, bool write, bool exact, bool recurrent = true) {
    const auto transfer = [write](void* address, std::size_t bytes, unsigned tag,
                                  const char* message) {
        if (write) {
            fill(address, bytes, tag);
        } else {
            compare(address, bytes, tag, message);
        }
    };
    transfer(request.states.front().history, gdn_layers * gdn_history_elements * sizeof(Bf16),
             recurrent ? 91 : 0, "GDN convolution history mismatch");
    transfer(request.states.front().recurrent, gdn_layers * gdn_state_elements * sizeof(float),
             recurrent ? 97 : 0, "GDN recurrent state mismatch");
    transfer(request.work.hidden, hidden_size * sizeof(Bf16), exact ? 105 : 0,
             "Hidden row mismatch");
    transfer(request.work.residual, hidden_size * sizeof(Bf16), exact ? 109 : 0,
             "Residual row mismatch");
    if (write) {
        check(cudaStreamSynchronize(nullptr), "Complete test state fill");
    }
}

void guard_state(Request& request) {
    check(cudaMemset(request.states.front().history, guard,
                     gdn_layers * gdn_history_elements * sizeof(Bf16)), "Set history guard");
    check(cudaMemset(request.states.front().recurrent, guard,
                     gdn_layers * gdn_state_elements * sizeof(float)), "Set recurrent guard");
    check(cudaMemset(request.work.hidden, guard, hidden_size * sizeof(Bf16)), "Set hidden guard");
    check(cudaMemset(request.work.residual, guard, hidden_size * sizeof(Bf16)),
          "Set residual guard");
    check(cudaStreamSynchronize(nullptr), "Complete guard initialization");
}

void restore_state(int tokens, bool remap_seed, bool remap_restored) {
    const int pages = (tokens + page - 1) / page;
    const std::vector<int> prompt(tokens, 17);
    const std::string_view salt("a\0b", 3);
    Request seed(0, (pages + 3) * page, std::shared_ptr<const void>{});
    Request restored(0, (pages + 5) * page, std::shared_ptr<const void>{});
    PrefixCache cache(0, 1);
    require(cache.stats().device_bytes == 0, "Unused cache allocated device memory");
    const auto source_map = page_map(seed, remap_seed ? 2 : 0, false);
    const auto destination_map = page_map(restored, remap_restored ? 1 : 0, remap_restored);
    const auto stream = restored.memory.stream();
    fill_kv(seed, source_map);
    check_state(seed, true, true);
    cache.store(seed, prompt, salt, tokens, stream);
    check(cudaStreamSynchronize(stream), "Complete checkpoint copy");
    check_map(seed, source_map);
    check_kv(seed, source_map, seed.pages.page_count);
    check_state(seed, false, true);
    guard_kv(seed);
    guard_state(seed);
    guard_kv(restored);
    guard_state(restored);
    require(cache.restore(restored, prompt, "a", stream) == 0, "Salt collision");
    auto mismatch = prompt;
    mismatch.back() = 42;
    require(cache.restore(restored, mismatch, salt, stream) == 0, "False last-token hit");
    auto short_prompt = prompt;
    short_prompt.pop_back();
    require(cache.restore(restored, short_prompt, salt, stream) == 0, "False short-prompt hit");
    check(cudaStreamSynchronize(stream), "Complete miss checks");
    check_kv(restored, destination_map, 0);
    check_state(restored, false, false, false);
    require(cache.stats().hits == 0, "Counted a miss as a hit");
    require(cache.restore(restored, prompt, salt, stream) == tokens, "Missing exact match");
    check(cudaStreamSynchronize(stream), "Complete exact restore");
    check_map(restored, destination_map);
    check_state(restored, false, true);
    check_kv(restored, destination_map, tokens / page, tokens % page);
    guard_kv(restored);
    guard_state(restored);
    auto suffix = prompt;
    suffix.push_back(41);
    require(cache.restore(restored, suffix, salt, stream) == tokens, "Missing suffix match");
    check(cudaStreamSynchronize(stream), "Complete suffix restore");
    check_state(restored, false, false);
    check_kv(restored, destination_map, tokens / page, tokens % page);
    cache.store(seed, suffix, salt, tokens, stream);
    require(cache.restore(restored, prompt, salt, stream) == tokens, "Lost repeated match");
    check(cudaStreamSynchronize(stream), "Complete repeated restore");
    check_state(restored, false, true);
    check_kv(restored, destination_map, tokens / page, tokens % page);
    const auto stats = cache.stats();
    require(stats.entries == 1 && stats.pages == pages && stats.stores == 1 &&
            stats.evictions == 0 && stats.hits == 3 &&
            stats.cached_tokens == 3 * static_cast<std::uint64_t>(tokens) &&
            stats.device_bytes == target_snapshot_bytes(pages), "Unexpected cache counters");
    cache.reset();
    const auto reset = cache.stats();
    require(reset.entries == 0 && reset.pages == 0 && reset.device_bytes == 0 &&
            reset.stores == stats.stores && reset.hits == stats.hits,
            "Reset did not release snapshots and retain counters");
}

void eviction() {
    Request request(0, 3 * page, std::shared_ptr<const void>{});
    fill_kv(request, page_map(request, 0, false));
    check_state(request, true, true);
    PrefixCache cache(0, 2);
    const std::vector<int> short_prompt(page, 17);
    const std::vector<int> long_prompt(2 * page, 17);
    cache.store(request, short_prompt, "shared", page, nullptr);
    cache.store(request, long_prompt, "shared", 2 * page, nullptr);
    require(cache.restore(request, long_prompt, "shared", nullptr) == 2 * page, "Not longest");
    cache.store(request, short_prompt, "other", page, nullptr);
    require(cache.restore(request, short_prompt, "shared", nullptr) == 0, "Evicted wrong entry");
    require(cache.restore(request, long_prompt, "shared", nullptr) == 2 * page, "Lost recent hit");
    require(cache.restore(request, short_prompt, "other", nullptr) == page, "Missing replacement");
    check(cudaDeviceSynchronize(), "Complete LRU checks");
    const auto stats = cache.stats();
    require(stats.stores == 3 && stats.entries == 2 && stats.pages == 3 && stats.evictions == 1,
            "Unexpected eviction counters");
}

template <typename Error, typename Operation>
void expect_throw(Operation operation, std::string_view message) {
    try {
        operation();
    } catch (const Error& error) {
        require(std::string_view(error.what()).find(message) != std::string_view::npos,
                "Unexpected guard diagnostic");
        return;
    }
    throw std::runtime_error("Expected test operation to fail");
}

void invalid_mappings() {
    Request request(0, 4 * page, std::shared_ptr<const void>{});
    const auto identity = page_map(request, 0, false);
    const auto stream = request.memory.stream();
    const std::vector<int> prompt(2 * page, 17);
    PrefixCache cache(0, 1);
    fill_kv(request, identity);
    check_state(request, true, true);
    cache.store(request, prompt, "resident", prompt.size(), stream);
    guard_kv(request);
    guard_state(request);
    const auto check_invalid = [&](std::string_view error) {
        expect_throw<std::invalid_argument>([&] {
            cache.store(request, prompt, "replacement", prompt.size(), stream);
        }, error);
        expect_throw<std::invalid_argument>([&] {
            cache.restore(request, prompt, "resident", stream);
        }, error);
    };
    const std::vector<std::vector<std::int32_t>> invalid{
        {-1, 1, 2, 3}, {4, 1, 2, 3}, {0, 0, 2, 3}, {0, 1, 0, 3}
    };
    for (std::size_t index = 0; index < invalid.size(); ++index) {
        install_map(request, invalid[index]);
        check_invalid(index == 3 ? "aliases a suffix page" : "Invalid prefix KV page mapping");
        check_map(request, invalid[index]);
    }
    install_map(request, identity);
    const auto table = request.pages;
    request.pages.device_pages = nullptr;
    check_invalid("Invalid prefix KV page capacity");
    request.pages = table;
    request.pages.page_count = 1;
    check_invalid("Invalid prefix KV page capacity");
    request.pages = table;
    expect_throw<std::invalid_argument>([&] {
        cache.store(request, prompt, "bad-size", prompt.size() + 1, stream);
    }, "Invalid prefix cache checkpoint");
    expect_throw<std::invalid_argument>([&] {
        cache.store(request, prompt, std::string(257, 'x'), page, stream);
    }, "Invalid prefix cache checkpoint");
    check(cudaStreamSynchronize(stream), "Complete guard checks");
    check_kv(request, identity, 0);
    check_state(request, false, false, false);
    const auto stats = cache.stats();
    require(stats.entries == 1 && stats.stores == 1 && stats.hits == 0 && stats.evictions == 0,
            "Invalid map changed cache state");
    require(cache.restore(request, prompt, "resident", stream) == 2 * page,
            "Invalid replacement erased the resident prefix");
    check(cudaStreamSynchronize(stream), "Complete intact prefix restore");
    check_kv(request, identity, 2);
    check_state(request, false, true);
}

void check_impossible_reclaim(PrefixCache& cache) {
    const auto before = cache.stats();
    free_bytes_override = decoder_shared_reserve;
    cache.reclaim_for_admission(before.device_bytes + 1, 0);
    free_bytes_override = 0;
    const auto after = cache.stats();
    require(after.entries == before.entries && after.device_bytes == before.device_bytes &&
            after.stores == before.stores && after.evictions == before.evictions,
            "Impossible admission reclaimed an unrelated snapshot");
}

void check_allocation_failure(PrefixCache& cache, const Request& seed, cudaStream_t stream) {
    const auto before = cache.stats();
    const std::vector<int> larger(3 * page, 18);
    failed_allocation_bytes = target_kv_bytes(3);
    cache.store(seed, larger, "larger", larger.size(), stream);
    require(failed_allocation_bytes == 0, "Snapshot allocation fault was not triggered");
    const auto after = cache.stats();
    require(after.entries == before.entries && after.device_bytes == before.device_bytes &&
            after.stores == before.stores && after.evictions == before.evictions,
            "Failed replacement allocation erased the resident snapshot");
}

void check_copy_error(const char* operation) {
    require(copy_budget == -1 && recorded_copy_error == rejected_copy_status &&
            (recorded_copy_error == cudaErrorInvalidValue ||
             recorded_copy_error == cudaErrorInvalidPitchValue),
            "Real Runtime copy fault was not triggered");
    check(cudaPeekAtLastError(), operation);
}

void copy_failures() {
    Request seed(0, 3 * page, std::shared_ptr<const void>{});
    Request request(0, 4 * page, std::shared_ptr<const void>{});
    const auto source = page_map(seed, 1, false);
    const auto destination = page_map(request, 1, true);
    const auto stream = request.memory.stream();
    const std::vector<int> prompt(2 * page, 17);
    PrefixCache cache(0, 1);
    fill_kv(seed, source);
    check_state(seed, true, true);
    cache.store(seed, prompt, "resident", prompt.size(), stream);
    check_impossible_reclaim(cache);
    check_allocation_failure(cache, seed, stream);
    guard_kv(request);
    guard_state(request);
    copy_budget = 1;
    expect_throw<std::runtime_error>([&] {
        cache.restore(request, prompt, "resident", stream);
    }, "Copy prefix KV layers");
    check_copy_error("Restore left a stale Runtime error");
    check_map(request, destination);
    check_state(request, false, false, false);
    for (int layer = 0; layer < full_attention_layers; ++layer) {
        for (std::size_t logical = 1; logical < destination.size(); ++logical) {
            kv_page(request.caches[layer], destination[logical], -1, layer, false);
        }
    }
    require(cache.stats().hits == 0 && cache.stats().cached_tokens == 0,
            "Interrupted restore counted a hit");
    require(cache.restore(request, prompt, "resident", stream) == 2 * page,
            "Interrupted restore corrupted the saved prefix");
    check(cudaStreamSynchronize(stream), "Complete retried restore");
    check_kv(request, destination, 2);
    recorded_copy_error = cudaSuccess;
    copy_budget = 1;
    expect_throw<std::runtime_error>([&] {
        cache.store(seed, prompt, "replacement", prompt.size(), stream);
    }, "Copy prefix KV layers");
    check_copy_error("Store left a stale Runtime error");
    const auto stats = cache.stats();
    require(stats.entries == 0 && stats.device_bytes == 0 && stats.stores == 1 &&
            stats.evictions == 1,
            "Interrupted store published a partial checkpoint");
    require(cache.restore(request, prompt, "resident", stream) == 0 &&
            cache.restore(request, prompt, "replacement", stream) == 0,
            "Interrupted checkpoint remained reachable");
    cache.store(seed, prompt, "replacement", prompt.size(), stream);
    require(cache.restore(request, prompt, "replacement", stream) == 2 * page,
            "Replacement could not reuse the interrupted entry");
    check(cudaStreamSynchronize(stream), "Complete store retry");
    check_kv(request, destination, 2);
    check_state(request, false, true);
}

}

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc == 2 ? argv[1] : "";
        require(argc == 1 || mode == "--identity" || mode == "--remapped" ||
                mode == "--full" || mode == "--guards",
                "Expected optional --identity, --remapped, --full, or --guards");
        require(PrefixCache::checkpoint(-1) == 0 && PrefixCache::checkpoint(127) == 127 &&
                PrefixCache::checkpoint(128) == 128 && PrefixCache::checkpoint(129) == 129 &&
                PrefixCache::checkpoint(8191) == 8191 && PrefixCache::checkpoint(8192) == 8192 &&
                PrefixCache::checkpoint(8193) == 8193 &&
                PrefixCache::checkpoint(262144) == 262144 &&
                PrefixCache::checkpoint(262145) == 262144, "Incorrect checkpoint limit");
        if (mode.empty() || mode == "--identity") {
            restore_state(page, false, false);
            eviction();
        }
        if (mode.empty() || mode == "--remapped") {
            restore_state(2 * page, true, false);
            restore_state(2 * page, false, true);
            restore_state(2 * page, true, true);
            for (const int tokens : {1, 31, 32, 43, 127, 129, 161}) {
                restore_state(tokens, true, true);
            }
        }
        if (mode == "--full") {
            restore_state(decoder_batch_prefix_checkpoint_interval, true, true);
        }
        if (mode.empty() || mode == "--guards") {
            invalid_mappings();
            copy_failures();
        }
        std::puts("prefix state passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
