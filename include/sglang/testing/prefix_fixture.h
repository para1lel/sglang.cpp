#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/decoder_internal.h"
#include "sglang/decoder_prefix_cache.h"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <thread>

namespace prefix_fixture {

inline sglang::DecoderBatchPrefixUpdate wait(sglang::DecoderBatch& batch, int slot) {
    using namespace std::chrono_literals;
    const auto deadline = std::chrono::steady_clock::now() + 20s;
    while (std::chrono::steady_clock::now() < deadline) {
        auto update = batch.poll_prefix(slot);
        if (!update.pending) return update;
        std::this_thread::sleep_for(100us);
    }
    throw std::runtime_error("Prefix transfer timed out");
}

inline sglang::decoder::LmCacheClient client() {
    using namespace sglang::decoder;
    const auto* socket = std::getenv("LMCACHE_SOCKET");
    if (!socket || !*socket) throw std::runtime_error("Set LMCACHE_SOCKET to the test daemon");
    LmCacheClient cache({socket, "sglang", 20000});
    if (!cache.ready()) throw std::runtime_error("Test LMCache daemon is not ready");
    return cache;
}

inline void attach(sglang::DecoderOptions& options, sglang::decoder::LmCacheClient cache,
                   const std::vector<int>& prompt, std::string_view salt,
                   std::function<void()> before_store = nullptr) {
    using namespace sglang::decoder;
    options.submit_prefix_transfer = [cache = std::move(cache), before_store,
                                      keys = PrefixCacheKeys(prompt, salt)](
            const PrefixDeviceTransfer& view, bool restore, const int* generated, int count) {
        auto current = keys;
        for (int index = 0; index < count; ++index) current.append(generated[index]);
        return std::async(std::launch::async, [cache, current = std::move(current), &view,
                                                restore, before_store] {
            if (!restore && before_store) before_store();
            const auto status = restore ? cache.retrieve(current, view)
                                        : cache.store(current, view);
            if (status != LmCacheStatus::complete) {
                throw std::runtime_error("Test LMCache transfer did not complete");
            }
            return true;
        }).share();
    };
}

inline std::shared_ptr<sglang::decoder::PrefixCacheRead> read_checkpoint(
        const sglang::decoder::LmCacheClient& cache, const std::vector<int>& prompt,
        std::string_view salt, int endpoint, bool draft = false) {
    using namespace sglang::decoder;
    const auto match = cache.lookup_prefix(PrefixCacheKeys(prompt, salt), endpoint, draft);
    if (match.gqa < endpoint || match.gdn != endpoint || (draft && match.draft != endpoint - 1)) {
        throw std::runtime_error("Missing test checkpoint at expected position");
    }
    return std::make_shared<PrefixCacheRead>(PrefixCacheRead{
        match.gqa, match.matched_gdn, match.gqa, match.gdn, match.draft});
}

// Only the test oracle reads device bytes. Cache transfer and restore still use the daemon.
inline std::string digest_region(const void* source, std::size_t size) {
    std::string bytes(size, '\0');
    sglang::decoder::check(cudaMemcpy(bytes.data(), source, size, cudaMemcpyDeviceToHost),
                           "Read test device state");
    return sglang::tokenizer::sha256(bytes);
}

inline std::array<std::string, 4> recurrent_digest(const sglang::decoder::Request& request,
        const sglang::decoder::Workspace& row) {
    using namespace sglang::decoder;
    const auto state = request.states.front();
    return {digest_region(state.history, gdn_layers * gdn_history_elements * sizeof(Bf16)),
            digest_region(state.recurrent, gdn_layers * gdn_state_elements * sizeof(float)),
            digest_region(row.hidden, hidden_size * sizeof(Bf16)),
            digest_region(row.residual, hidden_size * sizeof(Bf16))};
}

inline std::string target_digest(const sglang::decoder::Request& request, int end) {
    using namespace sglang;
    using namespace sglang::decoder;
    constexpr auto payload_row = mxfp8_payload_bytes_per_page / mxfp8_page_size;
    const int pages = end / mxfp8_page_size;
    const int tail = end % mxfp8_page_size;
    std::string digests;
    for (const auto& cache : request.caches) {
        const auto payload = pages * mxfp8_payload_bytes_per_page + tail * payload_row;
        const auto scales = pages * mxfp8_scale_bytes_per_page;
        for (const auto* plane : {cache.key_payload, cache.value_payload}) {
            digests += digest_region(plane, payload);
        }
        for (const auto* plane : {cache.key_scales, cache.value_scales}) {
            digests += digest_region(plane, scales);
            if (!tail) continue;
            std::array<std::uint8_t, mxfp8_scale_bytes_per_page> page;
            check(cudaMemcpy(page.data(), plane + scales, page.size(), cudaMemcpyDeviceToHost),
                  "Read test target scale tail");
            for (int head = 0; head < mxfp8_kv_heads; ++head) {
                for (int token = 0; token < tail; ++token) {
                    const int lane = head * mxfp8_group_size + token % mxfp8_group_size;
                    const int offset = (lane * (mxfp8_page_size / mxfp8_group_size) +
                                        token / mxfp8_group_size) * mxfp8_groups;
                    digests.append(reinterpret_cast<const char*>(page.data() + offset),
                                   mxfp8_groups);
                }
            }
        }
    }
    return tokenizer::sha256(digests);
}

}
