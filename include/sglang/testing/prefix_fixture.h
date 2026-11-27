#pragma once

#include "sglang/decoder_batch.h"
#include "sglang/runtime/cache/lmcache_client.h"

#include <array>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sglang::decoder {
struct Request;
struct Workspace;
}

namespace prefix_fixture {

std::string run_salt(std::string label);
sglang::DecoderBatchPrefixUpdate wait(sglang::DecoderBatch& batch, int slot);
sglang::decoder::LmCacheClient client();

struct Stores {
    std::vector<std::future<void>> receipts;

    void wait();
};

std::shared_ptr<Stores> attach(sglang::DecoderOptions& options,
        sglang::decoder::LmCacheClient cache, const std::vector<int>& prompt,
        std::string_view salt, std::function<void()> before_store = nullptr);

std::shared_ptr<sglang::decoder::PrefixCacheRead> read_checkpoint(
        const sglang::decoder::LmCacheClient& cache, const std::vector<int>& prompt,
        std::string_view salt, int endpoint, bool draft = false);

std::array<std::string, 4> recurrent_digest(const sglang::decoder::Request& request,
        const sglang::decoder::Workspace& row);

std::string target_digest(const sglang::decoder::Request& request, int end);

}
