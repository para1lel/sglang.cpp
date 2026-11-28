#include "sglang/decoder.h"
#include "sglang/decoder_batch.h"
#include "sglang/runtime/backend/tp_ar_counters.h"

#include <cstdlib>

namespace sglang {

struct Decoder::Model {};
#define SGLANG_TEST_TP_HEADER "sglang/runtime/execution/decoder_tp.h"
#if __has_include(SGLANG_TEST_TP_HEADER)
struct Decoder::TpRuntime {};
const decoder::TpRuntimeInfo* Decoder::tensor_parallel_info() const { return nullptr; }
#endif
Decoder::Decoder(const std::filesystem::path&, int) {}
Decoder::Decoder(const std::filesystem::path&, int, bool) {}
Decoder::Decoder([[maybe_unused]] const std::filesystem::path& checkpoint,
                 [[maybe_unused]] int kv_limit, [[maybe_unused]] bool mtp,
                 [[maybe_unused]] int tp_size, [[maybe_unused]] DecoderMemoryOptions options) {
}
Decoder::~Decoder() = default;
decoder::TpArCounters Decoder::tp_ar_counters() const { return {}; }

std::size_t DecoderBatch::request_device_allocation_bytes() const { return 0; }
DecoderPrefixCacheStats DecoderBatch::prefix_cache_stats() const { return {}; }
bool DecoderBatch::prefix_backup_pending() const {
    return prefix_cache_stats().pending_backups != 0;
}
DecoderBatchPrefixUpdate DecoderBatch::poll_prefix(int) { std::abort(); }
void DecoderBatch::release_retired_allocations() {}

}
