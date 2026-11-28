#pragma once

#include "sglang/decoder_media.h"
#include "sglang/runtime/cache/lmcache_client.h"
#include "sglang/runtime/cache/mxfp8_cache_geometry.h"
#include "sglang/tokenizer/digest.h"


#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sglang::decoder {

struct PrefixCacheLayout;
class PrefixPrefillSnapshot;
inline constexpr int prefix_gqa_chunk_tokens = 256;
inline constexpr int prefix_prefill_retention_limit = 12;
enum class PrefixPayloadKind { gqa, gdn, draft };

struct PrefixCacheRead {
    int matched_gqa = 0;
    int matched_gdn = 0;
    int usable_gqa = 0;
    int usable_gdn = 0;
    int usable_draft = 0;
    std::vector<int> matched_gdn_endpoints;
    std::vector<int> draft_ready_gdn_endpoints;
    int matched_draft = 0;
};

struct PrefixDeviceAddress {
    std::uint32_t allocation = 0;
    std::uint64_t offset = 0;
};

struct PrefixDeviceMxfp8 {
    int begin = 0;
    int end = 0;
    int layers = 0;
    int page_count = 0;
    int kv_heads = mxfp8_kv_heads;
    std::array<PrefixDeviceAddress, 4> planes{};
    std::vector<std::int32_t> physical_pages;
};

struct PrefixDeviceSpan {
    PrefixDeviceAddress address;
    std::uint64_t bytes = 0;
};

struct PrefixDeviceGdn {
    int end = 0;
    std::array<PrefixDeviceSpan, 4> planes{};
};

struct PrefixDeviceLayout {
    PrefixDeviceMxfp8 gqa;
    PrefixDeviceMxfp8 draft;
    PrefixDeviceGdn gdn;
};

struct PrefixDeviceTransfer {
    LmCacheGpuTransfer transfer;
    PrefixDeviceLayout layout;
    std::shared_ptr<void> ready;
    std::shared_ptr<const void> store_owner;
    std::shared_ptr<PrefixPrefillSnapshot> restore_snapshot;
    bool prefill_checkpoint = false;
    std::shared_ptr<bool> attention_only_restore;
    std::array<int, prefix_prefill_retention_limit> retention_endpoints{};
    int retention_endpoint_count = 0;
    bool retention_requires_draft = false;
};

std::size_t prefix_payload_bytes(PrefixPayloadKind kind, int begin, int end);
std::size_t prefix_payload_bytes(PrefixPayloadKind kind, int begin, int end,
                                 const PrefixCacheLayout& layout);

class PrefixCacheKeys {
public:
    PrefixCacheKeys(const std::vector<int>& prompt, std::string_view salt,
                    const std::vector<DecoderMedia>* media = nullptr,
                    const PrefixCacheLayout* layout = nullptr);
    explicit PrefixCacheKeys(std::shared_ptr<const PrefixCacheKeys> prefix);

    int tokens() const;
    int cacheable_tokens() const;
    std::string key(int end) const;
    void append(int token);
    bool matches_rank(int world_size, int rank) const;

  private:
    const tokenizer::Sha256Digest& digest(int end) const;
    void append_row(int token, MropePosition position, std::string_view media, int feature_row);

    std::shared_ptr<const PrefixCacheKeys> prefix_;
    int prefix_size_ = 0;
    std::vector<tokenizer::Sha256Digest> keys_;
    int tokens_ = 0;
    int next_position_ = 0;
    int world_size_ = 1;
    int rank_ = 0;
};

}
