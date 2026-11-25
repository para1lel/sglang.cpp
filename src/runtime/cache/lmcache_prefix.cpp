#include "sglang/runtime/cache/lmcache_client.h"

#include "sglang/runtime/cache/mxfp8_cache_geometry.h"
#include "sglang/runtime/cache/prefix_payload.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sglang::decoder {
namespace {

constexpr int chunk = prefix_gqa_chunk_tokens;
constexpr auto payload_row = mxfp8_payload_bytes_per_page / mxfp8_page_size;
constexpr auto scale_row = mxfp8_scale_bytes_per_page / mxfp8_page_size;

LmCacheKey record_key(const PrefixCacheKeys& keys, LmCacheFamily family, int end) {
    return {family, keys.key(end + (family == LmCacheFamily::draft))};
}

void append_scale_tail(LmCacheGpuRecord& record, PrefixDeviceAddress plane,
                       std::uint64_t device, std::uint64_t offset, int tokens) {
    constexpr auto lane_bytes = mxfp8_page_size / mxfp8_group_size * mxfp8_groups;
    constexpr auto head_bytes = mxfp8_page_size * mxfp8_groups;
    constexpr auto lanes = mxfp8_group_size * mxfp8_kv_heads;
    const auto complete = std::uint64_t(tokens / mxfp8_group_size * mxfp8_groups);
    const auto tail = static_cast<std::uint32_t>(tokens % mxfp8_group_size);
    if (complete) {
        record.regions.push_back({plane.allocation, device, lane_bytes, offset, complete,
                                  complete, lanes});
        offset += complete * lanes;
    }
    if (tail) {
        for (int head = 0; head < mxfp8_kv_heads; ++head) {
            record.regions.push_back({plane.allocation, device + head * head_bytes + complete,
                lane_bytes, offset, mxfp8_groups, mxfp8_groups, tail});
            offset += tail * mxfp8_groups;
        }
    }
}

void append_mxfp8_record(std::vector<LmCacheGpuRecord>& records, const PrefixCacheKeys& keys,
                         LmCacheFamily family, const PrefixDeviceMxfp8& view, int begin, int end) {
    const auto count = static_cast<std::uint32_t>(end - begin);
    const auto layers = static_cast<std::uint32_t>(view.layers);
    const auto payload_width = count * payload_row;
    const auto scale_width = count * scale_row;
    const auto payload_pitch = view.page_count * mxfp8_payload_bytes_per_page;
    const auto scale_pitch = view.page_count * mxfp8_scale_bytes_per_page;
    const auto full_scale = count / mxfp8_page_size * mxfp8_scale_bytes_per_page;
    LmCacheGpuRecord record{record_key(keys, family, end), 0, {}, count};
    for (int index = 0; index < 2; ++index) {
        const auto plane = view.planes[index];
        record.regions.push_back({plane.allocation, plane.offset + begin * payload_row,
            payload_pitch, record.bytes, payload_width, payload_width, layers});
        record.bytes += payload_width * layers;
    }
    for (int index = 2; index < 4; ++index) {
        const auto plane = view.planes[index];
        const auto first = plane.offset + begin * scale_row;
        if (full_scale) {
            record.regions.push_back({plane.allocation, first, scale_pitch, record.bytes,
                                       scale_width, full_scale, layers});
        }
        if (const int tail = count % mxfp8_page_size) {
            for (std::uint32_t layer = 0; layer < layers; ++layer) {
                append_scale_tail(record, plane, first + layer * scale_pitch + full_scale,
                                  record.bytes + layer * scale_width + full_scale, tail);
            }
        }
        record.bytes += scale_width * layers;
    }
    records.emplace_back(std::move(record));
}

void append_mxfp8(std::vector<LmCacheGpuRecord>& records, const PrefixCacheKeys& keys,
                  LmCacheFamily family, const PrefixDeviceMxfp8& view) {
    if (view.begin == view.end) return;
    if (view.begin < 0 || view.begin >= view.end || view.begin % chunk != 0 ||
        view.layers <= 0 || view.page_count <= 0 ||
        view.end > std::int64_t(view.page_count) * mxfp8_page_size) {
        throw std::invalid_argument("Invalid native MXFP8 cache extent");
    }
    for (int begin = view.begin; begin < view.end; begin += chunk) {
        append_mxfp8_record(records, keys, family, view, begin, std::min(view.end, begin + chunk));
    }
}

void append_gdn(std::vector<LmCacheGpuRecord>& records, const PrefixCacheKeys& keys,
                const PrefixDeviceGdn& view) {
    if (!view.end) return;
    LmCacheGpuRecord record{record_key(keys, LmCacheFamily::gdn, view.end), 0, {}};
    for (auto region : view.planes) {
        region.record_offset = record.bytes;
        record.bytes += region.width;
        record.regions.push_back(region);
    }
    records.emplace_back(std::move(record));
}

LmCacheGpuTransfer native_transfer(const PrefixCacheKeys& keys, const PrefixDeviceTransfer& view) {
    LmCacheGpuTransfer transfer{view.transfer.device, view.transfer.ready_event,
                               view.transfer.allocations, {}};
    append_mxfp8(transfer.records, keys, LmCacheFamily::gqa, view.layout.gqa);
    append_mxfp8(transfer.records, keys, LmCacheFamily::draft, view.layout.draft);
    append_gdn(transfer.records, keys, view.layout.gdn);
    return transfer;
}

}  // namespace

LmCachePrefix LmCacheClient::lookup_prefix(const PrefixCacheKeys& keys, int max_restore,
                                           bool draft) const {
    const int limit = keys.cacheable_tokens();
    if (limit <= 0) return {};
    std::string digests;
    digests.reserve(std::size_t(limit) * 64);
    for (int end = 1; end <= limit; ++end) digests += keys.key(end);
    return lookup_prefix(digests, std::clamp(max_restore, 0, limit), draft);
}

LmCacheStatus LmCacheClient::store(const PrefixCacheKeys& keys,
                                    const PrefixDeviceTransfer& transfer,
                                    LmCacheReceipt* receipt) const {
    return store(native_transfer(keys, transfer), receipt);
}

LmCacheStatus LmCacheClient::retrieve(const PrefixCacheKeys& keys,
                                       const PrefixDeviceTransfer& transfer,
                                       LmCacheReceipt* receipt) const {
    return retrieve(native_transfer(keys, transfer), receipt);
}

}  // namespace sglang::decoder
