#include "sglang/mxfp8_gqa.h"
#include "sglang/text/gqa_verify_batch.h"
#include "sglang/text/prefill_attn_sm100.h"

#include <limits>

namespace sglang {

bool gqa_geometry_valid(GqaGeometry geometry) {
    return geometry.query_heads == mxfp8_queries_per_kv_head * geometry.kv_heads &&
           (geometry.kv_heads == mxfp8_kv_heads || geometry.kv_heads == mxfp8_kv_heads / 2);
}

bool gqa_geometry_matches(GqaGeometry geometry, const Mxfp8KvCache& cache) {
    return gqa_geometry_valid(geometry) && geometry.kv_heads == cache.kv_heads;
}

int gqa_head_pairs(const Mxfp8KvCache& cache) {
    return gqa_query_heads(cache.kv_heads) / 2;
}

namespace gqa_decode {

bool supports_eager_wide_split(int position) {
    constexpr int max_average_chunks_per_split = 10;
    const int chunks = position / split_chunk_tokens + 1;
    return chunks <= max_average_chunks_per_split * split_count(position);
}

bool valid_buffers(const decoder::PagedDecodeBuffers& buffers) {
    const auto scratch = reinterpret_cast<std::uintptr_t>(buffers.scratch);
    return buffers.queries && buffers.output && scratch && scratch % alignof(Partial) == 0;
}

}

namespace decoder {

bool cache_covers(Mxfp8KvCache cache, Mxfp8PageTable pages, std::int64_t token_count) {
    if (token_count <= 0 || token_count > std::numeric_limits<int>::max()) {
        return false;
    }
    const auto last_page = (token_count - 1) / mxfp8_page_size;
    return last_page < cache.page_count && last_page < pages.page_count && pages.device_pages
        && cache.key_payload && cache.value_payload && cache.key_scales && cache.value_scales;
}

bool wants_long_serial_staging(int position, int rows) {
    return rows >= staged_serial_min_rows && rows <= 1024 && position >= 0 &&
        position <= staged_serial_max_tokens - rows && position + rows > staged_serial_min_position;
}

}

}
