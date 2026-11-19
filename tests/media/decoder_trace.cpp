#include "sglang/decoder_cuda.cuh"
#include "sglang/testing/media/decoder_fixture.h"
#include "sglang/testing/media/decoder_trace.h"

using namespace sglang;
using sglang::decoder::ExactGqaWorkspace;

namespace {

bool tracing = false;
bool force_fallback = false;
int rotary_calls = 0;
int append_calls = 0;
int attention_calls = 0;
int fallback_calls = 0;
constexpr int layers = 16;
constexpr int decode_rows = 3;
constexpr int tile_rows = 128;

std::pair<int, int> geometry(int call) {
    const int tile = call / layers;
    if (force_fallback) {
        return {tile, 1};
    }
    return tile < 2 ? std::pair{tile * tile_rows, tile_rows} :
        tile == 2 ? std::pair{2 * tile_rows, media_test::physical - 2 * tile_rows} :
        std::pair{media_test::physical + tile - 3, 1};
}

}

void media_test::start_trace(bool fallback) {
    tracing = true;
    force_fallback = fallback;
    rotary_calls = append_calls = attention_calls = fallback_calls = 0;
}

void media_test::finish_trace() {
    tracing = false;
    const int calls = layers * ((force_fallback ? physical : 3) + decode_rows);
    require(rotary_calls == calls && append_calls == calls && attention_calls == calls,
            "Unexpected number of rotary, cache append, or cache read calls");
    require(fallback_calls == int(force_fallback), "Prefill allocation fault was not exercised");
}

extern "C" cudaError_t __real_cudaMallocFromPoolAsync(
    void**, std::size_t, cudaMemPool_t, cudaStream_t);

extern "C" cudaError_t __wrap_cudaMallocFromPoolAsync(
    void** output, std::size_t bytes, cudaMemPool_t pool, cudaStream_t stream) {
    if (tracing && force_fallback) {
        ++fallback_calls;
        *output = nullptr;
        return cudaErrorMemoryAllocation;
    }
    return __real_cudaMallocFromPoolAsync(output, bytes, pool, stream);
}

extern "C" cudaError_t __real_cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);

extern "C" cudaError_t __wrap_cudaStreamBeginCapture(cudaStream_t stream,
                                                    cudaStreamCaptureMode mode) {
    media_test::require(!tracing, "Media tried to capture a text decode graph");
    return __real_cudaStreamBeginCapture(stream, mode);
}

using Bf16 = __nv_bfloat16;
template <class Position>
using Rotate = cudaError_t(Position, int, const Bf16*, const Bf16*, const Bf16*, const Bf16*,
                           Bf16*, Bf16*, cudaStream_t);

extern "C" Rotate<const MropePosition*> real_mrope;
extern "C" Rotate<const MropePosition*> traced_mrope;
extern "C" Rotate<int> real_rope;
extern "C" Rotate<int> traced_rope;

cudaError_t traced_mrope(const MropePosition* positions, int rows, const Bf16* query,
                         const Bf16* keys, const Bf16* qnorm, const Bf16* knorm,
                         Bf16* rotated_query, Bf16* rotated_keys, cudaStream_t stream) {
    if (tracing) {
        const auto [start, count] = geometry(rotary_calls++);
        media_test::require(rows == count && start + rows <= media_test::physical,
                            "Multimodal rotary call crossed into decode or wrong chunk");
        std::array<MropePosition, tile_rows> actual;
        media_test::cuda(cudaMemcpyAsync(actual.data(), positions, rows * sizeof(actual[0]),
                                         cudaMemcpyDeviceToHost, stream));
        media_test::cuda(cudaStreamSynchronize(stream));
        for (int row = 0; row < rows; ++row) {
            media_test::require(media_test::equal(actual[row], media_test::axes(start + row)),
                                "Wrong rotary axis or tile offset");
        }
    }
    return real_mrope(positions, rows, query, keys, qnorm, knorm, rotated_query, rotated_keys,
                      stream);
}

cudaError_t traced_rope(int position, int rows, const Bf16* query, const Bf16* keys,
                        const Bf16* qnorm, const Bf16* knorm, Bf16* rotated_query,
                        Bf16* rotated_keys, cudaStream_t stream) {
    if (tracing) {
        const auto [start, count] = geometry(rotary_calls++);
        media_test::require(start >= media_test::physical && rows == count &&
                                position == media_test::rotary + start - media_test::physical,
                            "Decode rotary position did not use the multimodal delta");
    }
    return real_rope(position, rows, query, keys, qnorm, knorm, rotated_query, rotated_keys,
                     stream);
}

using Append = cudaError_t(Mxfp8KvCache, Mxfp8PageTable, int, int, const Bf16*, const Bf16*,
                           cudaStream_t);
extern "C" Append real_append;
extern "C" Append traced_append;

cudaError_t traced_append(Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
                          const Bf16* keys, const Bf16* values, cudaStream_t stream) {
    if (tracing) {
        const auto [start, count] = geometry(append_calls++);
        media_test::require(position == start && rows == count && pages.page_count == 3,
                            "Cache append used the rotary index or the wrong physical tile");
    }
    return real_append(cache, pages, position, rows, keys, values, stream);
}

using Attend = cudaError_t(Mxfp8KvCache, Mxfp8PageTable, int, int, int, const Bf16*, Bf16*,
                           cudaStream_t);
extern "C" Attend real_attend;
extern "C" Attend traced_attend;

cudaError_t traced_attend(Mxfp8KvCache cache, Mxfp8PageTable pages, int valid, int position,
                          int rows, const Bf16* query, Bf16* output, cudaStream_t stream) {
    if (tracing) {
        const auto [start, count] = geometry(attention_calls++);
        media_test::require(position == start && rows == count && valid == start + count &&
                                start < media_test::physical,
                            "Causal prefill used the wrong physical index or valid length");
    }
    return real_attend(cache, pages, valid, position, rows, query, output, stream);
}

using Decode = cudaError_t(Mxfp8KvCache, Mxfp8PageTable, int, const Bf16*, Bf16*,
                           ExactGqaWorkspace, cudaStream_t);
extern "C" Decode real_decode;
extern "C" Decode traced_decode;

cudaError_t traced_decode(Mxfp8KvCache cache, Mxfp8PageTable pages, int position, const Bf16* query,
                          Bf16* output, ExactGqaWorkspace workspace, cudaStream_t stream) {
    if (tracing) {
        const auto [start, count] = geometry(attention_calls++);
        media_test::require(count == 1 && position == start && start >= media_test::physical,
                            "Causal decode used rotary position in place of physical cache index");
    }
    return real_decode(cache, pages, position, query, output, workspace, stream);
}
