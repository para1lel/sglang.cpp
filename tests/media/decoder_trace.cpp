#include "sglang/decoder_cuda.cuh"
#include "sglang/text/prefill_attn_sm100.h"
#include "sglang/testing/media/decoder_fixture.h"
#include "sglang/testing/media/decoder_trace.h"

#include <iostream>

using namespace sglang;
using sglang::decoder::ExactGqaWorkspace;
using sglang::decoder::PairedPrefillScratch;

namespace {

bool tracing = false;
bool force_fallback = false;
int rotary_calls = 0;
int append_calls = 0;
int attention_calls = 0;
int fallback_calls = 0;
constexpr int layers = 16;
int mtp_prompt_rows = 0;
int seed_embedding_rows = 0;
int seed_key_rows = 0;
int draft_first_queries = 0;
int shifted_feature_rows = 0;
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

void observe_prefill(int position, int rows, int valid) {
    const auto [start, count] = geometry(attention_calls++);
    media_test::require(position == start && rows == count && valid == start + count &&
                            start < media_test::physical,
                        "Causal prefill used the wrong physical index or valid length");
}

void observe_decode(int position) {
    const auto [start, count] = geometry(attention_calls++);
    media_test::require(count == 1 && position == start && start >= media_test::physical,
                        "Causal decode used rotary position in place of physical cache index");
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

void media_test::start_mtp_trace(int prompt_rows) {
    mtp_prompt_rows = prompt_rows;
    seed_embedding_rows = seed_key_rows = draft_first_queries = shifted_feature_rows = 0;
}

void media_test::finish_mtp_trace() {
    require(seed_embedding_rows == mtp_prompt_rows - 1 &&
                seed_key_rows == mtp_prompt_rows - 1 && draft_first_queries == 1,
            "MTP did not rotate the whole media seed and complete the first draft query");
    int visual_rows = 0;
    for (const auto& span : spans) visual_rows += span.height * span.width;
    require(shifted_feature_rows == visual_rows, "MTP seed did not consume all shifted features");
    std::cout << "media draft seed keys=" << seed_key_rows
              << " shifted features=" << shifted_feature_rows << '\n';
    mtp_prompt_rows = 0;
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
template <class Position, class... Args>
using Rotate = cudaError_t(Position, int, const Bf16*, const Bf16*, const Bf16*, const Bf16*, Bf16*,
                           Bf16*, cudaStream_t, Args...);

extern "C" Rotate<const MropePosition*> real_mrope;
extern "C" Rotate<const MropePosition*> traced_mrope;
extern "C" Rotate<const MropePosition*, GqaGeometry> real_mrope_model;
extern "C" Rotate<const MropePosition*, GqaGeometry> traced_mrope_model;
extern "C" Rotate<int, GqaGeometry> real_rope;
extern "C" Rotate<int, GqaGeometry> traced_rope;

cudaError_t traced_mrope(const MropePosition* positions, int rows, const Bf16* query,
                         const Bf16* keys, const Bf16* qnorm, const Bf16* knorm,
                         Bf16* rotated_query, Bf16* rotated_keys, cudaStream_t stream) {
    if (mtp_prompt_rows != 0 && seed_key_rows == mtp_prompt_rows - 1 && rows == 1) {
        MropePosition actual;
        media_test::cuda(
            cudaMemcpyAsync(&actual, positions, sizeof(actual), cudaMemcpyDeviceToHost, stream));
        media_test::cuda(cudaStreamSynchronize(stream));
        media_test::require(media_test::equal(actual, media_test::axes(mtp_prompt_rows - 1)),
                            "First MTP proposal did not rotate the final prompt row");
        ++draft_first_queries;
    }
    return real_mrope(positions, rows, query, keys, qnorm, knorm, rotated_query, rotated_keys,
                      stream);
}

cudaError_t traced_mrope_model(const MropePosition* positions, int rows, const Bf16* query,
                               const Bf16* keys, const Bf16* qnorm, const Bf16* knorm,
                               Bf16* rotated_query, Bf16* rotated_keys, cudaStream_t stream,
                               GqaGeometry heads) {
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
    return real_mrope_model(positions, rows, query, keys, qnorm, knorm, rotated_query, rotated_keys,
                            stream, heads);
}

cudaError_t traced_rope(int position, int rows, const Bf16* query, const Bf16* keys,
                        const Bf16* qnorm, const Bf16* knorm, Bf16* rotated_query,
                        Bf16* rotated_keys, cudaStream_t stream, GqaGeometry heads) {
    if (tracing) {
        const auto [start, count] = geometry(rotary_calls++);
        media_test::require(start >= media_test::physical && rows == count &&
                                position == media_test::rotary + start - media_test::physical,
                            "Decode rotary position did not use the multimodal delta");
    }
    return real_rope(position, rows, query, keys, qnorm, knorm, rotated_query, rotated_keys, stream,
                     heads);
}

using Append = cudaError_t(Mxfp8KvCache, Mxfp8PageTable, int, int, const Bf16*, const Bf16*,
                           cudaStream_t, GqaGeometry);
extern "C" Append real_append;
extern "C" Append traced_append;

cudaError_t traced_append(Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
                          const Bf16* keys, const Bf16* values, cudaStream_t stream,
                          GqaGeometry heads) {
    if (tracing) {
        const auto [start, count] = geometry(append_calls++);
        media_test::require(position == start && rows == count && pages.page_count == 3,
                            "Cache append used the rotary index or the wrong physical tile");
    }
    return real_append(cache, pages, position, rows, keys, values, stream, heads);
}

using Attend = cudaError_t(Mxfp8KvCache, Mxfp8PageTable, int, int, int, const Bf16*, Bf16*,
                           cudaStream_t, GqaGeometry);
extern "C" Attend real_attend;
extern "C" Attend traced_attend;

cudaError_t traced_attend(Mxfp8KvCache cache, Mxfp8PageTable pages, int valid, int position,
                          int rows, const Bf16* query, Bf16* output, cudaStream_t stream,
                          GqaGeometry heads) {
    if (tracing) observe_prefill(position, rows, valid);
    return real_attend(cache, pages, valid, position, rows, query, output, stream, heads);
}

using AttendTensor = cudaError_t(Mxfp8KvCache, Mxfp8PageTable, int, int, const Bf16*, Bf16*,
                                 cudaStream_t, PairedPrefillScratch);
extern "C" AttendTensor real_attend_tensor;
extern "C" AttendTensor traced_attend_tensor;

cudaError_t traced_attend_tensor(Mxfp8KvCache cache, Mxfp8PageTable pages, int position, int rows,
                                 const Bf16* query, Bf16* output, cudaStream_t stream,
                                 PairedPrefillScratch scratch) {
    if (tracing) observe_prefill(position, rows, position + rows);
    return real_attend_tensor(cache, pages, position, rows, query, output, stream, scratch);
}

using Decode = cudaError_t(Mxfp8KvCache, Mxfp8PageTable, int, const Bf16*, Bf16*,
                           ExactGqaWorkspace, cudaStream_t, std::size_t, bool);
extern "C" Decode real_decode;
extern "C" Decode traced_decode;

cudaError_t traced_decode(Mxfp8KvCache cache, Mxfp8PageTable pages, int position, const Bf16* query,
                          Bf16* output, ExactGqaWorkspace workspace, cudaStream_t stream,
                          std::size_t bytes, bool prefill) {
    if (tracing) {
        if (prefill) observe_prefill(position, 1, position + 1);
        else observe_decode(position);
    }
    return real_decode(cache, pages, position, query, output, workspace, stream, bytes, prefill);
}

using DecodeVector = cudaError_t(Mxfp8KvCache, Mxfp8PageTable, int, const Bf16*, Bf16*,
                                 ExactGqaWorkspace, cudaStream_t);
extern "C" DecodeVector real_decode_vector;
extern "C" DecodeVector traced_decode_vector;

cudaError_t traced_decode_vector(Mxfp8KvCache cache, Mxfp8PageTable pages, int position,
                                 const Bf16* query, Bf16* output, ExactGqaWorkspace workspace,
                                 cudaStream_t stream) {
    if (tracing) observe_decode(position);
    return real_decode_vector(cache, pages, position, query, output, workspace, stream);
}

using RotateKeys = cudaError_t(const MropePosition*, int, const Bf16*, const Bf16*, Bf16*,
                               cudaStream_t);
extern "C" RotateKeys real_seed_keys;
extern "C" RotateKeys traced_seed_keys;

cudaError_t traced_seed_keys(const MropePosition* positions, int rows, const Bf16* keys,
                             const Bf16* weight, Bf16* output, cudaStream_t stream) {
    if (mtp_prompt_rows != 0) {
        std::vector<MropePosition> actual(rows);
        media_test::cuda(cudaMemcpyAsync(actual.data(), positions, rows * sizeof(actual[0]),
                                         cudaMemcpyDeviceToHost, stream));
        media_test::cuda(cudaStreamSynchronize(stream));
        for (int row = 0; row < rows; ++row) {
            media_test::require(media_test::equal(actual[row],
                                                  media_test::axes(seed_key_rows + row)),
                                "Draft seed key used shifted or scalar rotary coordinates");
        }
        seed_key_rows += rows;
    }
    return real_seed_keys(positions, rows, keys, weight, output, stream);
}

using SeedNorm = cudaError_t(const Bf16*, const Bf16*, const Bf16*, const Bf16*, int, int,
                             Bf16*, cudaStream_t);
extern "C" SeedNorm real_seed_norm;
extern "C" SeedNorm traced_seed_norm;

cudaError_t traced_seed_norm(const Bf16* embedding, const Bf16* hidden, const Bf16* embedding_norm,
                             const Bf16* hidden_norm, int width, int rows, Bf16* output,
                             cudaStream_t stream) {
    if (mtp_prompt_rows != 0) {
        media_test::require(width == media_test::width, "Unexpected MTP seed embedding width");
        std::vector<std::uint16_t> actual(std::size_t(rows) * width);
        media_test::cuda(cudaMemcpyAsync(actual.data(), embedding, actual.size() *
                                         sizeof(actual[0]), cudaMemcpyDeviceToHost, stream));
        media_test::cuda(cudaStreamSynchronize(stream));
        for (int row = 0; row < rows; ++row) {
            const auto [item, source_row] = media_test::feature(seed_embedding_rows + row + 1);
            if (item < 0) continue;
            for (int channel = 0; channel < width; ++channel) {
                media_test::require(actual[std::size_t(row) * width + channel] ==
                    media_test::sample(item, source_row, channel),
                    "Draft seed feature was not shifted");
            }
            ++shifted_feature_rows;
        }
        seed_embedding_rows += rows;
    }
    return real_seed_norm(embedding, hidden, embedding_norm, hidden_norm, width, rows, output,
                          stream);
}
