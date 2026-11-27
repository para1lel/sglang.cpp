#pragma once


#include "sglang/decoder_prefill.h"
#include "sglang/decoder_batch.h"
#include "sglang/decoder_media_internal.h"
#include "sglang/decoder_mtp.h"
#include "sglang/prefill_fp8.h"
#include "sglang/runtime/execution/decoder_prefill_plan.h"
#include "sglang/text/decoder_logprobs.h"
#include "sglang/text/gdn_local.h"
#include "sglang/text/prefill_attn_sm100.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace sglang {

std::optional<cudaError_t> launch_partial_staged_tmem_prefill(
    const decoder::PagedPrefillSequence*, int, cudaStream_t, decoder::PairedPrefillScratch);
std::optional<cudaError_t> launch_causal_group_staged_tmem_prefill(
    const decoder::PagedPrefillSequence&, cudaStream_t, decoder::PairedPrefillScratch);

namespace decoder::prefill_detail {


struct ScratchLayout {
    std::byte* data = nullptr;
    std::size_t bytes = 0;

    Bf16* take_bf16(std::size_t count);
};

struct PrefillDims {
    int q_gate_width = text_gqa_q_gate_width;
    int kv_width = text_gqa_kv_width;
    int q_width = text_gqa_q_width;
    int gdn_qkv_width = text_gdn_qkv_width;
    int gdn_value_width = text_gdn_value_width;
    int gdn_heads = text_gdn_value_heads;
    int intermediate = intermediate_size;

    static PrefillDims from_layers(const std::array<Layer, layer_count>& layers);

    void validate() const;

    int attention_width() const;
    bool is_local_gdn() const;
    GqaGeometry gqa_geometry() const;
    std::size_t gdn_workspace_bytes(int rows) const;
};

struct SeedScratch {
    Bf16* concatenated = nullptr;
    Bf16* keys = nullptr;
    Bf16* values = nullptr;
    Bf16* rope_keys = nullptr;
};

std::size_t seed_scratch_bytes(int concat_rows, int kv_rows);

SeedScratch make_seed_scratch(Bf16* memory, int concat_rows, int kv_rows);

struct TileBuffers {
    TileBuffers(ScratchLayout& layout, int rows, const PrefillDims& tile_dims = PrefillDims{});

    PairedPrefillScratch gqa_scratch() const;

    TileBuffers attention_segment(int offset) const;

    PrefillDims dims;
    Bf16* hidden;
    Bf16* normalized;
    Bf16* residual;
    Bf16* projected_attention;
    Bf16* attention;
    Bf16* mlp_gate;
    Bf16* mlp_up;
    Bf16* q_gate;
    Bf16* keys;
    Bf16* values;
    Bf16* rope_queries;
    Bf16* rope_keys;
    Bf16* gdn_qkv;
    Bf16* gdn_z;
    Bf16* gdn_b;
    Bf16* gdn_a;
    Bf16* gdn_scratch;
    TextFp8Activation activation;
};

struct Chunk : PrefillProjections {
    TileBuffers work;
    int position;
    const MropePosition* mrope;
    int cached_gqa_tokens = 0;
    PairedPrefillScratch long_staging{};

    bool project_packed(TextFp8ActivationView input,
                        std::initializer_list<PrefillFp8Projection> projections) const;

    void seed_draft(MtpDraftRequest& draft, const int* shifted_tokens, int seed_rows,
                    const Bf16* final_norm, bool wmma, const MediaPrefill* media,
                    const SeedScratch* seed = nullptr) const;

    void gdn_input(const GdnLayer& layer) const;

    void gdn_recur(const GdnLayer& layer, TextGdnState state, int bank,
                   TextFp8Activation quantized) const;

    void gdn_attention(const GdnLayer& layer, TextGdnState state, int bank) const;

    void gqa_input(const GqaLayer& layer, bool reuse_all = false) const;

    void gqa_append(const GqaLayer& layer, Mxfp8KvCache cache, Mxfp8PageTable pages) const;

    void gqa_gate() const;

    void gqa_read(Mxfp8KvCache cache, Mxfp8PageTable pages, PairedPrefillScratch scratch) const;

    void gqa_attention(const GqaLayer& layer, Mxfp8KvCache cache, Mxfp8PageTable pages,
                       PairedPrefillScratch scratch, bool postgate = true) const;

    void project_scaled(Fp8Weight weight, TextFp8ActivationView input, Bf16* output,
                        bool eligible) const;

    void mlp(const Layer& layer) const;

    void normalize(const Bf16* branch, Bf16* residual, const Bf16* weight,
                   RmsQuantizeLaunch launch = RmsQuantizeLaunch::ordinary) const;

    void normalize_tp(const Bf16* branch, Bf16* residual, const Bf16* weight,
                      RmsQuantizeLaunch launch, bool save_normalized) const;
};

void gather_embeddings(const Bf16* embedding, const int* tokens, int rows, Bf16* residual,
                       cudaStream_t stream);

void stage_prefill_row(Request& request, const TileBuffers& work, int row, cudaStream_t stream);

void capture_prefill_scores(const DecoderModel& model, const PrefillGrant& grant,
                            const TileBuffers* buffers, int offset, cudaStream_t stream);

constexpr int layerwise_row_alignment = 128;
constexpr int max_layerwise_rows = 1023;
static_assert(max_layerwise_rows / 256 <= max_gdn_prefill_segments);
static_assert(text_gqa_kv_width * sizeof(Bf16) >= text_gdn_value_heads * sizeof(float2));

PrefillGroup prefill_group(const PrefillGrant* grants, int count);

Chunk prefill_attention_segment(const Chunk& chunk, const PrefillGrant& grant, int offset);

void prefill_group_gdn(const Chunk& chunk, const GdnLayer& layer, int bank,
                       const PrefillGrant* grants, int count, int offset);

bool is_full_partial_prefill_group(const PrefillGrant* grants, PrefillGroup group);

void prefill_partial_gdn(const Chunk& chunk, const GdnLayer& layer, int bank,
                         const PrefillGrant* grants, int count, bool grouped);

void prefill_serial_gqa(const Chunk& chunk, const GqaLayer& layer, int bank,
                        const PrefillGrant* grants, int count, int serial_capacity,
                        bool causal, bool postgate);

}
}
