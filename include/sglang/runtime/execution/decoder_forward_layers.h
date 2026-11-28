#pragma once

#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_verify.h"
#include "sglang/text/gdn_local.h"
#include "sglang/text/gqa_verify_batch.h"
#include "sglang/text/target_fp8.h"
#include "sglang/runtime/execution/tp_verify_graph.h"

#include <algorithm>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace sglang {
namespace decoder {
namespace forward_detail {


void project(Bf16Weight weight, const Bf16* input, Bf16* output, cudaStream_t stream);

struct GraphLane {
    Request* request;
    const int* position;
    const int* rotary_position;
    TextGdnMode mode;
    static constexpr bool narrow_gqa = true;
};

DecodeFp8Projection decode_projection(Fp8Weight weight, Bf16* output);

struct LayerProjection {
    Fp8Weight weight;
    decltype(&Workspace::hidden) output;
};

template <typename Lane>
Workspace& workspace(const Lane& lane);

Workspace& workspace(const VerificationRow& row);

Workspace& workspace(const BatchGraphLane& lane);

void embed_forward_lanes(const Bf16* embedding, const ForwardLane* lanes, int count,
                         cudaStream_t stream, const char* operation);

static_assert(wide_mtp_rows == text_target_fp8_m40_rows);
static_assert(2 * wide_mtp_rows == text_target_fp8_m80_rows);
static_assert(max_wide_mtp_rows == text_target_fp8_m160_rows);

template <typename Lane>
struct LayerBatch {
    const Lane* lanes;
    int count;
    cudaStream_t stream;
    bool ordered_attention = false;
    bool target_tc = false;
    bool vector_attention = false;
    bool grouped_target_attention = false;
    int target_split_end = 0;
    std::uint8_t* packed_activation = nullptr;
    
    Bf16* const* attention_output = nullptr;
    Bf16* const* hidden_output = nullptr;
    
    bool tensor_parallel = false;
    bool singleton_verify_activation = false;
    const TextFp8Activation* projection_override = nullptr;

    int gqa_attention_width() const;
    int gdn_attention_width() const;
    int gdn_gate_heads() const;
    int intermediate_width() const;
    GqaGeometry gqa_geometry() const;

    Bf16* lane_output(int index, Bf16* Workspace::*output) const;
    LayerBatch verification_packet(int first) const;

    bool tensor_core_lane(int index) const;

    bool single_decode() const;
    bool batch_decode() const;

    bool fuse_verify_activation() const;

    bool fuse_wide_verify_norm() const;

    bool packs_projection_activation(int width) const;

    TextFp8Activation projection_activation(int index, int width) const;

    int projection_rows(int group) const;

    void normalize(const Bf16* weight, const char* operation,
                   Bf16* Workspace::* branch = nullptr) const;

    void quantize(Bf16* Workspace::*source, int width) const;

    void quantize_verification(Bf16* Workspace::*source, int width,
                               Bf16* Workspace::*up = nullptr) const;

    void quantize_wide_verification(Bf16* Workspace::*source, int width,
                                    Bf16* Workspace::*up = nullptr) const;

    bool project_fused_mlp(const Layer& layer) const;

    void project(Fp8Weight weight, Bf16* Workspace::*output) const;

    static bool target_shape(Fp8Weight weight);

    void project_group(std::initializer_list<LayerProjection> layers) const;

    template <int Capacity = text_target_fp8_max_rows>
    bool project_target_chunk(std::initializer_list<LayerProjection> layers, int first,
                              int rows, bool allow_fallback) const;

    void project_target(std::initializer_list<LayerProjection> layers) const;

    void project_verification_packets(std::initializer_list<LayerProjection> layers, int first,
                                      int size) const;

  private:
    Bf16* const* output_override(Bf16* Workspace::* output) const;
};

template <typename Lane>
void run_gdn(const GdnLayer& layer, const LayerBatch<Lane>& batch, int state_index);

void run_gdn(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch, int state_index);

template <typename Lane>
void run_gqa(const GqaLayer& layer, const LayerBatch<Lane>& batch, int cache_index);

template <typename Lane>
void run_mlp(const Layer& layer, const LayerBatch<Lane>& batch);

}
}
}
