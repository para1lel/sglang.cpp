#pragma once

#include "sglang/decoder.h"
#include "sglang/text_attention.h"
#include "sglang/text_ops.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sglang {
class Checkpoint;
}

namespace sglang::decoder {

using Bf16 = __nv_bfloat16;

inline constexpr int hidden_size = 5120;
inline constexpr int intermediate_size = 17408;
inline constexpr int vocab_size = 248320;
inline constexpr int layer_count = 64;
inline constexpr int full_attention_interval = 4;
inline constexpr int full_attention_layers = layer_count / full_attention_interval;
inline constexpr int gdn_layers = layer_count - full_attention_layers;
inline constexpr std::size_t gdn_history_elements = text_gdn_qkv_width * text_gdn_history_taps;
inline constexpr std::size_t gdn_state_elements =
    text_gdn_value_heads * text_gdn_head_dim * text_gdn_head_dim;

constexpr bool is_full_attention(int layer_index) {
    return (layer_index + 1) % full_attention_interval == 0;
}

void check(cudaError_t status, const char* operation);

class DeviceScope {
public:
    explicit DeviceScope(int device);
    ~DeviceScope();
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

    void restore();

private:
    int previous_ = 0;
    bool changed_ = false;
};

struct TransferBuffers {
    std::unique_ptr<Checkpoint> checkpoint;
    std::vector<std::int32_t> logical_pages;
    TextLogitCandidate winner{};
    std::vector<Bf16> logits;
};

class Allocations {
public:
    explicit Allocations(int device);
    ~Allocations();
    Allocations(const Allocations&) = delete;
    Allocations& operator=(const Allocations&) = delete;

    template <typename Element>
    Element* allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element)) {
            throw std::overflow_error("CUDA allocation size overflow");
        }
        return static_cast<Element*>(allocate_bytes(count * sizeof(Element)));
    }

    std::size_t bytes() const;
    TransferBuffers& transfers();
    cudaStream_t stream() const;
    void open_stream(std::shared_ptr<const Allocations> weights = nullptr);
    void close_stream();
    void close();

private:
    struct State;
    void* allocate_bytes(std::size_t bytes);

    std::unique_ptr<State> state_;
};

struct Fp8Weight {
    const std::uint8_t* values;
    const Bf16* scales;
    int rows;
    int columns;
};

struct Bf16Weight {
    const Bf16* values;
    int rows;
    int columns;
};

struct GdnLayer {
    Fp8Weight qkv;
    Fp8Weight z;
    Bf16Weight b;
    Bf16Weight a;
    TextGdnWeights recurrent;
    Fp8Weight output;
};

struct GqaLayer {
    Fp8Weight q_gate;
    Fp8Weight key;
    Fp8Weight value;
    const Bf16* query_norm;
    const Bf16* key_norm;
    Fp8Weight output;
};

struct Layer {
    const Bf16* input_norm;
    const Bf16* attention_norm;
    Fp8Weight mlp_gate;
    Fp8Weight mlp_up;
    Fp8Weight mlp_down;
    GdnLayer gdn;
    GqaLayer gqa;
};

struct Workspace {
    explicit Workspace(Allocations& allocations);

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
    Bf16* logits;
    TextFp8Activation activation;
    TextLogitCandidate* argmax_scratch;
    TextLogitCandidate* argmax;
};

struct Request {
    Request(int device, int capacity, std::shared_ptr<const Allocations> weights);

    Allocations memory;
    Workspace work;
    Mxfp8PageTable pages;
    std::array<Mxfp8KvCache, full_attention_layers> caches;
    std::array<TextGdnState, gdn_layers> states;
};

}

namespace sglang {

struct Decoder::Model {
    Model(const std::filesystem::path& checkpoint, int cuda_device);

    void forward(decoder::Request& request, int token_id, int position, TextGdnMode mode) const;
    void compute_logits(decoder::Request& request) const;

    int device;
    std::shared_ptr<decoder::Allocations> memory;
    std::array<decoder::Layer, decoder::layer_count> layers{};
    const decoder::Bf16* embedding = nullptr;
    const decoder::Bf16* final_norm = nullptr;
    decoder::Bf16Weight lm_head{};
};

}
