#pragma once

#include "sglang/decoder.h"
#include "sglang/decoder_cuda.cuh"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace sglang {
class Checkpoint;
}

namespace sglang::decoder {

class MtpDraftWeights;
class MtpDraftRequest;
class PrefillWorkspace;
class TargetVerification;
struct VerificationSequence;

using Bf16 = __nv_bfloat16;

inline constexpr int hidden_size = 5120;
inline constexpr int intermediate_size = 17408;
inline constexpr int vocab_size = 248320;
inline constexpr int layer_count = 64;
inline constexpr int full_attention_interval = 4;
inline constexpr int full_attention_layers = layer_count / full_attention_interval;
inline constexpr int gdn_layers = layer_count - full_attention_layers;
inline constexpr int prefill_chunk_rows = text_embedding_max_rows;
inline constexpr std::size_t gdn_history_elements = text_gdn_qkv_width * text_gdn_history_taps;
inline constexpr std::size_t gdn_state_elements =
    text_gdn_value_heads * text_gdn_head_dim * text_gdn_head_dim;

constexpr bool is_full_attention(int layer_index) {
    return (layer_index + 1) % full_attention_interval == 0;
}

void check(cudaError_t status, const char* operation);
void validate_request(const std::vector<int>& prompt, const DecoderOptions& options,
                      bool allow_mtp = false);

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
    int winner = -1;
    std::vector<Bf16> logits;
    std::shared_ptr<void> verification_results;
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
    cudaGraphExec_t graph() const;
    void capture_graph(const std::function<void()>& enqueue);
    bool replay_batch_graph(int width, const std::function<void()>& prepare,
                            const std::function<void()>& capture,
                            const std::function<bool()>& bind);
    void reclaim_deferred();
    void open_stream(std::shared_ptr<const void> owners = nullptr);
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
    explicit Workspace(Allocations& allocations, Bf16* verification_attention = nullptr,
                       TextLogitCandidate* verification_winner = nullptr);

    ExactGqaWorkspace gqa_scores(int capacity) const;

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

using GdnBank = std::array<TextGdnState, gdn_layers>;
void allocate_gdn(Allocations& memory, GdnBank& states);

struct Request {
    explicit Request(int device, bool reusable = false);
    Request(int device, int capacity, std::shared_ptr<const void> owners, bool reusable = false);

    void initialize(int capacity, cudaStream_t stream);
    void reset(cudaStream_t stream);
    void reset();
    ExactGqaWorkspace batch_graph_scores();

    Allocations memory;
    Workspace work;
    int* decode_position;
    Mxfp8PageTable pages;
    std::array<Mxfp8KvCache, full_attention_layers> caches;
    GdnBank states;
    std::optional<ExactGqaWorkspace> long_graph_scores;
};

void copy_greedy_token(Request& request, cudaStream_t stream);
int read_greedy_token(Request& request);

struct ForwardLane {
    Request* request;
    int token_id;
    int position;
    TextGdnMode mode;
};

struct BatchDecodeBinding {
    TextLogitCandidate input;
    int position;
    ExactGqaWorkspace scores;
    Mxfp8PageTable pages;
    std::array<Mxfp8KvCache, full_attention_layers> caches;
    std::array<TextGdnState, gdn_layers> states;
};

struct BatchGraphLane {
    Workspace* work;
    const BatchDecodeBinding* binding;
};

}

namespace sglang {

struct Decoder::Model {
    struct MtpGeneration;

    Model(const std::filesystem::path& checkpoint, int cuda_device, Fp8PrefillPolicy fp8_prefill);

    Fp8PrefillPolicy fp8_prefill() const { return fp8_prefill_; }
    void forward(decoder::Request& request, int token_id, int position, TextGdnMode mode) const;
    void forward(const decoder::ForwardLane* lanes, int count, cudaStream_t stream) const;
    void forward_verify(decoder::TargetVerification& verification, int position,
                        const int* tokens, int count, cudaStream_t stream) const;
    void forward_verify(const decoder::VerificationSequence* sequences, int count,
                        cudaStream_t stream) const;
    void prefill_chunk(decoder::Request& request, const int* tokens, int rows, int position,
                       decoder::PrefillWorkspace& workspace,
                       decoder::MtpDraftRequest* draft = nullptr, int prompt_remaining = 0) const;
    void decode_graph(decoder::Request& request) const;
    void capture_batch_decode(const decoder::BatchGraphLane* lanes, int count,
                              cudaStream_t stream) const;
    void normalize_final(decoder::Workspace& work, cudaStream_t stream) const;
    void compute_logits(decoder::Request& request, bool decode = false) const;
    void compute_logits(decoder::Workspace& work, cudaStream_t stream, bool decode = false) const;
    void generate_serial(const std::vector<int>& prompt, const DecoderOptions& options,
                         const DecoderCallbacks& callbacks, DecoderResult& result) const;
    DecoderFinishReason generate_mtp(const std::vector<int>& prompt,
                                     const DecoderOptions& options,
                                     const DecoderCallbacks& callbacks,
                                     DecoderResult& result) const;

    template <typename Lane>
    void forward_layers(const Lane* lanes, int count, cudaStream_t stream) const;

    int device;
    std::shared_ptr<decoder::Allocations> memory;
    std::array<decoder::Layer, decoder::layer_count> layers{};
    const decoder::Bf16* embedding = nullptr;
    const decoder::Bf16* final_norm = nullptr;
    decoder::Bf16Weight lm_head{};
    std::shared_ptr<const decoder::MtpDraftWeights> mtp_weights;
    mutable std::atomic_flag request_in_use = ATOMIC_FLAG_INIT;
    mutable std::unique_ptr<decoder::Request> decode_request;

private:
    Model(const std::filesystem::path& checkpoint, int cuda_device);

    Fp8PrefillPolicy fp8_prefill_ = Fp8PrefillPolicy::exact;
};

}
