#pragma once

#include "sglang/decoder_internal.h"

namespace sglang::decoder {

struct DraftOperatorCapabilities;
struct DraftDepthScratch;
struct PrefillProjections;
struct PrefixCacheRead;
struct PrefixCacheLayout;
struct PrefixDeviceTransfer;

inline constexpr int draft_projection_rows = 16;

struct MtpPrefillBuffers {
    Bf16* normalized_hidden;
    Bf16* embedding;
    Bf16* concatenated;
    Bf16* layer_input;
    Bf16* keys;
    Bf16* values;
    Bf16* rope_keys;
};

struct MtpDraftInput {
    MtpDraftRequest* request;
    int position;
    int token;
    const Bf16* hidden;
    const MropePosition* mrope = nullptr;
    int rotary_position = -1;
};

using DraftRepairs = std::array<MtpDraftInput, decoder_batch_max_mtp_steps>;

class MtpDraftWeights {
public:
    MtpDraftWeights(const std::filesystem::path& checkpoint, int device,
                    std::shared_ptr<const Allocations> target_weights, const Bf16* embedding,
                    Bf16Weight lm_head, DraftOperatorCapabilities capabilities);
    ~MtpDraftWeights();

    std::size_t device_allocation_bytes() const;
    std::shared_ptr<const TcVocabWeights> shared_vocab() const;

private:
    struct State;
    std::unique_ptr<const State> state_;
    friend class MtpDraftRequest;
};

class MtpDraftRequest {
public:
    MtpDraftRequest(std::shared_ptr<const MtpDraftWeights> weights, int capacity,
                    RequestBlockCache* cache = nullptr);
    ~MtpDraftRequest();

    const Bf16* enqueue(int position, int shifted_token, const Bf16* hidden, bool score,
                        cudaStream_t stream, const ConstraintWorkspace* scratch = nullptr,
                        const ConstraintCandidates* allowed = nullptr,
                        const MropePosition* mrope = nullptr, int rotary_position = -1,
                        const Bf16* feature = nullptr);
    static void enqueue_wave(MtpDraftInput* rows, int count, bool score, cudaStream_t stream);
    static bool prepare_depth(MtpDraftRequest& lane, DraftDepthScratch& scratch,
                              std::uint8_t* activation, cudaStream_t stream);
    static void enqueue_depth(MtpDraftInput* rows, int count, const DraftDepthScratch& scratch,
                              cudaStream_t stream);
    static void repair_wave(const DraftRepairs* repairs, int count, Bf16* packed,
                            cudaStream_t stream);

    void seed_prefill(int position, const int* shifted_tokens, const Bf16* hidden,
                      const Bf16* residual, const Bf16* final_norm,
                      const PrefillProjections& projections, MtpPrefillBuffers buffers,
                      const MediaPrefill* media = nullptr);

    int read_token() const;
    bool retry_invalid_int8(cudaStream_t stream);

    int valid_prefix() const;
    void rewind(int valid_tokens);
    std::size_t device_allocation_bytes() const;
    void recycle_after_completion();

private:
    const Bf16* enqueue_impl(int position, int shifted_token, const Bf16* hidden, bool score,
                            cudaStream_t stream, const ConstraintWorkspace* scratch,
                            const ConstraintCandidates* allowed, bool defer_score,
                            const MropePosition* mrope, int rotary_position, const Bf16* feature);
    static void enqueue_wave_impl(MtpDraftInput* rows, int count, bool score,
                                  cudaStream_t stream, bool defer_score, Bf16* packed = nullptr);
    friend PrefixDeviceTransfer restore_prefix_view(Request&, const PrefixCacheRead&,
                                                    cudaStream_t, MtpDraftRequest*,
                                                    const PrefixCacheLayout*);
    friend PrefixDeviceTransfer export_prefix_view(const Request&, int, int, int, cudaStream_t,
                                                   const MtpDraftRequest*, const Workspace*, bool,
                                                   const PrefixCacheLayout*);
    friend int commit_restored_prefix(Request&, const PrefixDeviceTransfer&, MtpDraftRequest*);
    Mxfp8KvCache prefix_cache_view(int rows, bool storing) const;
    const Allocations& prefix_cache_memory() const;
    void commit_restored_prefix(int rows);

    struct State;
    std::unique_ptr<State> state_;
};

}
