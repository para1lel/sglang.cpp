#pragma once

#include "sglang/decoder_internal.h"
#include "sglang/text/prefill_attn_sm100.h"

#include <array>
#include <memory>

namespace sglang::decoder {

inline constexpr int batch_prefill_tile_rows = 1024;
inline constexpr int min_partial_prefill_rows = 256;

struct PrefillPool;

void reclaim_prefill_workspaces(int device);

struct PromptLogprobCapture {
    const std::vector<int>* prompt;
    DecoderPromptLogprobs* output;
    GeneratedLogprobReadback* readback;
    void* workspace;
    int top;
    int start;

    int score_end(int position, int rows) const;
    void enqueue(const DecoderModelAccess::Model& model, Workspace& work, int position,
                 int index, cudaStream_t stream) const;
    void collect(int rows, cudaStream_t stream) const;
};

void finish_prompt_logprobs(const DecoderOptions& options, DecoderPromptLogprobs& scores,
                            DecoderBatchStep& result);

struct PrefillGrant {
    Request* request;
    const int* tokens;
    int position;
    int rows;
    int tile_rows;
    int prompt_remaining;
    MtpDraftRequest* draft;
    const MediaPrefill* media;
    const PromptLogprobCapture* scores = nullptr;
};

struct PrefillProjections {
    TextFp8Activation activation;
    int rows;
    cudaStream_t stream;
    bool tensor_core_fp8 = false;
    bool grouped_projection = false;
    bool wmma_draft_seed = false;

    TextFp8ActivationView quantize(const Bf16* input, int width) const;
    void project(Fp8Weight weight, TextFp8ActivationView input, Bf16* output) const;
    void project(Bf16Weight weight, const Bf16* input, Bf16* output) const;
    void project_seed_fc(Bf16Weight weight, const Bf16* input, Bf16* output) const;
};

class PrefillWorkspace {
public:
    enum class Retention { existing, transient };

    PrefillWorkspace(cudaStream_t stream, int capacity);
    PrefillWorkspace(cudaStream_t stream, int capacity, int staged_visible);
    PrefillWorkspace(cudaStream_t stream, int capacity, Retention retention);
    ~PrefillWorkspace();
    PrefillWorkspace(const PrefillWorkspace&) = delete;
    PrefillWorkspace& operator=(const PrefillWorkspace&) = delete;

    std::size_t bytes() const { return bytes_; }
    std::size_t layout_bytes() const { return layout_bytes_; }
    void close();

    cudaStream_t stream() const { return stream_; }
    int capacity() const { return capacity_; }
    void* data() const { return data_; }
    PairedPrefillScratch long_gqa_staging() const { return staging_; }

private:
    friend class PrefillPeerNorm;
    bool allow_peer_access(int peer_device);
    PrefillWorkspace(cudaStream_t stream, int capacity, Retention retention, int staged_visible);
    cudaError_t release() noexcept;

    cudaStream_t stream_;
    int capacity_;
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    std::size_t layout_bytes_ = 0;
    PairedPrefillScratch staging_{};
    cudaError_t release_status_ = cudaSuccess;
    std::unique_ptr<PrefillPool> pool_;
};

class PrefillPeerNorm {
public:
    PrefillPeerNorm() = default;
    ~PrefillPeerNorm();
    PrefillPeerNorm(const PrefillPeerNorm&) = delete;
    PrefillPeerNorm& operator=(const PrefillPeerNorm&) = delete;

    void prepare(PrefillWorkspace& root, PrefillWorkspace& peer, int root_device, int peer_device);
    bool enqueue(const std::array<PrefillPeerNormRow, 2>& rows) const;

private:
    struct Rank {
        int device = -1;
        cudaStream_t stream = nullptr;
        cudaEvent_t produced = nullptr;
        cudaEvent_t consumed = nullptr;
    };
    std::array<Rank, 2> ranks_{};
    bool ready_ = false;
};

std::size_t prefill_workspace_bytes(int rows);

inline bool full_tile_prefill_grants(const PrefillGrant* grants, int count) {
    if (count < 2 || grants[0].rows < batch_prefill_tile_rows) {
        return false;
    }
    for (int index = 0; index < count; ++index) {
        const auto& grant = grants[index];
        if (grant.tile_rows != batch_prefill_tile_rows || grant.rows != grants[0].rows ||
            grant.rows % batch_prefill_tile_rows != 0) {
            return false;
        }
    }
    return true;
}

inline PrefillGrant prefill_tile(PrefillGrant grant, int offset) {
    grant.tokens += offset;
    grant.position += offset;
    grant.prompt_remaining -= offset;
    grant.rows = batch_prefill_tile_rows;
    return grant;
}

bool try_prefill_projection_groups(const DecoderModel& model, const PrefillGrant* grants,
                                   int count, cudaStream_t stream, DecoderBatchStepStats& stats);

}
