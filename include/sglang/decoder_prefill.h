#pragma once

#include "sglang/decoder_internal.h"

#include <memory>

namespace sglang::decoder {

inline constexpr int batch_prefill_tile_rows = 1024;

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

    TextFp8ActivationView quantize(const Bf16* input, int width) const;
    void project(Fp8Weight weight, TextFp8ActivationView input, Bf16* output) const;
    void project(Bf16Weight weight, const Bf16* input, Bf16* output) const;
};

class PrefillWorkspace {
public:
    PrefillWorkspace(cudaStream_t stream, int capacity);
    ~PrefillWorkspace();
    PrefillWorkspace(const PrefillWorkspace&) = delete;
    PrefillWorkspace& operator=(const PrefillWorkspace&) = delete;

    std::size_t bytes() const { return bytes_; }
    void close();

    cudaStream_t stream() const { return stream_; }
    int capacity() const { return capacity_; }
    void* data() const { return data_; }

private:
    cudaError_t release() noexcept;

    cudaStream_t stream_;
    int capacity_;
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    cudaError_t release_status_ = cudaSuccess;
    std::unique_ptr<PrefillPool> pool_;
};

}
