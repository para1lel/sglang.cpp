#pragma once

#include "sglang/decoder_internal.h"

#include <memory>

namespace sglang::decoder {

struct PrefillPool;

void reclaim_prefill_workspaces(int device);

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
