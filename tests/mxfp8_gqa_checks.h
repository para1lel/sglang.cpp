#pragma once

#include "sglang/mxfp8_gqa.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

namespace gqa_test {

void check_cuda(cudaError_t status, const char* operation);

template <class Element> class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    Element* get() const;
    void upload(const Element* source, cudaStream_t stream) const;
    void upload(const std::vector<Element>& source, cudaStream_t stream, int offset = 0) const;
    void fill(std::uint8_t sentinel, cudaStream_t stream) const;
    std::vector<Element> download(cudaStream_t stream) const;

  private:
    std::size_t bytes() const;
    void upload(const Element* source, std::size_t count, cudaStream_t stream, int offset) const;

    std::size_t count_;
    Element* data_ = nullptr;
};

class Graph {
  public:
    Graph();
    explicit Graph(cudaStream_t stream);
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = delete;
    Graph& operator=(Graph&&) = delete;
    ~Graph();

    void finish(cudaStream_t stream, cudaError_t launch_status);
    cudaGraph_t get() const;
    cudaError_t end_capture(cudaStream_t stream);
    void instantiate();
    void replay(cudaStream_t stream) const;

  private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
};

struct Pages {
    const std::vector<std::int32_t> mapping;
    DeviceBuffer<std::int32_t> device{mapping.size()};

    sglang::Mxfp8PageTable view() const;
};

constexpr std::uint8_t attention_sentinel = 0xff;
constexpr int prefill_guard = 8;

template <class Element>
void require_same(const std::vector<Element>& actual, const std::vector<Element>& expected,
                  const char* message);

template <class Element>
bool poisoned(const std::vector<Element>& data, std::size_t first, std::size_t end);

std::vector<__nv_bfloat16> download_prefill(const DeviceBuffer<__nv_bfloat16>& output,
                                            cudaStream_t stream, int guard = prefill_guard);
std::vector<__nv_bfloat16> make_queries(int start_token, int count);
double decode_value(std::uint8_t code, std::uint8_t scale);
void check_bound_gdn(cudaStream_t stream);
void check_raw_attention(cudaStream_t stream);
void verify_ordered_prefill(const char* phase, sglang::Mxfp8KvCache cache,
                            sglang::Mxfp8PageTable pages, int first, int rows,
                            const __nv_bfloat16* queries, const std::vector<__nv_bfloat16>& actual,
                            cudaStream_t stream);

std::size_t row_index(int token, int head, int heads);

void check_flash_prefill(cudaStream_t stream, bool repair_only = false);
void check_mixed_grouped_prefill(cudaStream_t stream);
void check_short_prefill_replay(cudaStream_t stream);
void check_tiled_prefill_scales(cudaStream_t stream);
}
