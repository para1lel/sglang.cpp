#pragma once

#include "sglang/text/gqa_verify_batch.h"
#include "sglang/text/gdn/replay_ssm.h"

#include <array>
#include <memory>

namespace sglang::decoder { class TensorParallel; }
namespace sglang::decoder::tp_verify_graph {

inline constexpr int rows = 10;
inline constexpr int lanes = 2;
inline constexpr int depth = rows / lanes;
inline constexpr int score_capacity = 256;

struct GqaBinding {
    PagedDecodeLane reads[rows];
};

struct RowExchange {
    Bf16* hidden;
    Bf16* residual;
};

class Graph;
using Input = std::array<const VerificationRow*, 2>;
struct Forward {
    void operator()(const Input& input) const;

    const void* state;
    void (*call)(const void*, const Input&);
};

bool replay(std::shared_ptr<Graph>& owner, bool& disabled, const Input& input,
            const std::array<int, 2>& devices, const std::array<cudaStream_t, 2>& streams,
            TensorParallel& group, const Forward& forward);

struct Submission {
    Submission(TensorParallel& transport, Graph* graph) noexcept;
    ~Submission();
    Submission(const Submission&) = delete;
    Submission& operator=(const Submission&) = delete;

private:
    TensorParallel& group;
    Graph* owner;
    int uncaught;
};
void record_fused(cudaStream_t stream);
struct HostBinding;
bool run_gdn(cudaStream_t stream, int layer, TextGdnWeights weights);
void attend_gqa(HostBinding& live, const GqaLayer& layer, const VerificationRow* packet,
                int cache_index, GqaGeometry geometry);
void record_regular(cudaStream_t stream);

struct DeviceBinding {
    int rotary[rows];
    GqaBinding gqa[full_attention_layers];
    gdn_replay_ssm::Lane gdn[gdn_layers][lanes];
    RowExchange resident[rows];
    RowExchange request[rows];
};

struct HostBinding {
    cudaStream_t stream;
    DeviceBinding* device;
    int gqa_calls = 0;
    int gdn_calls = 0;
    int fused_calls = 0;
    int regular_calls = 0;
};

extern thread_local HostBinding* active[2];

HostBinding* binding(cudaStream_t stream);

cudaError_t preprocess(const GqaVerifyRows<GqaVerifyPreprocessRow>& batch, const int* rotary,
                       GqaGeometry geometry, const Bf16* q_norm, const Bf16* k_norm,
                       cudaStream_t stream);
cudaError_t append(const GqaVerifyRows<GqaVerifyAppendRow>& batch, const GqaBinding* live,
                   int kv_heads, cudaStream_t stream);
cudaError_t read(const GqaBinding* live, int query_heads, cudaStream_t stream);
cudaError_t exchange_rows(const DeviceBinding* table, bool output, cudaStream_t stream);

}

namespace sglang::decoder {

template <>
tp_verify_graph::DeviceBinding*
Allocations::allocate<tp_verify_graph::DeviceBinding>(std::size_t count);

}
