#pragma once

#include "sglang/text/gqa_verify_batch.h"
#include "sglang/text/gdn/replay_ssm.h"

#include <array>
#include <memory>

namespace sglang::decoder { class TensorParallel; }
namespace sglang::decoder::tp_verify_graph {

inline constexpr int max_lanes = 2 * wide_mtp_lanes;
inline constexpr int max_rows = max_lanes * target_verify_max_rows;
inline constexpr int short_capacity = target_split_first_position;

struct SplitGroup {
    int first;
    int count;
    gqa_decode::SplitVariant variant;
    bool operator==(const SplitGroup& other) const;
};

struct GqaBinding {
    PagedDecodeLane reads[max_rows];
    gqa_decode::LongGraphBinding long_reads[max_lanes];
};

struct RowExchange {
    Bf16* hidden;
    Bf16* residual;
};

class Graph;

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
bool owns_collective(const cudaStream_t streams[], int rows);
void record_collective(cudaStream_t stream, bool two_shot = false);
struct HostBinding;
bool run_gdn(cudaStream_t stream, int layer, TextGdnWeights weights);
void attend_gqa(HostBinding& live, const GqaLayer& layer, const VerificationRow* packet,
                int cache_index, GqaGeometry geometry);
bool close(std::shared_ptr<Graph>& owner, TensorParallel& group) noexcept;

struct DeviceBinding {
    int tokens[decoder_batch_max_lanes];
    RowExchange resident[max_rows];
    RowExchange request[max_rows];
    BatchDecodeBinding decode[decoder_batch_max_lanes];
    int rotary[max_rows];
    GqaBinding gqa[full_attention_layers];
    gdn_replay_ssm::Lane gdn[gdn_layers][max_lanes];
};

struct HostBinding {
    cudaStream_t stream;
    DeviceBinding* device;
    int rows = 0;
    int lanes = 0;
    int depth = 0;
    std::array<int, max_rows> short_rows;
    int short_count = 0;
    std::array<int, max_rows> split_rows;
    std::array<SplitGroup, max_rows / verification_packet_rows> splits;
    int split_groups = 0;
    int long_count = 0;
    gqa_decode::LongDecodeShape long_shape{};
    std::uint32_t decode_split = 0;
    int gqa_calls = 0;
    int gdn_calls = 0;
    int collectives = 0;
    int two_shot = 0;
};

HostBinding* binding(cudaStream_t stream);

cudaError_t read(const GqaBinding* live, const std::array<int, max_rows>& indices, int rows,
                 int query_heads, cudaStream_t stream);
cudaError_t exchange_rows(const DeviceBinding* table, int rows, bool output, cudaStream_t stream);
}

namespace sglang::decoder {

template <>
tp_verify_graph::DeviceBinding*
Allocations::allocate<tp_verify_graph::DeviceBinding>(std::size_t count);

}
