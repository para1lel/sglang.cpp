#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>

namespace sglang::decoder {

struct MnnvlArError : std::runtime_error {
    MnnvlArError(const std::string& what, bool peer_enqueued_arg);

    bool peer_enqueued;
};

class MnnvlAllReduce {
public:
    static std::unique_ptr<MnnvlAllReduce> create(std::vector<int> devices);
    ~MnnvlAllReduce();

    MnnvlAllReduce(const MnnvlAllReduce&) = delete;
    MnnvlAllReduce& operator=(const MnnvlAllReduce&) = delete;

    int rank_count() const noexcept;
    int device(int rank) const;

    void allreduce_bf16(__nv_bfloat16* const buffers[], const cudaStream_t streams[],
                        std::size_t count, bool two_shot);

    std::size_t max_count() const noexcept;
    bool supports_twoshot(std::size_t count) const noexcept;

    bool healthy() const noexcept;

private:
    struct State;

    explicit MnnvlAllReduce(std::unique_ptr<State> state);
    void ensure_healthy() const;

    static void probe_devices(State& state);
    static std::uint64_t size_workspace(State& state, std::uint64_t requested);
    static void create_physical(State& state);
    static void map_workspace(State& state, std::uint64_t mc_granularity);
    static void init_lamport(State& state);
    static void preflight(State& state, __nv_bfloat16* const buffers[],
                          const cudaStream_t streams[], bool two_shot);

    std::unique_ptr<State> state_;
};

}
