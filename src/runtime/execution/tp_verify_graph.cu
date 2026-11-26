#include "sglang/runtime/execution/tp_verify_graph.h"
#include <cuda_runtime.h>

namespace sglang::decoder::tp_verify_graph {
namespace {

__device__ void copy_vectors(Bf16* to, const Bf16* from, int elements) {
    auto* dst = reinterpret_cast<uint4*>(to);
    const auto* src = reinterpret_cast<const uint4*>(from);
    for (int index = threadIdx.x; index < elements / 8; index += blockDim.x) {
        dst[index] = src[index];
    }
}

__global__ void row_exchange(const DeviceBinding* table, bool output) {
    const int row = blockIdx.x;
    const auto live = table->request[row];
    const auto fixed = table->resident[row];
    if (output) {
        copy_vectors(live.hidden, fixed.hidden, hidden_size);
        copy_vectors(live.residual, fixed.residual, hidden_size);
    } else {
        copy_vectors(fixed.residual, live.residual, hidden_size);
    }
}

__global__ void replay_exchange(const DeviceBinding* table) {
    const auto data = table->replay[blockIdx.x][blockIdx.y];
    copy_vectors(data.to.qkv, data.from.qkv, depth * GdnLocal::qkv_width);
    copy_vectors(data.to.convolved, data.from.convolved, depth * GdnLocal::qkv_width);
    copy_vectors(data.to.b, data.from.b, depth * GdnLocal::value_heads);
    copy_vectors(data.to.a, data.from.a, depth * GdnLocal::value_heads);
}
}

cudaError_t exchange_rows(const DeviceBinding* table, bool output, cudaStream_t stream) {
    row_exchange<<<rows, 128, 0, stream>>>(table, output);
    return cudaPeekAtLastError();
}

cudaError_t exchange_replay(const DeviceBinding* table, cudaStream_t stream) {
    replay_exchange<<<dim3(gdn_layers, lanes), 256, 0, stream>>>(table);
    return cudaPeekAtLastError();
}
}
