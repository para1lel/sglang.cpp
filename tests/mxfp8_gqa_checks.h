#pragma once

#include <cuda_runtime_api.h>

namespace gqa_test {

void check_flash_prefill(cudaStream_t stream, bool repair_only = false);
void check_mixed_grouped_prefill(cudaStream_t stream);
void check_short_prefill_replay(cudaStream_t stream);
void check_tiled_prefill_scales(cudaStream_t stream);
}
