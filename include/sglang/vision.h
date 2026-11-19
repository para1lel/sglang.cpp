#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace sglang {

class Checkpoint;

inline constexpr int vision_patch_width = 1536;
inline constexpr int vision_feature_width = 5120;
inline constexpr int vision_max_frame_patches = 4096;
inline constexpr int vision_max_patches = 16384;

struct VisionGrid {
    std::int64_t temporal_pairs;
    std::int64_t height;
    std::int64_t width;
};

/// Returns T*H*W/4. Requires positive T and even H/W, H*W <= 4096,
/// T*H*W <= 16384. Throws std::invalid_argument for unsupported grids.
std::size_t vision_feature_rows(VisionGrid grid);

class VisionEncoder {
public:
    /// Validates the vision schema and uploads immutable BF16 weights to the current device.
    explicit VisionEncoder(const Checkpoint& checkpoint);
    ~VisionEncoder();

    VisionEncoder(const VisionEncoder&) = delete;
    VisionEncoder& operator=(const VisionEncoder&) = delete;

    /// One preprocessed grid, with rows in (t, block_y, block_x, in_y, in_x) order.
    /// Each row contains finite FP32 values in (RGB, temporal plane, y, x) order.
    /// Four consecutive patches produce one [5120] BF16 feature row. Attention is
    /// bidirectional within each temporal pair. The pinned model has zero DeepStack levels.
    /// Input and output are disjoint contiguous device spans with exact element counts.
    /// The caller owns their allocations through return. Select the encoder's device
    /// and supply a stream on it outside graph capture. Synchronizes stream before return.
    /// Throws on failure. The caller discards the output on failure.
    void encode(VisionGrid grid, const float* patches, std::size_t patch_elements,
                __nv_bfloat16* features, std::size_t feature_elements, cudaStream_t stream) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
