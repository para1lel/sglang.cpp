#include "sglang/vision/detail.cuh"

#include "sglang/checkpoint.h"
#include "sglang/runtime/backend/device_arena.h"

#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace sglang::vision_detail {

void check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

bool cleanup(cudaError_t status, const char* operation) noexcept {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

}

namespace sglang {
namespace {

using namespace vision_detail;

template <typename Element>
class DeviceBuffer {
public:
  DeviceBuffer(const std::shared_ptr<decoder::DeviceArena>& arena, std::size_t count)
      : arena_(arena), data_(static_cast<Element*>(arena_->allocate(count * sizeof(Element)))) {
  }
    ~DeviceBuffer() {
        if (!quarantined_)
            arena_->release(data_);
    }

    void quarantine() noexcept {
        arena_->quarantine();
        quarantined_ = true;
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    Element* get() const { return data_; }

private:
  std::shared_ptr<decoder::DeviceArena> arena_;
  Element* data_ = nullptr;
  std::atomic<bool> quarantined_{false};
};

Grid checked_grid(VisionGrid grid) {
    const auto [time, height, width] = grid;
    if (time < 1 || time > vision_max_patches || height < 2 || width < 2 ||
        height > vision_max_frame_patches || width > vision_max_frame_patches ||
        height % 2 || width % 2 || height * width > vision_max_frame_patches ||
        time * height * width > vision_max_patches) {
        throw std::invalid_argument("Unsupported preprocessed vision grid");
    }
    return {static_cast<int>(time * height * width), static_cast<int>(height),
            static_cast<int>(width), static_cast<int>(height * width)};
}

template <typename Element>
void validate_span(const Element* pointer, std::size_t size, std::size_t expected, int device) {
    if (!pointer || size != expected || reinterpret_cast<std::uintptr_t>(pointer) %
        alignof(Element) != 0) {
        throw std::invalid_argument("Invalid vision device span");
    }
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, pointer), "Inspect vision device span");
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
        throw std::invalid_argument("Vision device span belongs to another memory space");
    }
}

void validate_stream(int device, cudaStream_t stream) {
    int current = 0;
    check(cudaGetDevice(&current), "Get current vision device");
    if (current != device) {
        throw std::invalid_argument("Current device differs from the vision encoder device");
    }
    int stream_device = 0;
    check(cudaStreamGetDevice(stream, &stream_device), "Get vision stream device");
    if (stream_device != device) {
        throw std::invalid_argument("Vision stream belongs to another device");
    }
    cudaStreamCaptureStatus capture{};
    check(cudaStreamIsCapturing(stream, &capture), "Check vision stream capture status");
    if (capture != cudaStreamCaptureStatusNone) {
        throw std::invalid_argument("Vision encoding requires a stream outside graph capture");
    }
}

void validate_spans(const float* patches, std::size_t patch_elements,
                    const Bf16* features, std::size_t feature_elements, Grid grid, int device) {
    validate_span(patches, patch_elements, std::size_t(grid.rows) * vision_patch_width, device);
    validate_span(features, feature_elements, std::size_t(grid.rows / 4) * vision_feature_width,
                  device);
    const auto input = reinterpret_cast<std::uintptr_t>(patches);
    const auto output = reinterpret_cast<std::uintptr_t>(features);
    if (input <= output ? output - input < patch_elements * sizeof(float)
                        : input - output < feature_elements * sizeof(Bf16)) {
        throw std::invalid_argument("Vision device spans overlap");
    }
}

struct Workspace {
    Workspace(Grid grid, const std::shared_ptr<decoder::DeviceArena>& arena)
        : patches(arena, static_cast<std::size_t>(grid.rows) * vision_patch_width),
          residual(arena, static_cast<std::size_t>(grid.rows) * hidden),
          norm(arena, static_cast<std::size_t>(grid.rows) * hidden),
          qkv(arena, static_cast<std::size_t>(grid.rows) * 3 * hidden),
          qk(arena, static_cast<std::size_t>(grid.rows) * 2 * hidden),
          context(arena, static_cast<std::size_t>(grid.rows) * hidden),
          projected(arena, static_cast<std::size_t>(grid.rows) * hidden),
          intermediate(arena, static_cast<std::size_t>(grid.rows) * feedforward),
          rotary(arena, static_cast<std::size_t>(grid.frame) * (head_width / 2)) {
    }

    void quarantine() noexcept {
        patches.quarantine();
        residual.quarantine();
        norm.quarantine();
        qkv.quarantine();
        qk.quarantine();
        context.quarantine();
        projected.quarantine();
        intermediate.quarantine();
        rotary.quarantine();
    }

    DeviceBuffer<Bf16> patches;
    DeviceBuffer<Bf16> residual;
    DeviceBuffer<Bf16> norm;
    DeviceBuffer<Bf16> qkv;
    DeviceBuffer<Bf16> qk;
    DeviceBuffer<Bf16> context;
    DeviceBuffer<Bf16> projected;
    DeviceBuffer<Bf16> intermediate;
    DeviceBuffer<float2> rotary;
};

void transformer(const Block& weights, Workspace& work, Grid grid, cudaStream_t stream) {
    const int rows = grid.rows;
    auto* residual = work.residual.get();
    auto* norm = work.norm.get();
    auto* qkv = work.qkv.get();
    auto* qk = work.qk.get();
    auto* context = work.context.get();
    auto* projected = work.projected.get();
    auto* middle = work.intermediate.get();
    normalize(residual, weights.norm1, norm, rows, stream);
    linear(norm, weights.qkv, qkv, rows, hidden, 3 * hidden, stream);
    rotate(qkv, work.rotary.get(), qk, grid, stream);
    attend(qk, qkv, context, grid, stream);
    linear(context, weights.attention, projected, rows, hidden, hidden, stream);
    add(residual, projected, rows * hidden, stream);
    normalize(residual, weights.norm2, norm, rows, stream);
    linear(norm, weights.fc1, middle, rows, hidden, feedforward, stream);
    gelu(middle, rows * feedforward, true, stream);
    linear(middle, weights.fc2, projected, rows, feedforward, hidden, stream);
    add(residual, projected, rows * hidden, stream);
}

void merge(const Weights& weights, Workspace& work, Grid grid, Bf16* output,
           cudaStream_t stream) {
    auto* norm = work.norm.get();
    auto* middle = work.projected.get();
    const int rows = grid.rows / 4;
    normalize(work.residual.get(), weights.merge_norm, norm, grid.rows, stream);
    linear(norm, weights.merge_fc1, middle, rows, merge_width, merge_width, stream);
    gelu(middle, rows * merge_width, false, stream);
    linear(middle, weights.merge_fc2, output, rows, merge_width, vision_feature_width, stream);
}

void forward(const Weights& weights, Workspace& work, Grid grid, const float* input,
             Bf16* output, cudaStream_t stream) {
    auto* patches = work.patches.get();
    auto* residual = work.residual.get();
    auto* temporary = work.projected.get();
    cast_patches(input, patches, grid.rows * vision_patch_width, stream);
    linear(patches, weights.patch, residual, grid.rows, vision_patch_width, hidden, stream);
    positions(weights.position, temporary, grid, stream);
    add(residual, temporary, grid.rows * hidden, stream);
    make_rotary(work.rotary.get(), grid, stream);
    for (const auto& block : weights.blocks) {
        transformer(block, work, grid, stream);
    }
    merge(weights, work, grid, output, stream);
}

int validated_device(const Checkpoint& checkpoint) {
    checkpoint.validate_vision();
    int device = 0;
    check(cudaGetDevice(&device), "Get vision encoder device");
    return device;
}

}

std::size_t vision_feature_rows(VisionGrid grid) {
    return checked_grid(grid).rows / 4;
}

namespace {

std::shared_ptr<decoder::DeviceArena> vision_arena(int device) {
    if (decoder::DeviceArena::active(device))
        return decoder::DeviceArena::current(device);
    auto arena = decoder::DeviceArena::start(device, decoder::default_gpu_memory_utilization);
    arena->commit();
    return arena;
}

}

struct VisionEncoder::Impl {
    explicit Impl(const Checkpoint& checkpoint)
        : device(validated_device(checkpoint)), arena(vision_arena(device)),
          memory(arena, weight_bytes / sizeof(Bf16)), weights(upload(checkpoint)) {
    }

    Weights upload(const Checkpoint& checkpoint) {
        try {
            return load_weights(checkpoint, memory.get());
        } catch (...) {
            if (cudaDeviceSynchronize() != cudaSuccess)
                memory.quarantine();
            throw;
        }
    }

    const int device;
    const std::shared_ptr<decoder::DeviceArena> arena;
    DeviceBuffer<Bf16> memory;
    const Weights weights;
};

VisionEncoder::VisionEncoder(const Checkpoint& checkpoint)
    : impl_(std::make_unique<Impl>(checkpoint)) {}

VisionEncoder::~VisionEncoder() = default;

void VisionEncoder::encode(VisionGrid shape, const float* patches, std::size_t patch_elements,
                          Bf16* features, std::size_t feature_elements, cudaStream_t stream) const {
    const Grid grid = checked_grid(shape);
    validate_stream(impl_->device, stream);
    validate_spans(patches, patch_elements, features, feature_elements, grid, impl_->device);
    Workspace work(grid, impl_->arena);
    try {
        forward(impl_->weights, work, grid, patches, features, stream);
    } catch (...) {
        if (!cleanup(cudaStreamSynchronize(stream), "Drain vision stream after launch failure")) {
            work.quarantine();
            impl_->memory.quarantine();
        }
        throw;
    }
    const auto status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        work.quarantine();
        impl_->memory.quarantine();
    }
    check(status, "Complete vision encoding");
}

}
