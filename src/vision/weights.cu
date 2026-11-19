#include "sglang/vision/detail.cuh"

#include "sglang/checkpoint.h"

#include <vector>

namespace sglang::vision_detail {
namespace {

class UploadStream {
public:
    UploadStream() {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
              "Create vision upload stream");
    }

    ~UploadStream() {
        if (!drained_) {
            cleanup(cudaStreamSynchronize(stream), "Drain vision upload stream after failure");
        }
        cleanup(cudaStreamDestroy(stream), "Destroy vision upload stream");
    }

    void finish() {
        const auto status = cudaStreamSynchronize(stream);
        drained_ = true;
        check(status, "Complete vision weight upload");
    }

    cudaStream_t stream = nullptr;

private:
    bool drained_ = false;
};

struct Copy {
    const HostTensorView& source;
    Bf16* destination;
};

class Loader {
public:
    Loader(const Checkpoint& checkpoint, Bf16* destination)
        : checkpoint_(checkpoint), destination_(destination) {
        copies_.reserve(333);
    }

    Affine affine(const std::string& name, std::initializer_list<std::uint64_t> shape,
                  std::uint64_t channels) {
        return {tensor(name + ".weight", shape), tensor(name + ".bias", {channels})};
    }

    const Bf16* tensor(const std::string& name, std::initializer_list<std::uint64_t> shape) {
        const auto& view = checkpoint_.tensor("model.visual." + name, TensorDtype::BF16, shape);
        if (view.size_bytes > weight_bytes - used_) {
            throw std::runtime_error("Vision weight payload exceeds capacity");
        }
        auto* destination = destination_ + used_ / sizeof(Bf16);
        copies_.push_back({view, destination});
        used_ += view.size_bytes;
        return destination;
    }

    void upload() const {
        if (copies_.size() != 333 || used_ != weight_bytes) {
            throw std::runtime_error("Incomplete vision weight payload");
        }
        UploadStream upload;
        for (const auto& copy : copies_) {
            check(cudaMemcpyAsync(copy.destination, copy.source.data, copy.source.size_bytes,
                                  cudaMemcpyHostToDevice, upload.stream), "Upload vision weights");
        }
        upload.finish();
    }

private:
    const Checkpoint& checkpoint_;
    Bf16* destination_;
    std::size_t used_ = 0;
    std::vector<Copy> copies_;
};

}

Weights load_weights(const Checkpoint& checkpoint, Bf16* destination) {
    Loader load(checkpoint, destination);
    Weights weights{};
    weights.patch = load.affine("patch_embed.proj", {hidden, 3, 2, 16, 16}, hidden);
    weights.position = load.tensor("pos_embed.weight", {table_side * table_side, hidden});
    for (int layer = 0; layer < depth; ++layer) {
        const auto prefix = "blocks." + std::to_string(layer) + ".";
        weights.blocks[layer] = {
            load.affine(prefix + "norm1", {hidden}, hidden),
            load.affine(prefix + "attn.qkv", {3 * hidden, hidden}, 3 * hidden),
            load.affine(prefix + "attn.proj", {hidden, hidden}, hidden),
            load.affine(prefix + "norm2", {hidden}, hidden),
            load.affine(prefix + "mlp.linear_fc1", {feedforward, hidden}, feedforward),
            load.affine(prefix + "mlp.linear_fc2", {hidden, feedforward}, hidden),
        };
    }
    weights.merge_norm = load.affine("merger.norm", {hidden}, hidden);
    weights.merge_fc1 = load.affine("merger.linear_fc1", {merge_width, merge_width}, merge_width);
    weights.merge_fc2 = load.affine("merger.linear_fc2",
                                   {vision_feature_width, merge_width}, vision_feature_width);
    load.upload();
    return weights;
}

}
