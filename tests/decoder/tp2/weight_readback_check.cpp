#include "sglang/checkpoint.h"
#include "sglang/decoder_internal.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using sglang::decoder::Bf16Weight;
using sglang::decoder::DecoderModel;
using sglang::decoder::Fp8Weight;
using sglang::decoder::GdnLayer;
using sglang::decoder::GqaLayer;
using sglang::decoder::WeightSharding;

void require(bool valid, const std::string& message) {
    if (!valid) throw std::runtime_error(message);
}

struct Slice {
    int row, rows, column, columns;
};

class Readback {
public:
    Readback(const std::filesystem::path& path, int rank, int world)
        : rank(rank), world(world), checkpoint(path), scratch(8u << 20) {}

    void check(const DecoderModel& model) {
        const std::string root = "model.language_model.";
        whole(root + "embed_tokens.weight", model.embedding);
        whole(root + "norm.weight", model.final_norm);
        if (rank == 0) {
            dimensions(model.lm_head, 248320, 5120, "lm_head");
            whole("lm_head.weight", model.lm_head.values);
        } else {
            require(!model.lm_head.values && !model.lm_head.rows && !model.lm_head.columns,
                    "rank 1 owns LM head");
        }
        require(model.layers.size() == 64, "unexpected layer count");
        for (int index = 0; index < 64; ++index) {
            const auto& layer = model.layers[index];
            auto prefix = root + "layers." + std::to_string(index) + ".";
            whole(prefix + "input_layernorm.weight", layer.input_norm);
            whole(prefix + "post_attention_layernorm.weight", layer.attention_norm);
            if ((index + 1) % 4 == 0) gqa(prefix + "self_attn.", layer.gqa);
            else gdn(prefix + "linear_attn.", layer.gdn);
            adjacent(layer.mlp_gate, layer.mlp_up, prefix + "gate/up packing");
            fp8(prefix + "mlp.gate_proj", layer.mlp_gate, rows(17408, 5120));
            fp8(prefix + "mlp.up_proj", layer.mlp_up, rows(17408, 5120));
            fp8(prefix + "mlp.down_proj", layer.mlp_down, columns(5120, 17408));
        }
        std::cout << "TP" << world << " rank " << rank << " device readback matched "
                  << compared_bytes << " bytes in " << regions << " regions\n";
        if (world == 2 && rank == 1) controls(model.layers[0]);
    }

private:
    const int rank, world;
    sglang::Checkpoint checkpoint;
    std::vector<std::byte> scratch;
    std::uint64_t compared_bytes = 0;
    unsigned regions = 0;

    void controls(const sglang::decoder::Layer& layer) {
        auto rejects = [&](const std::string& label, const std::string& name, const void* ptr,
                           Slice wrong, int size) {
            try {
                inspect(name, ptr, wrong, size);
            } catch (const std::runtime_error& error) {
                if (std::string(error.what()).rfind("device mismatch: " + name, 0) != 0) throw;
                std::cout << "rejected " << label << ": " << error.what() << '\n';
                return;
            }
            throw std::runtime_error("negative control accepted: " + label);
        };
        const std::string p = "model.language_model.layers.0.";
        rejects("swapped rank", p + "mlp.down_proj.weight", layer.mlp_down.values,
                {0, 1, 0, 8704}, 1);
        rejects("wrong scale block", p + "mlp.down_proj.weight_scale_inv", layer.mlp_down.scales,
                {0, 1, 0, 68}, 2);
        rejects("swapped GDN banks", p + "linear_attn.in_proj_qkv.weight", layer.gdn.qkv.values,
                {3072, 1, 0, 5120}, 1);
    }

    Slice rows(int height, int width, int base = 0) const {
        return {base + rank * (height / world), height / world, 0, width};
    }
    Slice columns(int height, int width) const {
        return {0, height, rank * (width / world), width / world};
    }

    template <class Weight>
    static void dimensions(const Weight& device, int h, int w, const std::string& name) {
        require(device.rows == h && device.columns == w, "wrong shape: " + name);
    }

    static void adjacent(const Fp8Weight& first, const Fp8Weight& second,
                         const std::string& name) {
        require(first.values && second.values && first.scales && second.scales, name);
        require(second.values == first.values + std::size_t(first.rows) * first.columns &&
                    second.scales == first.scales + std::size_t(first.rows / 128) *
                                                    (first.columns / 128), name);
    }

    void inspect(const std::string& name, const void* device, Slice s, int size) {
        const auto& tensor = checkpoint.tensor(name);
        const auto height = tensor.shape.at(0);
        const std::size_t pitch = tensor.size_bytes / height;
        require(device && s.row >= 0 && s.column >= 0 && s.rows > 0 && s.columns > 0 &&
                    std::uint64_t(s.row) + s.rows <= height &&
                    std::size_t(s.column + s.columns) * size <= pitch, "bad extent: " + name);
        const std::size_t row_bytes = std::size_t(s.columns) * size;
        const auto* actual = static_cast<const std::byte*>(device);
        const auto* expected = tensor.data + std::size_t(s.row) * pitch + s.column * size;
        for (int row = 0; row < s.rows;) {
            const int count = std::min<int>(s.rows - row, scratch.size() / row_bytes);
            require(count > 0, "row exceeds readback buffer: " + name);
            const auto error = cudaMemcpy(scratch.data(), actual, count * row_bytes,
                                          cudaMemcpyDeviceToHost);
            require(error == cudaSuccess, name + ": " + cudaGetErrorString(error));
            for (int i = 0; i < count; ++i) {
                if (std::memcmp(scratch.data() + i * row_bytes, expected + i * pitch, row_bytes)) {
                    throw std::runtime_error("device mismatch: " + name + " at checkpoint row " +
                                             std::to_string(s.row + row + i) + " column " +
                                             std::to_string(s.column));
                }
            }
            actual += count * row_bytes;
            row += count;
            if (row < s.rows) expected += count * pitch;
        }
        compared_bytes += std::size_t(s.rows) * row_bytes;
        ++regions;
    }

    void whole(const std::string& name, const void* device) {
        const auto& tensor = checkpoint.tensor(name);
        const int height = static_cast<int>(tensor.shape.at(0));
        const int width = static_cast<int>(tensor.size_bytes / height / 2);
        inspect(name, device, {0, height, 0, width}, 2);
    }

    void segment(const std::string& stem, const Fp8Weight& device, Slice s, int dst_row = 0) {
        require(device.values && device.scales, "empty FP8 tensor: " + stem);
        inspect(stem + ".weight", device.values + std::size_t(dst_row) * device.columns, s, 1);
        Slice scale{s.row / 128, s.rows / 128, s.column / 128, s.columns / 128};
        inspect(stem + ".weight_scale_inv",
                device.scales + std::size_t(dst_row / 128) * (device.columns / 128), scale, 2);
    }

    void fp8(const std::string& stem, const Fp8Weight& device, Slice s) {
        dimensions(device, s.rows, s.columns, stem);
        segment(stem, device, s);
    }

    void bf16_rows(const std::string& name, const Bf16Weight& device, Slice s) {
        dimensions(device, s.rows, s.columns, name);
        inspect(name, device.values, s, 2);
    }

    void gdn(const std::string& p, const GdnLayer& layer) {
        dimensions(layer.qkv, 10240 / world, 5120, p + "QKV");
        adjacent(layer.qkv, layer.z, p + "QKV/Z packing");
        int cursor = 0;
        for (const auto slice : {rows(2048, 5120), rows(2048, 5120, 2048),
                                 rows(6144, 5120, 4096)}) {
            segment(p + "in_proj_qkv", layer.qkv, slice, cursor);
            require(layer.recurrent.convolution, "empty convolution: " + p);
            inspect(p + "conv1d.weight", layer.recurrent.convolution + cursor * 4,
                    {slice.row, slice.rows, 0, 4}, 2);
            cursor += slice.rows;
        }
        fp8(p + "in_proj_z", layer.z, rows(6144, 5120));
        bf16_rows(p + "in_proj_a.weight", layer.a, rows(48, 5120));
        bf16_rows(p + "in_proj_b.weight", layer.b, rows(48, 5120));
        inspect(p + "A_log", layer.recurrent.a_log, rows(48, 1), 2);
        inspect(p + "dt_bias", layer.recurrent.dt_bias, rows(48, 1), 2);
        whole(p + "norm.weight", layer.recurrent.norm);
        fp8(p + "out_proj", layer.output, columns(5120, 6144));
    }

    void gqa(const std::string& p, const GqaLayer& layer) {
        adjacent(layer.q_gate, layer.key, p + "Q/K packing");
        adjacent(layer.key, layer.value, p + "K/V packing");
        fp8(p + "q_proj", layer.q_gate, rows(24 * 2 * 256, 5120));
        fp8(p + "k_proj", layer.key, rows(4 * 256, 5120));
        fp8(p + "v_proj", layer.value, rows(4 * 256, 5120));
        fp8(p + "o_proj", layer.output, columns(5120, 24 * 256));
        whole(p + "q_norm.weight", layer.query_norm);
        whole(p + "k_norm.weight", layer.key_norm);
    }
};
}

int main(int argc, char** argv) {
    try {
        require(argc == 4, "usage: weight-readback-check CHECKPOINT WORLD RANK");
        const int world = std::stoi(argv[2]);
        const int rank = std::stoi(argv[3]);
        require((world == 1 || world == 2) && rank >= 0 && rank < world, "invalid rank/world");
        Readback oracle(argv[1], rank, world);
        if (world == 1) oracle.check(DecoderModel(argv[1], 0));
        else oracle.check(DecoderModel(argv[1], 0, WeightSharding{rank, world}));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
