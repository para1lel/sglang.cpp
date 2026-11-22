#include "sglang/decoder_internal.h"

#include "sglang/checkpoint.h"
#include "sglang/text/mtp/tc_vocab.h"

#include <string>
#include <string_view>

namespace sglang {
namespace {

using namespace decoder;

class WeightUpload {
public:
    WeightUpload(const Checkpoint& checkpoint, Allocations& memory, cudaStream_t stream)
        : checkpoint_(checkpoint), memory_(memory), stream_(stream) {}

    const Bf16* bf16(const std::string& name, std::initializer_list<std::uint64_t> shape) {
        return reinterpret_cast<const Bf16*>(upload(name, TensorDtype::BF16, shape));
    }

    Bf16Weight bf16_matrix(const std::string& name, int rows, int columns) {
        return {bf16(name, {extent(rows), extent(columns)}), rows, columns};
    }

    Fp8Weight fp8_matrix(const std::string& stem, int rows, int columns) {
        const auto* values = upload(stem + ".weight", TensorDtype::F8_E4M3,
                                    {extent(rows), extent(columns)});
        const auto* scales = bf16(stem + ".weight_scale_inv",
                                  {extent(text_fp8_group_count(rows)),
                                   extent(text_fp8_group_count(columns))});
        return {values, scales, rows, columns};
    }

    template <int... Widths>
    std::array<Fp8Weight, sizeof...(Widths)> packed_fp8(
        const std::array<std::string, sizeof...(Widths)>& stems, int columns) {
        static_assert(((Widths > 0 && Widths % text_fp8_group_size == 0) && ...));
        constexpr int total_rows = (Widths + ...);
        constexpr std::array rows{Widths...};
        const int scale_columns = text_fp8_group_count(columns);
        auto* values = memory_.allocate<std::uint8_t>(std::size_t(total_rows) * columns);
        auto* scales = memory_.allocate<Bf16>(std::size_t(text_fp8_group_count(total_rows)) *
                                             scale_columns);
        std::array<Fp8Weight, sizeof...(Widths)> weights;
        for (std::size_t index = 0; index < rows.size(); ++index) {
            const int width = rows[index];
            const int scale_rows = text_fp8_group_count(width);
            const auto& stem = stems[index];
            upload(stem + ".weight", TensorDtype::F8_E4M3, {extent(width), extent(columns)},
                   values);
            upload(stem + ".weight_scale_inv", TensorDtype::BF16,
                   {extent(scale_rows), extent(scale_columns)},
                   reinterpret_cast<std::uint8_t*>(scales));
            weights[index] = {values, scales, width, columns};
            values += std::size_t(width) * columns;
            scales += std::size_t(scale_rows) * scale_columns;
        }
        return weights;
    }

private:
    static std::uint64_t extent(int size) {
        return static_cast<std::uint64_t>(size);
    }

    const std::uint8_t* upload(const std::string& name, TensorDtype dtype,
                               std::initializer_list<std::uint64_t> shape,
                               std::uint8_t* destination = nullptr) {
        const auto& tensor = checkpoint_.tensor(name, dtype, shape);
        if (!destination) {
            destination = memory_.allocate<std::uint8_t>(tensor.size_bytes);
        }
        check(cudaMemcpyAsync(destination, tensor.data, tensor.size_bytes,
                              cudaMemcpyHostToDevice, stream_), name.c_str());
        return destination;
    }

    const Checkpoint& checkpoint_;
    Allocations& memory_;
    cudaStream_t stream_;
};

GdnLayer load_gdn(WeightUpload& weights, const std::string& prefix) {
    const auto [qkv, z] = weights.packed_fp8<text_gdn_qkv_width, text_gdn_value_width>(
        {prefix + "in_proj_qkv", prefix + "in_proj_z"}, hidden_size);
    return {
        qkv, z,
        weights.bf16_matrix(prefix + "in_proj_b.weight", text_gdn_value_heads, hidden_size),
        weights.bf16_matrix(prefix + "in_proj_a.weight", text_gdn_value_heads, hidden_size),
        {
            weights.bf16(prefix + "conv1d.weight",
                         {text_gdn_qkv_width, 1, text_gdn_history_taps + 1}),
            weights.bf16(prefix + "A_log", {text_gdn_value_heads}),
            weights.bf16(prefix + "dt_bias", {text_gdn_value_heads}),
            weights.bf16(prefix + "norm.weight", {text_gdn_head_dim}),
        },
        weights.fp8_matrix(prefix + "out_proj", hidden_size, text_gdn_value_width),
    };
}

GqaLayer load_gqa(WeightUpload& weights, const std::string& prefix) {
    const auto [q_gate, key, value] =
        weights.packed_fp8<text_gqa_q_gate_width, text_gqa_kv_width, text_gqa_kv_width>(
            {prefix + "q_proj", prefix + "k_proj", prefix + "v_proj"}, hidden_size);
    return {
        q_gate, key, value,
        weights.bf16(prefix + "q_norm.weight", {mxfp8_head_dim}),
        weights.bf16(prefix + "k_norm.weight", {mxfp8_head_dim}),
        weights.fp8_matrix(prefix + "o_proj", hidden_size, text_gqa_q_width),
    };
}

void load_layer(WeightUpload& weights, Layer& layer, int index, const std::string& root) {
    const auto prefix = root + "layers." + std::to_string(index) + ".";
    layer.input_norm = weights.bf16(prefix + "input_layernorm.weight", {hidden_size});
    layer.attention_norm = weights.bf16(prefix + "post_attention_layernorm.weight", {hidden_size});
    if (is_full_attention(index)) {
        layer.gqa = load_gqa(weights, prefix + "self_attn.");
    } else {
        layer.gdn = load_gdn(weights, prefix + "linear_attn.");
    }
    const auto [gate, up] = weights.packed_fp8<intermediate_size, intermediate_size>(
        {prefix + "mlp.gate_proj", prefix + "mlp.up_proj"}, hidden_size);
    layer.mlp_gate = gate;
    layer.mlp_up = up;
    layer.mlp_down = weights.fp8_matrix(prefix + "mlp.down_proj", hidden_size, intermediate_size);
}

}

Decoder::Model::Model(const std::filesystem::path& directory, int cuda_device)
    : device(cuda_device), memory(std::make_shared<Allocations>(cuda_device)) {
    DeviceScope selected_device(device);
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "Read decoder CUDA device properties");
    if (properties.major != 10 || properties.minor != 0) {
        throw std::invalid_argument("The decoder requires an sm_100 CUDA device");
    }
    accelerator_name = properties.name;
    b200_ = accelerator_name == "NVIDIA B200";
    auto& checkpoint = memory->transfers().checkpoint;
    checkpoint = std::make_unique<Checkpoint>(directory);
    memory->open_stream();
    WeightUpload weights(*checkpoint, *memory, memory->stream());
    const std::string root = "model.language_model.";
    embedding = weights.bf16(root + "embed_tokens.weight", {vocab_size, hidden_size});
    final_norm = weights.bf16(root + "norm.weight", {hidden_size});
    lm_head = weights.bf16_matrix("lm_head.weight", vocab_size, hidden_size);
    for (int index = 0; index < layer_count; ++index) {
        load_layer(weights, layers[index], index, root);
    }
    memory->close_stream();
    selected_device.restore();
}

}
