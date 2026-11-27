#include "sglang/decoder_internal.h"

#include "sglang/decoder_batch.h"
#include "sglang/decoder_cuda.cuh"
#include "sglang/decoder_verify.h"
#include "sglang/runtime/execution/decoder_forward_layers.h"

namespace sglang {
namespace {

using namespace decoder;
using namespace decoder::forward_detail;

}

void Decoder::Model::forward(const ForwardLane* lanes, int count, cudaStream_t stream) const {
    if (!lanes || count < 1 || count > decoder_batch_max_lanes) {
        throw std::invalid_argument("Invalid decoder batch width");
    }
    embed_forward_lanes(embedding, lanes, count, stream, "Look up token embedding");
    forward_layers(lanes, count, stream);
}

void Decoder::Model::forward_verification_layers(const VerificationRow* rows, int count,
                                                  cudaStream_t stream,
                                                  std::uint8_t* packed_activation) const {
    forward_layers(rows, count, stream, packed_activation);
}

void Decoder::Model::decode_graph(Request& request) const {
    const auto stream = request.memory.stream();
    if (!request.memory.graph()) {
        auto* const position = request.decode_position;
        auto* const rotary = position + 1;
        request.memory.capture_graph([&] {
            check(decode_embedding(embedding, vocab_size, hidden_size, request.work.argmax,
                                   request.work.residual, stream),
                  "Look up decode token embedding");
            const GraphLane lane{&request, position, rotary, TextGdnMode::decode};
            forward_layers(&lane, 1, stream);
            compute_logits(request, true);
            check(advance_decode_position(position, rotary, stream),
                  "Advance decode position");
        });
    }
    check(cudaGraphLaunch(request.memory.graph(), stream), "Replay decode graph");
}

void Decoder::Model::capture_batch_decode(const BatchGraphLane* lanes, int count,
                                          cudaStream_t stream) const {
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        auto& work = *lane.work;
        check(decode_embedding(embedding, vocab_size, hidden_size, &lane.binding->input,
                               work.residual, stream), "Look up bound token embedding");
    }
    forward_layers(lanes, count, stream);
    DecodeHeadRows rows;
    if (count == static_cast<int>(rows.size())) {
        DecodeStateRows states;
        for (int index = 0; index < count; ++index) {
            rows[index] = lanes[index].work;
            states[index] = &lanes[index].binding->final_state;
        }
        compute_decode_logits(rows, stream, &states);
        return;
    }
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        compute_logits(*lane.work, stream, true, &lane.binding->final_state);
    }
}

template <typename Lane>
void Decoder::Model::forward_layers(const Lane* lanes, int count, cudaStream_t stream,
                                    std::uint8_t* packed_activation) const {
    LayerBatch<Lane> batch{
        lanes, count, stream,
        backend.ordered_attention,
        backend.tensor_core_decode,
        backend.vector_decode_attention,
        backend.grouped_target_attention,
        backend.target_split_end,
        packed_activation,
    };
    batch.tensor_parallel = tp != nullptr;
    for (int index = 0; index < layer_count; ++index) {
        const auto& layer = layers[index];
        const auto [full_attention, bank] = qwen_text_model.layer_mapping(index);
        batch.normalize(layer.input_norm, "Normalize layer input",
                        index == 0 ? nullptr : &Workspace::hidden);
        if (full_attention) {
            run_gqa(layer.gqa, batch, bank);
        } else {
            run_gdn(layer.gdn, batch, bank);
        }
        batch.normalize(layer.attention_norm, "Normalize post-attention residual",
                        &Workspace::projected_attention);
        run_mlp(layer, batch);
    }
}

}
