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
    for (int index = 0; index < count; ++index) {
        const auto& lane = lanes[index];
        auto* residual = lane.request->work.residual;
        if (lane.feature) {
            copy_cuda_async(residual, lane.feature, hidden_size * sizeof(Bf16),
                            cudaMemcpyDeviceToDevice, stream, "Substitute visual embedding");
        } else {
            check(text_embedding(embedding, vocab_size, hidden_size, lane.token_id, residual,
                                 stream), "Look up token embedding");
        }
    }
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
    const LayerBatch<BatchGraphLane> batch{lanes, count, stream};
    batch.each([&](const BatchGraphLane& lane, Workspace& work) {
        check(decode_embedding(embedding, vocab_size, hidden_size, &lane.binding->input,
                               work.residual, stream), "Look up bound token embedding");
    });
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
    batch.each([&](const BatchGraphLane& lane, Workspace& work) {
        compute_logits(work, stream, true, &lane.binding->final_state);
    });
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
    const auto normalize = [&](const Bf16* weight, const char* operation, const auto& residuals) {
        if (batch.fuse_verify_activation()) {
            VerifyRmsQuantizeRows rows;
            for (int index = 0; index < fused_verify_rows; ++index) {
                auto& work = workspace(lanes[index]);
                const auto residual_pair = residuals(work);
                rows.entries[index] = {
                    residual_pair.branch, residual_pair.residual, work.normalized,
                    batch.projection_activation(index, hidden_size)};
            }
            check(verify_rms_quantize(rows, weight, stream), operation);
            return;
        }
        if (batch.fuse_wide_verify_norm()) {
            const int capacity = packed_activation ? WideVerifyRmsQuantizeRows::capacity
                                                   : verification_packet_rows;
            for (int first = 0; first < count; first += capacity) {
                WideVerifyRmsQuantizeRows rows{};
                const int size = std::min(count - first, capacity);
                for (int index = 0; index < size; ++index) {
                    auto& work = workspace(lanes[first + index]);
                    const auto residual_pair = residuals(work);
                    rows.entries[index] = {
                        residual_pair.branch, residual_pair.residual, work.normalized,
                        batch.projection_activation(first + index, hidden_size)};
                }
                check(verify_rms_quantize_wide(rows, size, weight, stream), operation);
            }
            return;
        }
        batch.each([&](const Lane&, Workspace& work) {
            const auto residual_pair = residuals(work);
            check(text_rms_norm(residual_pair.branch, residual_pair.residual, weight,
                                hidden_size, work.normalized, residual_pair.residual, stream),
                  operation);
        });
    };
    execute_layers(layers, normalize,
                   [&](const GqaLayer& layer, int bank) { run_gqa(layer, batch, bank); },
                   [&](const GdnLayer& layer, int bank) { run_gdn(layer, batch, bank); },
                   [&](const Layer& layer) { run_mlp(layer, batch); });
}

}
