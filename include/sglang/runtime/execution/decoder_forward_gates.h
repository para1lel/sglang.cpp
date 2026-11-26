#pragma once

// Single-include implementation detail of
// include/sglang/runtime/execution/decoder_forward_layers.h: the GDN
// verification-gate projection lives here so that header stays under the
// canonical file-size limit. Include only from decoder_forward_layers.h at
// the marked site; the include site provides LayerBatch, workspace(),
// project_verification_bf16 and the decoder_cuda declarations.
inline void project_gdn_gates(const GdnLayer& layer, const LayerBatch<VerificationRow>& batch,
                              bool batched) {
    if (!batch.target_tc || batch.count < 2 ||
        layer.b.rows != batch.gdn_gate_heads() || layer.a.rows != batch.gdn_gate_heads() ||
        layer.b.columns != hidden_size || layer.a.columns != hidden_size) {
        project_verification_bf16(layer.b, batch.lanes, batch.count, &Workspace::gdn_b,
                                  batch.stream);
        project_verification_bf16(layer.a, batch.lanes, batch.count, &Workspace::gdn_a,
                                  batch.stream);
        return;
    }
    std::array<VerifyGdnGateRow, gdn_packet_rows> gates;
    const auto fill = [&](int first, int rows) {
        for (int index = 0; index < rows; ++index) {
            const auto& work = workspace(batch.lanes[first + index]);
            gates[index] = {work.normalized, work.gdn_b, work.gdn_a};
        }
        const auto gates_kernel =
            batch.tensor_parallel ? try_verify_bf16_gdn_gates_local : try_verify_bf16_gdn_gates;
        return gates_kernel(layer.b.values, layer.a.values, gates.data(), rows, batch.stream);
    };
    if (batched) {
        // The dedicated gate kernels accept 2..text_fp8_max_batch rows or one
        // wide gdn_packet_rows batch; a wider packet is filled in 20-row
        // chunks so the width-40 TP2 verify keeps the dedicated kernels
        // instead of silently falling back to the grouped loop.
        bool dedicated = (batch.count >= 2 && batch.count <= text_fp8_max_batch) ||
                         batch.count % gdn_packet_rows == 0;
        for (int first = 0; dedicated && first < batch.count; first += gdn_packet_rows) {
            const int rows = std::min(gdn_packet_rows, batch.count - first);
            if (const auto status = fill(first, rows)) {
                check(*status, "Run wide verification GDN gate projections");
            } else {
                dedicated = false;
            }
        }
        if (dedicated) {
            return;
        }
    }
    std::array<TextBf16Projection, text_fp8_max_batch> gate_b, gate_a;
    for (int first = 0, group = 0; first < batch.count; ++group) {
        const int rows = batch.projection_rows(group);
        if (const auto status = fill(first, rows)) {
            check(*status, "Run grouped BF16 verification GDN gate projections");
        } else {
            for (int index = 0; index < rows; ++index) {
                const auto& gate = gates[index];
                gate_b[index] = {gate.input, gate.output_b};
                gate_a[index] = {gate.input, gate.output_a};
            }
            check(text_bf16_gemm(layer.b.values, layer.b.rows, layer.b.columns, gate_b.data(),
                                rows, batch.stream), "Run batched BF16 GDN b projection");
            check(text_bf16_gemm(layer.a.values, layer.a.rows, layer.a.columns, gate_a.data(),
                                rows, batch.stream), "Run batched BF16 GDN a projection");
        }
        first += rows;
    }
}

