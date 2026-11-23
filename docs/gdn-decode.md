# Packed GDN decode

## Responsibilities and sources of truth

The historical packed K0 gated delta network (GDN) decode updated
recurrent state and convolution history for each active row. Current
verification uses its [grouped GDN forward path](../src/decoder_forward.cpp#L407)<!-- prefix: "void run_gdn" -->.

| Part | Responsibility | Source |
| --- | --- | --- |
| Batch graph | Bind each row to its request's state. | [Batch runtime](batch-graphs.md#runtime-path) |
| Scratch | Own normalized vectors and gate values for the batch. | [Scratch allocation](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L187)<!-- prefix: "rows_.normal" --> |
| Dispatch | Select the three launches by submitted width. | [GDN dispatch](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_ops.cu#L403)<!-- prefix: "if (rows >= " --> |
| Convolution | Prepare the inputs to recurrence and advance wide-path history. | [Packed convolution](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_ops.cu#L63)<!-- prefix: "__global__ v" --> |
| Recurrence | Update each request's FP32 state and write the BF16 observable. | [Scalar recurrence](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_ops.cu#L121)<!-- prefix: "__global__ v" --> · [Streaming recurrence](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_ops.cu#L181)<!-- prefix: "__global__ v" --> |
| Finish | Apply gated RMS normalization and advance narrow-path history. | [Packed finish](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_ops.cu#L259)<!-- prefix: "__global__ v" --> |

## Runtime path

The [packed input projection](long-paged-decode.md#packed-rmsnorm-programmatic-launch) produces QKV and Z. The GDN gate projection reads the BF16 normalized rows from the same RMSNorm. [Gate call](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L298)<!-- prefix: "check(text_b" -->

Eager execution and graph replay use the same packed dispatch. Convolution rounds its activated result to BF16 before normalizing Q and K. It writes BF16 convolved values and FP32 normalized vectors, decay, and beta to batch scratch. Each recurrence warp reuses these inputs across four value rows. Finish applies gated RMS normalization to the BF16 recurrence observable. The batch stream orders the three launches and allows the next layer to reuse the scratch.

Submitted width selects the path, including when the replay has a shorter live prefix. Below width 8, the scalar recurrence uses eight warps per CTA and finish shifts history in separate CTAs. At width 8 and above, each convolution thread loads a tap before writing it into the preceding history slot. The final tap appends the raw current QKV sample. Finish launches only the gated normalization CTAs. [History shift](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_ops.cu#L80)<!-- prefix: "if constexpr" -->

The streaming recurrence uses four warps per CTA, with 16 value rows per CTA and eight CTAs per request and value head. Each lane processes four K elements at a stride of 32. Streaming loads and stores move FP32 state directly between global memory and registers. Each FP32x2 instruction pairs independent value rows and preserves the scalar multiply, FMA, and XOR reduction order within each row. [State load](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_ops.cu#L214)<!-- prefix: "const float2" --> · [State store](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_ops.cu#L242)<!-- prefix: "__stcs(addre" --> · [Warp reduction](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_ops.cu#L40)<!-- prefix: "__device__ f" -->

FlashInfer v0.7.4's single-token FP32 pretranspose kernel provides a source comparison for SM100. For `V = K = 128`, it also assigns 16 value rows per CTA and eight CTAs per request and value head. It stages FP32 `[V,K]` state in shared memory with cache-global 128-bit `cp.async`, moving four contiguous K elements per lane. It then updates the state in registers, uses warp butterfly reductions, and writes to the destination pool slot. [FlashInfer launch](https://github.com/flashinfer-ai/flashinfer/blob/77039051b0be921788dc2784ec41423b558878ec/flashinfer/gdn_kernels/gdn_decode_pretranspose.py#L725-L793) · [FlashInfer recurrence](https://github.com/flashinfer-ai/flashinfer/blob/77039051b0be921788dc2784ec41423b558878ec/flashinfer/gdn_kernels/gdn_decode_pretranspose.py#L280-L367)

## Constraints and failure behavior

Each live row updates its bound request's FP32 `[V,K]` state and three BF16 history taps in place. The checkpoint uses 16 Q/K heads, 48 value heads, and head dimension 128. Each kernel checks the current active count before it accesses state or writes outputs. Padding can carry null bindings and leaves recurrent state, history, and outputs untouched, including when the active count is zero. [Checkpoint dimensions](../include/sglang/text_attention.h#L10)<!-- prefix: "inline const" -->

FlashInfer uses input and output pool indices. A negative input index skips the recurrence and writes zero output. A live input with a negative destination index writes state to the null slot. [FlashInfer pool binding](https://github.com/flashinfer-ai/flashinfer/blob/77039051b0be921788dc2784ec41423b558878ec/flashinfer/gdn_kernels/gdn_decode_pretranspose.py#L141-L177) · [FlashInfer padded output](https://github.com/flashinfer-ai/flashinfer/blob/77039051b0be921788dc2784ec41423b558878ec/flashinfer/gdn_kernels/gdn_decode_pretranspose.py#L355-L367)

A wide-path execution failure after convolution can leave history advanced. A target-only execution error outside graph fallback or prefix recovery disables the batch. See the [packed batch failure contract](batch-graphs.md#constraints-and-failure-behavior).

## Verification

Use the focused replay bundle on a B200 with the default Pixi CUDA 13.4 environment and CUDA compatibility library. The wrapper requires both shared GPU4 reservations. From a copy of the bundle, verify the executable and input hashes against the frozen checkpoint identity. Use fresh run tags.

```bash
set -euo pipefail
awk 'NF == 2 && $1 ~ /^[0-9a-f]+$/ && length($1) == 64' \
    runs/frozen-checkpoint.identity | sha256sum -c -
cmp source/replay-probe.sm100 build/bin/probe
bash gpu4.sh review-checkpoint checkpoint 13
bash gpu4.sh review-sweep sweep 13
```

Require bitwise parity for the BF16 observable, final BF16 output, BF16 convolution history, and FP32 recurrent state. Check inactive output sentinels and compare eager execution with graph replay after rebinding the state pointers. Cover zero, partial, and full active counts across all submitted buckets. Record which checkpoint inputs were reconstructed and which were synthetic. For sanitizer coverage, use unfiltered initcheck so it observes both the writer and the reader.

Use paired CUDA-event samples around convolution, recurrence, and finish within the same graph. Exclude graph capture and binding transfers. Record per-layer medians by submitted and active width, including partial buckets and the width-8 dispatch boundary. Compare coverage with the [W5 workload](performance.md#named-workloads). For an archive-weighted estimate, apply each matched timing ratio to the historical three-stage cost after subtracting previously credited savings. Record the layer-count-weighted timing difference separately. Both calculations project suffix savings from isolated timings. Use the serving workload to measure an end-to-end change.
