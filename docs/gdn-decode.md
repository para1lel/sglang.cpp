# Packed GDN decode

## Responsibilities and sources of truth

Packed gated delta network (GDN) decode updates recurrent state and convolution history for each active row.

| Part | Responsibility | Source |
| --- | --- | --- |
| Batch graph | Bind each row to its request's state. | [Batch runtime](batch-graphs.md#runtime-path) |
| Scratch | Own normalized vectors and gate values for the batch. | [Scratch allocation](../src/text/decoder_graph_packed.cpp#L185)<!-- prefix: "rows_.normal" --> |
| Convolution | Prepare the inputs to recurrence. | [Packed convolution](../src/text/decoder_graph_ops.cu#L51)<!-- prefix: "__global__ v" --> |

## Runtime path

Convolution rounds its result to BF16 before normalizing Q and K. It writes FP32 normalized vectors, decay, and beta to batch scratch. Each recurrence warp reuses these inputs across four value rows. The finish kernel applies the output gate and shifts convolution history. The batch stream orders the three launches and allows the next layer to reuse the scratch. [Recurrence](../src/text/decoder_graph_ops.cu#L104)<!-- prefix: "__global__ v" --> · [Finish](../src/text/decoder_graph_ops.cu#L165)<!-- prefix: "__global__ v" -->

## Constraints and failure behavior

Eager and graph replay use the same launches. The kernels read the live active row count and current device bindings. Padded rows leave recurrent state, history, and outputs untouched. Each request retains FP32 recurrent state and BF16 history. Scratch allocation follows the [packed setup failure contract](batch-graphs.md#constraints-and-failure-behavior).

## Verification

Compare both BF16 outputs, convolution history, and FP32 recurrent state across eager and replay with rebound state pointers, partial padding, and zero active rows. Use CUDA events around convolution, recurrence, and finish in the same graph. Use checkpoint dimensions, include fully and partially active buckets, and exclude capture and binding transfers. Report time per layer and the captured and active widths.
