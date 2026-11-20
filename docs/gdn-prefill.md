# Gated DeltaNet prefill

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| GDN entry | Accept the tile, workspace, and state selected by the decoder. | [Tensor-core entry](../src/text_attention_gdn.cu#L537)<!-- prefix: "cudaError_t " --> |
| GDN operator | Prepare rows, update FP32 recurrent state, and finish BF16 output and convolution history. | [Prefill submission](../src/text_attention_gdn.cu#L462)<!-- prefix: "cudaError_t " --> |
| Verification | Compare state, history, and output with the original CUDA path and the independent CPU oracle. | [Segmented probe](../tests/decoder/gdn_check.cu#L516)<!-- prefix: "int main([[m" --> · [Checkpoint probe](../tests/decoder/gdn_checkpoint.cpp#L99)<!-- prefix: "int main(int" --> |

## Runtime path

The paired kernel stages eight token rows at a time and assigns two adjacent value channels to each warp. Both channels share prepared Q, K, and gates. Gate preparation reuses key scratch. SM100 uses packed round-to-nearest FP32 multiply, fused multiply-add, and add. Other compiled architectures use scalar arithmetic in the paired kernel. Each channel keeps its token, FP32 update, and warp reduction order. The kernel stores FP32 carry and rounds output to BF16. The operator submits preparation, recurrence, and finish in order on the lane stream. [Paired kernel](../src/text_attention_gdn.cu#L203)<!-- prefix: "__global__ v" --> · [Arithmetic](../src/text_attention_gdn.cu#L170)<!-- prefix: "__device__ f" -->

## Constraints and failure behavior

The tensor-core prefill GDN route selects the paired kernel for at least 32 rows with 16-byte-aligned workspace. Other calls to that route use the original kernel. The dispatch keeps the existing [prefill precision policy](development.md#runtime-path). Launch errors propagate to the caller. [Alignment and row guard](../src/text_attention_gdn.cu#L474)<!-- prefix: "const bool p" -->

## Verification

On a reserved B200, build the segmented probe with `SGLANG_TEST_GDN_TILED` and wrap `cudaLaunchKernel` at link time. Run `segmented paired CAPTURE` on both the packed candidate and the scalar paired control. The probe starts with nonzero history and recurrent state and carries them through eight segments. It checks at least the 31-, 32-, and 33-row boundary and a two-byte workspace offset.

Require bitwise equality of BF16 output, FP32 carry, and BF16 history across the CUDA paths. Require exact history and apply the probe tolerances for output and carry against the scalar FP64 CPU oracle. Run `short paired CAPTURE` under memcheck, racecheck, initcheck, and synccheck. Require zero errors and hazards.

Run the checkpoint probe with 2,048 prompt IDs and eight fixed teacher IDs. Leave `SGLANG_SERIAL_PREFILL_ROWS` unset. Require two 1,024-row prefill chunks. Compare carries and histories across all 48 GDN layers after each chunk and after the decode sequence. Compare full BF16 logits and winners before and after the fixed decode steps.

Time recurrence alone and preparation through finish with CUDA events. Use the [frozen W2 and W3 protocols](performance.md) for serving measurements. Match the native tile schedule and graph settings across builds. Check native IDs and report the frozen workload gates separately.

On B200 GPU6, the packed and scalar paired builds matched subnormal carry bits, checkpoint state, and all W1 and W2 BF16 logits. Median 1,024-row GDN operator time fell from 699.904 to 627.616 µs. In the native tensor-core W2 comparison, p95 TTFT fell from 277.961 to 270.737 ms with matching greedy IDs. See `artifacts/takeover-gdn-packed-test/performance-comparison.json`.
