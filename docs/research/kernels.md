# SM100 kernel constraints and reference paths

## Scope

This note preserves durable Blackwell operator facts for native kernel work. It compares native behavior with pinned SGLang and vLLM paths while excluding transient benchmark conclusions.

## FP8 projections

The checkpoint has 407 `F8_E4M3` matrices with BF16 inverse-scale blocks of `128×128`. SGLang’s importable `sgl-deep-gemm` selects DeepGEMM on SM100 for eligible projections, requantizes ordinary checkpoint weights to UE8M0, and quantizes activation groups into TMA-aligned packed scales. If DeepGEMM is unavailable or rejects an input, SGLang falls back to FlashInfer TRTLLM groupwise or Triton.

vLLM tries FlashInfer and DeepGEMM before CUTLASS. On this checkpoint, the ordinary FlashInfer block-scale wrapper is gated to SM90. DeepGEMM excludes nested `qwen3_5_text` on SM100 for accuracy. CUTLASS accepts E4M3 activations, column-major FP32 per-token scales, and 128-channel groups. The CUDA per-token group quantizer is a separate operation in compilation mode `NONE`.

## GQA decode

Native stores four MXFP8 K/V planes with one E8M0 scale per 32 channels. The reference vLLM path predicts FlashInfer TRTLLM-Gen decode on SM100 when NVIDIA artifacts are available. With one token and six query heads per KV head, FlashInfer selects the grouped-swap generation kernel and Q tile 8. The default BF16-query/FP8-KV path transforms K and V separately, converts K to BF16 before the score MMA, and uses unit QK/PV KV factors.

Offline SASS inspection of the checksummed persistent variant confirms in-kernel E4M3-to-BF16 conversion and `UTCHMMA`. Its selection in live traces and its complete PV dataflow were not verified.

## GDN recurrence

vLLM ordinary decode uses a separate indexed causal convolution update, then packed Triton recurrence. The recurrence normalizes Q and K, computes GDN gates, applies decay and delta update, and updates FP32 state. SGLang’s automatic FlashInfer GDN decode requires BF16 state; with configured FP32 state it selects packed Triton recurrence. SGLang can fuse projection/convolution output unpacking with convolution-history update.

## Long prefill

Reconstructed vLLM settings predict FlashInfer FA2 prefill with 64-token kernel pages on SM100. The hybrid cache manager sizes GDN state as 3,207,168 bytes, equivalent to 1,566 FP8 KV tokens, rounded to a 1,600-token allocation block at 64-token alignment. With 128-token kernel pages, vLLM predicts TRTLLM-Gen prefill instead. Confirm the actual family and page size in a paired rerun before attributing a measured gap.

The inspected FA4 SM100 pipeline uses asynchronous `tcgen05` QK/PV, tensor memory, and separate load, MMA, softmax, correction, and store warps. The TRTLLM-Gen context cubin contains `UTMALDG.4D`, `UTCHMMA`, `LDTM`, and `MUFU.EX2`; its exact warp schedule was not verified.

## Scheduler and capture

vLLM schedules running requests before waiting requests and permits partial prefill within the shared token budget. Native shares prefill budget across lanes and uses per-lane tiles. A native pure-decode guard prevents CUDA graph replay on mixed prefill/decode steps. vLLM `FULL_AND_PIECEWISE` uses full graphs for uniform decode and piecewise graphs for eligible mixed steps. Its independent prefill token grid caps at 8,192, above the native 32K W5 forward.

The singleton MTP graph captures only a pure-decode K4 round. Prefill input changes break replay, so MTP remains on eager prefill in the native runtime. CUDA graph dispatch is fixed before model execution and cannot select MTP from target logits.

## Durable constraints

- Preserve FP32 scale arithmetic when packing FP8 projections.
- Pack only compatible 128-channel scale blocks.
- Keep MXFP8 group scales and page mapping explicit.
- Merge split-attention partials in FP32.
- Compare GDN convolution history and recurrent state over long forced continuations.
- Treat FA4 and TRTLLM-Gen performance claims as source observations, not measured native expectations.

## Provenance

- vLLM revision: `81a8cb063e6247fe254d7caf7e98a6691d71022a`
- SGLang revision: `f04d886bf79aa3d6237cbb12a3311838eb9f9bd8`
- FlashInfer revision: `77039051b0be921788dc2784ec41423b558878ec`
- DeepGEMM revision: `e3f6b84093dab1213f50cd1865d847365697ef23`
- CUTLASS revision: `b46b16d003484063bca4ed365e44095c4c6ed633`
- FA4 implementation revision: `aae6770da7dfd2bafd57d2d0ee3d1428b62f4334`
- TRTLLM-Gen artifact revision: `f50085594b9adcf9cbde7b8ea108c8587fa99b4c`

Historical native W3–W6 measurements are tied to revision `6b4f734c9c1a90ce16ff3b61ed9d3f0ad128c811`; W1–W2 comparisons use `bc07a3b56dc18a9bc32986d76c057a23953d8800`. This condensed research used static source reads and offline binary inspection; no GPU run was executed.
