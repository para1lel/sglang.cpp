# Packed GDN decode

## Responsibilities and sources of truth

Grouped target verification owns rank-local recurrent carry and convolution history. [Grouped forward](../include/sglang/runtime/execution/decoder_forward_layers.h#L470)<!-- prefix: "void run_gdn" --> · [Former packed K0 architecture](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-decode.md#responsibilities-and-sources-of-truth)

## Runtime path

Under two ranks, eager verify advance packs uniform-row rounds into one three-launch call per GDN layer. Packed lanes must share one stream and rank-local GDN state, up to sixteen lanes. Mixed rows, gate mismatches, or a declined route keep the per-layer launches; results are unchanged. [Packed replay](../include/sglang/text/gdn_local.h#L113)<!-- prefix: "cudaError_t " --> · [Lane gates](../include/sglang/runtime/execution/decoder_forward_layers.h#L558)<!-- prefix: "if (!verific" --> · [Fallback](../include/sglang/runtime/execution/decoder_forward_layers.h#L621)<!-- prefix: "} else if (!" -->

[Frozen packed K0 runtime and reference comparison](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-decode.md#runtime-path)

### TP2 80-row target verification

On B200 TP2, eager verification uses the wide GDN recurrence for 16 complete five-row K4 lanes with packed activations, target Tensor Cores, and rank-local state. Each rank has 24 value heads of 128 channels. Eight warps process four value channels each. Each block covers 32 channels, so each lane needs 96 blocks instead of 192. Scalar streaming FP32 state accesses preserve alignment requirements and per-channel arithmetic. Ordinary verification, fixed verifier replay, 32 local lanes, and full-model GDN retain their kernels. [Caller](../include/sglang/runtime/execution/decoder_forward_layers.h#L424)<!-- prefix: "inline bool " --> · [Geometry](../src/text/gdn_verify.cu#L149)<!-- prefix: "__global__ v" --> · [Rank-local dispatch](../src/text/gdn_verify.cu#L377)<!-- prefix: "if constexpr" -->

Each verification writes tentative history and recurrent state. A full commit swaps in the tentative tail. A partial commit replays the accepted prefix from the initial state and saved inputs before swapping. A zero-row commit keeps the initial state. Submission errors propagate through the existing cleanup path. [Commit](../src/decoder_verify.cpp#L512)<!-- prefix: "void TargetV" --> · [Launch status](../src/text/gdn_verify.cu#L351)<!-- prefix: "cudaError_t " -->

## Constraints and failure behavior

Each live row updates its bound request's FP32 `[V,K]` state and three BF16 history taps in place. The checkpoint uses 16 Q/K heads, 48 value heads, and head dimension 128. Each kernel checks the current active count before it accesses state or writes outputs. Padding can carry null bindings and leaves recurrent state, history, and outputs untouched, including when the active count is zero. [Checkpoint dimensions](../include/sglang/text_attention.h#L10)<!-- prefix: "inline const" -->

A wide-path execution failure after convolution can leave history advanced. A target-only execution error outside graph fallback or prefix recovery disables the batch. See the [packed batch failure contract](batch-graphs.md#constraints-and-failure-behavior).

## Verification

### TP2 80-row caller replay

Compare complete verifier results on both ranks. Replay one- through four-row prefixes and check carry mutation detection. Cover alignment, full-model and wider controls, guarded buffers, and direct operator capture. [Frozen W4 target interval and prefix replay receipts](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-decode.md#tp2-80-row-caller-replay)


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

Time convolution, recurrence, and finish together with paired CUDA events. Exclude capture and bindings. Report submitted and active widths and the width-eight boundary. [Frozen workload and isolated-to-serving projection](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-decode.md#verification)
