# GDN decode with ReplaySSM

## Responsibilities and sources of truth

Each target request keeps one FP32 recurrent checkpoint and a bounded replay log per GDN layer and rank. The log stores processed keys, recurrent updates and decays, and raw BF16 convolution inputs. A cursor identifies the oldest row and the committed and tentative lengths. [State](../include/sglang/text/gdn/replay_ssm.h#L18)<!-- prefix: "struct State" --> · [Request storage](../src/runtime/model/qwen_request.cpp#L55)<!-- prefix: "template <ty" --> · [ReplaySSM](https://dao-lab.ai/blog/2026/replayssm/)

## Runtime path

Decode reads the checkpoint and committed log, then advances with the current row. Verification adds tentative rows to the same log. A full or partial commit advances the cursor by the consumed prefix and discards the rest. A zero-row commit preserves the committed length. Once the log holds at least 16 committed rows, the next launch folds the oldest 16 into the checkpoint. K4 reserves 25 ring slots for the window, rollover and tentative rows. [Window](../include/sglang/text/gdn/replay_ssm.h#L7)<!-- prefix: "inline const" --> · [Cursor advance](../src/text/gdn_replay_ssm.cu#L419)<!-- prefix: "__global__ v" --> · [Target commit](../src/runtime/execution/decoder_verify_workspace.cpp#L48)<!-- prefix: "void TargetV" -->

[Frozen packed K0 runtime and reference comparison](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-decode.md#runtime-path)

### Packed eager and captured verification

The verifier launches convolution, recurrence and finish in order. Eager groups use single-lane or packed jobs of up to 16 lanes per submission. TP1 and TP2 graphs bind live request state and use the same ReplaySSM transitions. The verifier applies acceptance outside capture. [Dispatch](../src/runtime/execution/decoder_forward_attention.cpp#L75)<!-- prefix: "void run_gdn" --> · [Kernel sequence](../src/text/gdn_replay_ssm.cu#L554)<!-- prefix: "cudaError_t " --> · [TP2 bindings](../src/runtime/execution/tp_verify_graph.cpp#L247)<!-- prefix: "return true;" -->

The replay log stays in the request. Terminal LMCache stores materialize committed GDN into the usual convolution and recurrent spans on each rank. Prefill captures the state at each selected prompt endpoint. [Materialization](../src/runtime/model/qwen_request.cpp#L366)<!-- prefix: "void Request" --> · [Checkpoint boundaries](prefix-cache.md#checkpoint-boundaries)

## Constraints and failure behavior

GDN uses an FP32 `[V,K]` recurrent state and three BF16 history taps. The checkpoint has 16 Q/K heads, 48 value heads, and head dimension 128, with rank-local shards at TP2. [Checkpoint dimensions](../include/sglang/text_attention.h#L10)<!-- prefix: "inline const" -->

The 48-layer replay log and its cursors occupy 64,128,576 B per TP1 request or 32,064,576 B per TP2 rank. The existing recurrent checkpoint and convolution history have their own storage. [Log bytes](../src/runtime/model/qwen_request.cpp#L61)<!-- prefix: "static const" --> · [Rank shape](../include/sglang/text/gdn_local.h#L27)<!-- prefix: "using GdnLoc" -->

CUDA submission errors propagate. An execution failure can leave request state advanced. Target-only failure outside graph fallback or prefix recovery disables the batch. [Batch failure contract](batch-graphs.md#constraints-and-failure-behavior)

## Verification

For current ReplaySSM, compare outputs and materialized checkpoints against an independent recurrence. Exercise zero, partial and full consumption, replacement rows, rollover, and mixed-length packed lanes. Rebind requests before replaying each captured path and verify that an actual graph launch occurred. Compare fixed-history logits and continuation across restore. Report free-running output differences separately. [Operator qualification](performance.md#operator-qualification)

### Historical TP2 80-row caller replay

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
