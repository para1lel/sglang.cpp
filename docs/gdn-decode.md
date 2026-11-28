# GDN decode with ReplaySSM

## Responsibilities and sources of truth

Each target request keeps one FP32 recurrent checkpoint and a bounded replay log per GDN layer and rank. The log stores processed keys, recurrent updates and decays, and raw BF16 convolution inputs. A cursor identifies the oldest row and the committed and tentative lengths. [State](../include/sglang/text/gdn/replay_ssm.h#L18)<!-- prefix: "struct State" --> · [Request storage](../src/runtime/model/qwen_request.cpp#L56)<!-- prefix: "template \u003cty" --> · [ReplaySSM](https://dao-lab.ai/blog/2026/replayssm/)

## Runtime path

Decode reads the checkpoint and committed log, then advances with the current row. Verification adds tentative rows to the same log. A full or partial commit advances the cursor by the consumed prefix and discards the rest. A zero-row commit preserves the committed length. Once the log holds at least 16 committed rows, the next launch folds the oldest 16 into the checkpoint. K4 reserves 25 ring slots for the window, rollover and tentative rows. [Window](../include/sglang/text/gdn/replay_ssm.h#L7)<!-- prefix: "inline const" --> · [Cursor advance](../src/text/gdn_replay_ssm.cu#L450)<!-- prefix: "__global__ v" --> · [Target commit](../src/runtime/execution/decoder_verify_workspace.cpp#L342)<!-- prefix: "void TargetV" -->

[Frozen packed K0 runtime and reference comparison](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/gdn-decode.md#runtime-path)

### Packed eager and captured verification

The verifier launches convolution, recurrence and finish in order. Eager groups use single-lane or packed jobs of up to 16 lanes per submission. TP1 and TP2 graphs bind live request state and use the same ReplaySSM transitions. The verifier applies acceptance outside capture. [Dispatch](../src/runtime/execution/decoder_forward_attention.cpp#L136)<!-- prefix: "void run_gdn" --> · [Kernel sequence](../src/text/gdn_replay_ssm.cu#L585)<!-- prefix: "cudaError_t " --> · [TP2 bindings](../src/runtime/execution/tp_verify_graph.cpp#L768)<!-- prefix: "return true;" -->

Each recurrence warp carries four value channels and interleaves their independent work. Committed history replays in token order. For each channel, the tentative update preserves the original FP32 fused multiply-add order and XOR-shuffle reduction order. The shared path uses the same checkpoint and replay-log layout at both widths. [Warp shape](../src/text/gdn_replay_ssm.cu#L19)<!-- prefix: "constexpr in" --> · [History replay](../src/text/gdn_replay_ssm.cu#L292)<!-- prefix: "__device__ v" --> · [Tentative recurrence](../src/text/gdn_replay_ssm.cu#L351)<!-- prefix: "__device__ D" -->

Tensor Core target verification can project B/A together in one launch for any valid count of 2–160 rows, including unequal packet tails. It requires checkpoint gate dimensions, nonnull outputs, and 16-byte-aligned weights and inputs. Each block assigns one warp per row, and grid Y selects groups of up to 20 rows. Inactive tail warps skip output. Full and rank-local operators preserve the existing FP32 dot-product and reduction order and BF16 rounding. If the full group declines, the runtime tries grouped gates and then separate B/A projections. CUDA errors propagate. [Gate dispatch](../src/runtime/execution/decoder_forward_attention.cpp#L90)<!-- prefix: "void project" --> · [Kernel and tail](../src/text_decode_bandwidth.cu#L377)<!-- prefix: "const int ch" --> · [Packet launch](../src/text_decode_bandwidth.cu#L542)<!-- prefix: "std::optiona" -->

The replay log stays in the request. Terminal LMCache stores materialize committed GDN into the usual convolution and recurrent spans on each rank. Prefill captures the state at each selected prompt endpoint. [Materialization](../src/runtime/model/qwen_request.cpp#L421)<!-- prefix: "void Request" --> · [Checkpoint boundaries](prefix-cache.md#checkpoint-boundaries)

## Constraints and failure behavior

GDN uses an FP32 `[V,K]` recurrent state and three BF16 history taps. The checkpoint has 16 Q/K heads, 48 value heads, and head dimension 128, with rank-local shards at TP2. [Checkpoint dimensions](../include/sglang/text_attention.h#L10)<!-- prefix: "inline const" -->

The 48-layer replay log and its cursors occupy 64,128,576 B per TP1 request or 32,064,576 B per TP2 rank. The existing recurrent checkpoint and convolution history have their own storage. [Log bytes](../src/runtime/model/qwen_request.cpp#L61)<!-- prefix: "static const" --> · [Rank shape](../include/sglang/text/gdn_local.h#L27)<!-- prefix: "using GdnLoc" -->

CUDA submission errors propagate. An execution failure can leave request state advanced. Target-only failure outside graph fallback or prefix recovery disables the batch. [Batch failure contract](batch-graphs.md#constraints-and-failure-behavior)

## Verification

For current ReplaySSM, compare outputs and materialized checkpoints against an independent recurrence. Exercise zero, partial and full consumption, replacement rows, rollover, and mixed-length packed lanes. Rebind requests before replaying each captured path and verify that an actual graph launch occurred. Compare fixed-history logits and continuation across restore. Report free-running output differences separately. [Operator qualification](performance.md#operator-qualification)

The isolated B200 probe matched the packaged `81b477126d5b` build bitwise in 40 cases at full and rank-local widths. It compared outputs, replay state, and materialized checkpoints for 1, 2, 8 and 16 lanes with one to five rows. The 12-layer captured probe used 16 rank-local lanes accepting all five rows. Its median fell from 62.912 to 50.116 µs per layer, including convolution, recurrence, finish and commit. The packaged object's encoded instructions matched the timed variant.

The packaged TP1 checkpoint check passed six cases and graph reuse at accepted cuts of 5, 2 and 0 rows.

Compare every row count from 2 to 160 against packeted B/A for full and rank-local widths in eager and captured execution. Require bitwise BF16 output, intact guards and zero submissions for invalid counts or misalignment. The isolated B200 gate check covered all 159 counts at both widths. A fixed-input TP1 four-by-five model witness matched complete BF16 logits in warmup, capture and replay. Already coalesced packets can have the same gate count in both versions. [Gate fixture](../tests/decoder/decode_bandwidth_check.cpp#L110)<!-- prefix: "void verific" -->

### TP2 80-row caller checks

Compare complete verifier results on both ranks. Replay one- through four-row prefixes and check carry mutation detection. Cover alignment, wider controls, guarded buffers, and direct operator capture.

Use the [batched target GDN fixture](../tests/decoder/README.md#batched-target-gdn) for scalar comparison and graph reuse. In operator replay, require bitwise parity for BF16 observables, output and convolution history, and FP32 recurrent state. Check inactive output sentinels and compare eager execution with graph replay after rebinding state pointers. Cover zero, partial, and full active counts across the submitted buckets. Use unfiltered initcheck to observe both the writer and the reader.

Time convolution, recurrence, and finish together with paired CUDA events. Exclude capture and bindings. Report submitted and active widths and the width-eight boundary.
