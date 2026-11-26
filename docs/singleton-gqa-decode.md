# GQA decode and target verification

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| CUDA backend policy | Chooses the B200 split-KV policy or the generic SM100 path. | [Device policy](../src/runtime/backend/cuda_backend_policy.cpp#L18)<!-- prefix: "const bool b" --> |
| Batch executor | Selects direct decode lanes, binds scores and request state, and owns graph and scratch lifetimes. | [Batch selection](../src/decoder_batch.cpp#L354)<!-- prefix: "run = run_de" --> · [Graph dispatch](../src/text/decoder_graph.cpp#L83)<!-- prefix: "DecodeRun ru" --> · [Score allocation](../src/runtime/backend/cuda_resources.cpp#L509)<!-- prefix: "ExactGqaWork" --> |
| Target attention | Appends KV, reads each row's causal prefix, and gates the attention output before projection. | [Eager attention](../include/sglang/runtime/execution/decoder_forward_layers.h#L810)<!-- prefix: "if (batch.ta" --> · [Bound singleton](../include/sglang/runtime/execution/decoder_forward_layers.h#L686)<!-- prefix: "if (batch.ta" --> · [Bound verification](../include/sglang/runtime/execution/decoder_forward_layers.h#L706)<!-- prefix: "inline void " --> |

## Runtime path

Regular B200 batch replay supports one eligible direct decode lane. The first eligible step runs eagerly. The graph binds the live position, page table, KV, and score workspace for the selected request. At width one it requires inline score capacity through 5,120 visible tokens unless a shared score buffer is available. B200 does not use that buffer. Positive MTP uses a [separate verification graph policy](mtp.md#target-verification-graph). The standalone serial generator uses eager decode if its request can enter the split window. [Replay policy](../src/runtime/execution/decoder_execution_policy.cpp#L21)<!-- prefix: "bool Decoder" --> · [Warmup and score check](../src/text/decoder_graph.cpp#L30)<!-- prefix: "bool replay_" --> · [Serial selection](../src/decoder.cpp#L678)<!-- prefix: "bool Decoder" -->

### Generic SM100 singleton

The generic policy enables narrow attention for one direct batch decode lane. With sufficient workspace the page-parallel reader covers 1,536–2,560 visible tokens. Other rows with inline scores use ordered score and replay kernels. Eager rows beyond inline capacity use the scratch-free reader unless they qualify for shared scores. [Narrow selection](../src/text/decoder_graph.cpp#L92)<!-- prefix: "first.narrow" --> · [Narrow window](../src/mxfp8_gqa.cu#L196)<!-- prefix: "constexpr in" --> · [Inline scores](../src/runtime/model/qwen_request.cpp#L198)<!-- prefix: "ExactGqaWork" --> · [Direct reader](../include/sglang/runtime/execution/decoder_forward_layers.h#L617)<!-- prefix: "return mxfp8" -->

From 32,768 through 36,864 visible tokens, the generic batch can bind a single shared score buffer for eager or graph decode. Every GQA cache and the page table must cover the live rows. The batch allocates 3,538,944 bytes on first use for 24 FP32 scores per position. Each bind rounds its stride to a 128-token page. GQA layers and later requests reuse the buffer in stream order. [Eligibility and stride](../src/text/decoder_graph.cpp#L9)<!-- prefix: "ExactGqaWork" --> · [Replay binding](../src/text/decoder_graph.cpp#L57)<!-- prefix: "if (lane.sin" --> · [Score and replay](../include/sglang/text/gqa_ordered_launch.cuh#L4)<!-- prefix: "cudaError_t " -->

### Scratch-free ordered attention

Restored singleton prefill and allocation-limited prefill can reach the same reader as eager decode without score scratch. The public attention call keeps its tiny reader when the query end is below eight tokens. At eight or more it stages 32 keys for two query heads at a time. It applies the causal mask and accumulates values in token order. A standalone serial graph whose reserved KV exceeds inline score capacity uses the same tiled reader with a device position. Replay loads that position on each launch. [Eager dispatch](../src/mxfp8_gqa.cu#L803)<!-- prefix: "cudaError_t " --> · [Serial graph dispatch](../include/sglang/runtime/execution/decoder_forward_layers.h#L625)<!-- prefix: "if (capacity" --> · [Ordered kernel](../src/mxfp8_gqa_prefill.cu#L13)<!-- prefix: "__global__ v" --> · [Tile geometry](../include/sglang/text/mxfp8_gqa_prefill_common.cuh#L40-L41)<!-- prefix: "constexpr in" -->

## Split-KV target attention

B200 selects split-KV attention for eligible eager and bound text target rows from 2,048 through 262,144 visible tokens, including the current token. Bound kernels check the live position on device. Inside the window the reader decodes paged MXFP8 K/V to BF16 for Tensor Core attention, merges FP32 split partials, and rounds to BF16. Flagged heads use FP64 replay. Outside the window the target uses ordered attention. The split reductions can change scores and greedy IDs. [Window](../include/sglang/decoder_cuda.cuh#L312)<!-- prefix: "inline const" --> · [Bound route and fallback](../include/sglang/text/gqa_target_split.cuh#L98)<!-- prefix: "cudaError_t " --> · [Combine and FP64 replay](../include/sglang/text/gqa_target_split.cuh#L6)<!-- prefix: "__global__ v" -->

### Batched eager verification

The batched path requires vector attention, the full-context policy, and 2–20 rows inside the split window. It runs Q/K normalization and RoPE, KV append, split reads, and sigmoid gating in stream order. Each row keeps its workspace, KV page table, and causal position. Other groups run preprocessing, append, and gating per row. They can still group eligible reads. Bound replay uses its own dispatch. The [MTP verifier](mtp.md#runtime-path) owns acceptance and the committed KV prefix. [Eligibility and launch order](../include/sglang/text/gqa_verify_batch.h#L69)<!-- prefix: "inline bool " --> · [Eager fallback](../include/sglang/runtime/execution/decoder_forward_layers.h#L850)<!-- prefix: "if (!try_gqa" -->

TP2 vector-backend lanes below the split range batch their exact-scores and replay passes into one eager kernel pair. A side-effect-free eligibility check gates the route, and constrained or non-vector batches keep the per-row passes. [Short verify](../include/sglang/text/gqa/short_verify.cuh#L65)<!-- prefix: "cudaError_t " --> · [Eligibility](../include/sglang/text/gqa/short_verify.cuh#L45)<!-- prefix: "bool decoder" -->

Under the full-context policy, ordinary eager launches can group 10–20 rows. All inputs must satisfy Tensor Core alignment, and each row must average at most ten 64-token chunks per split. This route uses one shared-memory copy stage. It transposes QK across 16 keys and eight head slots (six active and two padded), then restores token order before masking and softmax. Other valid eager groups use at most five rows per launch. Narrow, singleton, and bound launches use two stages. Each row reserves scratch for up to 16 splits. [Grouping](../include/sglang/text/gqa_target_split.cuh#L65)<!-- prefix: "int eager_ta" --> · [Launch geometry](../src/text/gqa_decode_split.cu#L650)<!-- prefix: "cudaError_t " --> · [Transposed scores](../src/text/gqa_decode_split.cu#L289)<!-- prefix: "__device__ v" -->

### Shared KV reads for long K4 packets

A full 20-row eager packet can share KV reads across four distinct cache and page-table identities. Each request contributes five consecutive zero-based positions within one 64-token chunk. Positions must be at least 32,768 and below 40,960. Each row needs readable KV, valid scratch, and Tensor Core alignment. One CTA per request, KV head, and split stages K/V and scales for its five rows. Each row masks its own tail and writes independent split partials. Other packets use ordinary split attention. [Packet dispatch](../include/sglang/text/gqa_target_split.cuh#L136)<!-- prefix: "if (count ==" --> · [Eligibility](../include/sglang/text/gqa/gqa_decode_reuse.cuh#L95)<!-- prefix: "bool make_re" --> · [Causal operands](../include/sglang/text/gqa/gqa_decode_reuse.cuh#L66)<!-- prefix: "if (chunk \u003c " -->

### Tensor Core reads across long K4 packets

The full-context eager Tensor Core verifier first selects contiguous suffixes of one to five rows from positions 32,768 onward. Selected suffixes can cross a 64-token chunk or 128-token page boundary. Duplicate page tables in the wave prevent selection. Every selected row must be unconstrained and have readable KV and valid, aligned buffers. Earlier rows and ineligible requests run first in their original packets. The verifier then appends selected KV and submits one long reader and combine per GQA layer. [Sequence collection](../include/sglang/text/gqa_verify_batch.h#L159)<!-- prefix: "inline int c" --> · [Guard and suffix selection](../include/sglang/text/gqa_verify_batch.h#L191)<!-- prefix: "bool try_gqa" -->

One block per request, KV head, and split stages each 64-token chunk by TMA for up to five rows. Splits follow the first selected position and the final split extends through the newest row. Each row masks later keys. BF16 Tensor Core multiplies form FP32 scores. The value multiply accumulates the rounded BF16 weight and its rounded BF16 residual. This arithmetic can change scores relative to the ordinary split reader. [Paged read](../src/text/gqa_decode_long.cu#L178)<!-- prefix: "__device__ v" --> · [Split bounds](../src/text/gqa_decode_long.cu#L465-L467)<!-- prefix: "const int to" --> · [Probability split](../src/text/gqa_decode_long.cu#L356)<!-- prefix: "const auto h" -->

The reader uses one expanded BF16 K slot and alternates between two expanded BF16 V slots. It refills a slot after the last QK or PV consumer of that slot finishes. The long verifier combines a fixed 16 splits in the existing order. Flagged heads still replay in FP64. [Operand slots](../src/text/gqa_decode_long.cu#L84)<!-- prefix: "struct align" --> · [Refill barriers](../src/text/gqa_decode_long.cu#L252)<!-- prefix: "constexpr in" --> · [Fixed split combine](../include/sglang/text/gqa_target_split.cuh#L163)<!-- prefix: "cudaError_t " -->

## Constraints and failure behavior

A clean shared-score OOM makes the generic batch remember the allocation failure. Long singleton steps use direct eager attention, while short replay remains eligible. Other allocation errors reach batch recovery. A recoverable graph rejection drains the stream, destroys its graph handles, and disables replay for that width. Eager decode can still use the retained score buffer. Batch cleanup drains the stream and destroys graphs before freeing it. [Allocation fallback](../src/runtime/backend/cuda_resources.cpp#L509)<!-- prefix: "ExactGqaWork" --> · [Graph recovery](../src/runtime/backend/cuda_resources.cpp#L568)<!-- prefix: "const auto f" --> · [Teardown](../src/runtime/backend/cuda_resources.cpp#L247)<!-- prefix: "CleanupError" -->

For the 20-row shared reader, a clean decline or a recognized optional setup or launch rejection returns that packet to the ordinary split route. A later packet can try again. A preexisting or distinct recorded CUDA error prevents fallback and remains recorded. The code clears only a recorded status matching the failed call and gives the returned status precedence on hard failure. [Rejection handling](../include/sglang/text/gqa/gqa_decode_reuse.cuh#L134)<!-- prefix: "std::optiona" -->

A clean device, shared-memory, or tensor-map setup decline sends the selected rows to the existing packet split dispatch. That dispatch reads their appended KV before gating. Pending errors and failed setup, reader, or combine calls propagate without retry. A recorded terminal CUDA error takes precedence over a nonterminal returned status. [Long setup](../src/text/gqa_decode_long.cu#L566)<!-- prefix: "std::optiona" --> · [Fallback after append](../include/sglang/text/gqa_verify_batch.h#L260)<!-- prefix: "if (const au" --> · [Status precedence](../src/text/gqa_decode_long.cu#L559)<!-- prefix: "cudaError_t " -->

## Verification

- Follow the [batch CUDA Graph runbook](../tests/batch/README.md) on a reserved B200. The singleton mode compares IDs and full BF16 scores against eager at slot 31. It exercises request replacement and width-local launch rejection. It requires replay at 5,120 visible tokens and eager execution at 5,121. The `mxfp8-gqa-check` operator target compares score-backed and direct BF16 outputs on reversed pages. It also checks invalid argument guards and bound replay. [Singleton fixture](../tests/batch/batch_graph_lanes.cpp#L214)<!-- prefix: "void singlet" --> · [Exact operator check](../tests/mxfp8_gqa_check.cu#L541)<!-- prefix: "void check_e" -->
- Follow the [long decode runbook](../tests/decoder/README.md#wide-long-context-decode). It covers split and combine geometry, shuffled pages, alignment, full context, FP64 recovery, and decode dispatch. Enable `--long-decode-check=y` and build `gqa-split-fp64-check`. Run it on the same reserved GPU for independent FP64 attention checks. For a changed reduction, use the [operator qualification policy](performance.md#operator-qualification). [FP64 check](../tests/decoder/gqa_split_fp64_check.cpp#L181)<!-- prefix: "int main() t" -->
- Run the [K4 crossing fixture](../tests/decoder/README.md#k4-boundary-crossings) and [mixed-request fixture](../tests/decoder/README.md#mixed-k4-verification) for long attention. Check reader and combine counts, output IDs, ordinary remainder, clean decline, and error propagation. Follow the [MTP verification runbook](mtp.md#verification) for acceptance cuts and continuation.

### Scratch-free attention checks

Build both binaries with the [long decode runbook](../tests/decoder/README.md#wide-long-context-decode). Set `serial_ids` to a whitespace-delimited file of at least 8,195 valid checkpoint token IDs. Keep the reserved B200 and the `out` and `checkpoint` settings from that runbook. For the checkpoint fixture, start an isolated [LMCache daemon](serving/lmcache.md#start-the-daemon) on the same GPU with namespace `sglang`.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$out/bin/long-decode-check" serial
CUDA_VISIBLE_DEVICES="$GPU_UUID" LMCACHE_SOCKET="$LMCACHE_SOCKET" \
    "$out/bin/long-decode-routes" "$checkpoint" --serial-prefill "$serial_ids"
```

The operator fixture compares eager output against a single captured node while changing the device position. The checkpoint fixture checks restored singleton prefill and a three-row tail after a real arena allocation failure. It counts tiled reader launches and checks that a subsequent unforced request recovers. [Operator fixture](../tests/decoder/long_decode_check.cu#L470)<!-- prefix: "void check_s" --> · [Checkpoint fixture](../tests/decoder/long_decode_routes.cpp#L431)<!-- prefix: "void serial_" -->

The earlier packed-runtime notes and historical probes remain in the [archived guide](https://github.com/para1lel/sglang.cpp/blob/2676f3320cbfbe0f09f5d1193f67ec5abcf67639/docs/singleton-gqa-decode.md).
