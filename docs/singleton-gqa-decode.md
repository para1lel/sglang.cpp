# Score-backed singleton GQA decode

The score-backed singleton descriptions below cover the earlier packed
runtime. Current B200 text decode and verification are described under
[split-KV target attention](#split-kv-target-attention).

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Batch scheduler | Selects eligible text decode steps. | [Batch options](../src/decoder_batch.cpp#L620)<!-- prefix: "const auto r" --> · [Lane selection](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L486)<!-- prefix: "first.narrow" --> |
| Batch execution owner | Allocates and releases the shared score buffer. | [Allocation](../src/decoder_resources.cpp#L551)<!-- prefix: "ExactGqaWork" --> · [Device release](../src/decoder_resources.cpp#L259)<!-- prefix: "CleanupError" --> |
| Exact attention | Computes FP32 scores and consumes them in token order. | [Score and replay launches](../include/sglang/text/gqa_ordered_launch.cuh#L4)<!-- prefix: "cudaError_t " --> |

## Runtime path

The batch selects this path for one ordinary text decode row with 32,768–36,864 visible tokens, including the current token. The step needs zero prefill rows, no MTP work, and no rotary position override. Both FP8 prefill policies support it. Independent full-prefix-hit scoring can share the step. The standalone serial decoder and the eager remainder of a packed group keep their existing attention paths. [Window](../include/sglang/decoder_internal.h#L481)<!-- prefix: "inline const" --> · [Packed remainder](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L573)<!-- prefix: "if (remainin" -->

With tensor-core prefill and no MTP work, a prefill step can reuse the buffer for an ordinary text lane's final one-row prompt grant. It can also reuse it for an ordinary decode row when all other work in the step consists of multi-row prefill or cache-hit scoring. These eager rows qualify above their inline score capacity and through 36,864 visible tokens. Each layer's score and replay launches run in stream order before the next row uses the buffer. [Batch selection](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/decoder_batch.cpp#L675)<!-- prefix: "const auto e" --> · [Eager binding](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L421)<!-- prefix: "void prepare" -->

The batch stream owner allocates one buffer on first eligible use. It allocates before graph capture or eager submission. The bound is 3,538,944 device bytes (3.375 MiB) for 24 FP32 scores per position across 36,864 positions. All GQA layers and later requests share the buffer in stream order. It adds zero request-owned bytes. Request KV and existing graph scratch have separate allocations. [Query heads](../include/sglang/mxfp8_gqa.h#L11)<!-- prefix: "inline const" --> · [Resident admission](batch-graphs.md#resident-memory-admission)

The step rounds the live visible length up to a 128-token page for the per-head stride. It requires a page table and K/V payload and scale buffers with sufficient declared capacity in every attention layer. Eager decode passes the score descriptor by value. With graphs enabled, the first eligible width-one step warms up eagerly. Replay uploads the live position, page table, KV cache, and score descriptor. The same width-one executable can use short inline scores or the batch buffer. Both score-backed routes round the attention result to BF16. [Eligibility and stride](../src/text/decoder_graph.cpp#L9)<!-- prefix: "ExactGqaWork" --> · [Eager read](../src/decoder_forward.cpp#L510)<!-- prefix: "if (singleto" --> · [Warmup](../src/text/decoder_graph.cpp#L36)<!-- prefix: "auto& warmed" --> · [Replay binding](../src/text/decoder_graph.cpp#L66)<!-- prefix: "}" -->

Eager final-prompt replay stages packed V and its scales in two shared-memory buffers at 8,192 or more visible tokens when their bases are 16-byte and 8-byte aligned, respectively. Each CTA handles one query head and 64 value channels. Asynchronous copies follow the page table and zero-fill the invisible part of the last tile. The path reuses the score producer and preserves each channel's token-ordered FP32 recurrence. Eager decode, graph replay, shorter contexts, and misaligned buffers keep the existing replay. [Staged replay](../include/sglang/text/gqa_prefill_replay.cuh#L34)<!-- prefix: "__global__ v" -->

## Historical vectorized ordered replay

On B200, `tensor-core-mtp-target` uses vectorized V loads for score-backed text decode and verification. Each CTA stages 32 channels across a 32-token tile of decoded MXFP8 V. The kernel retains the score reduction, ascending FP32 recurrence, division, and BF16 rounding. When a tile's scores are at or below a finite running maximum, the history multiplier is one, so the numerator FMA reads the prior numerator directly. Each row reads only its visible positions, including after a partially accepted round. Prefill, draft attention, media, and GDN keep their existing dispatch.

## Split-KV target attention

The B200 text target selects split-KV attention for eligible decode and verification rows from 2,048 through 262,144 visible tokens. [Split target dispatch](../include/sglang/text/gqa_target_split.cuh#L6)<!-- prefix: "__global__ v" -->

Each verification row uses its own causal position. The kernel reads paged MXFP8 K/V, decodes it to BF16 for Tensor Core attention, and merges splits in FP32 before rounding to BF16. Ineligible heads in the window use the existing wide FP64 replay. The changed dot-product and softmax reductions can change target scores and greedy token IDs. Outside the window, rows use ordered attention on the current request state. Earlier split rows can affect that state.

Batch graph replay checks each live position on device. The standalone generator uses eager decode if the request can enter the window. Prefill, draft attention, media, and pointwise scoring use their existing paths. The split operator uses the existing checkpoint, KV capacity, and logits scratch.

In the ordinary eager route, the full-context policy groups 10–20 verification
rows when every row has aligned Tensor Core inputs and averages at most ten
64-token chunks per split. These launches use one shared-memory copy stage.
Other valid eager groups use launches of at most five rows. Singleton, narrow
eager, and bound launches use two stages. Each row has scratch for up to 16 splits.
[Eager grouping](../include/sglang/text/gqa_target_split.cuh#L48)<!-- prefix: "int eager_ta" --> ·
[Stage dispatch](../src/text/gqa_decode_split.cu#L645)<!-- prefix: "cudaError_t " -->

Each transposed QK multiply covers 16 keys and eight query head slots.
Six heads are active and two are zero padded. Warp shuffles restore token order
before the causal mask, softmax, and value accumulation. Each score keeps the
same 32-channel group order and scales. The split merge and FP64 replay stay shared.
[Transposed scores](../src/text/gqa_decode_split.cu#L297)<!-- prefix: "__device__ v" --> ·
[Single-stage call](../src/text/gqa_decode_split.cu#L576)<!-- prefix: "process_tile" -->

One stage reduces shared-memory use and lets three CTAs fit per B200 SM.
Each warp waits for its copies before reading the tile and synchronizes before
reusing the stage. The CTA computes one softmax weight per warp and query head
and reuses it across value channels. The merge preserves warp order for the
maximum, mass, and per-channel FP32 accumulation. The same combine kernel
reduces split partials and sends flagged heads to FP64 replay.
[Shared-memory bound](../src/text/gqa_decode_split.cu#L78)<!-- prefix: "static_asser" --> ·
[Tile loop](../src/text/gqa_decode_split.cu#L569)<!-- prefix: "if constexpr" --> ·
[Warp weights](../src/text/gqa_decode_split.cu#L499)<!-- prefix: "__device__ v" --> ·
[Cross-split combine](../include/sglang/text/gqa_target_split.cuh#L6)<!-- prefix: "__global__ v" -->

The earlier bounded and long profiles used explicit launch settings. Their
[historical paired results](https://github.com/para1lel/sglang.cpp/blob/f6c74b50d57dff48137d2e4a19c0a9c99a15bd20/docs/serving/paired-split-attention-results.md)
record the measured configuration.

### Batched eager verification

The B200 eager verifier batches Q/K preprocessing, KV append, and attention gating
for groups of 2–20 target rows. It requires vector attention and the full-context
split-KV policy. Every row must be in the split window. Other eager groups outside
the long Tensor Core route use per-row preprocessing, append, and gating.
They can still group eligible split reads.
Graph replay uses bound verification dispatch.
[Dispatch](../src/decoder_forward.cpp#L748)<!-- prefix: "if (!try_gqa" --> ·
[Eligibility](../include/sglang/text/gqa_verify_batch.h#L66)<!-- prefix: "inline bool " --> ·
[Bound verification](../src/decoder_forward.cpp#L600)<!-- prefix: "void attend_" -->

Wide K4 verification tries the [long Tensor Core reader](#tensor-core-reads-across-long-k4-packets)
first. If its geometry check declines, these rules apply separately to each 20-row packet.

Each row supplies its own workspace, absolute position, and request KV page table.
The stream orders BF16 Q/K normalization and RoPE, MXFP8 K/V append,
the existing grouped split attention, and in-place sigmoid gating.
Preprocessing, append, and gating each launch one block per row and applicable head.
Each attention row reads through its own position, even after later rows append KV.
Output quantization and projection follow gating. The [MTP verifier](mtp.md#runtime-path)
owns acceptance and the committed KV prefix.
[Launch sequence](../include/sglang/text/gqa_verify_batch.h#L91)<!-- prefix: "check(gqa_ve" --> ·
[Preprocessing](../src/text_attention_gqa.cu#L105)<!-- prefix: "__global__ v" --> ·
[Append](../include/sglang/text/gqa_target_split.cuh#L156)<!-- prefix: "__global__ v" --> ·
[Gating](../src/text_attention_gqa.cu#L172)<!-- prefix: "__global__ v" -->

### Shared KV reads for long K4 packets

The B200 full-context eager split path can share KV reads in a 20-row K4 target
packet. It needs four distinct cache and page-table identities, each with five
consecutive positions in one 64-token chunk. Every zero-based position must
be at least 32,768 and below 40,960. All rows need readable KV, valid scratch,
and aligned Tensor Core inputs. Each packet in a wider wave qualifies separately.
Short, partial, cross-chunk, unaligned, or otherwise ineligible packets use the
ordinary split rules. Bound replay, draft, and nonverification routes keep their
existing dispatch.
[Packet dispatch](../include/sglang/text/gqa_target_split.cuh#L119)<!-- prefix: "if (count ==" --> ·
[Eligibility](../include/sglang/text/gqa/gqa_decode_reuse.cuh#L94)<!-- prefix: "bool make_re" -->

One CTA per request, KV head, and split stages paged MXFP8 K/V and scales once
for its five query rows. The loader uses the newest row's visible tail. Each
query keeps its own causal mask and zeroes K and V operands past its position.
The reader keeps native BF16 MMA, FP32 accumulation and merge order, and the
existing FP64 replay for flagged heads. Each row writes its own split partials.
[Shared load](../include/sglang/text/gqa/gqa_decode_reuse.cuh#L48)<!-- prefix: "if (query_in" --> ·
[Causal operands](../include/sglang/text/gqa/gqa_decode_reuse.cuh#L66)<!-- prefix: "if (chunk \u003c " --> ·
[Split output](../include/sglang/text/gqa/gqa_decode_reuse.cuh#L81)<!-- prefix: "write_weight" -->

### Tensor Core reads across long K4 packets

The SM100 reader can group 20–160 rows from complete four-lane K4 packets.
It requires eager, unconstrained verification and the full 262,144-token context policy.
Each lane supplies five consecutive positions starting at 32,768 or later.
All five positions must fit in one 64-token chunk. Lanes can use different chunks.
Each lane needs a distinct page table, readable paged MXFP8 KV, and aligned buffers.
A geometry decline leaves the existing packet dispatch eligible.
[Group guard](../include/sglang/text/gqa_verify_batch.h#L100)<!-- prefix: "inline bool " --> ·
[Page and buffer checks](../include/sglang/text/gqa_verify_batch.h#L134)<!-- prefix: "for (int pri" -->

One block per lane, KV head, and split shares each 64-token KV chunk across
five rows and six query heads per row. TMA reads the existing paged payloads
and scales into shared memory. Each row keeps its own causal mask and FP32
split partials in the existing logits scratch. The combine retains FP64 replay
for flagged heads.
[Paged read](../src/text/gqa_decode_long.cu#L176)<!-- prefix: "__device__ v" --> ·
[Split partials](../src/text/gqa_decode_long.cu#L428)<!-- prefix: "auto& out = " --> ·
[Combine](../include/sglang/text/gqa_target_split.cuh#L145)<!-- prefix: "cudaError_t " -->

BF16 Tensor Core multiplies form FP32 scores. Online softmax uses FP32 mass
and rescaling. The value multiply accumulates two BF16 terms per weight:
the rounded weight and its rounded residual. This arithmetic can change scores
compared with the ordinary split reader.
[Probability split](../src/text/gqa_decode_long.cu#L350)<!-- prefix: "const auto h" --> ·
[Value accumulation](../src/text/gqa_decode_long.cu#L307)<!-- prefix: "mma(output, " -->

## Constraints and failure behavior

Short singleton replay remains eligible through 5,120 visible tokens. Decode-only singletons at 5,121–32,767 use direct ordered eager attention. Above 36,864 tokens or when the page table or a layer's buffers lack the required capacity, the new eager binding stays empty and existing dispatch rules apply. The batch buffer stays fixed even if the request reserves more KV. The legacy width-two through width-four graph keeps its separate request-owned long scores. [Replay limit](https://github.com/para1lel/sglang.cpp/blob/93648269d852e0fee78f7af4d1c3ffe4897aa361/src/text/decoder_graph_packed.cpp#L436)<!-- prefix: "if (count ==" --> · [Direct eager read](../src/decoder_forward.cpp#L515)<!-- prefix: "return mxfp8" --> · [Context limit](../include/sglang/text_attention.h#L9)<!-- prefix: "inline const" --> · [Legacy scores](batch-graphs.md#constraints-and-failure-behavior)

After clean score-buffer allocation OOM, the batch remembers the failure for its lifetime and uses direct ordered eager attention for long singletons. Short replay remains eligible. Other allocation errors propagate. A recoverable graph rejection retires the graph for that width. The eager fallback can use the retained score buffer. The buffer survives request retirement and KV growth. Batch cleanup drains the stream and destroys graph handles before releasing the allocation. Cleanup retains device resources while stream or graph completion is uncertain. [Allocation fallback](../src/decoder_resources.cpp#L559)<!-- prefix: "if (error.st" --> · [Graph recovery](../src/decoder_resources.cpp#L610)<!-- prefix: "const auto f" --> · [Graph teardown](../src/decoder_resources.cpp#L225)<!-- prefix: "CleanupError" -->

For the 20-row shared reader, a clean decline or recognized optional setup or
launch rejection uses ordinary split attention for that packet. A later packet can try again.
Hard setup, launch, and combine errors propagate. A pending CUDA error before
setup prevents shared dispatch and stays recorded. On setup or launch failure,
the reader clears only a recorded status that matches the returned status.
A distinct recorded error prevents optional fallback and stays recorded.
On hard failure, the returned status takes precedence.
[Rejection handling](../include/sglang/text/gqa/gqa_decode_reuse.cuh#L133)<!-- prefix: "std::optiona" --> ·
[Setup and launch](../include/sglang/text/gqa/gqa_decode_reuse.cuh#L147)<!-- prefix: "std::optiona" -->

Unsupported device, shared-memory, or tensor-map setup can prevent the long
Tensor Core read from launching. This decline requires clear CUDA state.
On that decline, the decoder reuses the KV already appended for each packet.
It reads through the existing split path before postgating. Pending errors and
failed CUDA setup, read, or combine calls propagate without retry.
[Long setup](../src/text/gqa_decode_long.cu#L556)<!-- prefix: "std::optiona" --> ·
[Packet fallback](../include/sglang/text/gqa_verify_batch.h#L157)<!-- prefix: "if (const au" -->

## Verification

Run the same-input dispatch fixture on a reserved B200. Compare eager and replay BF16 attention bitwise with ordered eager. Exercise 32,767–32,769 and 36,864–36,865 visible tokens. Rebind short and long requests to the same executable. Include a remapped page table and a partial last page. Verify one batch buffer allocation. Verify that request-owned score bytes stay constant.

The [short-prefill checker](../tests/mxfp8_gqa_short_prefill_check.cu#L119)<!-- prefix: "void check_s" --> compares staged and ordered replay at the visibility threshold and on reversed pages, including misaligned V and scales.

Inject score-buffer OOM and a recoverable graph launch rejection in separate runs. Check direct ordered fallback after OOM. Check score-backed eager after graph rejection. Record warm dispatch timing and serving timing as separate measurements. See the [batch graph checks](batch-graphs.md#verification) for HTTP parity and failure recovery.

For batched verification, compare the batched stages with per-row stages on a
reserved B200. Give both paths the same histories, packed KV, and grouped split plan.
Exercise 2, 5, 15, and 20 rows with remapped page tables and a page crossing.
Compare BF16 Q/K and gated attention byte for byte. Compare all packed K/V payloads
and scales, including untouched slots. Trace one launch per batched stage per GQA
layer. Exercise a group that straddles 2,047 and 2,048 visible tokens. Check that
this group uses the per-row stages. Use the [MTP verification checks](mtp.md#verification)
for acceptance and continuation.

On B200, compare transposed scoring against the ordinary one-stage loop. [Control revision](https://github.com/para1lel/sglang.cpp/commit/650f69d853380d5738955586ce37f30051485f8f)
Keep the BF16 queries, visible range, split partition, and packed KV identical.
Remap pages and include a partial tail. Compare BF16 outputs and written split
partials bitwise. Include a flagged head to check FP64 replay.

Trace 9 and 10 eligible rows. With 10 aligned rows, move one row from 10,240 to
10,241 visible tokens. Check that dispatch returns to narrow two-stage groups.

For bitwise regression of the 20-row shared reader against the ordinary reader,
disable the long Tensor Core reader in both arms. Use a reserved B200 with the
same checkpoint, histories, and paged MXFP8 KV. Trace full K4 packets with first positions at
32,768 and 36,864. Compare BF16 attention and written split-partial fields
byte for byte. Include remapped pages and a head that requires FP64 replay.
Exercise complete packets on both sides of 32,768 and 40,960.
Check ordinary dispatch for a cross-chunk group, a partial
packet, and bound replay.

Replay the same five target inputs per lane with both readers. Compare all 20
full BF16 vocabulary rows bitwise in a separate fixed-history diagnostic. Capture
those scores through the ordinary full vocabulary head. Run native greedy K4
separately with its certified winner path. Compare continuation IDs and MTP
row counts. Exercise both replacement and full acceptance.

Inject optional setup and launch rejections in separate runs. Check ordinary
dispatch for only the rejected packet. Verify shared dispatch on the next eligible
packet. Inject hard launch and combine errors separately. Check status propagation
without replay. Check that a preexisting or different recorded CUDA error blocks
fallback.

Compare the long Tensor Core reader's fixed-history full BF16 vocabulary rows
against a two-term control. In a separate native greedy run, compare continuation
IDs with the same control. Separately apply the
[numerical and quality checks](performance.md#operator-qualification) against
independent attention and end-to-end oracles. Exercise remapped pages, distinct
lane positions, and a group that crosses a 64-token boundary. Verify fallback
after a clean setup decline and error propagation after a failed launch.
Compare committed target and draft state through rejection and full acceptance.
Restore a checkpoint and run a further K4 round.
