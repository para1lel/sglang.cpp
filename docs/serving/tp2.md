# TP2 serving

`--tp` accepts 1 or 2 and defaults to 1. TP2 enables eligible [ordinary and verification replay](../batch-graphs.md#runtime-path) by default. `--eager` disables both. Each graph remains subject to rank-local state, topology, and shape eligibility. [TP option](../../apps/sglang-server.cpp#L189)<!-- prefix: "} else if (o" --> · [Graph options](../../apps/sglang-server.cpp#L408)<!-- prefix: "arguments.en" -->

TP2 uses one node-local LMCache daemon, socket and namespace across both ranks. Each rank still needs independent device and transfer receipts. [Ranked transport](lmcache.md#ranked-tp2-transport)

Both ranks own target workers and store 16,896 target KV bytes per token per rank. Each rank loads its draft weight shard and maintains rank-local draft KV. Restore requires matching ranked coverage and a common GDN endpoint. See [ranked cache checks](lmcache.md#standalone-verification). [Primary draft weights](../../src/decoder.cpp#L521)<!-- prefix: "model_->mtp_" --> · [Peer draft weights](../../src/decoder.cpp#L527)<!-- prefix: "peer.mtp_wei" -->

## Two-rank wide verification

Each rank executes draft projections, KV append, attention and its vocabulary shard. Rank 0 selects from the rank-local winners with deterministic ties. Both ranks also seed and repair draft state. Eligible complete greedy groups can use packed M40/M80 target verification. Shortened, sampled, constrained and scored groups use ordinary packet layouts as required. Source and device proofs for two-rank proposals are distinct from [target-forward replay](../batch-graphs.md#tp2-target-verification-graph). [Proposal](../../src/decoder_tp.cpp#L244)<!-- prefix: "bool Decoder" --> · [Ranked draft forward](../../src/text/draft_projection.cpp#L387)<!-- prefix: "void MtpDraf" --> · [Winner merge](../../src/text/draft_projection.cpp#L556)<!-- prefix: "int MtpDraft" --> · [Repair](../../src/decoder_tp.cpp#L305)<!-- prefix: "void Decoder" --> · [Seeding](../../src/decoder_tp.cpp#L341)<!-- prefix: "void Decoder" -->

Each rank gathers embeddings and scatters reduced states to row-owned buffers with one launch per stage. [Gather](../../src/decoder_verify.cpp#L834)<!-- prefix: "check(gather" --> · [Scatter](../../src/runtime/execution/decoder_forward_tp.cpp#L167)<!-- prefix: "check(scatte" -->

## Prerequisites and command

Use a sharded-checkpoint build and start [LMCache 0.5.8](lmcache.md#start-the-daemon) with both GPU UUIDs, omitting `--rank`. In the server shell expose the same GPUs in rank order. Use the daemon socket in the server command:

```sh
sglang-server --checkpoint MODEL --device 0 --port PORT --tp 2 \
  --mtp-steps 4 --lmcache-socket SOCKET
```

Add `--eager` to run the same policy without target graph replay.

## Two-rank allreduce transport

The decoder builds MNNVL from the project CUDA source. NCCL handles devices or calls that decline multicast VMM or kernel eligibility. [MNNVL build](../../xmake.lua#L504)<!-- prefix: "target(\"sgla" --> · [Dispatch](../../src/runtime/backend/tensor_parallel.cpp#L463)<!-- prefix: "const bool t" -->

Ordinary MNNVL needs 16-byte alignment and whole 5,120-element BF16 rows. One shot supports up to 51 rows (1,048,576-byte cap) in eager execution or joined capture. Eager two shot needs 80 rows, two 819,200-byte stages per Lamport slot, and 400 concurrently resident blocks per GPU. [Workspace](../../src/runtime/backend/mnnvl_ar.cu#L697)<!-- prefix: "state.twosho" --> · [Occupancy](../../src/runtime/backend/mnnvl_ar.cu#L224)<!-- prefix: "bool twoshot" -->

NCCL handles ordinary oversize calls and independent captures when the transport permits them. Packed M80 verifier replay can use the joined MNNVL two-shot path when both ranks qualify. Mixed eager and captured streams are rejected. Enqueue errors propagate. Later calls use NCCL if neither rank enqueued. Partial enqueue fails the backend. [Error handling](../../src/runtime/backend/tensor_parallel.cpp#L305)<!-- prefix: "if (error.pe" --> See [4,096-row fusion](../native-prefill-workspace.md#tp2-causal-prefill-reduction-fusion).

One host thread enqueues both ranks in order. Each rank reserves 536,870,912 bytes with triple Lamport buffers. The rank sum normalizes negative zero to protect the sentinel. Multicast packs use 128-bit `multimem.st.relaxed.sys`. Unicast scatter and receive use ordinary accesses. [Sentinel](../../src/runtime/backend/mnnvl_ar.cu#L34)<!-- prefix: "constexpr st" --> · [Multicast store](../../src/runtime/backend/mnnvl_ar.cu#L129)<!-- prefix: "__device__ v" -->

`/metrics` counts successful enqueues per rank by transport and six NCCL fallback reasons. Graph replays do not increment these counters. [Counters](../../apps/server/loads.cpp#L234)<!-- prefix: "emit_rank(\"t" --> · [Fallback reasons](../../src/runtime/backend/tensor_parallel.cpp#L99)<!-- prefix: "enum class A" -->

The [TP2 transport fixture](../../tests/decoder/tp2/serving_check.sh#L5)<!-- prefix: "printf '%s\\n" --> wraps NCCL submissions and compares output IDs across TP1 and TP2 and against target-only decode. Follow the [TP2 lease isolation checks](../../tests/prefix/README.md#tp2-release-isolation) for ranked cache behavior.

### Verification reduction fusion

TP2 fuses one-shot MNNVL, RMSNorm, and quantization for ten packed verification rows or [eligible eager singletons](../batch-graphs.md#eager-tp2-singleton-verification). Both ranks need aligned 5,120-wide operands and ten resident blocks per GPU. Clean declines use ordinary collectives and local preparation. Eligible [M10 graph replay](../batch-graphs.md#tp2-target-verification-graph) retains fusion. A historical ordinary joined-capture fixture used MNNVL at ten rows and NCCL at 52. [Selection](../../src/runtime/execution/decoder_forward_tp.cpp#L51)<!-- prefix: "struct TpRed" --> · [Capture](../../src/runtime/backend/tensor_parallel.cpp#L506)<!-- prefix: "bool TensorP" --> · [Preflight](../../src/runtime/backend/mnnvl_ar.cu#L886)<!-- prefix: "bool MnnvlAl" -->

Each rank fuses 64 attention reductions with MLP input normalization and 63 MLP reductions with next-layer input normalization. The final MLP reduction stays separate. The kernel preserves the BF16 sum, residual, and normalization before E4M3 quantization with one FP32 scale per 128 channels. [Layer loop](../../src/runtime/execution/decoder_forward_tp.cpp#L246)<!-- prefix: "const bool a" --> · [Arithmetic](../../src/text/rms_quantize.cu#L93)<!-- prefix: "template \u003cbo" -->

The host prepares both rank launches before submission. Partial enqueue fails the group and propagates without an NCCL retry. On narrow fusion, a launched rank can wait indefinitely for its peer and require process termination. [Submission](../../src/runtime/backend/mnnvl_ar.cu#L952)<!-- prefix: "ReduceNorm\u003cR" --> · [Failure](../../src/runtime/backend/tensor_parallel.cpp#L305)<!-- prefix: "if (error.pe" -->

For ten-row verification on two B200s, compare both ranks' residuals, normalized output and E4M3 values and scales with ordinary reduction and local preparation. Check eligible graph replay and clean fallback.

<a id="forty-row-verification-fusion"></a>

### Wide verification fusion

On B200, both ranks need eligible packed verification with 40 or 80 rows, external packed FP8 buffers, and Tensor Core packing. The fused one-shot reduction requires aligned operands, one resident block per row on each GPU, and a mapped rejection word. The 80-row route also requires a 1,638,400-byte Lamport slot. A clean decline uses ordinary allreduce and local preparation for the rest of the forward. Eligible graph replay requires two-shot or one-shot support for its packed layout. In eager mode the final 80-row MLP reduction can use ordinary two shot or NCCL. [Shape gate](../../src/runtime/execution/decoder_forward_tp.cpp#L140)<!-- prefix: "if (primary." --> · [Transport](../../src/runtime/backend/tensor_parallel.cpp#L535)<!-- prefix: "bool TensorP" --> · [Rejection word](../../src/runtime/backend/mnnvl_ar.cu#L491)<!-- prefix: "bool MnnvlAl" -->

If rank 1 rejects submission synchronously after rank 0 enqueues, the host publishes the launch epoch. Healthy rank-0 blocks poll it and exit. The host attempts to fence both streams, fails the group, and propagates the error. [Rank-0 poll](../../src/runtime/backend/mnnvl_ar.cu#L332)<!-- prefix: "PairAtomic(*" --> · [Host rejection](../../src/runtime/backend/mnnvl_ar.cu#L978)<!-- prefix: "PairAtomic(*" -->

For eager 40- and 80-row verification on two B200s, compare against ordinary reduction using identical inputs. Check both ranks' states, eligible graph replay, clean fallback, and rank-1 submission rejection.
