# TP2 serving status and acceptance target

`--tp` accepts 1 or 2 and defaults to 1. TP2 defaults to [eligible verification replay](../batch-graphs.md#tp2-target-verification-graph). `--eager` disables it. Ordinary TP2 decode is eager. Use `benchmarks/tp2/run.sh` for two-B200 campaigns. [TP option](../../apps/sglang-server.cpp#L168)<!-- prefix: "} else if (o" --> · [Graph policy](../../apps/sglang-server.cpp#L354)<!-- prefix: "arguments.ma" --> · [TP1 topology](../../include/sglang/runtime/server/batch_session.h#L15)<!-- prefix: "inline const" --> · [Launcher limit](../../benchmarks/paired/README.md#L72)<!-- prefix: "the number o" -->

TP2 with LMCache requires rank device receipts. The server derives one endpoint per rank from the base socket path. Each rank uses a private namespace and daemon. [Rank endpoints](../../include/sglang/runtime/cache/lmcache_client.h#L58)<!-- prefix: "std::string " --> · [Ranked transport](lmcache.md#ranked-tp2-transport)

Both ranks own target workers and store 16,896 target KV bytes per token per rank. Rank 0 owns draft weights and cache. Restore requires the same GDN endpoint on both ranks. [Cold, warm, and one-sided eviction receipts](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/serving/tp2.md#tp2-serving-status-and-acceptance-target)

## Campaign record

The archived `e912b39b` campaign (`6f7754f4-b4f7-4e0b-acb1-ea2f8df74e73`) passed W1–W3 at 1.0474×, 1.0117×, and 1.0639×. W4 took 29.046274476 s native and 25.335314580 s vLLM (1.146473804×), so the campaign stopped before W5. Both arms completed all sixteen W4 requests. Native stored eighty checkpoints per rank for W4, with zero restores or I/O errors. [Frozen results, identities, and receipts](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/serving/tp2.md#campaign-record)

<a id="w4-wide-verification-release"></a>
<a id="w1-diagnostic-profile"></a>
<a id="w2-divergence-evidence"></a>
<a id="w4-readiness-diagnostic"></a>
<a id="w3-wide-verify-diagnostic"></a>

[Archived W1–W4 profiles, readiness, and wide verification](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/serving/tp2.md#w4-wide-verification-release)

## Two-rank wide verification

The root owns greedy policy and drafts, and the peer mirrors its rounds. Eligible groups of eight or sixteen draft depth-major when reserves fit (696,320 or 1,392,640 bytes). Eight lanes use 409,600 bytes for live depth scoring. Otherwise drafting stays wave-major. Both ranks preflight mirrored storage, streams, and positions. Only complete greedy groups of eight or sixteen verify together, subject to native width, and score in certified forty-row passes. Sixteen lanes pack activations on both ranks. Sampled, constrained, partial, and logprob groups keep four-lane packets. Commit supports [sixteen-row repair](../long-paged-decode.md#sixteen-row-tp2-draft-repair) and [peer staging](request-memory.md#peer-draft-repair). `SGLANG_DISABLE_TP2_WIDE_VERIFY` forces four-lane dispatch. [Wide lane cap](../../include/sglang/runtime/execution/decoder_batch_state.h#L117)<!-- prefix: "wide_mtp_lim" --> · [Verification width](../../src/decoder_batch.cpp#L536)<!-- prefix: "static int v" --> · [Peer preflight](../../src/decoder_verify.cpp#L816)<!-- prefix: "throw std::i" --> · [Vocabulary pass](../../src/decoder_verify.cpp#L753)<!-- prefix: "bool score_t" -->

Each rank gathers embeddings and scatters reduced states to row-owned buffers with one launch per stage. [Gather](../../src/decoder_verify.cpp#L851)<!-- prefix: "check(gather" --> · [Scatter](../../src/runtime/execution/decoder_forward_tp.cpp#L206)<!-- prefix: "check(scatte" --> · [Frozen warm probes](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/serving/tp2.md#two-rank-wide-verification)

## Prerequisites and command

Use two visible GPUs, a sharded-checkpoint build, LMCache 0.5.8, and one daemon per rank started with `tools/lmcache/serve.sh --rank N`. Then run:

```sh
sglang-server --checkpoint MODEL --device 0 --port PORT --tp 2 --eager \
  --mtp-steps 4 --lmcache-socket SOCKET
```

The smoke arguments are in `artifacts/tp2-review/serving/cache-2328/complete/cache/ranked-cache.argv.json`. The campaigns omitted `--eager` and used `--port 17393 --tp 2 --max-active 16`.

## Two-rank allreduce transport

The decoder builds MNNVL from the project CUDA source. NCCL handles devices or calls that decline multicast VMM or kernel eligibility. [MNNVL build](../../xmake.lua#L471)<!-- prefix: "target(\"sgla" --> · [Dispatch](../../src/runtime/backend/tensor_parallel.cpp#L453)<!-- prefix: "const bool t" -->

MNNVL needs 16-byte alignment and whole 5,120-element BF16 rows. One shot supports up to 51 rows (1,048,576-byte cap) in eager execution or joined capture. Eager two shot needs 80 rows, two 819,200-byte stages per Lamport slot, and 400 concurrently resident blocks per GPU. [Workspace](../../src/runtime/backend/mnnvl_ar.cu#L672)<!-- prefix: "state.twosho" --> · [Occupancy](../../src/runtime/backend/mnnvl_ar.cu#L219)<!-- prefix: "bool twoshot" -->

NCCL handles other oversize calls, independent captures, and captured 80-row calls. Mixed eager and captured streams are rejected. Enqueue errors propagate. Later calls use NCCL if neither rank enqueued. Partial enqueue fails the backend. [Error handling](../../src/runtime/backend/tensor_parallel.cpp#L303)<!-- prefix: "if (error.pe" --> See [4,096-row fusion](../native-prefill-workspace.md#tp2-causal-prefill-reduction-fusion).

One host thread enqueues both ranks in order. Each rank reserves 536,870,912 bytes with triple Lamport buffers. The rank sum normalizes negative zero to protect the sentinel. Multicast packs use 128-bit `multimem.st.relaxed.sys`. Unicast scatter and receive use ordinary accesses. [Sentinel](../../src/runtime/backend/mnnvl_ar.cu#L32)<!-- prefix: "constexpr st" --> · [Multicast store](../../src/runtime/backend/mnnvl_ar.cu#L124)<!-- prefix: "__device__ v" -->

`/metrics` counts successful enqueues per rank by transport and six NCCL fallback reasons. Graph replays do not increment these counters. [Counters](../../apps/server/loads.cpp#L234)<!-- prefix: "emit_rank(\"t" --> · [Fallback reasons](../../src/runtime/backend/tensor_parallel.cpp#L99)<!-- prefix: "enum class A" -->

[Frozen NCCL and two-shot replay](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/serving/tp2.md#two-rank-allreduce-transport)

[Archived eager W1 verifier](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/benchmarks/tp2/verify_transport.py#L194)<!-- prefix: "if __name__ " --> compares one-shot MNNVL with forced NCCL using the original run, fixtures, trace, and executable and daemon receipts.

### Verification reduction fusion

TP2 fuses one-shot MNNVL, RMSNorm, and quantization for ten packed verification rows or [eligible eager singletons](../batch-graphs.md#eager-tp2-singleton-verification). Both ranks need aligned 5,120-wide operands and ten resident blocks per GPU. Clean declines use ordinary collectives and local preparation. Eligible [M10 graph replay](../batch-graphs.md#tp2-target-verification-graph) retains fusion. Ordinary joined capture used MNNVL at ten rows and NCCL at 52. [Selection](../../src/runtime/execution/decoder_forward_tp.cpp#L52)<!-- prefix: "struct TpRed" --> · [Capture](../../src/runtime/backend/tensor_parallel.cpp#L490)<!-- prefix: "bool TensorP" --> · [Preflight](../../src/runtime/backend/mnnvl_ar.cu#L861)<!-- prefix: "bool MnnvlAl" -->

Each rank fuses 64 attention reductions with MLP input normalization and 63 MLP reductions with next-layer input normalization. The final MLP reduction stays separate. The kernel preserves the BF16 sum, residual, and normalization before E4M3 quantization with one FP32 scale per 128 channels. [Layer loop](../../src/runtime/execution/decoder_forward_tp.cpp#L329)<!-- prefix: "const bool a" --> · [Arithmetic](../../include/sglang/text/quantization/rms_quantize.cuh#L99)<!-- prefix: "template \u003cbo" -->

The host prepares both rank launches before submission. Partial enqueue fails the group and propagates without an NCCL retry. On narrow fusion, a launched rank can wait indefinitely for its peer and require process termination. [Submission](../../src/runtime/backend/mnnvl_ar.cu#L911)<!-- prefix: "ReduceNorm\u003cR" --> · [Failure](../../src/runtime/backend/tensor_parallel.cpp#L303)<!-- prefix: "if (error.pe" -->

[Frozen ten-row parity, fallback audit, and timings](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/serving/tp2.md#verification-reduction-fusion)

### Forty-row verification fusion

On B200, both ranks need eager 40-row verification, external packed FP8 buffers, and Tensor Core packing. One-shot MNNVL requires aligned operands, 40 resident blocks per GPU, and a mapped rejection word. A clean decline uses ordinary allreduce and local preparation for the rest of the forward. [Shape gate](../../src/runtime/execution/decoder_forward_tp.cpp#L138)<!-- prefix: "if (primary." --> · [Transport](../../src/runtime/backend/tensor_parallel.cpp#L517)<!-- prefix: "bool TensorP" --> · [Rejection word](../../src/runtime/backend/mnnvl_ar.cu#L480)<!-- prefix: "bool MnnvlAl" -->

If rank 1 rejects submission synchronously after rank 0 enqueues, the host publishes the launch epoch. Healthy rank-0 blocks poll it and exit. The host attempts to fence both streams, fails the group, and propagates the error. [Rank-0 poll](../../src/runtime/backend/mnnvl_ar.cu#L325)<!-- prefix: "PairAtomic(*" --> · [Host rejection](../../src/runtime/backend/mnnvl_ar.cu#L935)<!-- prefix: "PairAtomic(*" -->

[Frozen forty-row rank parity, W1 replay exclusion, and W3 step timings](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/serving/tp2.md#forty-row-verification-fusion)

## Acceptance target

A passing campaign must satisfy every condition below.

- **Devices.** Run both engines alone on the same two B200 devices in the same rank order. Record both UUIDs, the rank assignment, interconnect, and distributed library.
- **Workloads.** Reuse the frozen W1–W5 fixture bytes. Offer 2, 2, 8, 16, and 8 requests total on W1–W5. This is one, one, four, eight, and four per GPU. Eight slots per GPU give sixteen slots total. W4 fills them exactly. W4 uses the first sixteen archived W5 lanes. W1 and W2 need one new distinct lane each. W3 and W5 need four more each. Freeze every new lane as distinct deterministic bytes in a separate additive TP2 corpus. Retain the TP1 lanes and bytes. Duplicate no request.
- **Native arm.** Run the native server with `--tp 2`, the default direct GPU LMCache path, and the LMCache daemon enabled. Verify the reported `tp_size` is two. [Server information](../native-model-info.md#server-information) Confirm effective direct cache activity on each rank, so a reported `tp_size` cannot hide a bypass. Record the adopted topology and the effective daemon configuration.
- **Cache.** Start one LMCache daemon per rank, each bound to its own GPU and rank namespace. [Ranked transport](lmcache.md#ranked-tp2-transport) Draft records live only in the rank 0 daemon. Drain and reset both rank daemons before cold timing.
- **vLLM arm.** Run the pinned checkout with `--tensor-parallel-size 2`. [TP option](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/engine/arg_utils.py#L1219)<!-- prefix: "\"--tensor-pa" --> Require `kv_transfer=false`, `kv_offloading_size=null`, and zero CPU weight offload in both workers' receipts. [Receipt policy](../performance.md#scheduler-and-reporting-policy)
- **Gate.** Submit each row's requests simultaneously through the original submission path. Run one warmup and one measured cohort per row and engine, and use the measured verdict. Record clocks and output lengths. Treat output content as reference only; no warmup-measured token identity is required. Require native TP2 makespan ≤1.10× pinned vLLM TP2 on each W1–W5 row independently. A passing row covers no other row. Apply the [scheduler policy](../performance.md#scheduler-and-reporting-policy) unchanged.
