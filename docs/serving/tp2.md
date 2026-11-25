# TP2 serving status and acceptance target

Tensor parallelism 2 is implemented for eager serving. Launch with `--tp 2`; the option accepts 1 or 2. [TP option](../../apps/sglang-server.cpp#L171)<!-- prefix: "} else if (o" --> Decode runs eager under tensor parallel: batch CUDA graphs stay TP1-only. [Graph policy](../../apps/sglang-server.cpp#L59)<!-- prefix: "bool enable_" --> TP1 keeps the single-device default topology. [TP1 topology](../../include/sglang/runtime/server/batch_session.h#L15)<!-- prefix: "inline const" --> The historical `benchmarks/paired` launcher supports one GPU. [Launcher limit](../../benchmarks/paired/README.md#L71)<!-- prefix: "the number o" --> `benchmarks/tp2/run.sh` is the separate two-B200 campaign entry point.

With LMCache enabled, TP2 requires rank device receipts from the loaded topology. The server derives one daemon endpoint per rank from the base socket path. [Rank endpoints](../../include/sglang/runtime/cache/lmcache_client.h#L64)<!-- prefix: "std::string " --> Each rank owns a private namespace and daemon. [Ranked transport](lmcache.md#ranked-tp2-transport)

A two-rank run is recorded under `artifacts/tp2-review/serving/cache-2328/complete/cache/`. The server reported `tp_size` 2 with `dp_size` 1 and `pp_size` 1, MTP4, and ranked LMCache active. [Server information](../native-model-info.md#server-information) The topology runs two rank-local target workers with a rank0-only full MTP draft; the receipts below carry draft bytes only on rank 0.

The run exercised cold, warm, one-sided eviction, and a longer tail on one 18-token input.

- **Cold.** The first request missed. Both ranks stored the 13-token prefix: rank 0 wrote 219,648 GQA bytes, 76,992,512 GDN bytes, and 25,344 draft bytes; rank 1 wrote the same GQA and GDN bytes and no draft bytes. Both receipts count 13 stored tokens.
- **Warm.** The repeated request restored the same prefix. The joint GET returned the stored bytes on both ranks: 219,648 and 76,992,512 per rank, draft only on rank 0.
- **One-sided eviction.** After rank 1 lost its records, the same request ran cold again. A prefix restores only when both ranks hold the same GDN endpoint.
- **Longer tail.** A 25-token extension restored 422,400 GQA bytes and 76,992,512 GDN bytes per rank, with 50,688 draft bytes on rank 0.

The raw receipts are `rank-receipts.json`; the counters are `metrics.json` in the same directory. Per-family byte counts follow the ranked geometry: 16,896 target bytes per token per rank and full-width draft bytes on rank 0 only.

## Campaign record

The current campaign runs the published release `337edce7`. W1 passes the gate. W2 has a native measurement but no valid comparison. W3 through W5 are unrun.

- **Identity.** Campaign `92b206b0-16e8-4ad0-a600-463926e0d6df`, mode measure, protocol `tp2-paired-v1` (manifest `c2c3480a…`), corpus `tp2-corpus-v1` (manifest `d65076eb…`). Native source tree `337edce7…`, executable SHA-256 `0100f4e2…`, checkpoint `Qwen3.8-27B-FP8` (`b54fe441…`). Evidence: `artifacts/tp2-bench/campaign-0222/`.
- **W1.** Native median 592.803737 ms; vLLM median 558.431928 ms; ratio 1.0615505798944935 against the ≤1.10 gate: W1 passes. Rank receipts are recorded for both ranks. Rank 0 `GPU-57c04c1a…` carries the LM head and MTP draft: 19,031,037,824 weight bytes. Rank 1 `GPU-b79a7675…` carries 14,734,676,480. A separate traced cohort accompanies the timed arms.
- **W2.** Native measured 1699.082051 ms. The vLLM arm failed the protocol determinism check: warmup and measured outputs disagree, first mismatches at 0-based offsets 49 and 303. The row has no valid ratio and is neither passed nor failed.
- **W3–W5.** Unrun on this campaign.
- **Historical context.** The first campaign (`24c2e092-25c7-4945-a251-aa9cb03358cb`, source `3a33e1c1…`, binary `fb03d111…`) measured W1 only: 925.746744 ms against 557.959022 ms, ratio 1.6591661887313294, a miss, and stopped. The [diagnostic profile](#w1-diagnostic-profile) below attributes that gap. A separate clean production diagnostic on the current binary measured 594.644780 ms for W1 before this campaign; it is not a campaign row.

## W1 diagnostic profile

A separate untimed profile traced both engines after the W1 miss. It guided optimization only and makes no campaign claim.

- **Cohort.** Two fixed requests with five verified positions each (MTP4), run on the native TP2 server and the pinned vLLM reference. Receipts record the eager flag off and batch CUDA graphs off. LMCache ran enabled and available on the default socket, one rank-local daemon per rank.
- **Sampling.** Both used the frozen 128-token W1 prompts. Native generated 24 output tokens per request and vLLM generated 128.
- **Devices.** The native run reserved devices 1 and 5; the reference reserved devices 4 and 5. Both pairs are NV18 NVLink-bridged B200 devices. The rank 0 device differs between the runs.
- **Width.** Width two means two active lanes; width one means one. Each lane carries five verified positions, independent of request length.
- **Method.** Nsight Systems traced the native server as one process hosting both ranks, one CPU host with two GPU timelines. The native target verification interval runs from the first `verify_rms` kernel through the last of the 128 row sums. The reference target graph span is the whole captured target-forward CUDA graph, including the vocabulary embedding reduction and its 128 fused reductions. Neither boundary includes the output head, rejection sampling, the four drafts, or the final GDN rollback. The native period is target start-to-start; the reference period is first step kernel to first step kernel. The reference used PyTorch profiler traces, four sampled steps per rank and width. Native width-2 medians use six rounds.
- **Rank 0 step medians at width two.** Period 18.746 ms native versus 10.237 ms reference. Native target verification interval 13.505 ms versus reference target graph span 8.016 ms. At width one: period 12.498 versus 10.177 ms; rank 0 target 9.828 versus 8.010 ms.
- **Rank 0 kernel-duration sums inside the target, width two.** FP8 projection 3.529 versus 4.024 ms. GQA attention 2.541 versus 0.451 ms. GDN attention 1.619 versus 1.048 ms.
- **Reductions.** Cross-rank elapsed sums after the last rank arrives, width two: 1.531 ms native versus 0.753 ms reference across 128 reductions. Both rank clocks contribute; the sum ends at the later completion.
- **Target head.** The native target head is a full-vocabulary BF16 projection on rank 0, outside the target span: two full-head grid launches, 31040 by one, 1.633 ms per width-2 step. The sharded reference spends 0.181 ms on the vocabulary GEMM plus 0.034 ms on the gather on rank 0, outside its span.
- **Evidence.** `artifacts/tp2-bench/native-profile-d0029/` holds the Nsight Systems database, `steps.json`, and `vs-reference.json`. `artifacts/tp2-bench/vllm-profile-0013/` holds `scoring.json`, `reduction-skew.json`, and per-rank traces with `profile.sha256` and `repro.sha256`. The analysis scripts are in `artifacts/tp2-bench/w1-diagnostics/`. The diagnostic native binary is SHA-256 `0acf4c24…`; the first campaign's binary is `fb03d111…`.

## Prerequisites and command

Use two visible GPUs, a sharded-checkpoint build, LMCache 0.5.8, and one daemon per rank started with `tools/lmcache/serve.sh --rank N`. Then run:

```sh
sglang-server --checkpoint MODEL --device 0 --port PORT --tp 2 --eager \
  --mtp-steps 4 --lmcache-socket SOCKET
```

The smoke run recorded under `artifacts/tp2-review/serving/cache-2328/complete/cache/` used this shape; `ranked-cache.argv.json` holds its exact arguments. The measured campaigns ran without `--eager`: the live argument record shows `--port 17393 --tp 2 --max-active 16`.

## Two-rank allreduce transport

The build enables the vendor transport only when the flashinfer wheel headers exist. Set `SGLANG_FLASHINFER_ROOT` to use a non-default installed header root; point it at a location without the wheel headers to reproduce the optional-absence build. `SGLANG_HOST_ROOT` only changes the enclosing shared root. A build without the headers rides NCCL for every allreduce. [Build gate](../../xmake.lua#L470)<!-- prefix: "local flashi" -->

The vendor transport embeds the FlashInfer TRT-LLM MNNVL one-shot kernel from flashinfer 0.7.4. It uses per-rank pinned allocations, a two-device multicast object, and triple Lamport buffers. One host thread enqueues both rank halves in rank order. [Transport](../../src/runtime/backend/mnnvl_ar.cu#L1)<!-- prefix: "// MNNVL one" -->

A call uses the vendor transport only for whole hidden-width rows of 5,120 BF16 elements, counts up to 51 rows, 16-byte aligned buffers, and either eager execution or one joined stream capture. Counts of 52 or more rows, and two independent rank captures, ride NCCL. [Eligibility](../../src/runtime/backend/tensor_parallel.cpp#L323)<!-- prefix: "count \u003e mnnv" -->

The rank sum normalizes negative zero to positive zero, so a zero result never aliases the Lamport sentinel bit pattern. [Sentinel](../../src/runtime/backend/mnnvl_ar.cu#L40)<!-- prefix: "constexpr st" -->

The multicast workspace reserves 536,870,912 bytes per rank on this platform, although the one-shot dispatch cap is 1,048,576 aggregated bytes. [Envelope](../../src/runtime/backend/mnnvl_ar.cu#L36)<!-- prefix: "constexpr st" -->

Six fallback reasons ride NCCL: disabled, setup unavailable, oversize, capture mismatch, unaligned, and after error. Each is counted per rank. [Fallback reasons](../../src/runtime/backend/tensor_parallel.cpp#L121)<!-- prefix: "enum class A" -->

`/metrics` exposes the `tp2_ar_` counters per rank. Counters record completed enqueues per transport, not graph replays. [Counters](../../apps/server/loads.cpp#L235)<!-- prefix: "emit_rank(\"t" -->

No startup flag selects the transport. Selection is automatic in a build that includes it.

## Acceptance target

W1 has passed the gate. W2 awaits a valid comparison. W3 through W5 are unrun. A passing campaign must satisfy every condition below.

- **Devices.** Run both engines alone on the same two B200 devices in the same rank order. Record both UUIDs, the rank assignment, interconnect, and distributed library.
- **Workloads.** Reuse the frozen W1–W5 fixture bytes. Offer 2, 2, 8, 16, and 8 requests total on W1–W5. This is one, one, four, eight, and four per GPU. Eight slots per GPU give sixteen slots total. W4 fills them exactly. W4 uses the first sixteen archived W5 lanes. W1 and W2 need one new distinct lane each. W3 and W5 need four more each. Freeze every new lane as distinct deterministic bytes in a separate additive TP2 corpus. Retain the TP1 lanes and bytes. Duplicate no request.
- **Native arm.** Run the native server with `--tp 2`, the default direct GPU LMCache path, and the LMCache daemon enabled. Verify the reported `tp_size` is two. [Server information](../native-model-info.md#server-information) Confirm effective direct cache activity on each rank, so a reported `tp_size` cannot hide a bypass. Record the adopted topology and the effective daemon configuration.
- **Cache.** Start one LMCache daemon per rank, each bound to its own GPU and rank namespace. [Ranked transport](lmcache.md#ranked-tp2-transport) Draft records live only in the rank 0 daemon. Drain and reset both rank daemons before cold timing.
- **vLLM arm.** Run the pinned checkout with `--tensor-parallel-size 2`. [TP option](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/engine/arg_utils.py#L1219)<!-- prefix: "\"--tensor-pa" --> Require `kv_transfer=false`, `kv_offloading_size=null`, and zero CPU weight offload in both workers' receipts. [Receipt policy](../performance.md#scheduler-and-reporting-policy)
- **Gate.** Require native TP2 makespan ≤1.10× pinned vLLM TP2 on each W1–W5 row independently. A passing row covers no other row. Apply the [scheduler policy](../performance.md#scheduler-and-reporting-policy) and [measurement](../performance.md#measurement) rules unchanged.
