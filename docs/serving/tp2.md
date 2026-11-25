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

## First measured result

The first official campaign measured W1 only. It missed the gate, so W2 through W5 were intentionally not run.

- **Identity.** Campaign `24c2e092-25c7-4945-a251-aa9cb03358cb`, mode measure, protocol `tp2-paired-v1` (manifest `c2c3480a…`), corpus `tp2-corpus-v1` (manifest `d65076eb…`). Native source tree `3a33e1c1…`, executable SHA-256 `fb03d111…`, checkpoint `Qwen3.8-27B-FP8` (`b54fe441…`). Evidence: `artifacts/tp2-bench/campaign-2345/W1/comparison.json`.
- **Devices.** Both engines ran alone on the same two B200 devices in the same rank order: rank 0 `GPU-57c04c1a…`, rank 1 `GPU-b79a7675…`. The native server is one process with NCCL on both ranks. Rank 0 carries the LM head and the MTP draft: 19,031,037,824 weight bytes against rank 1's 14,734,676,480. Rank daemons listen on ports 17655 and 17656.
- **Cohorts.** Each arm ran one warmup cohort, then one measured cohort of the two frozen W1 lanes: 128 prompt tokens, 128 output tokens, one request per GPU, two concurrent. The measured cohort is the single sample. Both arms verified all lanes generating before the first request finished.
- **Receipts.** Native: `tp_size` 2, MTP4, sixteen slots, 32,768 tokens per step, eager flag off, batch CUDA graphs off, LMCache enabled and available, no resident prefix entries. vLLM pin `81a8cb06`: TP2, `kv_transfer=false`, `kv_offloading_size=null`, zero CPU offload, local prefix cache on, FP8 E4M3 KV, CUDA graphs on.
- **Cache transfer.** The protocol reconciled each daemon's cold transfers with the joint receipt. Both ranks acknowledged every checkpoint store and restored nothing. GQA and GDN written bytes were equal on both ranks; draft bytes appeared only on rank 0.
- **Result.** Native median 925.746744 ms; vLLM median 557.959022 ms; ratio 1.6591661887313294 against the ≤1.10 gate: W1 fails. Native served 276.5 output tokens per second against vLLM's 458.8. Optimization work continues; no TP2 row has passed.
- **Untimed trace.** A separate traced cohort verified speculative decoding without timing claims. The native trace recorded 100 per-lane MTP rounds, 98 at depth four, with lane outputs matching the native warmup. The vLLM arm showed a draft-token delta of 840 and a depth-four verified-position delta of 210 over the timed arm's warmup and measured interval.

## Prerequisites and command

Use two visible GPUs, a sharded-checkpoint build, LMCache 0.5.8, and one daemon per rank started with `tools/lmcache/serve.sh --rank N`. Then run:

```sh
sglang-server --checkpoint MODEL --device 0 --port PORT --tp 2 --eager \
  --mtp-steps 4 --lmcache-socket SOCKET
```

The smoke run recorded under `artifacts/tp2-review/serving/cache-2328/complete/cache/` used this shape; `ranked-cache.argv.json` holds its exact arguments. The measured campaign server ran without `--eager`: its live argument record shows `--port 17393 --tp 2 --max-active 16`.

## Acceptance target

The first campaign stopped after the W1 miss. W2 through W5 remain unmeasured. A passing campaign must satisfy every condition below.

- **Devices.** Run both engines alone on the same two B200 devices in the same rank order. Record both UUIDs, the rank assignment, interconnect, and distributed library.
- **Workloads.** Reuse the frozen W1–W5 fixture bytes. Offer 2, 2, 8, 16, and 8 requests total on W1–W5. This is one, one, four, eight, and four per GPU. Eight slots per GPU give sixteen slots total. W4 fills them exactly. W4 uses the first sixteen archived W5 lanes. W1 and W2 need one new distinct lane each. W3 and W5 need four more each. Freeze every new lane as distinct deterministic bytes in a separate additive TP2 corpus. Retain the TP1 lanes and bytes. Duplicate no request.
- **Native arm.** Run the native server with `--tp 2`, the default direct GPU LMCache path, and the LMCache daemon enabled. Verify the reported `tp_size` is two. [Server information](../native-model-info.md#server-information) Confirm effective direct cache activity on each rank, so a reported `tp_size` cannot hide a bypass. Record the adopted topology and the effective daemon configuration.
- **Cache.** Start one LMCache daemon per rank, each bound to its own GPU and rank namespace. [Ranked transport](lmcache.md#ranked-tp2-transport) Draft records live only in the rank 0 daemon. Drain and reset both rank daemons before cold timing.
- **vLLM arm.** Run the pinned checkout with `--tensor-parallel-size 2`. [TP option](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/engine/arg_utils.py#L1219)<!-- prefix: "\"--tensor-pa" --> Require `kv_transfer=false`, `kv_offloading_size=null`, and zero CPU weight offload in both workers' receipts. [Receipt policy](../performance.md#scheduler-and-reporting-policy)
- **Gate.** Require native TP2 makespan ≤1.10× pinned vLLM TP2 on each W1–W5 row independently. A passing row covers no other row. Apply the [scheduler policy](../performance.md#scheduler-and-reporting-policy) and [measurement](../performance.md#measurement) rules unchanged.
