# TP2 serving status and acceptance target

Tensor parallelism 2 is implemented for eager serving. Launch with `--tp 2`; the option accepts 1 or 2. [TP option](../../apps/sglang-server.cpp#L171)<!-- prefix: "} else if (o" --> Decode runs eager under tensor parallel: batch CUDA graphs stay TP1-only. [Graph policy](../../apps/sglang-server.cpp#L59)<!-- prefix: "bool enable_" --> TP1 keeps the single-device default topology. [TP1 topology](../../include/sglang/runtime/server/batch_session.h#L15)<!-- prefix: "inline const" --> The paired benchmark launcher still supports one GPU. [Launcher limit](../../benchmarks/paired/README.md#L71)<!-- prefix: "the number o" -->

With LMCache enabled, TP2 requires rank device receipts from the loaded topology. The server derives one daemon endpoint per rank from the base socket path. [Rank endpoints](../../include/sglang/runtime/cache/lmcache_client.h#L64)<!-- prefix: "std::string " --> Each rank owns a private namespace and daemon. [Ranked transport](lmcache.md#ranked-tp2-transport)

A two-rank run is recorded under `artifacts/tp2-review/serving/cache-2328/complete/cache/`. The server reported `tp_size` 2 with `dp_size` 1 and `pp_size` 1, MTP4, and ranked LMCache active. [Server information](../native-model-info.md#server-information) The topology runs two rank-local target workers with a rank0-only full MTP draft; the receipts below carry draft bytes only on rank 0.

The run exercised cold, warm, one-sided eviction, and a longer tail on one 18-token input.

- **Cold.** The first request missed. Both ranks stored the 13-token prefix: rank 0 wrote 219,648 GQA bytes, 76,992,512 GDN bytes, and 25,344 draft bytes; rank 1 wrote the same GQA and GDN bytes and no draft bytes. Both receipts count 13 stored tokens.
- **Warm.** The repeated request restored the same prefix. The joint GET returned the stored bytes on both ranks: 219,648 and 76,992,512 per rank, draft only on rank 0.
- **One-sided eviction.** After rank 1 lost its records, the same request ran cold again. A prefix restores only when both ranks hold the same GDN endpoint.
- **Longer tail.** A 25-token extension restored 422,400 GQA bytes and 76,992,512 GDN bytes per rank, with 50,688 draft bytes on rank 0.

The raw receipts are `rank-receipts.json`; the counters are `metrics.json` in the same directory. Per-family byte counts follow the ranked geometry: 16,896 target bytes per token per rank and full-width draft bytes on rank 0 only.

## Prerequisites and command

Use two visible GPUs, a sharded-checkpoint build, LMCache 0.5.8, and one daemon per rank started with `tools/lmcache/serve.sh --rank N`. Then run:

```sh
sglang-server --checkpoint MODEL --device 0 --port PORT --tp 2 --eager \
  --mtp-steps 4 --lmcache-socket SOCKET
```

The recorded run used this command shape. See `ranked-cache.argv.json` for the exact arguments.

## Acceptance target

A TP2 performance campaign is remaining work. Per-workload timing is pending; no TP2 timing claim exists yet. It must satisfy every condition below.

- **Devices.** Run both engines alone on the same two B200 devices in the same rank order. Record both UUIDs, the rank assignment, interconnect, and distributed library.
- **Workloads.** Reuse the frozen W1–W5 fixture bytes. Offer 2, 2, 8, 16, and 8 requests total on W1–W5. This is one, one, four, eight, and four per GPU. Eight slots per GPU give sixteen slots total. W4 fills them exactly. W4 uses the first sixteen archived W5 lanes. W1 and W2 need one new distinct lane each. W3 and W5 need four more each. Freeze every new lane as distinct deterministic bytes in a separate additive TP2 corpus. Retain the TP1 lanes and bytes. Duplicate no request.
- **Native arm.** Run the native server with `--tp 2`, the default direct GPU LMCache path, and the LMCache daemon enabled. Verify the reported `tp_size` is two. [Server information](../native-model-info.md#server-information) Confirm effective direct cache activity on each rank, so a reported `tp_size` cannot hide a bypass. Record the adopted topology and the effective daemon configuration.
- **Cache.** Start one LMCache daemon per rank, each bound to its own GPU and rank namespace. [Ranked transport](lmcache.md#ranked-tp2-transport) Draft records live only in the rank 0 daemon. Drain and reset both rank daemons before cold timing.
- **vLLM arm.** Run the pinned checkout with `--tensor-parallel-size 2`. [TP option](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/engine/arg_utils.py#L1219)<!-- prefix: "\"--tensor-pa" --> Require `kv_transfer=false`, `kv_offloading_size=null`, and zero CPU weight offload in both workers' receipts. [Receipt policy](../performance.md#scheduler-and-reporting-policy)
- **Gate.** Require native TP2 makespan ≤1.10× pinned vLLM TP2 on each W1–W5 row independently. A passing row covers no other row. Apply the [scheduler policy](../performance.md#scheduler-and-reporting-policy) and [measurement](../performance.md#measurement) rules unchanged.
