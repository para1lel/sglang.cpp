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

The rows keep the release that measured them: W1–W4 ran on `337edce7`; W5 was re-measured on `cef34e34`. W1 and W2 pass the gate; W3, W4, and W5 miss it. Every row compares against the pinned vLLM reference `81a8cb06…`.

- **Identity.** Campaign `92b206b0-16e8-4ad0-a600-463926e0d6df`, mode measure, protocol `tp2-paired-v1` (manifest `c2c3480a…`), corpus `tp2-corpus-v1` (manifest `d65076eb…`). Native source tree `337edce7…`, executable SHA-256 `0100f4e2…`, checkpoint `Qwen3.8-27B-FP8` (`b54fe441…`). Evidence: `artifacts/tp2-bench/campaign-0222/`.
- **W1.** Native median 592.803737 ms; vLLM median 558.431928 ms; ratio 1.0615505798944935 against the ≤1.10 gate: W1 passes. Rank receipts are recorded for both ranks. Rank 0 `GPU-57c04c1a…` carries the LM head and MTP draft: 19,031,037,824 weight bytes. Rank 1 `GPU-b79a7675…` carries 14,734,676,480. A separate traced cohort accompanies the timed arms.
- **W2.** Native measured 1699.082051 ms. The pinned vLLM reference measured 2000.013573 ms, from earliest submission to final `[DONE]`. The ratio is 0.849535260128957: W2 passes. The W2 reference predates full driver completion: its driver exited before end and retirement, so the span ends at the final `[DONE]`.
- **W3.** A second campaign (`4e83b2ac-a36d-45dc-ab54-c4cc889bda5c`) measured W3: protocol name `tp2-paired-v1`, a distinct 36-file manifest `6d38e692…`, same corpus, source, binary, and devices. Native measured 6862.178762 ms; vLLM measured 4113.792758 ms; ratio 1.6680905348611146: W3 misses the gate. From earliest submission to latest first token, native spans 2125.991606 ms and vLLM 1605.194380 ms. The remaining spans through final `[DONE]` are 4736.187156 ms and 2508.598378 ms. Both arms completed eight 8192-token prompts to 1024 outputs with retirement verified. Evidence: `artifacts/tp2-bench/campaign-timing-0826/W3/comparison.json`.
- **W4.** Native measured 57338.227570 ms; vLLM measured 26346.666432 ms; ratio 2.176299142739304: W4 misses the gate. All sixteen lanes completed 32768-token prompts to 4096 outputs on both engines, with full backend and retirement verified; no request timed out. From earliest submission to the last lane's first token, native spans 18079.484509 ms and vLLM 14347.504268 ms. From that first token to final `[DONE]`, native spends 39258.743061 ms and vLLM 11999.162164 ms. Evidence: `artifacts/tp2-bench/campaign-timing-0826/W4/` and `artifacts/tp2-review/campaign/timing-only-0825/W4-phases.log`.
- **W5.** Release `cef34e34` re-measured W5: campaign `0dd0b75f-71ac-467d-8d8d-2c7ee524055c`, protocol `tp2-paired-v1` (manifest `6d38e692…`), binary `b26e278f…`, same corpus, checkpoint, and devices. Native median 55734.968000 ms; vLLM median 39394.467235 ms; ratio 1.4147917692990728 against the 43333.9139585 ms ceiling: W5 misses by 12401.0540415 ms. Thirty-two requests ran across four cold cohorts of eight 131072-token prompts to 1024 outputs. Every lane ended in `length` with `[DONE]`, and both arms fully retired. Native spans 49642.135752 ms from earliest submission to the last lane's first token, then 6092.832248 ms through final `[DONE]`. vLLM spans 36113.703510 ms and 3280.763725 ms. The prior native median on `337edce7` was 66235.146069 ms, 10500.178069 ms longer, with the last lane's first token at 60009.979660 ms versus 49642.135752 ms. Evidence: `artifacts/tp2-bench/campaign-w5-cef34e34-after-external-1032/W5/comparison.json` and `artifacts/tp2-bench/w5-plain-0943/named-phase-comparison.json`.
- **Historical context.** The first campaign (`24c2e092-25c7-4945-a251-aa9cb03358cb`, source `3a33e1c1…`, binary `fb03d111…`) measured W1 only: 925.746744 ms against 557.959022 ms, ratio 1.6591661887313294, a miss, and stopped. The [diagnostic profile](#w1-diagnostic-profile) below attributes that gap. A separate clean production diagnostic on `337edce7` measured 594.644780 ms for W1 before this campaign; it is not a campaign row.

## W4 wide-verification release

An isolated sixteen-lane build with rank-local packed activation staging ran one cold attribution against its eight-lane control on the full W4 shape, sixteen 32768-token prompts to 4096 outputs. Charged makespan fell from 50.815826650 s to 38.788724911 s, 23.67 percent shorter. All-lane overlap throughput rose from 2026.02 to 3564.05 tokens per second. From the last lane's first token to final `[DONE]`, the span fell from 32.323245 s to 18.781068 s; from earliest submission to the last lane's first token, the span rose from 18.492582 s to 20.007657 s. This single-cohort attribution is not a named campaign row. Evidence: `artifacts/tp2-bench/w4-named-1219/ordinary-W4-independent-replay.json` (`c38195a7…`), binaries `ab8a1e45…` control and `425fbffd…` sixteen-lane.

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

## W2 divergence evidence

The runs below reproduced the W2 divergence under controlled changes in request arrival. They are diagnostic only; they explain the superseded determinism rejection. The W2 row passes under the timing-only criterion.

- **Original-wire bursts.** Three cold, complete, original-wire W2 reference bursts with lane submissions 12.0–19.4 µs apart matched the earlier official warmup exactly: all 3×1024 tokens.
- **Stagger replay.** A 64-token diagnostic replayed the divergence with staggered submissions of 73.7–141.6 ms. The first divergence reappeared at 0-based offset 49 in a cold cohort. Lane 1 held nine output IDs before lane 0 was sent and seventeen before lane 0's first token. A 64-token logprob pair measured the top-2 candidates at offset 49: 1788 and 5215 tie at −0.6938778162002563, on the stagger cell only.
- **Submission-change rejections.** A true ordered submission and a symmetric final-byte burst each captured both complete W2 cohorts on the pinned vLLM reference. Both failed identically: lane 0 output 49 differs from warmup, expected 5215, observed 1788. The two lanes' final-byte writes started 6.382 µs apart in the warmup and 6.481 µs apart in the measured cohort. Neither change is released or installed, and no replacement protocol has qualified.
- **Final-byte lane detail.** In the final-byte measured cohort, lane 0 first differs at 0-based offset 49, 5215 to 1788, with 346 of 512 outputs changed. Lane 1 first differs at offset 303, 760 to 1919, with 87 of 512 changed. The final-byte warmup equals the official warmup. The final-byte measured lane 1 differs from the official measured lane 1 at 82 IDs. Engine iteration token totals show one large iteration in (4096, 8192] tokens in the warmup. The measured cohort shows two iterations in (2048, 4096] instead, consistent with joint versus split processing of the two 2048-token prompts. Lane 1 led by one output ID in the warmup and by four in the measured cohort, independently observed on server-sent events.
- **Evidence.** `artifacts/tp2-bench/w2-diagnostic-0239/outcomes.json` (SHA-256 `b5ef5b3a…`), `artifacts/tp2-bench/ordered-submit-0247/qualification.log`, and `artifacts/tp2-bench/final-byte-0255/` (`qualification-outcome.json`, `qualification-0308.log`).

## W4 readiness diagnostic

A bounded native-only diagnostic ran two cold cohorts on the long shapes before W4 was measured. It was a readiness check only; the W4 campaign row is recorded above.

- **Shorter shape.** Sixteen lanes with 2048-token prompts and eight outputs each ran twice cold. All sixteen lanes produced identical output IDs across the two cohorts.
- **Longer shape.** Sixteen lanes with 8192-token prompts and eight outputs each ran twice cold. Two of sixteen lanes differed between the cohorts. The first differing output was 24 to 2015 at offset 1 on one lane and 8156 to 279 at offset 0 on the other.
- **Cache.** Both ranks acknowledged every store: sixteen per rank on the shorter shape and thirty-two on the longer one. GQA and GDN written bytes are equal across ranks.
- **Evidence.** `artifacts/tp2-bench/long-readiness-0257/` (`results.json`, `analysis.log`).

## W3 wide-verify diagnostic

A W3-shaped diagnostic ran one candidate binary with the two-rank wide verifier enabled and disabled. The wide verifier combines eight greedy lanes into one forty-row round. Worst-lane end-to-end time fell from 7039.4 ms to 6573.9 ms; median per-output time fell from 5.171 ms to 4.925 ms. Both arms produced identical outputs. This single-cohort diagnostic is not a paired campaign measurement. Evidence: `artifacts/tp2-w3-throughput-dev/bench/w3_wide_verify_evidence.json`.

## Two-rank wide verification

Under two ranks, combination stops at eight lanes or the backend's native width, whichever is smaller. The root rank drafts proposals and owns the greedy policy; the peer rank mirrors each round without drafting. At exactly eight eligible lanes the wide draft runs depth-major, when the 696,320-byte wide activation reserves; its live depth-scoring extent is 409,600 bytes. Below eight lanes, or when the reservation fails, wave-major packets remain. Commit stays packet-sized under two ranks, repairing per row; verify width and wide groups are unchanged. Before any verifier begins, both targets must clear a mirrored storage, stream, and position preflight. Only an all-greedy complete eight-lane group verifies as one wide round. Sampled, constrained, partial, logprob, or five-to-seven-lane compositions keep four-lane packets. An all-greedy round scores through one certified forty-row vocabulary pass. Setting `SGLANG_DISABLE_TP2_WIDE_VERIFY` forces the four-lane reference dispatch. [Wide lane cap](../../include/sglang/runtime/execution/decoder_batch_state.h#L115)<!-- prefix: "wide_mtp_lim" --> · [Group dispatch](../../src/decoder_batch.cpp#L635)<!-- prefix: "const bool f" --> · [Peer preflight](../../src/decoder_verify.cpp#L811)<!-- prefix: "throw std::i" --> · [Vocabulary pass](../../src/decoder_verify.cpp#L751)<!-- prefix: "// Scores th" -->

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

W1 and W2 have passed the gate; W3, W4, and W5 have missed it. A passing campaign must satisfy every condition below.

- **Devices.** Run both engines alone on the same two B200 devices in the same rank order. Record both UUIDs, the rank assignment, interconnect, and distributed library.
- **Workloads.** Reuse the frozen W1–W5 fixture bytes. Offer 2, 2, 8, 16, and 8 requests total on W1–W5. This is one, one, four, eight, and four per GPU. Eight slots per GPU give sixteen slots total. W4 fills them exactly. W4 uses the first sixteen archived W5 lanes. W1 and W2 need one new distinct lane each. W3 and W5 need four more each. Freeze every new lane as distinct deterministic bytes in a separate additive TP2 corpus. Retain the TP1 lanes and bytes. Duplicate no request.
- **Native arm.** Run the native server with `--tp 2`, the default direct GPU LMCache path, and the LMCache daemon enabled. Verify the reported `tp_size` is two. [Server information](../native-model-info.md#server-information) Confirm effective direct cache activity on each rank, so a reported `tp_size` cannot hide a bypass. Record the adopted topology and the effective daemon configuration.
- **Cache.** Start one LMCache daemon per rank, each bound to its own GPU and rank namespace. [Ranked transport](lmcache.md#ranked-tp2-transport) Draft records live only in the rank 0 daemon. Drain and reset both rank daemons before cold timing.
- **vLLM arm.** Run the pinned checkout with `--tensor-parallel-size 2`. [TP option](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/engine/arg_utils.py#L1219)<!-- prefix: "\"--tensor-pa" --> Require `kv_transfer=false`, `kv_offloading_size=null`, and zero CPU weight offload in both workers' receipts. [Receipt policy](../performance.md#scheduler-and-reporting-policy)
- **Gate.** Submit each row's requests simultaneously through the original submission path. Run one warmup and one measured cohort per row and engine, and use the measured verdict. Record clocks and output lengths. Treat output content as reference only; no warmup-measured token identity is required. Require native TP2 makespan ≤1.10× pinned vLLM TP2 on each W1–W5 row independently. A passing row covers no other row. Apply the [scheduler policy](../performance.md#scheduler-and-reporting-policy) unchanged.
