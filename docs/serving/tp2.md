# TP2 serving status and acceptance target

TP2 serving uses eager decode. `--tp` accepts 1 or 2 and defaults to 1. Batch CUDA graphs are TP1-only. The `benchmarks/paired` launcher supports one GPU. Use `benchmarks/tp2/run.sh` for two-B200 campaigns. [TP option](../../apps/sglang-server.cpp#L169)<!-- prefix: "} else if (o" --> · [Graph policy](../../apps/sglang-server.cpp#L59)<!-- prefix: "bool enable_" --> · [TP1 topology](../../include/sglang/runtime/server/batch_session.h#L15)<!-- prefix: "inline const" --> · [Launcher limit](../../benchmarks/paired/README.md#L71)<!-- prefix: "the number o" -->

TP2 with LMCache requires rank device receipts. The server derives one endpoint per rank from the base socket path. Each rank uses a private namespace and daemon. [Rank endpoints](../../include/sglang/runtime/cache/lmcache_client.h#L58)<!-- prefix: "std::string " --> · [Ranked transport](lmcache.md#ranked-tp2-transport)

The run at `artifacts/tp2-review/serving/cache-2328/complete/cache/` reported TP2, DP1, PP1, MTP4, and ranked LMCache. Each rank runs a target worker. Rank 0 alone owns the full MTP draft and its cache bytes. [Server information](../native-model-info.md#server-information)

The run used one 18-token input and a longer extension.

- **Cold.** After a miss, both ranks stored 13 tokens, writing 219,648 GQA and 76,992,512 GDN bytes each. Rank 0 also wrote 25,344 draft bytes.
- **Warm.** The joint GET restored the same prefix and byte counts.
- **One-sided eviction.** Evicting rank 1 forced a cold run. Restore needs the same GDN endpoint on both ranks.
- **Longer tail.** A 25-token extension restored 422,400 GQA bytes and 76,992,512 GDN bytes per rank, with 50,688 draft bytes on rank 0.

See `rank-receipts.json` and `metrics.json` in that directory. Each token uses 16,896 target bytes per rank, with full-width draft storage on rank 0.

## Campaign record

The current campaign measured all five rows on one release, `168fdb1a`, against the pinned vLLM reference `81a8cb06…`; every row misses the ≤1.10 gate. Native ran TP2 MTP4 with ranked LMCache, eager off, batch CUDA graphs off, `SGLANG_GDN_VENDOR_PREFILL=1`, and vendor GQA off; the reference ran with KV offload, KV transfer, and CPU offload disabled. Executable SHA-256 `8385a2ba…`, protocol manifest `fda8adff…`. Evidence: `artifacts/tp2-bench/campaign-W124-GDN-168fdb1a-45-1559/five-rows.json` (`e33f270b…`). The historical rows below were measured on earlier releases.

- **W1.** Native 737.959724 ms; vLLM 493.984432 ms; ratio 1.493892673929449: miss.
- **W2.** Native 1617.663485 ms; vLLM 1315.224646 ms; ratio 1.2299522290125942: miss.
- **W3.** Native 4918.771611 ms; vLLM 4100.639623 ms; ratio 1.1995132621289604: miss.
- **W4.** Native 37267.981957 ms; vLLM 26244.656343 ms; ratio 1.4200217167995097: miss.
- **W5.** Native 48416.572765 ms; vLLM 39777.744062 ms; ratio 1.2171774419769759: miss.
- **Historical identity.** Campaign `92b206b0-16e8-4ad0-a600-463926e0d6df`, mode measure, protocol `tp2-paired-v1` (manifest `c2c3480a…`), corpus `tp2-corpus-v1` (manifest `d65076eb…`). Native source tree `337edce7…`, executable SHA-256 `0100f4e2…`, checkpoint `Qwen3.8-27B-FP8` (`b54fe441…`). Evidence: `artifacts/tp2-bench/campaign-0222/`.
- **W1.** Native median 592.803737 ms; vLLM median 558.431928 ms; ratio 1.0615505798944935 against the ≤1.10 gate: W1 passes. Rank receipts are recorded for both ranks. Rank 0 `GPU-57c04c1a…` carries the LM head and MTP draft: 19,031,037,824 weight bytes. Rank 1 `GPU-b79a7675…` carries 14,734,676,480. A separate traced cohort accompanies the timed arms.
- **W2.** Native measured 1699.082051 ms. The pinned vLLM reference measured 2000.013573 ms, from earliest submission to final `[DONE]`. The ratio is 0.849535260128957: W2 passes. The W2 reference predates full driver completion: its driver exited before end and retirement, so the span ends at the final `[DONE]`.
- **W3.** A second campaign (`4e83b2ac-a36d-45dc-ab54-c4cc889bda5c`) measured W3: protocol name `tp2-paired-v1`, a distinct 36-file manifest `6d38e692…`, same corpus, source, binary, and devices. Native measured 6862.178762 ms; vLLM measured 4113.792758 ms; ratio 1.6680905348611146: W3 misses the gate. From earliest submission to latest first token, native spans 2125.991606 ms and vLLM 1605.194380 ms. The remaining spans through final `[DONE]` are 4736.187156 ms and 2508.598378 ms. Both arms completed eight 8192-token prompts to 1024 outputs with retirement verified. Evidence: `artifacts/tp2-bench/campaign-timing-0826/W3/comparison.json`. Release `9dba4ab2` re-measured W3 as named campaign `59528e2e-…`, binary `7601ba51…`, same protocol manifest, corpus, checkpoint, and devices. Native median 5068.963697 ms faced fresh vLLM 4112.156203 ms: ratio 1.232677808615822, missing the gate by 545.5918737 ms against the 4523.3718233 ms ceiling. Native spans 2049.330194 ms to the latest first token, then 3019.633503 ms; vLLM spans 1609.036702 ms and 2503.119501 ms. Warmups ran 5247.990043 and 4317.087139 ms. Each native cohort recorded sixteen stores and no restores per rank; both arms retired. Evidence: `artifacts/tp2-bench/campaign-named-9dba4ab2-1315/W3/`.
- **W4.** Native measured 57338.227570 ms; vLLM measured 26346.666432 ms; ratio 2.176299142739304: W4 misses the gate. All sixteen lanes completed 32768-token prompts to 4096 outputs on both engines, with full backend and retirement verified; no request timed out. From earliest submission to the last lane's first token, native spans 18079.484509 ms and vLLM 14347.504268 ms. From that first token to final `[DONE]`, native spends 39258.743061 ms and vLLM 11999.162164 ms. Evidence: `artifacts/tp2-bench/campaign-timing-0826/W4/` and `artifacts/tp2-review/campaign/timing-only-0825/W4-phases.log`. Release `9dba4ab2` re-measured W4 as named campaign `8f1671ee-…`, binary `7601ba51…`. Native median 37971.121678 ms faced fresh vLLM 25554.467027 ms: ratio 1.485889791318323, missing the gate by 9861.2079483 ms against the 28109.9137297 ms ceiling. Native spans 19229.702746 ms to the last lane's first token, then 18741.418932 ms; vLLM spans 14290.140184 ms and 11264.326843 ms. Warmups ran 45936.967222 and 25669.308590 ms. Each native cohort recorded eighty stores and no restores per rank; both arms retired. The prior native median stood 19367.105892 ms higher. The measured source `9dba4ab2` predates the current default; the default adds only the default-off GDN prefill opt-in, leaving dispatch native. Evidence: `artifacts/tp2-bench/campaign-named-9dba4ab2-1315/W4/`.
- **W5.** Release `cef34e34` re-measured W5: campaign `0dd0b75f-71ac-467d-8d8d-2c7ee524055c`, protocol `tp2-paired-v1` (manifest `6d38e692…`), binary `b26e278f…`, same corpus, checkpoint, and devices. Native median 55734.968000 ms; vLLM median 39394.467235 ms; ratio 1.4147917692990728 against the 43333.9139585 ms ceiling: W5 misses by 12401.0540415 ms. Thirty-two requests ran across four cold cohorts of eight 131072-token prompts to 1024 outputs. Every lane ended in `length` with `[DONE]`, and both arms fully retired. Native spans 49642.135752 ms from earliest submission to the last lane's first token, then 6092.832248 ms through final `[DONE]`. vLLM spans 36113.703510 ms and 3280.763725 ms. The prior native median on `337edce7` was 66235.146069 ms, 10500.178069 ms longer, with the last lane's first token at 60009.979660 ms versus 49642.135752 ms. Evidence: `artifacts/tp2-bench/campaign-w5-cef34e34-after-external-1032/W5/comparison.json` and `artifacts/tp2-bench/w5-plain-0943/named-phase-comparison.json`.
- **Historical context.** The first campaign (`24c2e092-25c7-4945-a251-aa9cb03358cb`, source `3a33e1c1…`, binary `fb03d111…`) measured W1 only: 925.746744 ms against 557.959022 ms, ratio 1.6591661887313294, a miss, and stopped. The [diagnostic profile](#w1-diagnostic-profile) below attributes that gap. A separate clean production diagnostic on `337edce7` measured 594.644780 ms for W1 before this campaign; it is not a campaign row.

## W4 wide-verification release

One cold W4 comparison used sixteen 32,768-token prompts and 4,096 outputs each. Rank-local packed staging with sixteen lanes reduced charged makespan from 50.815826650 s to 38.788724911 s (23.67%) against the eight-lane control. All-lane overlap throughput rose from 2026.02 to 3564.05 tokens/s. Time from earliest submission to latest first token rose from 18.492582 s to 20.007657 s. The remaining span through final `[DONE]` fell from 32.323245 s to 18.781068 s. The named W4 row records this release. See `artifacts/tp2-bench/w4-named-1219/ordinary-W4-independent-replay.json` (`c38195a7…`) and binaries `ab8a1e45…` and `425fbffd…`.
A paired W3 session (`283cbb84-…`) priced the same candidate lineage. Native median 5119.357970 ms faced a fresh vLLM 4104.882148 ms: ratio 1.2471388423402796, missing the gate by 603.9876072 ms. The eight-lane control measured 6168.325329 ms. The named W3 row above prices the released executable. Evidence: `artifacts/tp2-bench/w3-scratch-final-after-lock-1241/W3-candidate/W3/comparison.json` (`9c4e8c85…`).

## W1 diagnostic profile

An untimed profile traced both engines after the W1 miss.

- **Cohort.** Two fixed MTP4 requests on native TP2 and pinned vLLM. Eager and batch graphs were off. LMCache used the default socket and one daemon per rank.
- **Sampling.** Frozen 128-token W1 prompts generated 24 native or 128 vLLM output tokens per request.
- **Devices.** Native used B200 devices 1 and 5 and vLLM used 4 and 5. Both pairs used NV18 NVLink.
- **Width.** One or two active lanes, five verification rows per lane.
- **Method.** Nsight Systems traced both native GPU timelines together. The native span starts at `verify_rms` and ends at the 128th sum. The vLLM target graph includes the embedding sum and 128 fused reductions. Both exclude scoring, rejection sampling, four drafts, and final GDN rollback. Period uses native target starts and vLLM first step kernels. PyTorch sampled four steps per rank and width. Native width-two medians cover six rounds.
- **Rank 0 medians.** At width two, native and reference periods were 18.746 and 10.237 ms, with target spans of 13.505 and 8.016 ms. At width one, periods were 12.498 and 10.177 ms, with target spans of 9.828 and 8.010 ms.
- **Rank 0 kernel-duration sums inside the target, width two.** FP8 projection 3.529 versus 4.024 ms. GQA attention 2.541 versus 0.451 ms. GDN attention 1.619 versus 1.048 ms. See [M10 projection replay](../long-paged-decode.md#ten-row-tp2-verification-projections).
- **Reductions.** At width two, 128 cross-rank spans sum to 1.531 ms native and 0.753 ms reference. Each span runs from the last arrival to the last completion. See [M10 fusion](#ten-row-verification-reduction-fusion).
- **Target head.** Outside the target span, native rank 0 spent 1.633 ms per width-two step on two full BF16 vocabulary grids of 31040 by one. Reference rank 0 spent 0.181 ms on sharded GEMM and 0.034 ms on gather.
- **M10 operator replay.** Rank 0 can use the [M10 certificate](../native-model-info.md#certified-tensor-core-target-head) for two complete eager greedy TP2 K4 lanes. Scored, sampled, constrained, or partial packets keep ordinary scoring. On one B200, four captured W1 packets averaged 577.210 µs direct packed BF16 versus 432.327 µs certified across their host-visible medians, with exact IDs and BF16 scores for all 40 winners. An independent check covered bounds for 14,899,200 BF16 logits and matched 60 winners. Serving latency remains unmeasured. See `artifacts/tp2-resume/short-score/candidate/result.log` and `artifacts/tp2-resume/short-review/run-3.log`.
- **Evidence.** `artifacts/tp2-bench/native-profile-d0029/` holds the Nsight Systems database, `steps.json`, and `vs-reference.json`. `artifacts/tp2-bench/vllm-profile-0013/` holds `scoring.json`, `reduction-skew.json`, and per-rank traces with `profile.sha256` and `repro.sha256`. The analysis scripts are in `artifacts/tp2-bench/w1-diagnostics/`. The diagnostic native binary is SHA-256 `0acf4c24…`; the first campaign's binary is `fb03d111…`.

## W2 divergence evidence

The runs below reproduced the W2 divergence under controlled changes in request arrival. They are diagnostic only; they explain the superseded determinism rejection. The `337edce7` W2 row passed under the timing-only criterion; the `168fdb1a` cohort missed.

- **Original-wire bursts.** Three cold, complete, original-wire W2 reference bursts with lane submissions 12.0–19.4 µs apart matched the earlier official warmup exactly: all 3×1024 tokens.
- **Stagger replay.** A 64-token diagnostic replayed the divergence with staggered submissions of 73.7–141.6 ms. The first divergence reappeared at 0-based offset 49 in a cold cohort. Lane 1 held nine output IDs before lane 0 was sent and seventeen before lane 0's first token. A 64-token logprob pair measured the top-2 candidates at offset 49: 1788 and 5215 tie at −0.6938778162002563, on the stagger cell only.
- **Submission-change rejections.** A true ordered submission and a symmetric final-byte burst each captured both complete W2 cohorts on the pinned vLLM reference. Both failed identically: lane 0 output 49 differs from warmup, expected 5215, observed 1788. The two lanes' final-byte writes started 6.382 µs apart in the warmup and 6.481 µs apart in the measured cohort. Neither change is released or installed, and no replacement protocol has qualified.
- **Final-byte lane detail.** In the final-byte measured cohort, lane 0 first differs at 0-based offset 49, 5215 to 1788, with 346 of 512 outputs changed. Lane 1 first differs at offset 303, 760 to 1919, with 87 of 512 changed. The final-byte warmup equals the official warmup. The final-byte measured lane 1 differs from the official measured lane 1 at 82 IDs. Engine iteration token totals show one large iteration in (4096, 8192] tokens in the warmup. The measured cohort shows two iterations in (2048, 4096] instead, consistent with joint versus split processing of the two 2048-token prompts. Lane 1 led by one output ID in the warmup and by four in the measured cohort, independently observed on server-sent events.
- **Evidence.** `artifacts/tp2-bench/w2-diagnostic-0239/outcomes.json` (SHA-256 `b5ef5b3a…`), `artifacts/tp2-bench/ordered-submit-0247/qualification.log`, and `artifacts/tp2-bench/final-byte-0255/` (`qualification-outcome.json`, `qualification-0308.log`).

## W4 readiness diagnostic

Two cold native-only cohorts checked long-shape readiness before the W4 campaign.

- **Shorter shape.** All sixteen lanes matched across cold cohorts of 2,048-token prompts and eight outputs.
- **Longer shape.** With 8,192-token prompts and eight outputs, two of sixteen lanes differed. First differences were 24 to 2015 at offset 1 and 8156 to 279 at offset 0.
- **Cache.** Each rank acknowledged sixteen stores at the shorter shape and thirty-two at the longer shape. GQA and GDN byte counts matched across ranks.
- **Evidence.** `artifacts/tp2-bench/long-readiness-0257/` (`results.json`, `analysis.log`).

## W3 wide-verify diagnostic

One W3-shaped cohort tested the same binary with and without wide verification. Eight greedy lanes formed forty-row rounds. Worst-lane makespan fell from 7039.4 to 6573.9 ms. Median time per output fell from 5.171 to 4.925 ms. Outputs matched. See `artifacts/tp2-w3-throughput-dev/bench/w3_wide_verify_evidence.json`.

The proposed [W3 shared-KV route](../singleton-gqa-decode.md#tensor-core-reads-across-long-k4-packets) covers first positions 8,191–10,235 for eight complete five-row groups with 12Q/2KV per rank. Its [operator probe](../singleton-gqa-decode.md#w3-shared-kv-operator-probe) measured 35–39% lower stream time. Candidate W3 makespan awaits measurement.

## Two-rank wide verification

The root owns greedy policy and drafts, and the peer mirrors its rounds. Eligible groups of eight or sixteen draft depth-major when reserves fit (696,320 or 1,392,640 bytes). Eight lanes use 409,600 bytes for live depth scoring. Otherwise drafting stays wave-major. Both ranks preflight mirrored storage, streams, and positions. Only complete greedy groups of eight or sixteen verify together, subject to native width, and score in certified forty-row passes. At sixteen lanes both ranks stage packed activations. Sampled, constrained, partial, and logprob groups keep four-lane packets. Commit can batch [sixteen-row draft repair](../long-paged-decode.md#sixteen-row-tp2-draft-repair). `SGLANG_DISABLE_TP2_WIDE_VERIFY` forces four-lane dispatch. [Wide lane cap](../../include/sglang/runtime/execution/decoder_batch_state.h#L117)<!-- prefix: "wide_mtp_lim" --> · [Group dispatch](../../src/decoder_batch.cpp#L635)<!-- prefix: "const bool f" --> · [Peer preflight](../../src/decoder_verify.cpp#L813)<!-- prefix: "throw std::i" --> · [Vocabulary pass](../../src/decoder_verify.cpp#L753)<!-- prefix: "void score_t" -->

Narrow and wide TP2 verification gather embeddings and scatter reduced hidden states into row-owned buffers, with one launch per stage per rank. Separate warm probes on one B200 against `e0a56f4` measured paired median stream savings of 13.504/18.464 µs at 10 rows and 131.712/170.464 µs at 80 rows, respectively. Serving remains unmeasured. See `artifacts/tp2-resume/verify-scatter/receipt.txt`. [Gather](../../src/decoder_verify.cpp#L848)<!-- prefix: "check(gather" --> · [Scatter](../../src/runtime/execution/decoder_forward_tp.cpp#L132)<!-- prefix: "check(scatte" -->

## Prerequisites and command

Use two visible GPUs, a sharded-checkpoint build, LMCache 0.5.8, and one daemon per rank started with `tools/lmcache/serve.sh --rank N`. Then run:

```sh
sglang-server --checkpoint MODEL --device 0 --port PORT --tp 2 --eager \
  --mtp-steps 4 --lmcache-socket SOCKET
```

The smoke arguments are in `artifacts/tp2-review/serving/cache-2328/complete/cache/ranked-cache.argv.json`. The campaigns omitted `--eager` and used `--port 17393 --tp 2 --max-active 16`.

## Two-rank allreduce transport

The optional transport embeds FlashInfer 0.7.4 TRT-LLM MNNVL kernels. Set `SGLANG_FLASHINFER_ROOT` to override the wheel header path. `SGLANG_HOST_ROOT` changes the enclosing shared root. Builds without headers use NCCL. [Build gate](../../xmake.lua#L472)<!-- prefix: "local flashi" -->

Both kernels require 16-byte aligned buffers and whole 5,120-element BF16 rows. One shot accepts up to 51 rows under its 1,048,576-byte aggregate cap, in eager execution or joined capture. Exactly 80 eager rows use two shot when each Lamport slot holds both 819,200-byte stages and both GPUs can run all 400 blocks concurrently. [Dispatch](../../src/runtime/backend/tensor_parallel.cpp#L443)<!-- prefix: "const bool t" --> · [Workspace check](../../src/runtime/backend/mnnvl_ar.cu#L407)<!-- prefix: "state.twosho" --> · [Occupancy check](../../src/runtime/backend/mnnvl_ar.cu#L37)<!-- prefix: "bool twoshot" -->

NCCL handles other oversize calls, independent captures, and captured 80-row calls. Mixed eager and captured streams are rejected. A vendor enqueue error propagates. Later calls use NCCL if neither rank enqueued. A partial enqueue marks the backend failed. [Error handling](../../src/runtime/backend/tensor_parallel.cpp#L292)<!-- prefix: "if (error.pe" -->

One host thread enqueues both ranks in order. The multicast workspace reserves 536,870,912 bytes per rank with triple Lamport buffers. The rank sum normalizes negative zero to protect the Lamport sentinel. [Sentinel](../../src/runtime/backend/mnnvl_ar.cu#L29)<!-- prefix: "constexpr st" -->

`/metrics` counts successful enqueues per rank by transport and six NCCL fallback reasons. Graph replays do not increment these counters. [Counters](../../apps/server/loads.cpp#L234)<!-- prefix: "emit_rank(\"t" --> · [Fallback reasons](../../src/runtime/backend/tensor_parallel.cpp#L98)<!-- prefix: "enum class A" -->

The 80-row operator probe on two B200s measured a 4.032 µs saving per collective. Median worst-rank stream time fell from 21.472 µs with NCCL to 17.440 µs across 81 samples without induced lag. W4 makespan awaits measurement. See `artifacts/tp2-resume/w4-ar/` (`stock-45.log`, `vendor-45.log`).

### Ten-row verification reduction fusion

Eager TP2 uses one-shot MNNVL plus RMSNorm and quantization for ten packed verification rows of width 5,120 on both ranks. It requires healthy MNNVL, ten resident blocks per GPU, and aligned operands. Capture and other clean declines use ordinary collective dispatch and normalization. In joined capture, the protocol test used MNNVL at ten rows and NCCL at 52. [Selection](../../src/runtime/execution/decoder_forward_tp.cpp#L49)<!-- prefix: "struct TpRed" --> · [Capture](../../src/runtime/backend/tensor_parallel.cpp#L457)<!-- prefix: "bool TensorP" --> · [Preflight](../../src/runtime/backend/mnnvl_ar.cu#L615)<!-- prefix: "bool MnnvlAl" -->

Each rank fuses 64 attention reductions with MLP input normalization and 63 MLP reductions with the next layer’s input normalization. The last MLP reduction stays separate before scatter. The kernel preserves the BF16 sum and residual and quantizes the BF16 normalized row to E4M3 with one FP32 scale per 128 channels. [Layer loop](../../src/runtime/execution/decoder_forward_tp.cpp#L252)<!-- prefix: "const bool a" --> · [Arithmetic](../../include/sglang/text/quantization/rms_quantize.cuh#L99)<!-- prefix: "template \u003cbo" -->

The host prepares both rank launches before submission. A partial enqueue fails the group and propagates the error without an NCCL retry. An already-running kernel can wait indefinitely for its peer, which can require process termination. [Submission](../../src/runtime/backend/mnnvl_ar.cu#L638)<!-- prefix: "ReduceNorm l" --> · [Failure](../../src/runtime/backend/tensor_parallel.cpp#L292)<!-- prefix: "if (error.pe" -->

On two B200s in both rank orders, 32 paired isolated target traversals saved a median 0.562 ms (8.32%). Both ranks’ greedy and sampled digests and MTP work matched, including forced NCCL and alignment refusal controls. An independent operator audit matched 921,600 entries each for reduced BF16, residual BF16, and E4M3 activations, plus 7,200 FP32 scales. Full W1 serving remains unmeasured. See `artifacts/tp2-resume/w1-reduce-norm/production/tracked/target.5ns0DEvu/` and `artifacts/tp2-resume/reduce-norm-review/review.md`.

## Acceptance target

W1 and W2 passed on `337edce7`; W3–W5 missed on their releases. The `168fdb1a` cohort missed all five rows. A passing campaign must satisfy every condition below.

- **Devices.** Run both engines alone on the same two B200 devices in the same rank order. Record both UUIDs, the rank assignment, interconnect, and distributed library.
- **Workloads.** Reuse the frozen W1–W5 fixture bytes. Offer 2, 2, 8, 16, and 8 requests total on W1–W5. This is one, one, four, eight, and four per GPU. Eight slots per GPU give sixteen slots total. W4 fills them exactly. W4 uses the first sixteen archived W5 lanes. W1 and W2 need one new distinct lane each. W3 and W5 need four more each. Freeze every new lane as distinct deterministic bytes in a separate additive TP2 corpus. Retain the TP1 lanes and bytes. Duplicate no request.
- **Native arm.** Run the native server with `--tp 2`, the default direct GPU LMCache path, and the LMCache daemon enabled. Verify the reported `tp_size` is two. [Server information](../native-model-info.md#server-information) Confirm effective direct cache activity on each rank, so a reported `tp_size` cannot hide a bypass. Record the adopted topology and the effective daemon configuration.
- **Cache.** Start one LMCache daemon per rank, each bound to its own GPU and rank namespace. [Ranked transport](lmcache.md#ranked-tp2-transport) Draft records live only in the rank 0 daemon. Drain and reset both rank daemons before cold timing.
- **vLLM arm.** Run the pinned checkout with `--tensor-parallel-size 2`. [TP option](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/engine/arg_utils.py#L1219)<!-- prefix: "\"--tensor-pa" --> Require `kv_transfer=false`, `kv_offloading_size=null`, and zero CPU weight offload in both workers' receipts. [Receipt policy](../performance.md#scheduler-and-reporting-policy)
- **Gate.** Submit each row's requests simultaneously through the original submission path. Run one warmup and one measured cohort per row and engine, and use the measured verdict. Record clocks and output lengths. Treat output content as reference only; no warmup-measured token identity is required. Require native TP2 makespan ≤1.10× pinned vLLM TP2 on each W1–W5 row independently. A passing row covers no other row. Apply the [scheduler policy](../performance.md#scheduler-and-reporting-policy) unchanged.
