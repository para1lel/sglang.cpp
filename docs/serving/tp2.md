# TP2 serving status and acceptance target

`--tp` accepts 1 or 2 and defaults to 1. TP2 defaults to [eligible verification replay](../batch-graphs.md#tp2-target-verification-graph). `--eager` disables it. Ordinary TP2 decode is eager. Use `benchmarks/tp2/run.sh` for two-B200 campaigns. [TP option](../../apps/sglang-server.cpp#L169)<!-- prefix: "} else if (o" --> · [Graph policy](../../apps/sglang-server.cpp#L355)<!-- prefix: "arguments.ma" --> · [TP1 topology](../../include/sglang/runtime/server/batch_session.h#L15)<!-- prefix: "inline const" --> · [Launcher limit](../../benchmarks/paired/README.md#L71)<!-- prefix: "the number o" -->

TP2 with LMCache requires rank device receipts. The server derives one endpoint per rank from the base socket path. Each rank uses a private namespace and daemon. [Rank endpoints](../../include/sglang/runtime/cache/lmcache_client.h#L58)<!-- prefix: "std::string " --> · [Ranked transport](lmcache.md#ranked-tp2-transport)

The `artifacts/tp2-review/serving/cache-2328/complete/cache/` run used TP2, DP1, PP1, MTP4, and ranked LMCache. Both ranks run target workers. Rank 0 owns draft weights and cache.

The run used one 18-token input and a longer extension.

- **Cold.** After a miss, both ranks stored 13 tokens, writing 219,648 GQA and 76,992,512 GDN bytes each. Rank 0 also wrote 25,344 draft bytes.
- **Warm.** The joint GET restored the same prefix and byte counts.
- **One-sided eviction.** Evicting rank 1 forced a cold run. Restore needs the same GDN endpoint on both ranks.
- **Longer tail.** A 25-token extension restored 422,400 GQA bytes and 76,992,512 GDN bytes per rank, with 50,688 draft bytes on rank 0.

See `rank-receipts.json` and `metrics.json`. Target storage is 16,896 bytes per token per rank. Rank 0 stores the full draft.

## Campaign record

Official `6f7754f4-b4f7-4e0b-acb1-ea2f8df74e73` measured `e912b39b` on B200s 4/5. Binary `dcfee059…` and pinned vLLM `81a8cb06…` used `tp2-paired-v1`. Makespans are native/vLLM. Native cohorts had zero LMCache restores or I/O errors.

- W1. 498.350240/475.794275 ms (1.047406970×, pass). 2 stores per rank.
- W2. 1228.592599/1214.372621 ms (1.011709732×, pass). 2 stores per rank.
- W3. 4074.498811/3829.800530 ms (1.063893218×, pass). 16 stores per rank.
- W4. 29046.274476/25335.314580 ms (1.146473804×, miss). 80 stores per rank. Both arms completed sixteen requests.

W4 was the first miss, 1177.428438 ms above the 1.10× limit. W5 was skipped. See `artifacts/tp2-resume/combined-release/e912b39bec428ce93728d62b4cae18899d7d3462/official-campaign/` for identity, per-row comparisons, and measured ranked cache snapshots.

Historical `0d745375` on B200s 4/5 passed W1 at 498.339692/474.724356 ms (1.049745364×) and W2 at 1286.342700/1214.911601 ms (1.058795306×). W3 missed by 42.308 ms at 4261.832270/3835.931200 ms (1.111029382×), with 16 LMCache stores per rank and zero restores or I/O errors. See `artifacts/tp2-resume/w1-graph-qualification/0d7453751ddf1699a50f5370ecc6a7b9a02e8623/official-4-5/W{1,2,3}/comparison.json` (`dd2aa96c`, `0925c245`, `f8b6d66b`).

Earlier W1 misses on B200s 4/5 included `a5f76d36` (545.106259/477.516500 ms, 1.141544342×), `edca6a` (592.471368/474.298792 ms, 1.249152×), and `6e71fceb` (1.142710×). `edca6a` skipped W2–W5 after the 6/3 interruption and scored 4/5 run. Receipts under `artifacts/tp2-resume/` include `combined-release/a5f76d36af3641e4f7f3f1fff2d6a5651a76bbac/official-campaign/W1/comparison.json` and `w1-draft-caller/release-3d2cf7dc02238cd79c0abe87582422eb057bd6ba/qualification-45/W1/comparison.json`.

The bounded `432daa8f` W1 HTTP diagnostic measured 508.721→499.538 ms median wire time over six runs per arm on B200s 4/5, with exact IDs and rank-state bytes. See `artifacts/tp2-resume/w1-fused-http/timed.x1bMco3R/graph-comparison.json`.

Release `36cb430a` missed W1 by 0.513175 ms (524.241268/476.116448 ms, 1.101077835×) and passed W2 (1302.201491/1218.488949 ms, 1.068702×). Both had two LMCache stores per rank, zero restores. See `artifacts/tp2-resume/w1-fusion-release/36cb430a3fddd479ce787aa34926cb6f46480f81/` (`requalified-campaign/W1`, `official-campaign/W2`).

Historical `168fdb1a`: TP2 MTP4, ranked LMCache, vendor GDN, native GQA, eager flag and batch graphs off. Pinned vLLM `81a8cb06…` disabled KV offload, KV transfer, and CPU offload. Binary `8385a2ba…`, manifest `fda8adff…`. All five missed. See `artifacts/tp2-bench/campaign-W124-GDN-168fdb1a-45-1559/five-rows.json` (`e33f270b…`).

- **W1.** 737.959724 ms native, 493.984432 ms vLLM (1.493892673929449).
- **W2.** 1617.663485 ms native, 1315.224646 ms vLLM (1.2299522290125942).
- **W3.** 4918.771611 ms native, 4100.639623 ms vLLM (1.1995132621289604).
- **W4.** 37267.981957 ms native, 26244.656343 ms vLLM (1.4200217167995097).
- **W5.** 48416.572765 ms native, 39777.744062 ms vLLM (1.2171774419769759).
- **Historical identity.** Campaign `92b206b0-16e8-4ad0-a600-463926e0d6df`, mode measure, protocol `tp2-paired-v1` (manifest `c2c3480a…`), corpus `tp2-corpus-v1` (manifest `d65076eb…`). Native source tree `337edce7…`, executable SHA-256 `0100f4e2…`, checkpoint `Qwen3.8-27B-FP8` (`b54fe441…`). Evidence: `artifacts/tp2-bench/campaign-0222/`.
- **W1.** Native 592.803737 ms, vLLM 558.431928 ms (1.0615505798944935, pass). Rank 0 `GPU-57c04c1a…` holds 19,031,037,824 weight bytes, including LM head and draft. Rank 1 `GPU-b79a7675…` holds 14,734,676,480. Both have rank receipts. A separate cohort captured traces.
- **W2.** Native 1699.082051 ms, vLLM 2000.013573 ms (0.849535260128957, pass). The reference spans earliest submission to final `[DONE]` because its driver exited before end and retirement checks.
- **W3.** Campaign `4e83b2ac-a36d-45dc-ab54-c4cc889bda5c` reused the corpus, source, binary, and devices with `tp2-paired-v1` manifest `6d38e692…` (36 files). Native/vLLM makespans were 6862.178762/4113.792758 ms (1.6680905348611146×, miss). First-token spans were 2125.991606/1605.194380 ms and remaining spans 4736.187156/2508.598378 ms. Both arms retired eight 8,192-token prompts after 1,024 outputs each. See `artifacts/tp2-bench/campaign-timing-0826/W3/comparison.json`. Campaign `59528e2e-…` on `9dba4ab2` (binary `7601ba51…`) reused the manifest, corpus, checkpoint, and devices. It measured 5068.963697/4112.156203 ms (1.232677808615822×), 545.5918737 ms over the 4523.3718233 ms ceiling. First-token spans were 2049.330194/1609.036702 ms, remaining 3019.633503/2503.119501 ms, warmup 5247.990043/4317.087139 ms. Native stored sixteen per rank with zero restores. Both arms retired. See `artifacts/tp2-bench/campaign-named-9dba4ab2-1315/W3/`.
- **W4.** Native/vLLM makespans were 57338.227570/26346.666432 ms (2.176299142739304×, miss). Both arms verified the full backend and retired sixteen 32,768-token prompts at 4,096 outputs each without timeouts. First-token spans were 18079.484509/14347.504268 ms and remaining spans 39258.743061/11999.162164 ms. See `artifacts/tp2-bench/campaign-timing-0826/W4/` and `artifacts/tp2-review/campaign/timing-only-0825/W4-phases.log`. Campaign `8f1671ee-…` on `9dba4ab2` (binary `7601ba51…`) measured 37971.121678/25554.467027 ms (1.485889791318323×), 9861.2079483 ms over the 28109.9137297 ms ceiling. First-token spans were 19229.702746/14290.140184 ms, remaining 18741.418932/11264.326843 ms, warmup 45936.967222/25669.308590 ms. Each native cohort stored eighty per rank with zero restores. Both arms retired. Prior native was 19367.105892 ms slower. `9dba4ab2` predates the GDN prefill opt-in, which defaults to native dispatch. See `artifacts/tp2-bench/campaign-named-9dba4ab2-1315/W4/`.
- **W5.** Release `cef34e34`, campaign `0dd0b75f-71ac-467d-8d8d-2c7ee524055c`, protocol `tp2-paired-v1` (manifest `6d38e692…`), binary `b26e278f…`, same corpus, checkpoint, and devices. Native/vLLM medians were 55734.968000/39394.467235 ms (1.4147917692990728×), 12401.0540415 ms over the 43333.9139585 ms ceiling. Thirty-two requests ran across four cold cohorts of eight 131,072-token prompts each retired at 1,024 outputs, all ending in `length` with `[DONE]`. Earliest submission to latest first-token spans were 49642.135752/36113.703510 ms and remaining spans 6092.832248/3280.763725 ms. On `337edce7`, native took 66235.146069 ms (10500.178069 ms longer) with a 60009.979660 ms first-token span. See `artifacts/tp2-bench/campaign-w5-cef34e34-after-external-1032/W5/comparison.json` and `artifacts/tp2-bench/w5-plain-0943/named-phase-comparison.json`.
- **Historical context.** The first campaign (`24c2e092-25c7-4945-a251-aa9cb03358cb`, source `3a33e1c1…`, binary `fb03d111…`) stopped after W1 missed at 925.746744 versus 557.959022 ms (1.6591661887313294). The [diagnostic profile](#w1-diagnostic-profile) attributes the gap. A separate pre-campaign production diagnostic on `337edce7` measured 594.644780 ms.

## W4 wide-verification release

In a cold W4 comparison with sixteen 32,768-token prompts and 4,096 outputs each, sixteen-lane rank-local staging cut makespan from 50.815826650 to 38.788724911 s (23.67%) versus eight lanes. All-lane overlap rose from 2026.02 to 3564.05 tokens/s. Time from earliest submission to latest first token rose from 18.492582 to 20.007657 s. The span through final `[DONE]` fell from 32.323245 to 18.781068 s. See `artifacts/tp2-bench/w4-named-1219/ordinary-W4-independent-replay.json` (`c38195a7…`) and binaries `ab8a1e45…`, `425fbffd…`.
Paired W3 (`283cbb84-…`) on that lineage missed by 603.9876072 ms (5119.357970/4104.882148 ms, 1.2471388423402796×). The eight-lane native control took 6168.325329 ms. See `artifacts/tp2-bench/w3-scratch-final-after-lock-1241/W3-candidate/W3/comparison.json` (`9c4e8c85…`).

## W1 diagnostic profile

An untimed profile traced both engines.

- **Cohort.** Two fixed native TP2 MTP4 and pinned vLLM requests. Eager flag and batch graphs off. Default LMCache socket with one daemon per rank.
- **Sampling.** Frozen 128-token W1 prompts generated 24 native or 128 vLLM output tokens per request.
- **Devices.** NV18 B200s 1/5 for native and 4/5 for vLLM.
- **Width.** One or two active lanes, five verification rows per lane.
- **Method.** Nsight Systems traced both native GPU timelines together. The native span starts at `verify_rms` and ends at the 128th sum. The vLLM target graph includes the embedding sum and 128 fused reductions. Both exclude scoring, rejection sampling, four drafts, and final GDN rollback. Period uses native target starts and vLLM first step kernels. PyTorch sampled four steps per rank and width. Native width-two medians cover six rounds.
- **Rank 0 medians.** At width two, native and reference periods were 18.746 and 10.237 ms, with target spans of 13.505 and 8.016 ms. At width one, periods were 12.498 and 10.177 ms, with target spans of 9.828 and 8.010 ms.
- **Rank 0 kernel-duration sums inside the target, width two.** FP8 projection 3.529 versus 4.024 ms. GQA attention 2.541 versus 0.451 ms. GDN attention 1.619 versus 1.048 ms. See [M10 projection replay](../long-paged-decode.md#ten-row-tp2-verification-projections).
- **Reductions.** At width two, 128 cross-rank spans sum to 1.531 ms native and 0.753 ms reference. Each span runs from the last arrival to the last completion. See [Verification fusion](#verification-reduction-fusion).
- **Target head.** Outside the target span, native rank 0 spent 1.633 ms per width-two step on two full BF16 vocabulary grids of 31040 by one. Reference rank 0 spent 0.181 ms on sharded GEMM and 0.034 ms on gather.
- **M10 operator replay.** Rank 0 can use the [M10 certificate](../native-model-info.md#certified-tensor-core-target-head) for two complete eager greedy TP2 K4 lanes. Scored, sampled, constrained, or partial packets keep ordinary scoring. On one B200, four captured W1 packets averaged 577.210 µs direct packed BF16 versus 432.327 µs certified across their host-visible medians, with exact IDs and BF16 scores for all 40 winners. An independent check covered bounds for 14,899,200 BF16 logits and matched 60 winners. Serving latency remains unmeasured. See `artifacts/tp2-resume/short-score/candidate/result.log` and `artifacts/tp2-resume/short-review/run-3.log`.
- **Draft caller replay.** The [native FC and K/V route](../native-model-info.md#grouped-draft-projections) matched B200 checkpoint BF16 normalized rows, logits, IDs, and MXFP8 KV prefixes. Median paired K4 proposal and repair savings were 174.378/117.106 µs at one/two rows, with 21/21 wins each. HTTP impact remains unmeasured. See `artifacts/tp2-resume/w1-draft-stream/tracked/caller.log`.
- **Evidence.** `artifacts/tp2-bench/native-profile-d0029/` holds the Nsight Systems database, `steps.json`, and `vs-reference.json`. `artifacts/tp2-bench/vllm-profile-0013/` holds `scoring.json`, `reduction-skew.json`, and per-rank traces with `profile.sha256` and `repro.sha256`. The analysis scripts are in `artifacts/tp2-bench/w1-diagnostics/`. The diagnostic native binary is SHA-256 `0acf4c24…`; the first campaign's binary is `fb03d111…`.

## W2 divergence evidence

Changing arrival timing reproduced the superseded W2 determinism rejection. Under the timing-only gate, `337edce7` passed and `168fdb1a` missed.

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

The [W3 shared-KV probe](../singleton-gqa-decode.md#w3-shared-kv-operator-probe) reported 35–39% lower stream time. [Compact FP8](../long-paged-decode.md#compact-forty-row-projections), [eight-row repair](../batch-graphs.md#eight-row-tp2-draft-repair), [peer repair](request-memory.md#peer-draft-repair), and [40-row fusion](#forty-row-verification-fusion) have direct replay results. See [official W3](#campaign-record).

## Two-rank wide verification

The root owns greedy policy and drafts, and the peer mirrors its rounds. Eligible groups of eight or sixteen draft depth-major when reserves fit (696,320 or 1,392,640 bytes). Eight lanes use 409,600 bytes for live depth scoring. Otherwise drafting stays wave-major. Both ranks preflight mirrored storage, streams, and positions. Only complete greedy groups of eight or sixteen verify together, subject to native width, and score in certified forty-row passes. Sixteen lanes pack activations on both ranks. Sampled, constrained, partial, and logprob groups keep four-lane packets. Commit supports [sixteen-row repair](../long-paged-decode.md#sixteen-row-tp2-draft-repair) and [peer staging](request-memory.md#peer-draft-repair). `SGLANG_DISABLE_TP2_WIDE_VERIFY` forces four-lane dispatch. [Wide lane cap](../../include/sglang/runtime/execution/decoder_batch_state.h#L117)<!-- prefix: "wide_mtp_lim" --> · [Verification width](../../src/decoder_batch.cpp#L536)<!-- prefix: "static int v" --> · [Peer preflight](../../src/decoder_verify.cpp#L816)<!-- prefix: "throw std::i" --> · [Vocabulary pass](../../src/decoder_verify.cpp#L753)<!-- prefix: "bool score_t" -->

Each rank gathers embeddings and scatters reduced states to row-owned buffers with one launch per stage. On one B200 against `e0a56f4`, separate warm probes measured median paired gather/scatter savings of 13.504/18.464 µs at 10 rows and 131.712/170.464 µs at 80. See `artifacts/tp2-resume/verify-scatter/receipt.txt`. [Gather](../../src/decoder_verify.cpp#L851)<!-- prefix: "check(gather" --> · [Scatter](../../src/runtime/execution/decoder_forward_tp.cpp#L206)<!-- prefix: "check(scatte" -->

## Prerequisites and command

Use two visible GPUs, a sharded-checkpoint build, LMCache 0.5.8, and one daemon per rank started with `tools/lmcache/serve.sh --rank N`. Then run:

```sh
sglang-server --checkpoint MODEL --device 0 --port PORT --tp 2 --eager \
  --mtp-steps 4 --lmcache-socket SOCKET
```

The smoke arguments are in `artifacts/tp2-review/serving/cache-2328/complete/cache/ranked-cache.argv.json`. The campaigns omitted `--eager` and used `--port 17393 --tp 2 --max-active 16`.

## Two-rank allreduce transport

Optional FlashInfer 0.7.4 TRT-LLM MNNVL uses wheel headers. `SGLANG_FLASHINFER_ROOT` overrides their path and `SGLANG_HOST_ROOT` the shared root. Builds without headers use NCCL. [Build gate](../../xmake.lua#L475)<!-- prefix: "local flashi" -->

Both routes need 16-byte alignment and whole 5,120-element BF16 rows. One shot supports up to 51 rows (1,048,576-byte cap) in eager execution or joined capture. Eager two shot needs 80 rows, two 819,200-byte stages per Lamport slot, and 400 concurrently resident blocks per GPU. [Dispatch](../../src/runtime/backend/tensor_parallel.cpp#L453)<!-- prefix: "const bool t" --> · [Workspace check](../../src/runtime/backend/mnnvl_ar.cu#L484)<!-- prefix: "state.twosho" --> · [Occupancy check](../../src/runtime/backend/mnnvl_ar.cu#L41)<!-- prefix: "bool twoshot" -->

NCCL handles other oversize calls, independent captures, and captured 80-row calls. Mixed eager and captured streams are rejected. Enqueue errors propagate. Later calls use NCCL if neither rank enqueued. Partial enqueue fails the backend. [Error handling](../../src/runtime/backend/tensor_parallel.cpp#L303)<!-- prefix: "if (error.pe" --> See [4,096-row fusion](../native-prefill-workspace.md#tp2-causal-prefill-reduction-fusion).

One host thread enqueues both ranks in order. The multicast workspace reserves 536,870,912 bytes per rank with triple Lamport buffers. The rank sum normalizes negative zero to protect the Lamport sentinel. [Sentinel](../../src/runtime/backend/mnnvl_ar.cu#L33)<!-- prefix: "constexpr st" -->

`/metrics` counts successful enqueues per rank by transport and six NCCL fallback reasons. Graph replays do not increment these counters. [Counters](../../apps/server/loads.cpp#L234)<!-- prefix: "emit_rank(\"t" --> · [Fallback reasons](../../src/runtime/backend/tensor_parallel.cpp#L99)<!-- prefix: "enum class A" -->

Across 81 balanced samples on two B200s without induced lag, median worst-rank stream time fell from 21.472 µs NCCL to 17.440 µs two shot. See `artifacts/tp2-resume/w4-ar/` (`stock-45.log`, `vendor-45.log`).

### Verification reduction fusion

TP2 fuses one-shot MNNVL, RMSNorm, and quantization for ten packed verification rows or [eligible eager singletons](../batch-graphs.md#eager-tp2-singleton-verification). Both ranks need aligned 5,120-wide operands and ten resident blocks per GPU. Clean declines use ordinary collectives and local preparation. Eligible [M10 graph replay](../batch-graphs.md#tp2-target-verification-graph) retains fusion. Ordinary joined capture used MNNVL at ten rows and NCCL at 52. [Selection](../../src/runtime/execution/decoder_forward_tp.cpp#L52)<!-- prefix: "struct TpRed" --> · [Capture](../../src/runtime/backend/tensor_parallel.cpp#L490)<!-- prefix: "bool TensorP" --> · [Preflight](../../src/runtime/backend/mnnvl_ar.cu#L697)<!-- prefix: "bool MnnvlAl" -->

Each rank fuses 64 attention reductions with MLP input normalization and 63 MLP reductions with next-layer input normalization. The final MLP reduction stays separate. The kernel preserves the BF16 sum, residual, and normalization before E4M3 quantization with one FP32 scale per 128 channels. [Layer loop](../../src/runtime/execution/decoder_forward_tp.cpp#L329)<!-- prefix: "const bool a" --> · [Arithmetic](../../include/sglang/text/quantization/rms_quantize.cuh#L99)<!-- prefix: "template \u003cbo" -->

The host prepares both rank launches before submission. Partial enqueue fails the group and propagates without an NCCL retry. On narrow fusion, a launched rank can wait indefinitely for its peer and require process termination. [Submission](../../src/runtime/backend/mnnvl_ar.cu#L747)<!-- prefix: "ReduceNorm\u003cR" --> · [Failure](../../src/runtime/backend/tensor_parallel.cpp#L303)<!-- prefix: "if (error.pe" -->

Across 32 paired isolated ten-row traversals on two B200s in both rank orders, median savings were 0.562 ms (8.32%). Greedy and sampled rank digests and MTP work matched, including forced NCCL and alignment refusal. An independent audit matched 921,600 entries each for reduced BF16, residual BF16, and E4M3, plus 7,200 FP32 scales. See `artifacts/tp2-resume/w1-reduce-norm/production/tracked/target.5ns0DEvu/` and `artifacts/tp2-resume/reduce-norm-review/review.md`.

### Forty-row verification fusion

On B200, both ranks need eager 40-row verification, external packed FP8 buffers, and Tensor Core packing. One-shot MNNVL requires aligned operands, 40 resident blocks per GPU, and a mapped rejection word. A clean decline uses ordinary allreduce and local preparation for the rest of the forward. [Shape gate](../../src/runtime/execution/decoder_forward_tp.cpp#L138)<!-- prefix: "if (primary." --> · [Transport](../../src/runtime/backend/tensor_parallel.cpp#L517)<!-- prefix: "bool TensorP" --> · [Rejection word](../../src/runtime/backend/mnnvl_ar.cu#L287)<!-- prefix: "bool MnnvlAl" -->

If rank 1 rejects submission synchronously after rank 0 enqueues, the host publishes the launch epoch. Healthy rank-0 blocks poll it and exit. The host attempts to fence both streams, fails the group, and propagates the error. Qualification used a host model and source inspection. [Rank-0 poll](../../src/runtime/backend/mnnvl_ar.cu#L125)<!-- prefix: "__device__ b" --> · [Host rejection](../../src/runtime/backend/mnnvl_ar.cu#L770)<!-- prefix: "PairAtomic(*" -->

Twelve paired eight-lane W3 steps saved a paired median 0.541486 ms (arm medians 11.910723 and 11.372932 ms). Across 202 rounds that projects 109.4 ms. W3 makespan awaits measurement. The tracked build matched outputs and rank digests across seven W3 traversals, with 127 fused and one ordinary one-shot reduction per rank. Nine W1 graph replays per rank matched and bypassed M40. See `artifacts/tp2-resume/w3-m40-reduce-norm/` (`trial.83cPe0rq/comparison.json`, `tracked-e0cd4/gpu.IAch6JZ5/result.log`).

## Acceptance target

A passing campaign must satisfy every condition below.

- **Devices.** Run both engines alone on the same two B200 devices in the same rank order. Record both UUIDs, the rank assignment, interconnect, and distributed library.
- **Workloads.** Reuse the frozen W1–W5 fixture bytes. Offer 2, 2, 8, 16, and 8 requests total on W1–W5. This is one, one, four, eight, and four per GPU. Eight slots per GPU give sixteen slots total. W4 fills them exactly. W4 uses the first sixteen archived W5 lanes. W1 and W2 need one new distinct lane each. W3 and W5 need four more each. Freeze every new lane as distinct deterministic bytes in a separate additive TP2 corpus. Retain the TP1 lanes and bytes. Duplicate no request.
- **Native arm.** Run the native server with `--tp 2`, the default direct GPU LMCache path, and the LMCache daemon enabled. Verify the reported `tp_size` is two. [Server information](../native-model-info.md#server-information) Confirm effective direct cache activity on each rank, so a reported `tp_size` cannot hide a bypass. Record the adopted topology and the effective daemon configuration.
- **Cache.** Start one LMCache daemon per rank, each bound to its own GPU and rank namespace. [Ranked transport](lmcache.md#ranked-tp2-transport) Draft records live only in the rank 0 daemon. Drain and reset both rank daemons before cold timing.
- **vLLM arm.** Run the pinned checkout with `--tensor-parallel-size 2`. [TP option](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/engine/arg_utils.py#L1219)<!-- prefix: "\"--tensor-pa" --> Require `kv_transfer=false`, `kv_offloading_size=null`, and zero CPU weight offload in both workers' receipts. [Receipt policy](../performance.md#scheduler-and-reporting-policy)
- **Gate.** Submit each row's requests simultaneously through the original submission path. Run one warmup and one measured cohort per row and engine, and use the measured verdict. Record clocks and output lengths. Treat output content as reference only; no warmup-measured token identity is required. Require native TP2 makespan ≤1.10× pinned vLLM TP2 on each W1–W5 row independently. A passing row covers no other row. Apply the [scheduler policy](../performance.md#scheduler-and-reporting-policy) unchanged.
