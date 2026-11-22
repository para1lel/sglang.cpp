# November 22, 2026 paired W1–W6 measurement ledger

This ledger records the builds and results behind the [matched W1–W6 comparison](../performance-comparison.md#matched-w1w6-at-8192-tokens). Each pair used the same physical B200 and checkpoint, with vLLM first. Both engines used a limit of 8,192 scheduled tokens per step.

## Measurement method

The client measured cohort makespan from the earliest submit to the last valid `[DONE]`. Concurrent clients connected all lanes before releasing a shared barrier. The protocol capped the span between lane submissions at 50 ms. The clients sent all request bodies before the first valid `[DONE]`.

Each arm ran three warmups and nine measured cohorts for each of W1 and W2. Each arm ran two warmups and five measured cohorts for each of W3–W6. All measured requests completed successfully. The table reports the median and observed minimum–maximum across measured cohorts. Latency percentiles pool requests from measured cohorts within each workload and arm. The client sorts the samples and linearly interpolates at zero-based rank `p × (n − 1)` for percentile fraction `p`. [Percentile calculation](../../benchmarks/streaming/client.py#L238)<!-- prefix: "def percenti" --> · [Paired protocol](../../benchmarks/paired/README.md#observe-and-compare)

## Measured results

The sole paired performance gate was strictly lower native median makespan. Output rate, time to first token (TTFT), time per output token (TPOT), and end-to-end latency (E2E) are diagnostics. The request count is measured completions over attempts per arm.

| Workload | Arm | Requests | Makespan median (range), s | Output rate median (range), IDs/s | TTFT p50 / p95, s | TPOT p50 / p95, ms/ID | E2E p50 / p95, s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | Native | 9/9 | 1.026 (0.985–1.068) | 124.713 (119.904–129.886) | 0.031 / 0.070 | 7.703 / 8.037 | 1.026 / 1.061 |
| W1 | vLLM | 9/9 | 1.237 (1.236–1.237) | 103.490 (103.462–103.550) | 0.020 / 0.020 | 9.582 / 9.583 | 1.237 / 1.237 |
| W2 | Native | 9/9 | 4.132 (4.121–4.244) | 123.898 (120.642–124.240) | 0.104 / 0.127 | 7.880 / 8.032 | 4.132 / 4.215 |
| W2 | vLLM | 9/9 | 5.002 (4.998–5.003) | 102.366 (102.333–102.433) | 0.081 / 0.081 | 9.630 / 9.632 | 5.002 / 5.003 |
| W3 | Native | 20/20 | 10.536 (10.465–10.735) | 388.769 (381.571–391.383) | 1.803 / 1.962 | 8.520 / 8.646 | 10.535 / 10.734 |
| W3 | vLLM | 20/20 | 12.379 (12.378–12.383) | 330.871 (330.783–330.902) | 1.205 / 1.393 | 10.910 / 11.727 | 12.366 / 12.382 |
| W4 | Native | 80/80 | 41.147 (40.999–41.261) | 398.180 (397.085–399.619) | 28.978 / 29.264 | 11.778 / 12.736 | 41.033 / 41.127 |
| W4 | vLLM | 80/80 | 46.368 (46.365–46.386) | 353.347 (353.210–353.367) | 17.239 / 31.914 | 28.171 / 42.637 | 46.066 / 46.365 |
| W5 | Native | 160/160 | 125.918 (125.672–126.069) | 1040.931 (1039.685–1042.973) | 57.577 / 58.699 | 16.616 / 17.226 | 125.794 / 125.948 |
| W5 | vLLM | 160/160 | 154.131 (154.087–154.143) | 850.391 (850.326–850.637) | 33.507 / 62.863 | 29.250 / 36.042 | 153.290 / 154.065 |
| W6 | Native | 20/20 | 68.811 (66.793–69.098) | 59.526 (59.278–61.324) | 54.746 / 54.778 | 11.797 / 13.894 | 66.804 / 68.989 |
| W6 | vLLM | 20/20 | 87.358 (87.325–87.368) | 46.887 (46.882–46.905) | 45.761 / 72.710 | 40.387 / 67.087 | 87.076 / 87.361 |

For W3–W6, every lane produced an ID before the first stream ended in every cohort. Three measured native W6 cohorts had a gap of about 2.3 s before one lane's last ID. The reported range and median include all five measured cohorts.

The paired replay checked output counts, usage, `length` finish, `[DONE]`, and repeat IDs within each arm. Including warmups, it covered 112 streams per W4 arm and 224 per W5 arm. The separate W6 replay checked 28 streams and 28,672 output IDs per arm. Native W6 IDs matched fresh vLLM and the four frozen W6 continuations.

## Measured builds

| Workloads | Native source | Native binary SHA-256 |
| --- | --- | --- |
| W1–W3 | `919885edc1b70937915ca6894a40f5f98ec3f202` | `03cf404e719d03c60dec21e100212c4af75e68ba8fe97e6632322820d4d3abf8` |
| W4–W5 | Base `e28e8983aad6045aef509719a380b8b91c5b7e70` plus the three W4–W5 patches below | `0871bbd2a6035c63220634ef58824b8d52b45cba31b9f893096ebd47920cbe97` |
| W6 | XMake Release build from base `250b9cb302d8b7b3c790baf21d0042055ae1100c` plus `source.patch`, source tree `5559ddf2cf9982b33c0ac7d34e457061e8bc71a5` | `75a45ec6cd28b95006cd0676e62933619b0fc1fba0efbc256fa0682555d62d79` |

The vLLM revision was `81a8cb063e6247fe254d7caf7e98a6691d71022a` for every workload. The retained wheel manifests record these SHA-256 values.

| Wheel | SHA-256 |
| --- | --- |
| vLLM `0.34.0rc1.dev787+g81a8cb063` | `25358fa092260117176d78f4efeabfd06a0e49731d29bbcffd717dfedc498ad9` |
| PyTorch `2.14.0+cu132` | `0b059371a41b7c868589b00ff60a9f72118bc9b3c0bf12e820c06eb2a3aab55b` |

| Workloads | Patch path under `artifacts/` | SHA-256 |
| --- | --- | --- |
| W4–W5 | `takeover-w4-scheduler/tail256/generalized-scheduler.patch` | `064b7cc25e37f3fd7e01aa334a5dff3cd29587a2180e4727efa15706ed648054` |
| W4–W5 | `takeover-w4-combined/native-tile-pair.patch` | `63f8cbdc350abfd90fdc220489c611ea37f97270766f0800ae531b64b20487ca` |
| W4–W5 | `takeover-w4-mailbox/mailbox.patch` | `f21cc59b085417c950022810b8431a218810aefbb7d0f758fbc6369bcbe2a711` |
| W6 formal | `takeover-baseline/paired-w6.0f6AV2Ck/native-formal/source.patch` | `9ab7e7c8d6090e1003e244f64e94f7969b07c9f7c98e06abee6410d243d3ba50` |
| W6 short guard | `takeover-w6-gqa/production-prefill.patch` | `29157799c88e4af72e57b3c15d3077d769d2e116b4810560400b71cfd6eb81bd` |

The two W6 patch files contain the same source change. The formal patch adds Git diff headers and a hunk context label. The formal patch bytes match the CUDA diff from the measured base to integration commit `d7e5af322efe6373b28aeb1b3a8afc078ea0429b`. That commit's tree is `acfee63f43612edab9fbf3eebfe3aca9f6a136e2`. The measured W6 tree ID comes from the formal run evidence.

## Evidence and recovery

These receipt paths resolve under `artifacts/takeover-baseline/` on the benchmark host. The tracked protocol can produce a fresh measurement. Recovering the original receipt bytes from another checkout requires a copy of the host artifacts with the matching hashes. A separate approved archive for those receipt bytes remains unverified.

| Workloads | Comparison receipt | SHA-256 |
| --- | --- | --- |
| W1–W2 | `paired-w1w2.FrMn3y/native/W1-W2/comparison.json` | `02fec92d80ae5156c166bbde6b2d3ffd5c89cdb810b476423ab94afa8e075278` |
| W3 | `paired-w3.XFQMl5fD/native/W3/comparison.json` | `bbf147f280acfe7a54c68592e4156cae894be9ab2f1f2a276af6129ca889cb97` |
| W4 | `paired-w4.XUPFw89u/native/W4/comparison.json` | `b3c68f9d2f313946985f543c3b0f8fe0ecce7f09265980598017e17bc589ed5c` |
| W5 | `paired-w5.oSXO12wz/native/W5/comparison.json` | `bfbb1b249fa4edd58313226eb0e9d6d901794bc3ca5fcd4027276509b78da1ff` |
| W6 | `paired-w6.0f6AV2Ck/native-formal/native/W6/comparison.json` | `5be85492a3e4bd72c65e996120ea5a7d877df656d175b9f98cd495970ce4f290` |

The `measurement/summary.json` files supply the rates and latency distributions. Within each run directory, the native summary is under `native/<group>/` and the vLLM summary is under `vllm/vllm-<group>/<group>/`. Use `W1-W2` for the shared W1 and W2 group. The W4 vLLM reference and summaries are under `vllm-w4.t17EISjA/`. For W6, use `paired-w6.0f6AV2Ck/native-formal/` for native and `paired-w6.0f6AV2Ck/restart/` for vLLM.

| Workload | Additional host evidence under `artifacts/takeover-baseline/` |
| --- | --- |
| W4 | `paired-w4.XUPFw89u/diagnostics.json` and `paired-w4.XUPFw89u/evidence.sha256` |
| W5 | `paired-w5.oSXO12wz/reference.json` and `paired-w5.oSXO12wz/independent-test/replay.json` |
| W6 | `paired-w6.0f6AV2Ck/native-formal/replay-result.json` and `paired-w6.0f6AV2Ck/native-formal/evidence.sha256` |

## W1 K4 qualification

The November 22, 2026 W1 run checked native release `dce521eaecabfca707b6a3e6d72547ac4699b3af` and the pinned vLLM revision above. Native default K4 matched all 128 IDs from explicit K0 with exact prefill. Separate live checks confirmed the same IDs for default K4, explicit K4, and exact K0 in the generator and HTTP server. The historical native tensor-core K0 continuation first differed from the exact K0 control at output token 29.

Live vLLM K0 and K4 matched for 28 output tokens. At token 29, K0 selected ID `524` with a 0.125 logprob lead over ID `321`. K4 reported equal logprobs for those IDs and selected `321`. This mismatch failed the v4 campaign's then-required vLLM K0/K4 ID check.

The [maintained matched K4 protocol](../../benchmarks/paired/README.md#matched-k4-timing-protocol) supersedes that v4 ID check. It requires strict native exact-K0 parity and valid streams from both arms. vLLM serves as a timing reference, and valid K4 streams qualify regardless of their output ID values. The archived v4 receipts retain their original checks and results.

The v4 campaign measured nine W1 cohorts per arm. The median makespan was 1.930878220 s for native and 0.638064157 s for vLLM. The native median was 3.026 times the vLLM median, so W1 missed the strict timing gate on its own. The campaign stopped at W1.

The receipts are on the benchmark host. The paths below are relative to `artifacts/`.

| Receipt | SHA-256 |
| --- | --- |
| `session-1122-mtp4-bench/campaigns/release-mtp4-v4/W1/comparison.json` | `50009ce7472c62b90b8c93b9343950fe14ad86395f28997a6cb5a85c33b1d616` |
| `session-1122-mtp4-bench/campaigns/release-mtp4-v3/W1/native/w1-qualification.json` | `bea83bf50bb1892c2a405c4277eae0caf7c9ef9a7304e9d0789db32d4972835a` |
| `session-1122-mtp4-bench/campaigns/release-mtp4-v3/W1/vllm/w1-qualification.json` | `b7ae0ecaf3147a799239104699bba093741f831835ba346ae180e690d13bd5b4` |
| `session-1122-default-live/cli-summary.json` | `0538c989b9b4213884ed2fd589fd94b97ac94263ee1b28ba6af92a0aab05694a` |
| `session-1122-default-live/http-omitted-depth/qualification.json` | `c550d3ee63ad246b19d6f74e4ba4e67f98bac88615b0158ea158870905896832` |

### Combined native W1 follow-up

On B200 GPU7, four fresh Release servers ran in base, combined, combined, base order. Each server ran three warmups and three timed requests. All 24 requests, including warmups, matched the frozen sequence of 128 output IDs. Both arms used exact FP8 prefill, default K4, CUDA graphs, 32 slots, and an 8,192-token step budget.

| Native arm | Timed requests | E2E median (range), ms | TTFT median, ms | TPOT median, ms/ID |
| --- | ---: | ---: | ---: | ---: |
| Base | 6 | 1,852.73 (1,842.65–1,975.96) | 382.38 | 11.58 |
| Combined | 6 | 1,758.35 (1,754.28–1,784.02) | 382.37 | 10.83 |

The combined E2E median was 5.09% lower. The client measures E2E from submission to a valid SSE `[DONE]`. This focused native-only result sits outside the formal paired gate. The archived 638.06 ms vLLM median came from GPU2 and serves as context only.

The measured source used base `d4388852209908d988907deffbfed9192b2f25a1` plus the two patches below. The five changed source files match the shipped attention and MLP commits `2b698a6518d2585185b92d559f86d7f31eac84af` and `dc9d7da1afd1dbbadab273c160cc95a9cea6dc99`. Independent operator checks are in `artifacts/mtp4-draft-attention-test/qualification-report.md` and `artifacts/mtp4-op-test/DISPOSITION.md`.

| Evidence path under `artifacts/mtp4-combined-test/` | SHA-256 |
| --- | --- |
| `draft.patch` | `0468e504fd721c10c46be452a952dd6970e0afd950b539f491f03469eaf05819` |
| `mlp.patch` | `22df57ae2fd0b8e8e4169d4771ddbcf83673e8a3712328b6b353fa403c4fd72d` |
| `base/build/bin/sglang-server` | `36c2f8bb31921848fb5896e3694f3c257655123c29b16d557aa44bff3ba86b42` |
| `combined/build/bin/sglang-server` | `cbd5aeee399a226018e36f1cfe68c6f96d4456aeb859e76810f85017b8492da3` |
| `run.LBHOaZYi/DISPOSITION.md` | `33fda44aeacce93a28abfa3a0555d0f2ffcb166ab43c7e12d0db9d7dfb56e3c7` |
| `run.LBHOaZYi/summary.json` | `786f6d24b2238b7310d7bd80239329c08fb7d74325bd51c84c0ec16add883c15` |

### Native W1 K4 pilot with exact prefill tiling and verifier fusion

On November 22, 2026, both Release builds ran on B200 GPU6 (`GPU-d4d87d81-ea84-5dd0-7167-3e1c48fde29b`). The candidate includes pointwise score, wide exact FP8 prefill tiles, and five-row verifier fusion. Both arms used exact FP8 prefill, default K4 and graphs, 32 maximum active requests, and an 8,192-token step budget. Both disabled prefix reuse.

Four fresh K4 servers ran in baseline, candidate, candidate, baseline order. Each server ran three warmups and three measured requests, with one request in flight. The two explicit K0 controls established the reference for each arm. All 24 unprofiled K4 streams matched their own arm's K0 control and all 128 frozen native output IDs. All 26 streams returned HTTP 200 and passed the ordinary SSE checks for prompt echo, single-ID deltas, and `length` finish. Each reported 128 prompt and 128 completion tokens, then sent one `[DONE]` and drained without trailing data.

The client measured E2E from submission to `[DONE]` and TTFT from submission to the first output ID. TPOT divides the first-to-last-ID span by 127. Percentiles use linear interpolation as described above. Each arm contributed six measured requests.

| Metric | Baseline p50 / p95 | Candidate p50 / p95 | Baseline range | Candidate range |
| --- | ---: | ---: | ---: | ---: |
| E2E, ms | 1,805.96 / 1,915.52 | 1,528.47 / 3,386.75 | 1,780.07–1,945.71 | 1,502.54–3,983.29 |
| TTFT, ms | 404.80 / 492.29 | 293.79 / 352.23 | 388.83–518.07 | 283.42–362.29 |
| TPOT, ms/ID | 11.063 / 11.214 | 9.768 / 23.969 | 10.952–11.241 | 9.385–28.511 |

The candidate E2E median fell 15.37%, and p95 increased. One measured candidate stream took 3,983.29 ms, including a 2,467.83 ms gap before the 127th output ID. It completed the exact sequence and remains in all measured summaries. The cause of the gap remains unknown.

Separate CUDA captures used one profiled request per arm after three warmups. Both captured requests matched the frozen IDs and completed 52 decode rounds, with 205 proposals, 75 accepted drafts, and 257 target verification rows. GPU-active time counts overlapping kernel, copy, and memset intervals once within each phase.

| Phase | Baseline GPU-active time, ms | Candidate GPU-active time, ms |
| --- | ---: | ---: |
| Prefill through winner readback | 379.21 | 260.75 |
| Draft waves | 124.42 | 124.40 |
| Target verification | 1,150.10 | 964.09 |
| Commit and repair | 34.23 | 34.27 |

The archived v4 vLLM W1 median of 638.064 ms came from B200 GPU2 and serves only as historical timing context. The candidate median was 890.41 ms higher, or 2.395 times that value. The [formal matched K4 gate](../../benchmarks/paired/README.md#matched-k4-timing-protocol) remains open.

| Arm | Measured source | Measured tree | Release server SHA-256 |
| --- | --- | --- | --- |
| Baseline | `5b4c370e730afb0b67394a8ed62ca55696aaf305` | `c01b246cee463111b65e299ae73cdc73aebef3d1` | `d9ee4550bb69c14fbd23b898faa429857001f7f4f27c3a32100a8dc1f06989f4` |
| Candidate | `0f0e77413bc06b4daa8ea0e162d69e4c81b9d900` plus the eight patches in `inputs/patches.sha256` | `77d4778ab1bdb3a3524fc043c8a87083d23d9192` | `fa6f515400d375cbf38374066c0806da862b06c4fc00ad742f833a9fecb93f52` |

The original receipts are under `artifacts/continue-combined-test/` on the benchmark host. Retain a copy with the matching manifest to recover the original bytes from another checkout.

| Evidence path | SHA-256 |
| --- | --- |
| `REPORT.md` | `e14894a0676744158d4976498fe1c8bf45e0da9059056bc1f06159f0cf5b081d` |
| `inputs/patches.sha256` | `ab649303dcd5ffcffb21965b23c41109c75474de6d3801c9e7179d2f1f204f84` |
| `run.zsDDGo49/summary.json` | `edc0a11ff5eb46e8b3efa2c557460530fe5e762fb2ed8cedf6508869ada4cac0` |
| `SHA256SUMS` | `efeee1f19bbe43e85a4e1432137e4a8b971bf2f7e7617ed962ffee274e1dc27e` |

### Native W1 K4 component trials

The HTTP trials used the Qwen3.8-27B-FP8 checkpoint, exact FP8 prefill,
K4, CUDA graphs, and no prefix reuse. Each trial used one B200 for both arms.
The W1 request used 128 prompt IDs and generated 128 output IDs. Each trial
checked the frozen native output sequence. Each ran four fresh servers in
control, candidate, candidate, control order.

| Change | Warmups / timed requests per arm | Median TTFT, ms, control → candidate | Median TPOT, ms/ID, control → candidate | Median E2E, ms, control → candidate |
| --- | ---: | ---: | ---: | ---: |
| Draft KV-only repair | 2 / 8 | 277.818 → 264.403 | 9.483 → 9.140 | 1,492.579 → 1,436.429 |
| Five-row verifier MLP | 6 / 6 | 264.261 → 264.127 | 9.535 → 8.672 | 1,476.946 → 1,365.444 |
| Wide 4×8 exact FP8 prefill | 4 / 10 | 290.189 → 234.490 | 9.338 → 9.358 | 1,479.418 → 1,427.579 |

**Draft KV-only repair.** A CUDA-event trial on a different B200 measured
all 75 repairs per request. Across eight timed requests per arm, the median
total repair time fell from 17.083 to 4.526 ms. The W1 state capture
matched 593,600 bytes across 205 scored appends and 75 repair appends.
The HTTP E2E median fell 56.150 ms. Variation across server starts and
the change in TTFT limit attribution of that difference to repair.
The independent checkpoint probe matched 2,351,040 state bytes across six
seed, repair, and continuation cases.

**Five-row verifier MLP.** The separate CUDA captures used one profiled
request per arm after three warmups. Verifier GPU-active time fell from
965.746 to 895.643 ms. The summed gate/up kernel time fell from 247.860
to 211.001 ms, and down kernel time fell from 169.869 to 129.793 ms.
The 51 five-row rounds used eager execution. The final two-row round
replayed a graph. The E2E median fell 7.55% in this sample of six timed
requests per arm. This trial predates draft KV-only repair. The selected
MLP layout reproduced the measured verifier instructions and resource
counts in an offline SM100 build.

**Wide 4×8 exact FP8 prefill.** Both orderings showed lower median TTFT,
with a pooled decrease of 19.2%. One ordering showed higher median E2E.
The candidate's 2,161.410 ms request retains its place in the sample.
In one separate profiled request per arm, 368 of 402 exact projection
launches switched to 4×8. Summed projection kernel time fell from
242.127 to 178.153 ms. Active prefill GPU time fell from 260.672 to
196.662 ms. The held-out checkpoint probes matched 1,599,842,260 state
bytes per arm. The timing trial includes draft KV-only repair and
predates the five-row MLP layout.

Each comparison uses its own measured control. The combined stack has
no timing in these trials. The [formal matched K4 gate](../../benchmarks/paired/README.md#matched-k4-timing-protocol)
for exact FP8 remains open.

| Trial | Measured control | Measured candidate |
| --- | --- | --- |
| Draft KV-only repair | `e44330fce30bb913ebb53e73d57b56822185651f` plus verifier fusion | Same control plus KV-only repair |
| Five-row verifier MLP | Tree `549a14d6b022ecf3491c149063f1d8870dfeb96c` | Tree `8c82c3784cab1a0cc3e545811c6f7f249312e9da` |
| Wide 4×8 exact FP8 prefill | Tree `a0fd6dd13a6468571a18a959556da8dfc6c5fbb7` | Tree `0667a21ed646ca745f247897bf00158ff7ab1e5f` |

The receipt paths are relative to `artifacts/` on the benchmark host.
For the repair trial, `continue-draft-audit/source-provenance.txt` identifies
the verifier fusion patch and `continue-draft-audit/handoff.md` identifies
the repair patch. The reports identify the binaries and qualification
probes for each trial.

| Receipt | SHA-256 |
| --- | --- |
| `continue-draft-audit/summary.json` | `08ced9565e4b06e453538e8cac0fe2012ea2157f255729d01341ee87ef310333` |
| `continue-exact-mlp-test/run.WvqYCkH6/summary.json` | `c95b57cc7ea9dfd90865a5ac75d3cd43aa7b31d7c15dd39eaa4d1b32034ea58e` |
| `continue-exact-mlp-test/profile.NLGh9LQn/profile-baseline/analysis-summary.json` | `43696966fd9d032adfb40593096fe3440839b28fafe5d29f187211ede63e2640` |
| `continue-exact-mlp-test/profile.NLGh9LQn/profile-candidate/analysis-summary.json` | `fc96d99558eeadb53d7cbf14310557aa98d045e74500d395e5ac36a7e43a799f` |
| `continue-wide-next-test/summary.json` | `e599ed16d86fed9149a471443049eba607542a77c018f18d0e836d0b483d6053` |

### Cumulative native W1 K4 pilot

The build combines draft KV-only repair, five-row verifier MLP, wide 4×8
exact FP8 prefill, and Ollama chat compatibility. On B200 GPU7, four
servers ran three warmups and five timed W1 requests each in ABBA order.
All 34 streams, including two K0 controls, matched the 128 frozen output
IDs, usage, and length finish.

| HTTP metric, ms | Control | Cumulative |
| --- | ---: | ---: |
| Median E2E | 1,476.372 | 1,283.083 |
| E2E range | 1,427.502–2,163.835 | 1,278.761–1,306.611 |
| Median TTFT | 264.036 | 200.004 |
| Median TPOT, ms/ID | 9.346 | 8.520 |

Median E2E fell 13.09% overall and fell in both orderings. The control
range includes a 716.840 ms gap before output ID 127 of unknown cause.
Separate CUDA captures, one request per arm, measured active prefill at
260.623 → 196.649 ms and active verify at 963.610 → 894.408 ms. Both
recorded 52 rounds, 205 drafted IDs, 75 accepted IDs, and 257 verify rows.

The control was commit `267d1ded2adc7c26f0d467e16673859802b64c5c`. The
cumulative build used tree `0a86c34dc38ce3764dc6bc543ff6f13caf776a18`.
The archived vLLM median of 638.064 ms used a different B200. The
[formal matched K4 gate](../../benchmarks/paired/README.md#matched-k4-timing-protocol)
remains open.

| Receipt under `artifacts/continue-cumulative-w1/` | SHA-256 |
| --- | --- |
| `REPORT.md` | `5c849ff2ddfeaa7374390164152eec817e768d6c6d1b06786e829086d330f2e3` |
| `run.wtALVRtm/summary.json` | `64b8999a4173eff1c49045c7cbc89224f6be563f33c67c144846114f46ee1e68` |

### Matched W1 K4 with ordered Tensor Core and INT8 draft

On November 22, 2026 UTC, both timed arms ran on B200 GPU3
(`GPU-ceeff2f7-3f39-50b4-8d29-576159781b9d`), with vLLM first.
The Release native source was `99c523d95c47788f6c872c6c5428c28418d395eb`.
The vLLM arm used the pinned revision above. Both used the same checkpoint,
32 request slots, and an 8,192-token scheduling cap. The
[opt-in profile](../../benchmarks/paired/README.md#ordered-tensor-core-with-int8-draft)
uses `tensor-core-mtp-target` and an INT8 draft head at explicit K4.
Exact FP8 remains the server default. This campaign has a separate gate from
the earlier exact K4 and K0 comparisons.

| Workload | Warmups / measured per arm | Native median (range), ms | vLLM median (range), ms | Strict native median gate |
| --- | ---: | ---: | ---: | --- |
| W1 | 3 / 9 | 649.368673 (614.321563–1,275.759181) | 726.240970 (725.871508–727.201134) | Pass, 76.872297 ms lower (10.585%) |

Independent replay checked wire format, length, usage, and drain for all 27
streams, including warmups and controls. Every native K4 stream matched the
128 output IDs from its own policy's K0 control. The separate native trace
accepted 77 of 198 proposals and reached depth four in 49 of 50 rounds. The
vLLM counters recorded 2,544 new draft tokens and 636 fourth-position
verifications. Four measured native streams took longer than the slowest
measured vLLM stream.

The ordered K0 and exact K0 controls differed at 62 output positions. The
first difference was token 67, with ordered ID `9976` and exact ID `19720`.
This diagnostic sits outside the timing gate. vLLM serves as a timing
comparator. The stream and parity checks cover this W1 fixture. Independent
model quality is a separate question, and the held-out target study predates
the INT8 draft head.

The raw receipts are under
`artifacts/continue-policy-campaign-test/campaign-published-99c523d-gpu3-retry-1795372885/W1/`
on the benchmark host. `comparison.json` has SHA-256
`bb09972e3eba03e29f715fd262c14b15ce2b90c8701939dcb62a3a53c2c1fd8b`.
The arm directories hold raw SSE under `measurement/`. See
`identity.json`, `independent-live-audit.json`, `independent-sse-replay.json`,
`exact-k0-compatibility.json`, `native-k4-trace/mtp-activity.json`, and
`vllm-k4/mtp-activity.json` for the configuration and correctness receipts.

### Matched W2 K4 with ordered Tensor Core and INT8 draft

W2 used the same campaign identity and GPU, with native first. Each arm ran
three warmups and nine measured streams of the 2,048-input, 512-output fixture.
The native median makespan was 3,743.794408 ms (3,731.274364–3,787.751962 ms).
The pinned vLLM median was 1,740.743117 ms (1,739.599098–1,745.821485 ms).
Native missed the strict median gate by 2,003.051291 ms. The campaign stopped
at W2, so W3–W6 did not run.

Independent SSE replay measured median TTFT of 135.070 ms for native and
76.089 ms for vLLM. Median time from first output to `[DONE]` was
3,609.546 ms and 1,665.127 ms, respectively. Median TPOT was 7.064 and
3.258 ms/ID, respectively.

All 26 streams passed the wire, length, usage, and drain checks. Every native
K4 stream matched all 512 IDs from its own policy's K0 control. The exact K0
diagnostic differed at 489 of 512 positions, first at zero-based index 13
(ordered ID `314`, exact ID `440`). The vLLM timing arm recorded 1,656
fourth-position verifications.

The raw receipts are in the same campaign's `W2/` directory. The SHA-256 of
`comparison.json` is
`ea445344d76449dca202e78642ad70044f5603824cf91becbf87ab251ae07015`.
See `independent-sse-replay.json`, `exact-k0-compatibility.json`, and
`vllm-k4/mtp-activity.json`.

See the [matched split-attention K4 campaign](paired-split-attention-results.md)
for the subsequent W1 and W2 comparison.
