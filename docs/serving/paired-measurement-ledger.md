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

Live vLLM K0 and K4 matched for 28 output tokens. At token 29, K0 selected ID `524` with a 0.125 logprob lead over ID `321`. K4 reported equal logprobs for those IDs and selected `321`. W1 failed the exact-ID requirement.

The v4 campaign measured nine W1 cohorts per arm. The median makespan was 1.930878220 s for native and 0.638064157 s for vLLM. The native median was 3.026 times the vLLM median, so W1 also failed the timing requirement. The timing results are descriptive only. The K4 campaign stopped at W1.

The receipts are on the benchmark host. The paths below are relative to `artifacts/`.

| Receipt | SHA-256 |
| --- | --- |
| `session-1122-mtp4-bench/campaigns/release-mtp4-v4/W1/comparison.json` | `50009ce7472c62b90b8c93b9343950fe14ad86395f28997a6cb5a85c33b1d616` |
| `session-1122-mtp4-bench/campaigns/release-mtp4-v3/W1/native/w1-qualification.json` | `bea83bf50bb1892c2a405c4277eae0caf7c9ef9a7304e9d0789db32d4972835a` |
| `session-1122-mtp4-bench/campaigns/release-mtp4-v3/W1/vllm/w1-qualification.json` | `b7ae0ecaf3147a799239104699bba093741f831835ba346ae180e690d13bd5b4` |
| `session-1122-default-live/cli-summary.json` | `0538c989b9b4213884ed2fd589fd94b97ac94263ee1b28ba6af92a0aab05694a` |
| `session-1122-default-live/http-omitted-depth/qualification.json` | `c550d3ee63ad246b19d6f74e4ba4e67f98bac88615b0158ea158870905896832` |
