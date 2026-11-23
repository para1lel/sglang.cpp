# Matched W1 and W2 K4 with native server defaults

Native Release `94e8bdca73ce2017e501b6b8b6abda9156ce0c89` passed the strict
median makespan gate on W1 and W2 on November 23, 2026 UTC. Formal W3–W6
remain unmeasured.

The [matched K4 timing protocol](../../benchmarks/paired/README.md#matched-k4-timing-protocol)
uses three warmups and nine timed single-request cohorts per arm for W1 and W2.
Client makespan covers submission through `[DONE]`. The gate requires the
native median to be strictly lower than the pinned vLLM median.

| Workload (input/output IDs) | Native median, ms | Pinned vLLM median, ms | Timing gate |
| --- | ---: | ---: | --- |
| W1 (128/128) | 613.630 | 728.106 | Pass |
| W2 (2,048/512) | 1,405.331 | 1,739.562 | Pass |

## Measured builds and launch

| Item | Identity |
| --- | --- |
| Native source tree | `ee39752ba62271c5d77c935ab5bfcca1c516af15` |
| Source archive SHA-256 | `1cbec5e37360745c99c38d33d42f4111ecfdb79674dbb8922ed49dde14162c4d` |
| Release server executable SHA-256 | `45d313e009ea0f7e7f90fecc57dbf7edb876a3aa73a037900d80e3876a5f7ceb` |
| `benchmarks` subtree | `d09757c21332ab018cb436673ed6363a28a3b5b0` |
| Protocol manifest SHA-256 | `5b63731b64c38a913de468e05560b81d860f5ce52425a618e7c0e7fb743be9d9` |
| Pinned vLLM | `81a8cb063e6247fe254d7caf7e98a6691d71022a` |
| Checkpoint manifest SHA-256 | `b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303` |
| Campaign identity SHA-256 | `fd2c5e88e3dc0b9ca45be78cf4daef9996944528daaacac4a0346618249f05e1` |

Both arms used `/data0/shared/Qwen3.8-27B-FP8` and B200
`GPU-ceeff2f7-3f39-50b4-8d29-576159781b9d`. Server CPUs were 24–33 and client
CPUs were 34–35, with local NUMA allocation.

The native timed command was `sglang-server --port 17424`. The
[server defaults](../../apps/sglang-server.cpp#L28)<!-- prefix: "struct Argum" -->
set MTP4, 32 request slots, an 8,192-token scheduling cap, CUDA graphs on,
and prefix caching off. The saved server settings show the INT8 draft head,
online draft GQA, and tensor-core FP8 prefill. The pinned vLLM arm selected
MTP4 explicitly.

## Controls and raw evidence

Each timed arm emitted one repeatable sequence per row across all three
warmups and nine timed cohorts. The separate W1 native trace verified
depth four and matched all 128 native IDs in every timed W1 cohort.

Cross-engine output IDs differed in every paired cohort. W1 differed at
100 of 128 positions, first at zero-based index 28 (native 524, vLLM 321).
W2 differed at 497 of 512 positions, first at index 0 (native 2,981, vLLM 314).
The vLLM arm supplies the timing reference. Exact output parity is outside the
timing gate.

The benchmark host retains SSE captures, parity reports, and launch receipts
under `artifacts/refactor-baseline/architecture-http/formal-commit-94e8bdc.oXQknTsS/`.
Use the protocol's replay procedure with `campaign/W1` or `campaign/W2` as
both the fixture and capture directory.

| Raw `comparison.json` | Finished (UTC) | SHA-256 |
| --- | --- | --- |
| W1 | 04:21:25 | `7331898daebd042a2ffa670322c529b6ecae59792351ef74f8a46ef793f39e4a` |
| W2 | 04:26:34 | `4a455118f4a99b3a0477699ca5f552fbabc38f33abe60a88aefdc1af4544b368` |

The [prior published-runtime comparison](https://github.com/para1lel/sglang.cpp/blob/94e8bdca73ce2017e501b6b8b6abda9156ce0c89/docs/serving/paired-native-defaults-results.md)
retains the earlier W1 and W2 results.
