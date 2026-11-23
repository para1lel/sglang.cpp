# Matched W1 and W2 K4 with native server defaults

Native Release `f46973d433fcf19cee3d80b6e2d0953f5a616324` passed the strict
median makespan gate on W1 and W2 on November 23, 2026 UTC. Formal W3–W6
remain unmeasured.

The [matched K4 timing protocol](../../benchmarks/paired/README.md#matched-k4-timing-protocol)
uses three warmups and nine timed single-request cohorts per arm for W1 and W2.
Client makespan covers submission through `[DONE]`. The gate requires the
native median to be strictly lower than the pinned vLLM median.

| Workload (input/output IDs) | Native median, ms | Pinned vLLM median, ms | Timing gate |
| --- | ---: | ---: | --- |
| W1 (128/128) | 613.831 | 649.219 | Pass |
| W2 (2,048/512) | 1,420.752 | 1,740.642 | Pass |

## Measured builds and launch

| Item | Identity |
| --- | --- |
| Native source tree | `7959c1e9643b8de0f3f514dae21fb563b67ed22b` |
| Source archive SHA-256 | `cc2ac21f4e6308106cb0f53c7803373efc30180d7fa75538cf72da96bbb525b3` |
| Release server executable SHA-256 | `f5b1fa327ef6282efa5dc7f7834439bf07568a070b9c8e1e38dd380e96c699f1` |
| `benchmarks` subtree | `d09757c21332ab018cb436673ed6363a28a3b5b0` |
| Protocol manifest SHA-256 | `5b63731b64c38a913de468e05560b81d860f5ce52425a618e7c0e7fb743be9d9` |
| Pinned vLLM | `81a8cb063e6247fe254d7caf7e98a6691d71022a` |
| Checkpoint manifest SHA-256 | `b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303` |
| Campaign identity SHA-256 | `8c429f490d50364f98413ba059119b2f8beaf28f9890ed657bfd20c5113ad587` |

Both arms used `/data0/shared/Qwen3.8-27B-FP8` and B200
`GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c`. Server CPUs were 24–33 and client
CPUs were 34–35, with local NUMA allocation.

The native timed command was `sglang-server --port 17424`. The
[server defaults](../../apps/sglang-server.cpp#L28)<!-- prefix: "struct Argum" -->
set MTP4, 32 request slots, an 8,192-token scheduling cap, CUDA graphs on,
and prefix caching off. The saved settings show the INT8 draft head,
certified INT8 verification head, online draft GQA, and tensor-core FP8 prefill.
They also show grouped-split target GQA and paired prefill. The pinned vLLM
arm selected MTP4 explicitly.

## Controls and raw evidence

Each timed arm emitted one repeatable sequence per row across all three
warmups and nine timed cohorts. The separate W1 native trace verified
depth four and matched all 128 native IDs in every timed W1 cohort.
Both engines reproduced their own output IDs from the prior formal run.

Cross-engine output IDs differed in every paired cohort. W1 differed at
100 of 128 positions, first at zero-based index 28 (native 524, vLLM 321).
W2 differed at 497 of 512 positions, first at index 0 (native 2,981, vLLM 314).
The vLLM arm supplies the timing reference. Exact output parity is outside the
timing gate.

The benchmark host retains SSE captures, parity reports, and launch receipts
under `artifacts/refactor-open-w1w2/formal-f46973d.9fRDcgMK/`.
Use the protocol's replay procedure with `campaign/W1` or `campaign/W2` as
both the fixture and capture directory.

| Raw `comparison.json` | Finished (UTC) | SHA-256 |
| --- | --- | --- |
| W1 | 05:56:04 | `35fcc391383a982db3f63c291e3d59860c51c6220cfa011031194b37c61f6d1c` |
| W2 | 06:01:11 | `dc76638d0da21db0b83222bb1f4e23f491a6582734e1e409f28b16b22946a605` |

The [prior formal comparison](https://github.com/para1lel/sglang.cpp/blob/fd5bd6cd73238bfbc6c41fcf6edd9021bdafae47/docs/serving/paired-native-defaults-results.md)
retains the W1 and W2 results for native Release `94e8bdc`.
