# Matched W1 and W2 K4 with native server defaults

Native Release `9987dc8463ce14d249b558af10ed660d44590927` passed the strict
median makespan gate on W1 and W2 on November 23, 2026 UTC. The formal
campaign ended after W2. Formal W3–W6 remain unmeasured.

The [matched K4 timing protocol](../../benchmarks/paired/README.md#matched-k4-timing-protocol)
uses three warmups and nine timed single-request cohorts per arm for W1 and W2.
The client measures makespan from submission through `[DONE]`. The gate
requires the native median to be strictly lower than the pinned vLLM median.

| Workload (input/output IDs) | Native median, ms | Pinned vLLM median, ms | Native makespan reduction | Timing gate |
| --- | ---: | ---: | ---: | --- |
| W1 (128/128) | 612.290869 | 644.133185 | 4.94% | Pass |
| W2 (2,048/512) | 1,370.155616 | 1,723.769346 | 20.51% | Pass |

## Measured builds and launch

| Item | Identity |
| --- | --- |
| Native source tree | `95967c33a08a408542ac311b54c7b6971b1f8d00` |
| Source archive SHA-256 | `2b35c8564e3a6312dd8b8ed5adb36d81a73d69c737a5a6b32da6259f41584bbc` |
| Release server executable SHA-256 | `64665213b39d5bf7dd47c5e4e2cf634bfbef38dc0888a18a4a264313ed4296e0` |
| `benchmarks` subtree | `d09757c21332ab018cb436673ed6363a28a3b5b0` |
| Protocol | `matched-k4-native-defaults-v1` |
| Protocol manifest SHA-256 | `5b63731b64c38a913de468e05560b81d860f5ce52425a618e7c0e7fb743be9d9` |
| Pinned vLLM | `81a8cb063e6247fe254d7caf7e98a6691d71022a` |
| Checkpoint manifest SHA-256 | `b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303` |
| Campaign ID | `66353793-b37f-49fa-9744-15c80057b235` |
| Campaign identity SHA-256 | `938376d6d0e9d6db9504fd096891dd6183514e8da4e21d354266e0b658ef1e71` |

Both arms used `/data0/shared/Qwen3.8-27B-FP8` and B200
`GPU-d4d87d81-ea84-5dd0-7167-3e1c48fde29b`. Server CPUs were 72–81 and client
CPUs were 82–83, with local NUMA allocation.

The native timed command was `sglang-server --port 17424`. It used MTP4,
32 request slots, an 8,192-token scheduling cap, CUDA batch graphs, and
no prefix caching. The saved settings show the INT8 MTP head and
tensor-core FP8 prefill. The pinned vLLM arm selected MTP4 explicitly.

## Controls and raw evidence

Both launchers exited with status zero under the same campaign identity.
Each engine reproduced its own output IDs from the prior formal run at native
revision `b7c36ded75985788adae04ced54ca2948c5c3478` across all 12 cohorts per row.
Native matched 1,536 IDs on W1 and 6,144 on W2. A separate W1 native trace
matched all 128 IDs and recorded 49 depth-four rounds out of 50.

In the first timed pair, cross-engine output IDs differed at 100 of 128
positions on W1, first at zero-based index 28 (native 524, vLLM 321).
On W2 they differed at 497 of 512 positions, first at index 0
(native 2,981, vLLM 314). Pinned vLLM supplies the timing reference.

Checkpoint cleanup removes the raw SSE captures, parity reports, and launch
receipts. This summary retains the request and comparison digests.
Replay requires the original receipts.

| Workload | Request SHA-256 | Comparison receipt SHA-256 |
| --- | --- | --- |
| W1 | `939228b6cc828db1c453115c7497b7146b29b9799d574aacd50f1d016cd9341a` | `b1c94c57150c5f91b9a989c220f1fd5e0eb49f86b9ce805242ca60d79889fd6c` |
| W2 | `f17aac9d1b94d9eb6a2185b9616fe9b659b41cc8508843d636fac4887859a34e` | `2b26dfcb4ab4e9208a59b9606bd052058eb629dbfee0c2c92aafd1b5e3f458bc` |
