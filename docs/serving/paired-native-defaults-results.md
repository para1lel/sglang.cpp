# Matched W1 and W2 K4 with native server defaults

The November 23, 2026 UTC comparison used the `matched-k4-native-defaults-v1`
protocol. Each timed arm ran three warmups and nine single-request cohorts.
Client makespan covers submission through `[DONE]`. The gate requires the
native median to be strictly lower than the pinned vLLM median.

| Workload (input/output IDs) | Native median (range), ms | Pinned vLLM median (range), ms | Strict median gate |
| --- | ---: | ---: | --- |
| W1 (128/128) | 613.877352 (611.439676–700.943523) | 648.982268 (648.049638–654.312857) | Pass, 35.104916 ms lower (5.409%) |
| W2 (2,048/512) | 1,407.243063 (1,405.205157–1,414.474298) | 1,952.642850 (1,948.837352–1,954.547000) | Pass, 545.399787 ms lower (27.931%) |

W1 ran vLLM first. One native sample exceeded the slowest vLLM sample.
W2 ran native first. Every native sample was faster than every vLLM sample.
Both rows exited 0. This record covers W1 and W2.

| Workload | Native / vLLM median TTFT, ms | Native / vLLM median first output to `[DONE]`, ms | Native / vLLM median TPOT, ms/ID |
| --- | ---: | ---: | ---: |
| W1 | 18.598 / 33.281 | 594.938 / 615.440 | 4.684 / 4.846 |
| W2 | 125.618 / 77.770 | 1,281.555 / 1,874.354 | 2.508 / 3.668 |

TTFT ends at the first output ID. TPOT measures the time between the first
and last ID divided by the number of subsequent IDs.

## Measured builds and launch

| Item | Identity |
| --- | --- |
| Published native runtime | `8b4e3f62e0a7a053ffdefe85da99a841fabeb5d3` |
| Release build archive | `b84a05c27eebd1c056ec32cc5d0c4da2884d0ec3`, with the same runtime sources |
| Native server executable SHA-256 | `239fe068a5705b0ed250d09b0ce0f6487ad907511b7d7ae42a90923ac0f31c8b` |
| Isolated measured protocol tree | `547d926d2c8795519f8fa46f66f35228b6c18ed5` |
| Measured `benchmarks` subtree | `d09757c21332ab018cb436673ed6363a28a3b5b0` |
| Protocol receipt SHA-256 | `5b63731b64c38a913de468e05560b81d860f5ce52425a618e7c0e7fb743be9d9` |
| Pinned vLLM | `81a8cb063e6247fe254d7caf7e98a6691d71022a` |
| Checkpoint | `/data0/shared/Qwen3.8-27B-FP8` |
| Checkpoint manifest SHA-256 | `b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303` |

The measured runtime predates automatic online draft attention and the later
early causal prefill change. The benchmark archive adds the one-line vLLM
live-argv receipt repair to the published runtime's benchmark sources.
The SHA-256 of the repair is
`7a8d922ac0367e5e753b273cf671276b06750d2a27095a094c16277caec84be1`.
The initial W1 launch stopped at the original argv check before timed native
capture. The table uses the fresh campaign with the repaired launcher.

Both arms used B200 GPU3 (`GPU-ceeff2f7-3f39-50b4-8d29-576159781b9d`),
server CPUs 24–33, client CPUs 34–35, and local NUMA allocation. Both had
32 request slots and an 8,192-token scheduling cap.

The native timed argv was the sealed `sglang-server` executable followed by
`--port 17424`. It resolved MTP4, the INT8 draft head, `tensor-core` FP8
prefill, CUDA graphs on, and prefix caching off.
The vLLM launch explicitly selected MTP4 with
`--speculative-config '{"method":"mtp","num_speculative_tokens":4}'`.
It used FlashInfer TRTLLM attention, BF16 queries, FP8 E4M3 KV, a 2,560-token
context limit and 10 GiB of KV memory. The full measured argv for each arm
is in its NUL-delimited `live-argv.bin`. Both rows recorded the same bytes
for each timed arm.

| Timed argv receipt | SHA-256 |
| --- | --- |
| Native | `5cb6a8c31c86c1ea5b2924da542bdb252084f494fc742daff26c6a95cef8cc7a` |
| vLLM | `e3b8c0eb2bc64aa75199f58183371778955e97bb32612f4f536be54befc6f1ea` |

## Controls and raw evidence

The separate W1 native control added `--trace-batches`. It reached depth four
in 49 of 50 rounds and emitted the same 128 IDs as the timed native arm.
Independent replay verified SSE format, length, usage, and drain across
25 W1 and 24 W2 streams. Output IDs repeated within each timed arm.
The vLLM W1 counters recorded 2,544 new draft tokens and 636 fourth-position
verifications. The W2 counters recorded 6,720 and 1,656, respectively.
This protocol uses vLLM for timing and has no K0 or cross-engine output
parity control.

On the benchmark host, the raw record is in
`artifacts/default-formal-mtp4/published/published-8b4e3f6-gpu3-argv-retry/campaign/`.
Each row holds the original SSE captures and per-arm launch and worker
receipts. Replay `k4.py assess` from the measured benchmark archive using
the row directory for both fixtures and captures.

| Raw `comparison.json` | Finished (UTC) | SHA-256 |
| --- | --- | --- |
| W1 | 01:23:02 | `431d5c19eccc5588deff1b947111b081e6aaa7352637a8a16eee2fbdd56defb4` |
| W2 | 01:29:40 | `a027bfbae30fd6188de7d9d8624faf3be9e114f614cce5117d4a6871f7ea02b2` |
