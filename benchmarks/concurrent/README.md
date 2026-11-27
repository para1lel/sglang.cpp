# W3–W6 concurrent streaming baseline

| Row | Prompt IDs per lane | Output IDs per lane | Lanes | Required / usable KV pages |
| --- | ---: | ---: | ---: | ---: |
| W3 | 8,192 | 1,024 | 4 | 36 / 2,620 |
| W4 | 32,768 | 1,024 | 16 | 400 / 2,620 |
| W5 | 32,768 | 4,096 | 32 | 864 / 2,620 |
| W6 | 131,072 | 1,024 | 4 | 344 / 2,620 |

The November 18 vLLM reference uses B200 `GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c`. Pair on that physical GPU. All 280 timed requests finished with `length`, zero cached tokens and exact lane goldens, as did the warmup and diagnostic cohorts. [Frozen capture details](https://github.com/para1lel/sglang.cpp/blob/aab1101e145507c4c40435602e3a2d3c5d88111d/benchmarks/concurrent/README.md)

## Regenerate and measure

Run from the repository root. Fixture generation uses Python 3.12.3 and `tokenizers` 0.23.3
from the capture environment. The client, offline tests, and replay consume standard-library
JSON and SSE, with `pytest` needed to run the tests.

```sh
mkdir -p "$RESULTS"
python -B -m benchmarks.concurrent.freeze_prompts "$CHECKPOINT/tokenizer.json" "$FIXTURES"
python -B -m benchmarks.concurrent.client W3 http://127.0.0.1:17382 "$FIXTURES" "$RESULTS/W3"
python -B -m pytest -q -p no:cacheprovider tests/test_concurrent_baseline.py
```

Pass a new fixture directory and a new output directory per row. Run W3 through W6 in order.
The generator checks the tokenizer digest, each text digest, and all 56 request digests.
All first IDs differ across lanes and rows. The prompts contain authored inventory prose
and a literal assistant prefill. Servers receive the frozen direct IDs without applying
tokenization, special-token insertion, or a chat template.

The client shares the [W1/W2 request and SSE contract](../streaming/README.md#wire-contract).
For a paired run, it checks the echoed prompt IDs, exact output count, `length` finish,
zero cached prompt tokens, usage, and stream completion. Both engines retain every
output ID. Their KV precision differs, so cross-engine sequence equality is diagnostic
only. Offline vLLM replay requires exact matches against the frozen vLLM lane sequences.
If a server provides an OpenAI completion ID, the ID must be stable and unique in the
cohort. A reported `total_tokens` must equal prompt plus output.

Preconnect one loopback socket per lane, release submissions on a barrier, and require
the submit span to stay within 50 ms. Every request body must be sent before any output
ID arrives. All streams must overlap, with at least one ID received from every lane
before the first `[DONE]`. Run one warmup and one measured cohort. Compare each
lane's full outputs between cohorts. The per-socket I/O, barrier, and cohort
timeouts are 90, 15, and 360 seconds. Stop the row at the first
invalid cohort, retaining every attempt and its original workload name. Gzip JSON
preserves the SSE frames, read boundaries, timestamps, IDs, usage, stops, and errors.

Use one `perf_counter_ns` clock for all lanes. Each frame takes the timestamp of the
HTTP read that completed it. TTFT ends at the first ID, TPOT divides first-to-last ID
time by output count minus one, and E2E ends at `[DONE]`. Cohort output IDs/s divides
the sum of lane output IDs by the time from earliest submit to latest `[DONE]`.
The summary also reports request and input-plus-output rates.

## Conditions and gates

The [reference](gpu2-reference.json#L2)<!-- prefix: "\"format\": \"s" --> stores launch arguments, runtime and checkpoint identities, scheduler, KV backing, profiles and full-precision cutoffs.

Use vLLM `81a8cb063e6247fe254d7caf7e98a6691d71022a`, one device, BF16 Q, E4M3 K/V with FP32 unit scales, BF16 GDN convolution and FP32 recurrence. Disable prefix caching, MTP and KV/weight offload. The reference used 8,192 scheduled tokens, 32 sequence slots, a 132,096-token context and `FULL_AND_PIECEWISE` with compile mode `NONE` and `enforce_eager=false`. Independently verify hardware, KV backing, native MXFP8 and dispatch. Check FlashInfer FA2 and chunked GDN in prefill, TRTLLM-Gen GQA in mixed steps and decode, and packed recurrent GDN in full decode. [Frozen scheduler and profiles](https://github.com/para1lel/sglang.cpp/blob/aab1101e145507c4c40435602e3a2d3c5d88111d/benchmarks/concurrent/README.md#conditions-and-gates)

The standalone client fails if the measured cohort misses any stored gate. Median cohort output IDs/s must exceed the baseline median multiplied by `1 + max(0.01, 2 × sample SD / median)`. Each pooled request p95 for TTFT, TPOT and E2E must be at most the baseline p95 plus `max(0.01 × p95, 2 × sample SD)`. Interpolate at `(n−1) × p`. Use the unrounded JSON values. Archived replay checks two warmups and five measured cohorts and derives the cutoffs from measured samples.

## Corpus registration

V2 froze before measurement and uses the checkpoint's no-thinking assistant prefill. Its exact repeatability covers the authored repetitive continuations. V1 failed that prerequisite and contributed no measured cohorts. [Frozen registration and row receipts](https://github.com/para1lel/sglang.cpp/blob/aab1101e145507c4c40435602e3a2d3c5d88111d/benchmarks/concurrent/README.md#corpus-registration)

## Provenance and offline replay

The [fixture manifest](fixture-manifest.json#L3)<!-- prefix: "\"tokenizer_s" --> pins the tokenizer and request digests. Concatenate the golden base64 chunks in order and decode lane-major little-endian `uint32` IDs. The loader checks the decoded digest. Checkpoint weights and tokenizer files are separate inputs. [Frozen encoding and provenance](https://github.com/para1lel/sglang.cpp/blob/aab1101e145507c4c40435602e3a2d3c5d88111d/benchmarks/concurrent/README.md#provenance-and-offline-replay)

Replay requires a separately retained `gpu2.20261118T1319Z-evidence.tar.gz`. Check `archive.bundle_sha256`, extract the bundle and pass its directory to the checker from the [archived replay revision](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/benchmarks/concurrent/README.md#provenance-and-offline-replay). It checks raw digests, exact requests and outputs, overlap, latency, gate arithmetic, scheduler and kernel dispatch. `files.sha256` covers the remaining evidence.
