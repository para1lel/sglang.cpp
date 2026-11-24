# W3–W6 concurrent streaming baseline

| Row | Prompt IDs per lane | Output IDs per lane | Lanes | Required / usable KV pages |
| --- | ---: | ---: | ---: | ---: |
| W3 | 8,192 | 1,024 | 4 | 36 / 2,620 |
| W4 | 32,768 | 1,024 | 16 | 400 / 2,620 |
| W5 | 32,768 | 4,096 | 32 | 864 / 2,620 |
| W6 | 131,072 | 1,024 | 4 | 344 / 2,620 |

The frozen vLLM run completed all four rows on B200
`GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c`. A paired comparison uses that physical GPU.
The 280 timed requests emitted 778,240 IDs, all with a `length` finish and zero cached prompt
tokens. All 112 warmup and 56 diagnostic requests also matched their exact lane goldens.
The reference represents vLLM results. Candidate performance and correctness remain unmeasured
by this publication.

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

The [reference](gpu2-reference.json#L2)<!-- prefix: "\"format\": \"s" -->
records the exact launch arguments, runtime identities, checkpoint hash, live scheduler and
cache settings, profile signatures, and full-precision gates.
Use the vLLM pin `81a8cb063e6247fe254d7caf7e98a6691d71022a`, BF16 Q, and per-tensor E4M3 K/V
with FP32 unit scales. Disable prefix caching, MTP, and KV and weight offload. Use one
device and BF16 GDN convolution with FP32 recurrence. The baseline used 8,192 scheduled
tokens, 32 sequence slots, and 132,096 maximum sequence length. It allocated
137,415,884,800 bytes of physical KV backing, including one reserved page.

vLLM used `FULL_AND_PIECEWISE` CUDA graphs with compile mode `NONE` and `enforce_eager=false`.
The separate diagnostic cohort profiled four initial steps and four full-batch decode
steps per row. Profile replay checks FlashInfer FA2 paged GQA and Blackwell chunked GDN
in prefill, with TRTLLM-Gen GQA in any mixed steps. Full decode uses TRTLLM-Gen GQA and
packed recurrent GDN. Establish effective hardware, cache, and dispatched
backend conditions independently for any paired run.
Use native MXFP8 KV for the candidate as specified by the project benchmark contract.

The standalone client checks the measured cohort against the frozen thresholds
and reports failure on a miss. The archived capture used two warmups and five
measured cohorts. Its replay checks the original wire and timing gates.

Require median cohort output IDs/s to **strictly exceed** the baseline median times
`1 + max(0.01, 2 × sample SD / median)`. Require each request-level p95 to stay at or
below its baseline p95 plus `max(1% × p95, 2 × sample SD)`. Interpolate at `(n−1) × p`
and pool measured requests for latency. The 1% throughput floor governed all four
rows. W4 and W6 TTFT caps exceed their respective E2E caps because the independent
TTFT samples have larger spreads. Both caps still apply. JSON holds the unrounded
numbers, and replay derives them from the original samples.

| Row | Median output IDs/s must exceed | TTFT p95 cap (ms) | TPOT p95 cap (ms) | E2E p95 cap (ms) |
| --- | ---: | ---: | ---: | ---: |
| W3 | 383.319743 | 2,234.493277 | 11.000384 | 10,903.335753 |
| W4 | 358.722515 | 50,226.957122 | 59.994477 | 46,590.145259 |
| W5 | 886.518814 | 99,323.812404 | 43.649717 | 150,758.159149 |
| W6 | 48.709172 | 113,118.073927 | 104.925433 | 85,806.011537 |

## Corpus registration

V1 was registered on November 18, 2026 at 13:28:35 UTC with exact lane repeatability
as a prerequisite. Two W3 exploratory warmups completed. Lane 0 first diverged at
zero-based output index 11, emitting 264 versus 279. Across the 1,024 positions,
1,013 differed. The other three lanes matched. V1 contributed zero measured cohorts.

V2 included the checkpoint's no-thinking assistant prefill so the requested repeated
text began outside the reasoning turn. It was registered at 13:30:04 UTC before its
warmups. The first measured cohort began at 13:30:26.464082 UTC. The policy file was
byte-identical in both registrations. V2 met the exact-repeat criterion in every
warmup, measured, and diagnostic cohort. Row gates froze at 13:37:14.448140,
13:38:57.191827, 13:59:10.887623, and 14:10:35.670787 UTC, respectively, before any
paired candidate measurements. The repetitive continuation narrows the test's
coverage. Exact candidate agreement on this corpus would leave open agreement on
less constrained or numerically sensitive continuations, including V1 reasoning.

## Provenance and offline replay

The generator contains original synthetic fixture prose. The compact goldens contain
observed numeric output IDs. The loader joins string chunks in order to recover the exact
base64 payloads, kernel signatures, profiler argument, and offload configuration. The
profiles refer to the shared kernel name list by zero-based index. The golden payloads
decode to lane-major, little-endian unsigned 32-bit IDs and are checked against a digest
of the full decoded sequence. Checkpoint weights and tokenizer files are separate inputs.
The captured checkpoint distribution includes the Apache 2.0 `LICENSE` and a model card
for attribution.
vLLM and its runtime dependencies retain their own distribution terms.

The capture input tree was `5a83bcc0d7aba2ac1ccbf476818b635c68208cf5`. This publication
uses `5119e6436892a28553b3cfae55e7b648176d548d`. Checkpoint cleanup leaves the tracked
reference and fixture generator. Offline replay requires a separately retained copy of
the original `gpu2.20261118T1319Z-evidence.tar.gz`. Check its SHA-256 against
`archive.bundle_sha256` before extracting. The archive contains raw cohorts, counters,
profiles, and the original launch and worker observer sources.

A fresh live run can regenerate the fixtures, evidence, and warmed runtime cache on
the pinned checkpoint, runtime, and GPU using the recorded launch settings. The
original launcher used `concurrent_observer.py` and seeded its cache from the earlier
W1/W2 run. Repeating that setup requires the original launcher and observer from a
separate copy of the capture sources and the W1/W2 seed cache.

```sh
mkdir -p "$EXTRACTED"
tar -xzf "$BUNDLE" -C "$EXTRACTED"
python -B -m benchmarks.concurrent.replay "$EXTRACTED"
```

Replay checks raw cohort and trace digests, exact requests and outputs, overlap,
latencies, full-precision gate arithmetic, and scheduler and kernel dispatch. The
bundle's `files.sha256` supports checking the remaining archived evidence as well.
