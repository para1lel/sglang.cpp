# W1 and W2 streaming baseline

| Workload | Prompt tokens | Output tokens | Active requests |
| --- | ---: | ---: | ---: |
| W1 | 128 | 128 | 1 |
| W2 | 2,048 | 512 | 1 |

The JSON token lists are the inputs. Both come from scripted English prose,
encoded with the frozen local checkpoint `tokenizer.json` and truncated by token
position. `freeze_fixtures.py` records the source text and verifies the tokenizer
hash. Direct IDs remove any dependency on a tokenizer or chat template in either
server. No special tokens or chat expansion are applied to the prompts.

## Wire contract

Send the compact JSON returned by `client.py` to `POST /v1/completions`.
The model name is `benchmark`. `prompt` is the JSON array of input IDs.
Use `max_tokens` from the fixture, `temperature=0`, `seed=0`, `ignore_eos=true`,
`add_special_tokens=false`, `return_token_ids=true`, `stream_interval=1`,
`stream=true` and `stream_options={"include_usage":true}`. Use one choice,
no stop strings, no speculation, and disabled prefix reuse.

Respond with HTTP 200 and `Content-Type: text/event-stream`. Emit JSON `data:`
frames with one entry in `choices`, `index=0`, and `token_ids` containing only
the newly emitted IDs. Echo `prompt_token_ids` once, on the first choice,
even if its output ID array is empty. The final choice reports
`finish_reason="length"` and may also carry token IDs. Send a subsequent usage frame with `choices=[]`,
`usage.prompt_tokens`, `usage.completion_tokens`, and
`usage.prompt_tokens_details.cached_tokens=0`. End with `data: [DONE]`.
The client validates all lengths, the prompt echo, usage, and completion reason.
It compares the complete emitted ID sequence across repetitions and against the
baseline when `--reference` is supplied. The reference also checks client,
fixture and request hashes and reports the measured throughput ratio.

## Frozen measurement protocol

Live collection runs one warmup and one measured round. Each round sends W1,
then W2. Submit the next request after storing the previous request's events.
The measured round supplies the point estimates. The archived GPU3 capture
used three warmup and nine measured rounds.
Each request uses a fresh localhost HTTP connection. Set GPU visibility to the
selected physical UUID. Keep the prefix cache disabled for startup, warmup,
measurement, and dispatch observation. Preserve all attempts without retries
or outlier removal. Start measurements after server health and warmups.

All timestamps use the client process's `perf_counter_ns`. Submission is
immediately before the HTTP connection and send. A frame takes the timestamp
of the HTTP-body read that completes it. Frames completed by the same read
share a timestamp. IDs in one frame also share that timestamp. Empty frames
have no token time. TTFT runs to the first observed ID. TPOT divides the
first-to-last ID interval by the actual output count minus one. E2E runs to
`[DONE]`. Output tokens/s divides the useful output count by E2E seconds.
Raw JSONL includes the SSE lines, read timestamps, output IDs, counts, and
failures. Disk writes happen between requests.

`--reference` applies saved thresholds when the reference includes them. A
reference captured with one measured round supplies a throughput ratio.

The original GPU3 protocol used the following statistics and limits.

Report median and range of throughput, and p50/p95 per-request latencies
using linear interpolation at `(n - 1) × p`. Compute sample standard deviation
over all nine measurements for each metric. For each workload, require
candidate median throughput divided by baseline median throughput to strictly
exceed `1 + max(0.01, 2 × baseline throughput stdev / baseline median throughput)`.
Each candidate p95 latency must be at or below baseline p95 plus
`max(0.01 × baseline p95, 2 × baseline latency stdev)`. These are engineering
noise allowances with a 1% floor. Accept a workload only when every request
passes the stream and greedy ID checks and the hardware and runtime policies
match the comparison contract.

## GPU3 reference

`gpu3-reference.json` contains the exact greedy output IDs, summary statistics
and full-precision thresholds from run `run.unEIDU`, frozen on November 18, 2026
at 12:47:05 UTC. The run used B200 UUID `GPU-ceeff2f7-3f39-50b4-8d29-576159781b9d`
and recorded project HEAD `2477f3704986a0e5863e60a13097fe1b7e0cd4c8`.

| Workload | Median output tokens/s | p95 TTFT (ms) | p95 TPOT (ms) | p95 E2E (ms) |
| --- | ---: | ---: | ---: | ---: |
| W1 | 115.87776 | 20.90009 | 8.53934 | 1105.11860 |
| W2 | 114.66872 | 79.49577 | 8.58548 | 4466.23870 |

The reference `manifest` records the checkout's protocol, client and fixture
SHA-256 hashes. `archive` records the measured versions of those files, the
measurement summary, the raw events and the evidence bundle. The fixtures
retain the same token IDs. Both clients produce byte-identical HTTP requests,
whose SHA-256 hashes are in `manifest.request_sha256`. The checkout client
enforces the first-choice echo rule. All 24 archived warmup and measured
streams pass replay with their original recorded timestamps, and both
archived dispatch traces pass the checkout validator. `verification` records
the replay count and the dispatch validator and trace hashes. The archive
retains the original files and output under `artifacts/workload-baselines/run.unEIDU`.

## Run

Use the [archived GPU3 capture procedure](https://github.com/para1lel/sglang.cpp/blob/acbf95b0de14108b37c37bdc3dc40daaf2db2099/benchmarks/streaming/README.md#run) for the original launcher and its W1 and W2 probes.
That procedure uses the pinned vLLM environment, a UUID lock, checkpoint
and wheel identity checks, and a loopback server. It enables the normal
vLLM graph policy, FlashInfer BF16 queries, E4M3 KV, and BF16/FP32 GDN state.
It reads live tensor metadata and scales before warmup and collects one
untimed dispatch profile per workload.

For a native server with the wire contract, run
the checkout client under the same device and host controls. The compact
reference works without the archived run.

```sh
/usr/bin/python3 -IB benchmarks/streaming/client.py \
    --url http://127.0.0.1:PORT --output NEW_DIR \
    --reference benchmarks/streaming/gpu3-reference.json
```

Use the archived client and fixtures to reproduce the original multi-round run.
Its `measurement/summary.json` remains evidence for that protocol.
