# W1 and W2 streaming baseline

| Workload | Prompt tokens | Output tokens | Active requests |
| --- | ---: | ---: | ---: |
| W1 | 128 | 128 | 1 |
| W2 | 2,048 | 512 | 1 |

Send the exact [W1](W1.json#L3)<!-- prefix: "\"prompt_toke" --> and [W2](W2.json#L3)<!-- prefix: "\"prompt_toke" --> JSON IDs directly. To regenerate from scripted English prose, use the frozen checkpoint tokenizer with `tokenizers` installed. The generator checks its hash and truncates by token position without special tokens or template expansion.

```sh
mkdir -p "$FIXTURES"
python -B benchmarks/streaming/freeze_fixtures.py "$CHECKPOINT/tokenizer.json" "$FIXTURES"
```

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
The client validates lengths, prompt echo, usage and finish, and compares exact greedy IDs across repetitions and against `--reference`. It checks fixture, request and client digests, with a client digest exception for the exact published GPU3 reference.

## Frozen measurement protocol

Run one warmup and one measured round, each in W1–W2 order. After server health, submit on fresh loopback connections, saving each attempt before the next. Use the selected physical GPU UUID and keep prefix reuse off during startup, warmup, timing and dispatch observation. Preserve all attempts without retries or outlier removal.

Use the client’s `perf_counter_ns`, starting just before connect and send. A frame and its IDs take the timestamp of the HTTP body read that completed it. Empty frames add no token timestamp. TTFT runs to the first ID. TPOT divides first-to-last ID time by output count minus one. E2E runs to `[DONE]`, and output tokens/s divides the output count by E2E seconds. Raw JSONL keeps SSE, read times, IDs, counts and failures.

When `--reference` has thresholds, require valid streams, a median throughput ratio strictly above `1 + max(0.01, 2 × baseline sample SD / baseline median)` and each p95 latency at most baseline p95 plus `max(0.01 × baseline p95, 2 × baseline sample SD)`. Use the unrounded limits. The archived three-warmup, nine-measurement protocol uses sample standard deviations and interpolates p95 at `(n−1) × p`. A one-round reference without thresholds reports the ratio.

## GPU3 reference

The [GPU3 reference](gpu3-reference.json#L1)<!-- prefix: "{" --> contains exact greedy IDs, statistics and thresholds from `run.unEIDU` on B200 `GPU-ceeff2f7-3f39-50b4-8d29-576159781b9d`. Its `manifest.protocol_sha256` identifies the [published frozen protocol bytes](https://github.com/para1lel/sglang.cpp/blob/e681f90867aa9c59bb0c11bf0e796557f36ce088/benchmarks/streaming/README.md#frozen-measurement-protocol). The `archive` identifies the measured inputs and evidence bundle. `verification` records 24 replayed streams and both dispatch trace digests. [Frozen evidence details](https://github.com/para1lel/sglang.cpp/blob/aab1101e145507c4c40435602e3a2d3c5d88111d/benchmarks/streaming/README.md#gpu3-reference)

## Run

Use the [archived GPU3 procedure](https://github.com/para1lel/sglang.cpp/blob/e681f90867aa9c59bb0c11bf0e796557f36ce088/benchmarks/streaming/README.md#run) for the original three-warmup, nine-measurement vLLM launch and untimed dispatch probes. Its `measurement/summary.json` records the multi-round results.

Run the current client against a native server on the same GPU under the same host controls. The compact reference works without the archived run.

```sh
/usr/bin/python3 -IB benchmarks/streaming/client.py \
    --url http://127.0.0.1:PORT --output NEW_DIR \
    --reference benchmarks/streaming/gpu3-reference.json
```

Use `--fixtures "$FIXTURES"` for a regenerated fixture directory. `--freeze-baseline` records a single-round reference without `--reference`. `--probe W1` or `--probe W2` runs one untimed dispatch probe.
