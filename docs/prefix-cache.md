# Native prefix cache

## Responsibilities and sources of truth

| Part | Responsibility | Source |
| --- | --- | --- |
| Server | Configure prefix reuse, validate salts, and report usage. | [CLI](../apps/sglang-server.cpp#L80)<!-- prefix: "} else if (o" --> · [Salt parser](../apps/server/text_request.cpp#L255)<!-- prefix: "void set_cac" --> · [Usage writer](../apps/server/text_completion.cpp#L27)<!-- prefix: "void append_" --> |
| Batch | Restore request state, schedule remaining prompt rows, and publish completed checkpoints. | [Admission](../src/decoder_batch.cpp#L292)<!-- prefix: "int DecoderB" --> · [Checkpoint publication](../src/decoder_batch.cpp#L634)<!-- prefix: "cache-\u003estore" --> |
| Cache | Match checkpoint keys and own the copies until eviction or batch teardown. | [Lookup](../src/decoder_prefix_cache.cpp#L119)<!-- prefix: "int PrefixCa" --> · [Restore](../src/decoder_prefix_cache.cpp#L138)<!-- prefix: "int PrefixCa" --> · [Store](../src/decoder_prefix_cache.cpp#L161)<!-- prefix: "void PrefixC" --> |

## Runtime path

Use `--prefix-cache-entries 1` through `4` for native prefix reuse. The default `0` disables reuse. Enabling the cache requires `--fp8-prefill exact` and `--mtp-steps 0`. Omit `--eager`. An enabled cache selects the batch route even at `--max-active 1`. Batch graph opt-in requires capacity two or higher. [CLI validation](../apps/sglang-server.cpp#L105)<!-- prefix: "if (argument" --> · [Route selection](../apps/sglang-server.cpp#L220)<!-- prefix: "if (argument" -->

Native text, chat, and benchmark ID requests accept an optional `cache_salt` string of up to 256 UTF-8 bytes. Omission uses an empty string. Invalid salts return HTTP 400 even when reuse is disabled. Native text and chat support greedy, sampled, and finite-choice requests with prefix reuse. The benchmark ID route keeps its greedy policy.

C++ callers set `DecoderBatchOptions::prefix_cache_entries` and `DecoderOptions::prefix_cache_salt`. The batch requires exact prefill and rejects salts longer than 256 bytes. `DecoderBatch::admit` returns the restored token count. [Batch options](../include/sglang/decoder_batch.h#L43)<!-- prefix: "struct Decod" --> · [Token-ID parser](../apps/server/completions.cpp#L26)<!-- prefix: "CompletionRe" -->

JSON usage and SSE usage count the full prompt in `prompt_tokens`. `prompt_tokens_details.cached_tokens` counts restored prompt IDs. It is zero on a miss, with the cache off, or for a zero output budget. [Token-ID usage](../apps/server/completions.cpp#L95)<!-- prefix: "std::string " -->

The checkpoint boundary is the largest multiple of 128 within the prompt length and 6,400-token limit. Prompts shorter than 128 tokens have no checkpoint. Lookup chooses the longest stored checkpoint within this boundary with matching prefix IDs and salt. A miss prefills from the start. A partial hit restores state into private request allocations and prefills the suffix. [Boundary](../src/decoder_prefix_cache.cpp#L115)<!-- prefix: "int PrefixCa" -->

On a full hit, eager scoring uses the restored hidden and residual rows at output index zero. Selection uses the current request's [grammar](text-serving.md#finite-choice-constraints) and [sampling options](text-serving.md#sampling), including its seed. Its first score stays outside the packed decode width and graph mask. Eligible decode peers can replay on the same step. Subsequent decode follows [batch graph eligibility](batch-graphs.md#runtime-path). [Full hit](../src/decoder_batch.cpp#L447)<!-- prefix: "if (lane.res" --> · [Token selection](../src/decoder_batch.cpp#L408)<!-- prefix: "void Decoder" -->

## Constraints and failure behavior

An entry stores four MXFP8 K/V payload and scale planes for each of 16 attention layers. It stores BF16 convolution history and FP32 recurrent state for 48 GDN layers, plus BF16 hidden and residual rows. A full entry uses 50 logical pages. Each configured entry reserves 370,233,344 device bytes when the batch starts. Four entries reserve 1,480,933,376 bytes in addition to request allocations. [Allocations](../src/decoder_prefix_cache.cpp#L104)<!-- prefix: "PrefixCache:" -->

Store gathers the four KV planes into logical page order. Restore scatters them into the destination request's physical pages. The copy validates the covered pages and rejects aliases into the suffix. A lane owns its restored copy and can continue after eviction. A store publishes the checkpoint after its copies complete on the batch stream. A hit or repeated key refreshes recency. A new key evicts the least recently used entry when all entries are occupied. Cancellation can leave an already published checkpoint available. Zero-budget requests and positive-depth C++ MTP lanes skip reuse and publication. [Page mapping](../src/decoder_prefix_cache.cpp#L17)<!-- prefix: "PageMapping " --> · [Completed copy](../src/decoder_prefix_cache.cpp#L183)<!-- prefix: "check(cudaSt" -->

An error while restoring a K0 lane disables the batch. A publication error disables the batch on a K0-only step. An invalid sampled or constrained selection retires only its lane. [Admission recovery](../src/decoder_batch.cpp#L338)<!-- prefix: "} catch (..." --> · [Step recovery](../src/decoder_batch.cpp#L653)<!-- prefix: "if (!contain" -->

With `--trace-batches`, `entries` and `pages` count published state, and `device_bytes` counts the reservation. `hits`, `cached_tokens`, `prefill_tokens`, `stores`, and `evictions` are cumulative for the batch. `prefill_tokens` counts cache-eligible prompt rows processed on the batch path. Request memory and batch scratch have separate counters. [Trace](../apps/server/batch_service.cpp#L258)<!-- prefix: "void trace_p" --> · [Counters](../src/decoder_prefix_cache.cpp#L196)<!-- prefix: "DecoderPrefi" -->

## Verification

Follow the [prefix cache CUDA and route checks](../tests/prefix/README.md). Cover identity and remapped page tables, transfer failures, all four KV planes, and the 6,400-token limit. Compare greedy cold, full-hit, and suffix-hit routes from the same binary with exact prefill. Check page boundaries, changed prefix IDs, changed salts, zero budget, cancellation, and eviction. Compare committed IDs and finish reasons. In usage, require the expected cached-token count and identical prompt and completion counts.

Repeat full and partial hits with seeded sampling and a second seed or sampling policy. Change the finite-choice grammar while reusing the same prefix and salt. Compare each warm result with its own cold request, including the first output ID. Check that a full hit stays out of the replay mask while eligible peers replay. Inject a selection failure and check that the failed lane retires and its peers finish.

### W3-prefix

Use the [frozen variant fixtures](../benchmarks/variants/README.md#reproduce-on-the-baseline-host). For both native arms, use `--max-active 4 --batch-cuda-graphs --fp8-prefill exact --mtp-steps 0 --prefix-cache-entries 4`. Start a fresh native process before each cohort to clear cache state. Wait for HTTP readiness. For the warm arm, submit the 6,401-ID off-branch prime with a one-ID output budget. Exclude the prime from timing. Then submit four 8,192-ID prompts with 1,024 output IDs each.

Use `prime` and `prefix` for the warm arm. Use `cold` for the same prompts with distinct salts, no prime, and zero expected hits. Match all output IDs between the two native arms. In a separate traced warm run, require four hits, 6,400 restored IDs per lane, and 1,792 suffix rows per lane. Expect one 50-page entry and zero evictions in that run. [Fixture loader](../benchmarks/variants/fixtures.py#L118)<!-- prefix: "def load_req" --> · [Capture](../benchmarks/variants/client.py#L233)<!-- prefix: "def capture(" -->

Run the command from the repository root. Set `endpoint` to the ready server origin. Set `fixtures` to the frozen fixture directory. Set `mode` to `cold`, `prime`, or `prefix`. Set `output` to the evidence directory. Use a unique `tag` for each capture.

```bash
/data1/shared/dongwu.chen/.pixi/envs/default/bin/python -B - \
    "$endpoint" "$fixtures" "$mode" "$output" "$tag" <<'PY'
from pathlib import Path
from sys import argv
from urllib.parse import urlsplit
from benchmarks.variants.client import capture
from benchmarks.variants.fixtures import load_requests

endpoint, fixtures, mode, output, tag = argv[1:]
directory = Path(output)
directory.mkdir(parents=True, exist_ok=True)
capture(urlsplit(endpoint), load_requests(Path(fixtures), mode), tag, directory)
PY
```

Compare `requests[].output_ids` in the cold and warm `*-replayed.json` reports. A failed capture or mismatch fails the check.

For acceptance, use the [frozen variant gates](../benchmarks/variants/README.md#evidence-and-comparison). Keep two warmup and five measured cohorts. Require stream overlap and a submission span of at most 50 milliseconds. Keep the 90-second socket, 15-second barrier, and 360-second cohort limits. Disable tracing and profiling for timing. Report cache reservation and peak device memory separately.

The earlier GPU6 diagnostic used tracing and disabled the cache in its cold arm. Its exact-prefill cold arm exceeded 90 seconds. The diagnostic used a 300-second socket limit. See `artifacts/session-prefix-dev/gpu6-w3-final/`. Frozen W3-prefix acceptance is pending.
