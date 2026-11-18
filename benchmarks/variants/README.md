# W3 prefix and MTP baselines

Both rows use four synchronized requests with 8,192 input IDs and 1,024 output IDs
per request. The frozen vLLM runs use B200 `GPU-7e25f95a-29b7-9f94-ce9a-2a2871cf7c0c`
and source pin `81a8cb063e6247fe254d7caf7e98a6691d71022a`.
Run the two rows on separate engine processes under one physical-device lock.

| Row | Real prefix hits per lane | Draft positions | Median IDs/s | p95 TTFT / TPOT / E2E, ms |
| --- | ---: | ---: | ---: | ---: |
| W3-prefix | 6,400 | 0 | 417.662 | 349.828 / 9.4805 / 9,808.280 |
| W3-MTP | 0 | 1 | 610.524 | 1,465.871 / 6.1484 / 6,709.283 |

The prefix fixture shares four 1,600-token blocks and branches immediately after them.
Reset the cache before every cohort and run one off-branch prime with 6,401 input
IDs and one output ID. Exclude reset and prime from timing. The cold control uses
the same four requests with distinct cache salts after a reset. Warm output IDs
match the cold vLLM output IDs in every lane.

The MTP fixture uses the exact archived W3 request bytes, with prefix caching off.
The live worker loads `Qwen3_5MTP` with one draft position. All five measured
cohorts each show 2,048 proposed, verified, and accepted IDs in vLLM engine counters.
Each MTP lane matches the archived vLLM W3 greedy sequence with MTP off.

Check prompt echo, output count, `length` finish, usage, expected cache hits, and
complete streams for a paired comparison. Retain both engines' output IDs for
diagnostics. Cross-engine token equality is diagnostic because KV precision
differs. Compare native MTP directly with native greedy on the same fixture and
precision. Compare native warm and cold prefix output IDs within the native engine.

## Reproduce on the baseline host

Run from the repository root with the pinned environment, checkpoint, and
concurrent client and reference. The launcher reads `fixtures/`, `cache/`,
`concurrent_observer.py`, `vllm_observer.py`, and `wheel.sha256` from the prior-run
directory below. A fresh W3–W6 run can regenerate the fixtures and warm a cache.
Restore `concurrent_observer.py` from a separately retained copy of the W3–W6
capture sources. The streaming observer and wheel manifest have tracked copies.
Repeating the original cache setup requires the original warmed W3–W6 cache. See the
[concurrent settings](../concurrent/README.md#conditions-and-gates) for the shared
engine and wire controls. Reserve with an empty compute list and an exclusive lock
on the archived physical GPU UUID. The launcher records the full server argv,
effective environment, device telemetry, and process logs.

```sh
runtime=/data1/shared/dongwu.chen/environments/vllm-81a8cb063e62
python="$runtime/venv/bin/python"
prior=artifacts/concurrent-baselines/gpu2.20261118T1319Z
checkpoint=/data0/shared/Qwen3.8-27B-FP8
mkdir -p artifacts/variant-baselines
run=$(mktemp -d "$PWD/artifacts/variant-baselines/gpu2.XXXXXXXX")
"$python" -B -m benchmarks.variants.fixtures \
    "$prior/fixtures" "$checkpoint/tokenizer.json" "$run/fixtures"
date -u +%FT%TZ > "$run/fixtures-frozen-utc.txt"
bash benchmarks/variants/reserve.sh "$run" > "$run/reservation.log" 2>&1 &
```

After each row's `ready.txt` appears, obtain the live snapshot and verify its
settings and page capacity before capture. The first ready row is `W3-prefix`,
and its release starts `W3-MTP`. Preserve `startup-failed.txt` and the server log
if a row fails to become ready. If capture fails, retain the original outcome and
release the row.

```sh
url=$(cat "$run/$row/ready.txt")
curl -fsS --max-time 20 "$url/collective_rpc" \
    -H 'Content-Type: application/json' -d '{"method":"variant_snapshot"}' \
    > "$run/$row/worker-before.json"
"$python" -B -c \
    'from benchmarks.variants.verify import verify_worker; from pathlib import Path; import sys
verify_worker(Path(sys.argv[1]), sys.argv[2])' "$run" "$row"
"$python" -B -m benchmarks.variants.client "$row" "$url" \
    "$run/fixtures" "$run/$row/capture" > "$run/$row/capture.log"
date -u +%FT%TZ > "$run/$row/release.txt"
"$python" -B -m benchmarks.variants.verify "$run" "$row" > "$run/$row/frozen.json"
```

## Evidence and comparison

The client uses two warmup and five measured cohorts, plus one diagnostic cohort
with separate four-step prefill and decode profiles. It uses the concurrent
barrier, timeouts, overlap requirements, and monotonic-clock definitions. The
first prefix profile shows the two GDN align kernels and replays exactly 1,792
input IDs per lane. Live page geometry makes the cached state column 3 and the
new state column 5. The pinned precopy kernel copies the state when those
columns differ. Prefill verifies FlashInfer FA2 BF16 Q/E4M3 K/V and Blackwell
chunked GDN. Decode verifies TRTLLM-Gen GQA and the BF16/FP32 GDN route for
each mode. MTP also checks the draft gather and argmax kernels, target rejection
sampling kernel, and four full-batch steps at two positions per request.

The row `frozen.json` records full-precision gates, profiles, counters, per-metric
samples, and the output IDs digest. The same 1% or two-sample-SD rule as the
concurrent benchmark applies. Both throughput ratios must strictly exceed
1.01, and every p95 must meet its row's ceiling.

| Row | Median IDs/s must exceed | p95 TTFT ceiling, ms | TPOT ceiling, ms | E2E ceiling, ms |
| --- | ---: | ---: | ---: | ---: |
| W3-prefix | 421.838530 | 564.269662 | 9.682372 | 9,906.363240 |
| W3-MTP | 616.629012 | 2,368.831773 | 7.003080 | 6,776.376327 |

Checkpoint cleanup leaves the tracked `comparison.json` and fixture generator.
Raw replay requires a separately retained copy of the original
`gpu2.lHKJfOaM-evidence.tar.gz`. Extract it, verify `files.sha256` from the archive
root, and invoke `python -B -m benchmarks.variants.verify "$run" "$row" --check`
from the repository root for each row. Use the full-precision gates in
`comparison.json` for paired evaluation.
