# W1 image and video reference

The pinned vLLM baseline passed both workloads on November 18, 2026.
Each workload used 128 effective input positions, 128 output tokens, and one active request.
All three warmups and nine measured requests per workload returned identical IDs, `length`,
zero cached prompt tokens, usage of 128/128/256, and `[DONE]`.
Use the vLLM IDs for historical diagnostics. Verify native greedy selection and numerical
correctness under the candidate's own precision.

[The reference](gpu0-reference.json#L1)<!-- prefix: "{" --> records the exact vLLM output IDs,
statistics, and unrounded thresholds.
[The timing rows](gpu0-timings.json#L1)<!-- prefix: "{" --> preserve the 18 measured samples
as three nanosecond offsets from client submission. The offsets mark the first output ID,
the last output ID, and `[DONE]`.

## Frozen inputs

| Workload | Case | Exact HTTP body | Decoded input | Grid and visual tokens |
| --- | --- | --- | --- | --- |
| W1-image | [Case](W1-image.case.json#L1)<!-- prefix: "{" --> | [Request](W1-image.http-body) | PNG, 256 × 256 RGB, 719 bytes | `[1,16,16]`, 64 |
| W1-video | [Case](W1-video.case.json#L1)<!-- prefix: "{" --> | [Request](W1-video.http-body) | MPEG-4 MP4, eight 128 × 128 RGB frames at 4 fps, 3,360 bytes | `[2,8,8]`, 32 |

Each request contains the complete original media file in its base64 data URL.
Send the `.http-body` bytes as-is with `Content-Type: application/json`.
The case records byte counts, request and media SHA-256 digests, and exact processor-expanded IDs.
Join each segment array without a separator to recover the original string.
The checkpoint chat template uses one user message, with thinking disabled.
The frozen text extends the natural image and video prompts from 91 and 81 positions to 128.
The OpenCV loader selects frames `[0,2,4,7]` at `[0,0.5,1,1.75]` seconds.
The processor uses these frames without resampling. The temporal-pair centers are `[0.25,1.375]`.

## Run identity

The timed window was 14:51:51–14:52:21 UTC on GPU0, an NVIDIA B200 with driver `590.48.01`.
The device UUID was `GPU-40f23920-6ba0-cdc4-e7e7-64b6cb470b32`.
The device had no active compute or resident memory before launch and after shutdown.

| Component | Identity |
| --- | --- |
| Project source | `98b5b35916b7fcadd36859c30abf069e03def377` |
| vLLM source and wheel | `81a8cb063e6247fe254d7caf7e98a6691d71022a`, `0.34.0rc1.dev787+g81a8cb063` |
| GPU runtime | Torch `2.14.0+cu132`, CUDA `13.2`, FlashInfer `0.7.4` |
| Media preprocessing | Transformers `5.23.0`, OpenCV `5.1.0.95`, `opencv` loader |
| Checkpoint | Qwen3.8-27B-FP8, manifest SHA-256 `b54fe44144f1701a9369e8b912997ee5cab77cd176c8f97ad3fe63e78cf3c303` |

The worker reported BF16 model activations, a 1 GiB KV cache, and `FULL_AND_PIECEWISE`
CUDA graphs. All 16 GQA layers used FlashInfer with E4M3 KV, BF16 Q, and unit Q/K/V
scales in FP32 on the device. GQA projections used E4M3 weights. GDN used BF16
convolution and FP32 recurrence.

An independent renderer check matched both uint8 patch arrays against offline preprocessing.
Each separate three-iteration profile had one fused input normalization, 27 BF16 SM100
vision attention calls, 16 FA2 GQA prefill calls, and 32 TRTLLM-Gen GQA decode calls.
Each profile also had 48 chunked GDN prefill calls and 96 packed recurrent decode calls.
The visual embeddings had shapes `[64,5120]` and `[32,5120]`. The language MRoPE deltas
were −56 and −24. Both profiled streams reproduced the corresponding vLLM output IDs.
Only the warmed, uninstrumented requests contribute to the timing reference.

## Verify offline

Run from the repository root with Python 3.10 or newer:

```bash
python -B -m benchmarks.media.collect verify
```

The verifier checks the exact request bytes, embedded media hashes, and input ID hashes.
It recomputes the reference statistics and thresholds from the compact timing rows.

## Reproduce the measurement

Use the [pinned vLLM environment](../vllm/README.md#environment) and a matching local checkpoint.
Provide absolute paths for the paths in this command. Use the existing device-specific lock.
Set `CACHE_SEED` to a matching provisioned cache to reproduce the seeded launch.
Set `LD_LIBRARY_PATH` to the required host CUDA compatibility library directory.
Run from the repository root.

```bash
export VLLM_ENV CHECKPOINT GPU_UUID GPU_LOCK_FILE RUN_DIR
bash benchmarks/media/serve.sh
```

The launcher verifies the wheel, installed native binaries, and checkpoint before serving.
Use the same physical GPU UUID when pairing measurements.
After `/health` succeeds, run the collector from a second terminal:

```bash
"$VLLM_ENV/venv/bin/python" -B -m benchmarks.media.collect measure \
    --url http://127.0.0.1:17462 --output "$RUN_DIR/measurement"
```

Stop the foreground server with Ctrl-C after collection. Keep new raw streams in the run directory.
To audit a new collection, run the same command with `replay` in place of `measure`.

Both requests use greedy decoding, seed 0, an output budget of 128, and EOS ignored.
Prefix reuse, processor caching, and speculative decoding are disabled.
The collector verifies the worker state and two `/tokenize` responses per case before timing.
It sends three image-then-video warmup cohorts and nine measured cohorts in the same order.
Each request uses a new loopback connection. Every attempted stream enters validation.

TTFT starts at client submission and ends on the first output ID.
TPOT divides first-to-last output time by 127.
End-to-end latency ends on `[DONE]`. Throughput divides 128 by end-to-end seconds.
The calculation uses inclusive linear percentiles and sample standard deviation.
The candidate median throughput ratio must exceed `1 + max(0.01, 2 × stdev / median)`.
Each candidate p95 latency must meet the ceiling `baseline p95 + max(0.01 × baseline p95, 2 × stdev)`.
Use the fixed limits in the reference for comparison with this historical run.
