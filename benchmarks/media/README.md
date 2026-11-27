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

The November 18 capture ran on GPU0, an NVIDIA B200. Only warmed, uninstrumented requests contribute timing samples. The [frozen run identity](https://github.com/para1lel/sglang.cpp/blob/aab1101e145507c4c40435602e3a2d3c5d88111d/benchmarks/media/README.md#run-identity) records the device, runtime, checkpoint and separate worker and renderer profiles.

## Verify offline

Run from the repository root with Python 3.10 or newer:

```bash
python -B -m benchmarks.media.collect verify
```

The verifier checks the exact request bytes, embedded media hashes, and input ID hashes.
It recomputes the reference statistics and thresholds from the compact timing rows.

Each case requires median tokens per second and its ratio to the reference median to exceed their respective unrounded thresholds. The p95 TTFT, TPOT and end-to-end latencies must each be at or below their stored ceiling. [W1-image cutoffs](gpu0-reference.json#L170)<!-- prefix: "\"thresholds\"" --> · [W1-video cutoffs](gpu0-reference.json#L346)<!-- prefix: "\"thresholds\"" -->

For native codec parity with the same bodies, run the [optional codec test](../../tests/media/http-contract.md#optional-codecs) on a reserved B200. It checks decoded RGB8, compressed requests and eager decode across JSON and SSE.

## Reproduce the measurement

The [November 18 procedure](https://github.com/para1lel/sglang.cpp/blob/e681f90867aa9c59bb0c11bf0e796557f36ce088/benchmarks/media/README.md#reproduce-the-measurement) runs three warmups and nine timed requests per case. Its replay validates saved streams and recomputes statistics and cutoffs. The [later single-round procedure](https://github.com/para1lel/sglang.cpp/blob/aab1101e145507c4c40435602e3a2d3c5d88111d/benchmarks/media/README.md#reproduce-the-measurement) runs one warmup and one timed request per case and can replay saved streams in the original format.
