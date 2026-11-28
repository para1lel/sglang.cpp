# Decoded media HTTP contract

Build the Release server in the project Pixi environment. Reserve a B200 and export its
UUID as `CUDA_VISIBLE_DEVICES`. For the default and `--feature-complete` runs, start a
dedicated [LMCache daemon](../../docs/serving/lmcache.md#start-the-daemon) on that GPU
with namespace `sglang`. Set `LMCACHE_SOCKET` to its absolute socket path. Both runs
reset the daemon store. Give each run a new evidence directory under an existing parent.

```bash
checkpoint=/data0/shared/Qwen3.8-27B-FP8
python tests/media/http_routes.py build/bin/sglang-server "$checkpoint" /tmp/media-http \
    --lmcache-socket "$LMCACHE_SOCKET"
```

The test creates small RGB8 frames in memory. It sends a mixed image, video and text request
through JSON and SSE, and compares prompt IDs, token IDs, usage and finish reasons. It repeats
the media request with K1 and checks identical output against K0.

Invalid dimensions, RGB8 and video metadata, unsupported options and compressed inputs return
HTTP 400. The test sends malformed media after a valid attachment and checks that none of the
rejected requests or the zero-budget request starts decode. It resets a live streaming socket
and a pending connection, then checks that the surviving request returns identical IDs. Plain
chat and text completions remain usable.

Use `--mixed-service` with a fresh evidence directory to compare image, video, text,
and pointwise label results between isolated and concurrent requests. It runs cache-off
and needs no daemon.

Run `--feature-complete` against the dedicated daemon with a fresh evidence directory.

```bash
python tests/media/http_routes.py build/bin/sglang-server "$checkpoint" /tmp/media-feature \
    --lmcache-socket "$LMCACHE_SOCKET" --feature-complete
```

In the saved `/server_info` responses, require `prefix_cache_entries` to equal zero in
both modes. `lmcache_enabled` and `lmcache_available` must both be true for the daemon
worker and false for the explicit cache-off control. The test verifies K4, the
32,768-row budget, cold text, and checkpoint hits. Ordered label scores and
aligned prompt and output logprobs are compared with cache-off workers using
matched prefill grants. The unaligned score replay checks JSON and SSE against
the same resident GQA schedule with zero usable GDN rows.
Decoded image and video branches reuse the exact `b−1,024` checkpoint and return
identical IDs over JSON and SSE. Changing pixels, grid geometry, video positions,
or a binary cache salt prevents reuse.
The test also cancels a live media stream and checks that a replacement succeeds.
The evidence directory contains the requests and responses.

## Optional codecs

Build the default server and a second server with `--media-codec-prefix`. Use a pinned codec
prefix and follow the RGB8 decoding steps in [chat serving](../../docs/chat-serving.md).
Put the concatenated frame bytes for each case in `<rgb>/W1-{image,video}/input/frames.rgb8`.
The test verifies each frame against the frozen fixture before submission.

```bash
python tests/media/codec_routes.py "$default_server" "$codec_server" "$codec_prefix" \
  "$checkpoint" "$rgb" /tmp/media-codecs
```

The test records direct and resolved codec dependencies, the frozen PNG and MP4 requests,
and raw JSON and SSE responses. It checks codec-disabled and malformed requests, PNG bounds,
and exact native parity across decoded RGB8, compressed input and eager decode.
