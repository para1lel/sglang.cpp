# Decoded media HTTP contract

Build the default release server in the project Pixi environment. Run on a reserved SM100
GPU with the pinned checkpoint and a fresh evidence directory.

```bash
checkpoint=/data0/shared/Qwen3.8-27B-FP8
python tests/media/http_routes.py build/bin/sglang-server "$checkpoint" /tmp/media-http
```

The test creates small RGB8 frames in memory. It sends a mixed image, video and text request
through JSON and SSE, and compares prompt IDs, token IDs, usage and finish reasons. It repeats
the media request with K1 and checks identical output against K0.

Invalid dimensions, RGB8 and video metadata, unsupported options and compressed inputs return
HTTP 400. The test sends malformed media after a valid attachment and checks that none of the
rejected requests or the zero-budget request starts decode. It resets a live streaming socket
and a pending connection, then checks that the surviving request returns identical IDs. Plain
chat and text completions remain usable.

Append `--mixed-service` to compare image, video, text, and pointwise label results between
isolated and concurrent requests with the cache explicitly disabled.

Append `--feature-complete` on B200 to compare default serving and explicit cache-off
controls. The test verifies K4 and the 8192-row budget, cold text, exact and partial hits,
ordered label scores, and exact prompt/output logprobs against cache-off with matched
prefill grants. The test compares decoded image and video JSON/SSE, checks exact reuse of
non-page-aligned media prompts, and rejects stale state after changes to pixels, grids,
video positions, suffixes, or binary cache salts. It then cancels a live cache-on stream
and checks a warm replacement. The evidence directory contains the requests and responses.

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
