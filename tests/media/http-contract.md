# Decoded media HTTP contract

Build the default release server in the project Pixi environment. Run on a reserved SM100
GPU with the pinned checkpoint and a fresh evidence directory.

```bash
checkpoint=/data0/shared/Qwen3.8-27B-FP8
python tests/media/http_routes.py build/bin/sglang-server "$checkpoint" /tmp/media-http
```

The test creates small RGB8 frames in memory. It sends a mixed image, video and text request
through JSON and SSE, and compares prompt IDs, token IDs, usage and finish reasons. It repeats
the media request with MTP enabled for ordinary text to check that media stays on K0.

Invalid dimensions, RGB8 and video metadata, unsupported options and compressed inputs return
HTTP 400. The test sends malformed media after a valid attachment and checks that none of the
rejected requests or the zero-budget request starts decode. It resets a live streaming socket
and a pending connection, then checks that the surviving request returns identical IDs. Plain
chat and text completions remain usable. Continuous and cache-enabled servers also serve
media, with zero cached media tokens.

Append `--ordered-fp8` and use a fresh evidence directory to test each ordered text policy.
The suite compares real image and video outputs and pointwise label scores with exact.
It also checks that the combined policy changes the frozen W1 text continuation and that
concurrent media, text, and score responses match their serial controls. For the serial C++
contract, run `media-contract "$checkpoint" --ordered-fp8` to compare BF16-derived top scores
for mixed features across prefill chunks.

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
