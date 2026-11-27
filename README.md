# sglang.cpp

sglang.cpp is a native C++ and CUDA inference engine for **Qwen3.8-27B-FP8**.
It provides a loopback HTTP server on one or two NVIDIA B200 GPUs, a token-ID
generator, and a C++ decoder.

## Build on the project host

On the Linux x86-64 project host, use the checkpoint at
`/data0/shared/Qwen3.8-27B-FP8` and the locked Pixi environment with GCC 15 and
CUDA 13.4. Run these commands from the repository root in Bash.

```bash
export PATH="/data1/shared/dongwu.chen/bin:$PATH"
host() {
    pixi run --locked --manifest-path /data1/shared/dongwu.chen/pixi.toml \
        --environment default "$@"
}

bash tools/toolchain-xmake.sh --install
host xmake f -P . -y -p linux -a x86_64 -m release -o build
host xmake build -P . -y sglang-server
host xmake build -P . -y sglang-generate
```

Use `host ./build/bin/sglang-generate --help` for token-ID generation and
`host ./build/bin/sglang-server --help` for serving options.

<a id="serve-token-ids-over-http"></a>

## Serve chat completions

Select the UUID of your reserved B200 and start the server in the same shell.

```bash
nvidia-smi -L
read -r -p 'Reserved B200 UUID: ' GPU_UUID
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    host ./build/bin/sglang-server
```

The server listens on `127.0.0.1:17384`. Send a streaming request from another shell.

```bash
curl --fail-with-body -N http://127.0.0.1:17384/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary '{
        "model": "Qwen3.8-27B-FP8",
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 64,
        "stream": true
    }'
```

To serve on two reserved B200s, set `GPU_UUIDS` to their comma-separated UUIDs in
rank order and run `CUDA_VISIBLE_DEVICES="$GPU_UUIDS" host ./build/bin/sglang-server --tp 2`.
Use `GET /server_info` to verify the effective configuration. [Server information](docs/native-model-info.md#server-information)

LMCache prefix reuse is enabled by default. Start one node-local daemon with
visibility of every GPU serving the model. [LMCache setup](docs/serving/lmcache.md)

## API and performance

- The default build accepts text and decoded RGB8 images and video. Inline PNG
  and MP4 require the optional libpng and FFmpeg build. [Chat API](docs/chat-serving.md)
- Text completions accept UTF-8 prompts and expose the benchmark token-ID API.
  [Text API](docs/text-serving.md) · [Token-ID wire contract](benchmarks/streaming/README.md#wire-contract)
- See the HTTP guide for discovery, readiness, and admission limits. [HTTP server](docs/http-server.md)
- The performance gate requires native LMCache makespan at or below 1.10× pinned
  vLLM with KV offload disabled on every W1–W5 workload at each TP width. Run the
  TP2 driver’s separate route qualification before timing. [Benchmark protocol](docs/performance.md) · [TP2 runbook and results](docs/serving/tp2.md)
- Find component responsibilities and development commands in the [development guide](docs/development.md).
