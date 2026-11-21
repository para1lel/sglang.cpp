# sglang.cpp

sglang.cpp is a native C++ and CUDA inference engine in development for
**Qwen3.8-27B-FP8 on one NVIDIA B200**. It provides a loopback HTTP server,
a token-ID generator, and a C++ decoder.

## Build on the project host

Use the Linux x86-64 project host and the local checkpoint at
`/data0/shared/Qwen3.8-27B-FP8`. The checkpoint includes the model weights,
tokenizer, and chat template. The default Pixi environment provides GCC 15
and CUDA 13.4. The toolchain script verifies XMake 3.1.2.

From the repository root, run the following commands in Bash.

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

The executables are `./build/bin/sglang-server` and `./build/bin/sglang-generate`.
Use `host ./build/bin/sglang-generate --help` for token-ID generation options.

## Serve token IDs over HTTP

In the same shell, select the UUID of your reserved B200 and start the server.

```bash
nvidia-smi -L
read -r -p 'Reserved B200 UUID: ' GPU_UUID
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    host ./build/bin/sglang-server --device 0 --port 17384
```

When the server listens on `127.0.0.1:17384`, send a chat request from another shell.
The example streams assistant output as server-sent events and ends with `[DONE]`.

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

## API and limits

- `GET /v1/models` lists the served model. See [model discovery](docs/http-server.md#model-discovery).
- [Chat completions](docs/chat-serving.md) accept `messages` at `POST /v1/chat/completions`.
  The guide covers images, video, reasoning, and tool calls.
- [Text completions](docs/text-serving.md) accept a UTF-8 `prompt` at `POST /v1/completions`.
  Use model `benchmark` on the same endpoint for the [token-ID API](benchmarks/streaming/README.md#wire-contract).
- Chat and text return JSON or server-sent events. See [output constraints](docs/output-constraints.md)
  for regular expressions, literal choices, and the supported JSON Schema subset.
- The prompt and output budget share a 262,144-token context. Available GPU memory can
  reduce the admitted capacity. The server accepts up to 32 active requests by default.
- `GET /ready` reports worker readiness. `GET /health` reports liveness and scheduler limits.
  See the [HTTP server guide](docs/http-server.md) for admission limits and startup options.

See the [measured vLLM comparison](docs/performance-comparison.md) for results
and the [benchmark policy](docs/performance.md#scheduler-and-reporting-policy)
for the comparison contract and acceptance rules.
