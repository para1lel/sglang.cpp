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

Both executables default to four MTP draft steps. Pass `--mtp-steps 0` or `--mtp-k 0`
to disable draft generation. Eligible greedy requests for per-token logprobs use the
configured depth. Active output penalties, logit bias, and minimum-token constraints
use target-only decoding. See [token logprobs](docs/serving/openai-logprobs.md#request-requirements).

On B200, prefill selects projection and attention kernels from the request shape.
Uncached text rows use the same policy whether prefix caching is enabled or disabled.
See [launch migration](docs/native-model-info.md#migrate-old-launch-commands)
if you use older launch options.

The HTTP server enables CUDA Graph replay for eligible decode steps by default.
Pass `--eager` to disable graph replay, including when `--batch-cuda-graphs` is present.
The HTTP server attempts [LMCache prefix reuse](docs/prefix-cache.md#runtime-path) by default.
Basic generation works without the daemon. To reuse prefixes across requests, start the
[LMCache 0.5.8 daemon](docs/serving/lmcache.md#start-the-daemon) before starting the server.
Run both as the same Linux user with the same physical GPU UUID. Both use `$XDG_RUNTIME_DIR/sglang-lmcache.sock` when `XDG_RUNTIME_DIR`
is absolute, or `/tmp/sglang-lmcache-<euid>.sock` otherwise. For a custom socket,
pass `--socket /absolute/path.sock` to the daemon and `--lmcache-socket /absolute/path.sock`
to the server. Pass `--prefix-cache-entries 0` to disable reuse.

## Serve token IDs over HTTP

In the same shell, select the UUID of your reserved B200 and start the server.

```bash
nvidia-smi -L
read -r -p 'Reserved B200 UUID: ' GPU_UUID
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved B200 UUID}" \
    host ./build/bin/sglang-server
```

When the server listens on `127.0.0.1:17384`, send a chat request from another shell.
The example streams assistant output as server-sent events and ends with `[DONE]`.
Use [`GET /server_info`](docs/native-model-info.md#server-information) to verify
the resolved settings. `lmcache_enabled` reports configuration and `lmcache_available`
reports whether the server can reach the daemon.

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
- [Chat completions](docs/chat-serving.md) accept `messages` at `POST /v1/chat/completions`,
  with image and video inputs available on the default launch. The vision encoder loads
  on its first use. The guide also covers reasoning and tool calls.
- [Label scoring](docs/serving/pointwise-score.md) is available at `POST /v1/score`.
- [Text completions](docs/text-serving.md) accept a UTF-8 `prompt` at `POST /v1/completions`.
  Use model `benchmark` on the same endpoint for the [token-ID API](benchmarks/streaming/README.md#wire-contract).
- Chat and text return JSON or server-sent events. See [output constraints](docs/output-constraints.md)
  for regular expressions, literal choices, and the supported JSON Schema subset.
- The prompt and output budget share a 262,144-token context. Available GPU memory can
  reduce the admitted capacity. The server defaults to eight active requests per participating GPU.
- The scheduler defaults to 32,768 tokens per step across requests. Set
  `--max-num-batched-tokens` to a value from 1 to 32,768 to override it.
- `GET /ready` reports worker readiness. `GET /health` reports liveness and scheduler limits.
  See the [HTTP server guide](docs/http-server.md) for admission limits and startup options.

## Development and performance

- Find the current component responsibilities in the [native architecture guide](docs/development.md#current-architecture-and-scope).
- Find the benchmark results and archived run records in [Matched K4 evidence](docs/performance.md#matched-k4-evidence).
