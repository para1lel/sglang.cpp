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
to disable draft generation. On a default server launch, requests for logprobs use zero
draft steps. An explicit positive server depth rejects requests for logprobs.

On B200, text prefill uses Tensor Cores when prefix caching is off.
Prefix caching, media, and label scoring use exact prefill.
See [launch migration](docs/native-model-info.md#migrate-old-launch-commands)
if you use older launch options.

The HTTP server enables CUDA Graph replay for eligible decode steps by default.
`--batch-cuda-graphs` remains accepted. Pass `--eager` to disable graph replay,
even when `--batch-cuda-graphs` is also present. Prefix caching requires omitting `--eager`.

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
the resolved settings.

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
- The scheduler defaults to 8,192 tokens per step across requests. Set
  `--max-num-batched-tokens` to a value from 1 to 32,768 to override it.
- `GET /ready` reports worker readiness. `GET /health` reports liveness and scheduler limits.
  See the [HTTP server guide](docs/http-server.md) for admission limits and startup options.

The formal K4 native-default comparison used native revision `94e8bdca73ce2017e501b6b8b6abda9156ce0c89`.
Native median client makespan was 15.7% lower on W1 and 19.2% lower on W2 than pinned vLLM.
Output token IDs repeated within each engine and differed across engines in all nine timed pairs on each workload.
W3–W6 remain unmeasured under the formal protocol. See the
[K4 native-default results](docs/serving/paired-native-defaults-results.md) and
[benchmark policy](docs/performance.md#scheduler-and-reporting-policy).
