# Prefix cache checks

## CPU reader queue

Run from the repository root in the managed Pixi default environment. The test drives the
production HTTP reader and completion queue over loopback. It uses a decoder-service double
and explicit barriers for final responses and reset acknowledgement. It needs no GPU or weights.

```bash
out="$PWD/build/prefix-readers"
mkdir -p "$out"
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -pthread -ffunction-sections -fdata-sections \
    -Iinclude tests/prefix/reader_queue.cpp \
    apps/server/{prefix_controls,flush_cache,http,loads,completions,ollama_discovery}.cpp \
    apps/server/{parse_function_call,native_sessions,responses,native_generate,ollama_generate}.cpp \
    apps/server/{text_request,text_completion,tokenization_validation,tool_literal,tool_union}.cpp \
    src/runtime/model/{qwen_chat_output,qwen_tool_output,qwen_chat_template}.cpp \
    src/tokenizer/*.cpp src/{json,utf8}.cpp \
    -Wl,--gc-sections \
    -Wl,--wrap=accept4 -o "$out/reader-queue"
for mode in --cache-off --cache-on --reader-failure --worker-failure --flush; do
    timeout 20s "$out/reader-queue" "$mode"
done
```

Each mode prints `HTTP reader queue passed` on success. The cache modes check health and
readiness during partial header and body reads, replacement of the oldest stalled reader, and
completion queue overflow, cancellation, FIFO order, and slot reuse. Fully buffered 48 KiB
requests reach idle workers or wait in the saturated queue. Requests beyond the queue capacity
receive HTTP 503. An incomplete header and body both expire on the original read deadline even
when the clients send more bytes. A held final response and a blocked write to a client that stops
reading make reset return HTTP 409 until the completion releases its slot. Health and readiness
respond while reset acknowledgement is held, and the queued completion starts after reset.
Shutdown closes active, queued, and partial connections. The failure modes inject a fatal accept
error and an unavailable decoder worker, and check error propagation, readiness, and descriptor
cleanup.

The flush mode holds a completion at its final response. It checks ordered timeout budgets,
busy replies, reader responsiveness, and separate reset acknowledgements after the lane is idle.

## LMCache route checks

Use the [real-daemon route procedure](../../docs/prefix-cache.md#verification) with a Release server and pinned checkpoint. It checks selective GQA and GDN coverage, shifted draft continuation, reuse across worker restarts, cancellation, and peer progress during held requests.

## Device state and graph controls

Use the managed Pixi default environment, current Release archives, pinned checkpoint, and a reserved B200. Start the [LMCache daemon](../../docs/serving/lmcache.md#start-the-daemon) on the same GPU with namespace `sglang`. Pass its socket and a fresh evidence directory.

```bash
GPU_UUID="$GPU_UUID" LMCACHE_SOCKET="$LMCACHE_SOCKET" \
    bash tests/prefix/run_lmcache_state.sh "$out" "$checkpoint" "$evidence"
```

Require `PASS` in both `state.log` and `graph.log`. The state check compares payloads and scores, counts KV writes with imported GQA, and covers unequal GQA and GDN coverage, a hole in GQA, and K4 continuation. The graph check advances a K0 peer while the checkpoint transfer is held. It compares the checkpoint state and continuation scores with the control after the copy completes. [Device runner](run_lmcache_state.sh#L4)<!-- prefix: "build=$(real" -->

## Historical resident-cache checks

Use the following procedures with the source, build, and runners from their linked revisions. The fixtures exercise the former GPU-resident implementation.

- [CUDA state, remapped pages, and grammar routes](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/tests/prefix/README.md#cuda-state-and-routes)
- [Committed target state and periodic checkpoints](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/tests/prefix/README.md#committed-target-state)
- [Tensor Core cache parity](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/tests/prefix/README.md#historical-tensor-core-cache-parity)
- [Resident prefix cache with MTP](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/tests/prefix/README.md#prefix-cache-with-mtp)
