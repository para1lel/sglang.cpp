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
    apps/server/detail/{batch_state,text_stop}.cpp \
    apps/server/qwen/{qwen_chat_output,qwen_tool_output,qwen_tool_syntax,qwen_chat_template}.cpp \
    src/runtime/server/batch_executor.cpp src/runtime/execution/decoder_options.cpp \
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

Use a Release server, the frozen W3 fixtures, and a private LMCache 0.5.8 daemon with at least 4 GiB on the same GPUs. Set `TP` to `1` or `2` and pass a fresh evidence directory. Add `--alternate-checkpoint` with a compatible checkpoint containing different weights to check weight isolation.

```bash
PYTHONPATH="$PWD" python3 -B tests/prefix/lmcache_routes.py \
    "$SGLANG_SERVER" "$CHECKPOINT" "$W3_FIXTURES" "$LMCACHE_SOCKET" "$EVIDENCE" \
    --tp "$TP"
```

Require an empty `failures` object in `summary.json`. The fixture checks resident repeats, suffix forks, cache salt isolation, selective misses across cache families, failed GET recovery, and CPU-backed reuse after the writer exits. TP1 also checks image identity. TP2 checks that the chosen recurrent checkpoint is present on both ranks when their deeper endpoints differ. Keep `relay.json` and the metric scrapes as evidence of the CPU backups and reloads induced by ordinary pool pressure.

The [pool verification procedure](../../docs/prefix-cache.md#verification) covers backup overlap with an active peer, cancellation, reset, and anchor retention. The [archived direct-STORE procedure](https://github.com/para1lel/sglang.cpp/blob/bdaec36dbb985edd57e54b3565315ba321290a23/docs/prefix-cache.md#verification) contains the held IPC, eager publication, aligned terminal STORE order, and wide eager checkpoint cases for that release.

## Device state and graph controls

Use the managed Pixi default environment, pinned checkpoint, and a reserved B200. Set `release_build` to the Release build directory containing the native archives under `lib/`. Start the [LMCache daemon](../../docs/serving/lmcache.md#start-the-daemon) on the same GPU with namespace `sglang`. Pass its socket and a fresh evidence directory.

```bash
GPU_UUID="$GPU_UUID" LMCACHE_SOCKET="$LMCACHE_SOCKET" \
    bash tests/prefix/run_lmcache_state.sh "${release_build:?set the Release build directory}" "$checkpoint" "$evidence"
```

Require exit zero and `PASS` in `state.log`, `graph.log`, and `release.log`. The runner bounds each binary, including its STORE worker and receipt waits. `LMCACHE_TEST_TIMEOUT` defaults to `5m`. At the deadline GNU `timeout` sends SIGTERM, then SIGKILL if the process survives another five seconds. A timeout fails the run. [Device runner](run_lmcache_state.sh#L57)<!-- prefix: "CUDA_VISIBLE" -->

The state check compares payloads and scores, counts KV writes with imported GQA, and covers unequal GQA and GDN coverage, a hole in GQA, and K4 continuation. It checks nonterminal views across allocation growth and a failed store. Terminal views retain their source blocks after retirement without copying them. The fixture queues a nonterminal copy before cancelling and replacing its source request, and injects allocation, copy, and event failures. [Nonterminal view](lmcache_state.cpp#L541)<!-- prefix: "void check_d" --> · [Terminal view](lmcache_state.cpp#L566)<!-- prefix: "void check_t" --> · [Delayed copy](lmcache_state.cpp#L616)<!-- prefix: "static void " --> · [Copy failure](lmcache_state.cpp#L669)<!-- prefix: "void check_c" -->

The state test checks that the STORE worker returns its GDN slice before acknowledging completion. A second case retains an alias until after acknowledgement and checks that the slice becomes available after the alias releases it.

The graph check advances a K0 or K1 peer while a checkpoint transfer is held, then compares native state and continuation scores with the control. For direct K1 generation, it checks a restored hit and a cold restart after an exceptional GET. Both must match cache-off output IDs and produce a deferred terminal STORE receipt. A failed STORE must surface through its receipt and leave the successful-store count at zero.

The release check holds a detached arena lease across an idle drain and a competing allocation. It injects event record and query errors and verifies that a block becomes available only after completion is confirmed.

The release probe also holds the pressure worker at IPC export, before transport counts the STORE. It checks that the residency manager reports pending during the hold and clears it after the injected export failure and reset.

### TP2 release isolation

Reserve two B200 GPUs and set `GPU_UUIDS` to their comma-separated UUIDs in rank order. Set `release_evidence` to a fresh directory path. This mode runs the release checks without an LMCache daemon. [Runner mode](run_lmcache_state.sh#L24)<!-- prefix: "if [[ $mode " -->

```bash
GPU_UUID="$GPU_UUIDS" bash tests/prefix/run_lmcache_state.sh \
    "${release_build:?set the Release build directory}" "$checkpoint" "$release_evidence" \
    --serving-release-only
```

Require exit zero and `PASS` in `release.log`. The fixture holds simultaneous detached leases on both ranks and verifies that releasing rank 1 leaves rank 0 charged. Each rank also exercises the event failure checks.

## Historical resident-cache checks

Use the following procedures with the source, build, and runners from their linked revisions. The fixtures exercise the former GPU-resident implementation.

- [CUDA state, remapped pages, and grammar routes](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/tests/prefix/README.md#cuda-state-and-routes)
- [Committed target state and periodic checkpoints](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/tests/prefix/README.md#committed-target-state)
- [Tensor Core cache parity](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/tests/prefix/README.md#historical-tensor-core-cache-parity)
- [Resident prefix cache with MTP](https://github.com/para1lel/sglang.cpp/blob/a7583f7b6346b5eb91aba2cba4748a6461e60fb4/tests/prefix/README.md#prefix-cache-with-mtp)
