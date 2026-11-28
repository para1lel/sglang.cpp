# Native output constraints

From the repository root in managed Pixi, set `build` to the Release server's build directory, `checkpoint` to Qwen3.8-27B-FP8, and `evidence` to a fresh directory for results.

```bash
tests=tests/constraints
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
nccl=$(python -I -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl")')
host_flags=(-std=c++17 -O2 -Wall -Wextra -Werror -Iinclude)
decoder_lib=(
    -Wl,--start-group "$build/lib/"*.a -Wl,--end-group
    -L"$toolkit/lib" -L"$toolkit/lib/stubs" -Wl,-rpath,"$toolkit/lib" -Wl,-rpath,"$nccl/lib"
    "$nccl/lib/libnccl.so.2" -lcuda -lcudart -pthread -ldl -lrt
)
g++ "${host_flags[@]}" "$tests/token_probe.cpp" \
    "$build/lib/libsglang_tokenizer.a" "$build/lib/libsglang_json.a" -o "$build/bin/token-probe"
g++ "${host_flags[@]}" -I"$toolkit/include" "$tests/mtp_guard.cpp" \
    "${decoder_lib[@]}" -o "$build/bin/mtp-guard"
"$build/bin/mtp-guard" "$checkpoint/tokenizer.json"
g++ "${host_flags[@]}" -I"$toolkit/include" "$tests/mtp_native.cpp" \
    "${decoder_lib[@]}" -o "$build/bin/mtp-native"
"$build/bin/mtp-native" "$checkpoint"
python "$tests/token_oracle.py" "$build/bin/token-probe" \
    "$checkpoint/tokenizer.json" "$evidence/masks"
```

The CPU oracle independently derives eligible IDs and minimum token counts for literal byte languages. Check alternate segmentations, split UTF-8, overlapping repeats, budget, EOS, unsupported patterns, and dense mask versus ID list.

The HTTP suite saves JSON and SSE. Select an idle GPU and set `LMCACHE_SOCKET` to an isolated, ready [daemon on that GPU](../../docs/serving/lmcache.md#start-the-daemon) for the cache case.

```bash
server="$build/bin/sglang-server"
python "$tests/http_contract.py" "$server" "$checkpoint" "$evidence/width-one"
python "$tests/http_contract.py" "$server" "$checkpoint" "$evidence/eager" --eager
python "$tests/http_contract.py" "$server" "$checkpoint" "$evidence/graph" \
    --width 4 --graph --mtp 1
for width in 1 4; do
    for depth in 1 2 3 4; do
        python "$tests/schema_integration.py" "$server" "$checkpoint" \
            "$evidence/w$width-k$depth" --width "$width" --graph --mtp "$depth"
    done
done
python "$tests/integration.py" "$server" "$checkpoint" "$evidence/prefix" \
    --width 4 --graph --lmcache-socket "$LMCACHE_SOCKET"
```

The native probe matches serial and batched IDs with depth zero. Counters cover full, partial, and zero acceptance, replay, terminal drafts, and mixed-depth lanes.

HTTP requires `jsonschema` 4.26.0. `http_contract.py --width 4 --graph --mtp 1` checks JSON, SSE, and verifier activity on one service. Each cache-off positive-depth `schema_integration.py` run matches token IDs and finish reasons with a depth-zero server. The schema runs check draft and verifier activity on both routes for finite-choice EBNF, regex, and JSON Schema. At width four, they also check joint verification and retirement after mixed requests.

For media rejection and recovery, append `--media-body "$media_request" --media-sha256 "$media_hash"` to a capacity-one command. The body must be a valid unconstrained chat request and must succeed before and after the rejected constrained media request.
