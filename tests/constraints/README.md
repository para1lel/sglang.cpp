# Native output constraints

Build `sglang-server` in Release mode. Run the commands from the repository
root inside the project's Pixi environment. Set `build` to the build output
directory, `checkpoint` to the Qwen3.8-27B-FP8 checkpoint directory and
`evidence` to a directory for raw results.

```bash
tests=tests/constraints
toolkit="$CONDA_PREFIX/targets/x86_64-linux"
nccl=$(python -I -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/nccl")')
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude "$tests/token_probe.cpp" \
    "$build/lib/libsglang_tokenizer.a" "$build/lib/libsglang_json.a" \
    -o "$build/bin/token-probe"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    -I"$toolkit/include" "$tests/mtp_guard.cpp" \
    -Wl,--start-group "$build/lib/"*.a -Wl,--end-group \
    -L"$toolkit/lib" -L"$toolkit/lib/stubs" -Wl,-rpath,"$toolkit/lib" -Wl,-rpath,"$nccl/lib" \
    "$nccl/lib/libnccl.so.2" -lcuda -lcudart -pthread -ldl -lrt \
    -o "$build/bin/mtp-guard"
"$build/bin/mtp-guard" "$checkpoint/tokenizer.json"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    -I"$toolkit/include" "$tests/mtp_native.cpp" \
    -Wl,--start-group "$build/lib/"*.a -Wl,--end-group \
    -L"$toolkit/lib" -L"$toolkit/lib/stubs" -Wl,-rpath,"$toolkit/lib" -Wl,-rpath,"$nccl/lib" \
    "$nccl/lib/libnccl.so.2" -lcuda -lcudart -pthread -ldl -lrt \
    -o "$build/bin/mtp-native"
"$build/bin/mtp-native" "$checkpoint"
python "$tests/token_oracle.py" "$build/bin/token-probe" \
    "$checkpoint/tokenizer.json" "$evidence/masks"
```

The CPU oracle computes eligible checkpoint token IDs from literal byte
languages and an independent minimum-token segmentation. It checks alternate
segmentations, split UTF-8, overlapping repeated choices, budget and EOS,
unsupported patterns, and agreement of the dense mask with the ID list.

The HTTP checks preserve raw JSON responses and event streams. Select an idle
GPU before starting the server.
Set `LMCACHE_SOCKET` to an isolated, ready [LMCache daemon on the same GPU](../../docs/serving/lmcache.md#start-the-daemon) for the cache case.

```bash
server="$build/bin/sglang-server"
python "$tests/http_contract.py" "$server" "$checkpoint" "$evidence/w1"
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

The native probe compares serial and batched token IDs with depth-zero controls.
Its counters cover full, partial, and zero proposal acceptance, replay, terminal
drafts, and a batch with both positive-depth and depth-zero lanes.

The HTTP integration requires `jsonschema` 4.26.0. Each cache-off positive-depth run
starts a matching depth-zero server and compares the full token ID lists and finish reasons.
The trace shows draft and verification
activity for finite-choice EBNF, regular expressions, and JSON Schema on both
routes. At capacity four, the integration check also verifies joint verification
and retirement after mixed requests.

Append `--media-body "$media_request" --media-sha256 "$media_hash"` to a
capacity-one command. The check sends the valid image request before and after
it verifies that media requests with constraints are rejected. The body must be
a valid, unconstrained chat request.
