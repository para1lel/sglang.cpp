# Native regex constraints

Build `sglang-server` in Release mode. Run the commands from the repository
root inside the project's Pixi environment. Set `build` to the build output
directory, `checkpoint` to the Qwen3.8-27B-FP8 checkpoint directory and
`evidence` to a directory for raw results.

```bash
tests=tests/constraints
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude "$tests/token_probe.cpp" \
    "$build/lib/libsglang_tokenizer.a" "$build/lib/libsglang_json.a" \
    -o "$build/bin/token-probe"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
    -I"$CONDA_PREFIX/targets/x86_64-linux/include" "$tests/mtp_guard.cpp" \
    -Wl,--start-group "$build/lib/"*.a -Wl,--end-group \
    -L"$CONDA_PREFIX/lib" -lcudart -pthread -ldl -lrt \
    -o "$build/bin/mtp-guard"
"$build/bin/mtp-guard" "$checkpoint/tokenizer.json"
python "$tests/token_oracle.py" "$build/bin/token-probe" \
    "$checkpoint/tokenizer.json" "$evidence/masks"
```

The CPU oracle computes eligible checkpoint token IDs from literal byte
languages and an independent minimum-token segmentation. It checks alternate
segmentations, split UTF-8, overlapping repeated choices, budget and EOS,
unsupported patterns, and agreement of the dense mask with the ID list.

The HTTP checks preserve raw JSON responses and event streams. Select an idle
GPU before starting the server.

```bash
server="$build/bin/sglang-server"
python "$tests/http_contract.py" "$server" "$checkpoint" "$evidence/serial"
python "$tests/http_contract.py" "$server" "$checkpoint" "$evidence/eager" --eager
python "$tests/http_contract.py" "$server" "$checkpoint" "$evidence/graph" \
    --width 4 --graph --mtp 1
python "$tests/integration.py" "$server" "$checkpoint" "$evidence/depth-serial" --mtp 1
python "$tests/integration.py" "$server" "$checkpoint" "$evidence/depth-mixed" \
    --width 4 --graph --mtp 1
python "$tests/integration.py" "$server" "$checkpoint" "$evidence/tensor-prefix" \
    --width 4 --graph --prefill tensor-core --cache-entries 4
```

Append `--media-body "$media_request" --media-sha256 "$media_hash"` to a serial
command to check a decoded image request before and after rejecting constraints
on that route. The body must be a valid, unconstrained chat request.
