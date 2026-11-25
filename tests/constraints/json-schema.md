# Bounded JSON Schema constraints

Apply the native regex test overlay first. Use the same `build`, `checkpoint`,
`tests`, `server` and `evidence` variables as in [README.md](README.md).
Build the CPU probes with the same commands and add
`-DSGLANG_CHECK_JSON_SCHEMA` to both compiler invocations. The Python
`jsonschema` package supplies the Draft 2020-12 oracle. The validated
version is 4.26.0 and is required only in the test environment.

Set `LMCACHE_SOCKET` to an isolated, ready [LMCache daemon on the same GPU](../../docs/serving/lmcache.md#start-the-daemon) for the cache case.

```bash
"$build/bin/mtp-guard" "$checkpoint/tokenizer.json"
python "$tests/schema_oracle.py" "$build/bin/token-probe" \
    "$checkpoint/tokenizer.json" "$evidence/schema-masks"
python "$tests/schema_http.py" "$server" "$checkpoint" "$evidence/schema-serial"
python "$tests/schema_http.py" "$server" "$checkpoint" "$evidence/schema-eager" --eager
python "$tests/schema_http.py" "$server" "$checkpoint" "$evidence/schema-graph" \
    --width 4 --graph --mtp 1
python "$tests/schema_integration.py" "$server" "$checkpoint" "$evidence/schema-depth-serial" \
    --mtp 1
python "$tests/schema_integration.py" "$server" "$checkpoint" "$evidence/schema-depth-mixed" \
    --width 4 --graph --mtp 1
python "$tests/schema_integration.py" "$server" "$checkpoint" "$evidence/schema-tensor-prefix" \
    --width 4 --graph --prefill tensor-core --lmcache-socket "$LMCACHE_SOCKET"
```

The checks validate values with the independent evaluator and verify compact
serialization, key order and unique keys independently. The finite token
oracle compares eligible IDs before each staged token, and the HTTP test
compares full JSON and SSE transcripts. The serial scripts also accept the
media fixture flags from the regex test.
