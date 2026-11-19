"""Check constrained routing alongside positive MTP depth or tensor prefix reuse."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import re
from threading import Barrier

from http_contract import (SAMPLED, arguments, assert_piece_text, body, media_guard, serve,
                           trace_slice, verify_verifier)
from token_oracle import checkpoint_bytes


def grammars():
    return {"ebnf": ('root ::= "Café" | "Café!"', 'root ::= "other"', "other"),
            "regex": ("^(?:Café|Café!){1,2}$", "^other$", "other")}


def field_pattern(field):
    return "JSON Schema" if field == "response_format" else field


def normalize_cache(row, cached):
    usage = row["usage"].copy()
    assert usage.pop("prompt_tokens_details")["cached_tokens"] == cached, row
    return row | {"usage": usage}


def reject_overrides(http, log, fields):
    for field, (spec, _, _) in fields.items():
        offset = log.stat().st_size
        for route in ("text", "chat"):
            request = body(route, 8, **{field: spec})
            for depth in (0, 1, 4):
                label = f"{field}-{route}-override-{depth}"
                http.call(label, route, request | {"mtp_steps": depth}, stream=bool(depth),
                          error="Unsupported .* option: mtp_steps")
            for option in ({"ignore_eos": True}, {"stop_token_ids": []}):
                name, = option
                http.call(f"{field}-{route}-{name}", route, request | option,
                          stream=True, error="EOS")
        assert not trace_slice(log, offset), field


def check_mtp(http, args, fields, validate):
    log = args.output / "server.log"
    reject_overrides(http, log, fields)
    cases = [(f"{field}-{route}-{mode}", route,
              body(route, 12, **{field: spec}, **options))
             for field, (spec, _, _) in fields.items() for route in ("text", "chat")
             for mode, options in (("greedy", {}), ("sampled", SAMPLED))]
    offset = log.stat().st_size
    http.pairs(cases, args.width, validate)
    if args.width > 1:
        verify_verifier(log, offset, False)
        if args.graph:
            assert re.search(r"^batch graph_slots=[1-9]\d*$", trace_slice(log, offset), re.M)
    ordinary = body("text", 18, ignore_eos=True, **SAMPLED)
    offset = log.stat().st_size
    control = http.call("ordinary-sampled-alone", "text", ordinary)
    if args.width > 1:
        verify_verifier(log, offset, True)
        mixed(http, log, cases, ordinary, control)
    else:
        for field, (spec, _, _) in fields.items():
            media_guard(http, args, {field: spec}, field_pattern(field), label=field)


def validate_constraint(pieces, schema_validator, label, request, row):
    assert row["reason"] == "stop" and 0 < len(row["tokens"]) <= request["max_tokens"]
    assert_piece_text(pieces, row)
    if "ebnf" in request:
        choices = request["ebnf"].removeprefix("root ::= ").split(" | ")
        assert row["text"] in tuple(map(json.loads, choices)), (label, row)
    if "regex" in request:
        assert re.fullmatch(request["regex"], row["text"]), (label, row)
    if "response_format" in request:
        schema = request["response_format"]["json_schema"]["schema"]
        schema_validator(schema, row["text"])


def mixed(http, log, cases, ordinary, control):
    constrained = cases[::4]
    jobs = constrained + [("mixed-ordinary", "text", ordinary)]
    barrier = Barrier(len(jobs))
    offset = log.stat().st_size
    with ThreadPoolExecutor(len(jobs)) as pool:
        pending = [pool.submit(http.call, f"mixed-{label}", route, request,
                               True, barrier=barrier) for label, route, request in jobs]
        rows = [job.result() for job in pending]
    assert rows[-1] == control
    for (label, _, _), row in zip(constrained, rows[:-1], strict=True):
        assert row == http.summary[label], (label, row)
    verify_verifier(log, offset, True)
    trace = trace_slice(log, offset)
    steps = re.split(r"(?=^batch width=)", trace, flags=re.M)
    assert any(re.search(r"^batch width=[2-4] decode=[2-4] ", step, re.M)
               and re.search(r"^batch verify lanes=1 rows=[1-9]\d* ", step, re.M)
               for step in steps), (log, trace)
    assert not re.search(r"^batch verify lanes=[2-9]", trace, re.M), (log, trace)


def check_prefix(http, fields, validate):
    prompt = "Hello" + " hello" * 127
    for field, (spec, other, expected) in fields.items():
        def request(pattern, salt):
            return body("text", 12, prompt=prompt, cache_salt=salt,
                        **{field: pattern}, **SAMPLED)

        cold = request(spec, field)
        initial = http.call(f"{field}-cold", "text", cold)
        assert initial["usage"]["prompt_tokens"] == 128
        warm = http.call(f"{field}-warm", "text", cold, stream=True)
        assert normalize_cache(initial, 0) == normalize_cache(warm, 128)
        validate(field, cold, warm)
        changed = request(other, field)
        fresh = request(other, f"{field}-fresh")
        result = http.call(f"{field}-changed-warm", "text", changed, stream=True)
        reference = http.call(f"{field}-changed-cold", "text", fresh)
        assert normalize_cache(result, 128) == normalize_cache(reference, 0)
        assert result["text"] == expected and result["reason"] == "stop"
        validate(field, changed, result)
        http.summary[field] = {"cold": initial, "warm": warm, "changed": result}


def run(args, fields, schema_validator=None):
    pieces, _ = checkpoint_bytes(args.checkpoint / "tokenizer.json")
    validate = partial(validate_constraint, pieces, schema_validator)
    with serve(args, pieces) as http:
        if args.cache_entries:
            assert args.width > 1 and args.graph and args.prefill == "tensor-core"
            check_prefix(http, fields, validate)
        else:
            assert args.mtp > 0
            check_mtp(http, args, fields, validate)
        (args.output / "summary.json").write_text(json.dumps(http.summary, ensure_ascii=True))


def main():
    run(arguments(), grammars())


if __name__ == "__main__":
    main()
