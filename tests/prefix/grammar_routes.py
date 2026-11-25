"""Compare seeded EBNF output on fresh exact-prefill and cache-enabled batch servers."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier

from routes import MODEL, Routes


TEXT = "/v1/completions"
CHAT = "/v1/chat/completions"
OPTIONS = {"model": MODEL, "max_tokens": 32, "return_token_ids": True, "ignore_eos": False}
SAMPLE = {"temperature": 8.5, "top_k": 19, "top_p": 0.89, "seed": -(2**63)}


def grammar(choices):
    return "root ::= " + " | ".join(json.dumps(choice) for choice in choices)


def fixtures():
    exact = "Hello" + " hello" * 127
    choices = ("Alpha raven", "Beta heron", "Gamma wren", "Delta lark",
               "Felix owl", "Omega swift", "Zeta tern", "Kappa jay")
    isolated = ("ZX: 765432109876543210",)
    chat = ("Caf\u00e9 \U0010fffd", "Caf\u00e9 \U0010fffd!")
    options = OPTIONS | SAMPLE
    body = options | {"prompt": exact, "ebnf": grammar(choices)}
    return {
        "exact": (TEXT, body, choices),
        "exact-other-seed": (TEXT, body | {"seed": 2**63 - 1}, choices),
        "partial": (TEXT, body | {"prompt": exact + " world"}, choices),
        "new-grammar": (TEXT, body | {"ebnf": grammar(isolated)}, isolated),
        "chat": (CHAT, options | {
            "messages": [{"role": "user", "content": exact}],
            "enable_thinking": False, "ebnf": grammar(chat)}, chat),
    }


def stream_result(raw, route, name):
    frames = raw.rstrip(b"\n").split(b"\n\n")
    assert frames.pop() == b"data: [DONE]", name
    data = [json.loads(frame.removeprefix(b"data: ")) for frame in frames]
    usage = data.pop()["usage"]
    choices = [choice for frame in data for choice in frame["choices"]]
    assert [row["finish_reason"] for row in choices if row.get("finish_reason")] == ["stop"]
    pieces = [(row.get("text") if route == TEXT else row["delta"].get("content")) or ""
              for row in choices]
    tokens = [token for row in choices for token in row.get("token_ids", ())]
    return "".join(pieces), tokens, usage


def check(routes, name, case, hit, salt=None, stream=False):
    route, body, choices = case
    if salt is not None:
        body = body | {"cache_salt": salt}
    if stream:
        body = body | {"stream": True, "stream_options": {"include_usage": True}}
    raw, media_type = routes.raw(name, body, route=route)
    if stream:
        assert media_type == "text/event-stream", name
        text, tokens, usage = stream_result(raw, route, name)
    else:
        assert media_type == "application/json", name
        data = json.loads(raw)
        usage = data["usage"]
        choice, = data["choices"]
        assert choice["finish_reason"] == "stop", (name, choice)
        assert len(choice["prompt_token_ids"]) == usage["prompt_tokens"], name
        text = choice["text"] if route == TEXT else choice["message"]["content"]
        tokens = choice["token_ids"]
    assert text in choices and 0 < len(tokens) <= body["max_tokens"], (name, text, tokens)
    Routes.usage(name, usage, usage["prompt_tokens"], len(tokens), hit)
    return {"ids": tokens, "text": text, "prompt": usage["prompt_tokens"]}


def warm(routes, cases, reference):
    def verify(name, case, hit, salt, stream=False):
        actual = check(routes, name, cases[case], hit, salt, stream)
        assert actual == reference[case], (name, actual, reference[case])
        return actual

    text_hit = reference["exact"]["prompt"]
    chat_hit = reference["chat"]["prompt"]
    verify("cold-exact", "exact", 0, "text")
    verify("warm-exact", "exact", text_hit, "text", True)
    verify("cold-chat", "chat", 0, "chat")
    verify("warm-chat", "chat", chat_hit, "chat", True)
    route, body, _ = cases["exact"]
    invalid = body | {"cache_salt": "text", "ebnf": "root ::= [a-z]"}
    routes.raw("invalid-grammar", invalid, code=400, route=route)
    verify("after-invalid", "exact", text_hit, "text")

    barrier = Barrier(4, timeout=10)
    lanes = (("warm-new-grammar", "new-grammar", text_hit, "text"),
             ("warm-suffix", "partial", text_hit, "text"),
             ("warm-other-seed", "exact-other-seed", text_hit, "text"),
             ("cold-mixed", "exact", 0, "unseen"))

    def concurrent(lane):
        barrier.wait()
        verify(*lane, stream=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(concurrent, lanes))
    verify("after-mixed", "exact", text_hit, "text")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "warm"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir()
    cases = fixtures()
    routes = Routes(args.port, args.evidence)
    if args.mode == "reference":
        results = {name: check(routes, name, case, 0) for name, case in cases.items()}
        assert results["exact"]["prompt"] == 128 and results["partial"]["prompt"] == 129
        assert results["exact"]["ids"][0] != results["exact-other-seed"]["ids"][0]
        args.reference.write_text(json.dumps(results, indent=2) + "\n")
    else:
        warm(routes, cases, json.loads(args.reference.read_text()))


if __name__ == "__main__":
    main()
