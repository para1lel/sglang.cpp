"""Exercise the native chat request compiler with real checkpoint token masks."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

CHECKPOINT = Path("/data0/shared/Qwen3.8-27B-FP8")
EOS = (248044, 248046)
REFERENCE_SHA = "97eb43d8cd7169ec2e44ace8fc9d141abbf9c3bef2cfb4fb519740913ae7c152"


def tool(name, properties, required=(), *, strict=True):
    return {"type": "function", "function": {"name": name, "strict": strict, "parameters": {
        "type": "object", "properties": properties, "required": list(required),
        "additionalProperties": False,
    }}}


WEATHER = tool("weather", {
    "city": {"type": "string", "enum": ["北京", "Boston"]},
    "days": {"type": "integer", "minimum": 1, "maximum": 4},
    "unit": {"type": "string", "enum": ["c", "f"]},
}, ["city"])
PING = tool("ping", {})


def named(name):
    return {"type": "function", "function": {"name": name}}


def body(mode, *, thinking=False, budget=192, **overrides):
    tools = [WEATHER, PING]
    if mode == "required-loose":
        tools = [item | {"function": item["function"] | {"strict": False}} for item in tools]
    selected = named("weather") if mode.startswith("named") else mode.split("-")[0]
    return {"model": CHECKPOINT.name, "messages": [{"role": "user", "content": "fixture"}],
            "tools": tools, "tool_choice": selected, "enable_thinking": thinking,
            "max_tokens": budget, "parallel_tool_calls": mode.endswith("parallel")} | overrides


def call(name, *parameters):
    args = "\n".join(f"<parameter={key}>\n{value}\n</parameter>" for key, value in parameters)
    return f"<tool_call>\n<function={name}>\n{args}\n</function>\n</tool_call>"


@dataclass(frozen=True)
class Case:
    name: str
    request: dict
    output: str | list
    accept: bool
    eos: bool


def portable_cases():
    city = call("weather", ("city", "北京"), ("days", 3), ("unit", "c"))
    ping = call("ping")
    required = body("required-single")
    parallel = body("required-parallel")
    selected = body("named-single")
    thinking = body("named-single", thinking=True)
    return [
        Case("initial", required, "", True, False),
        Case("prose-first", required, "Here is the result", False, False),
        Case("first-eos-text", required, [EOS[0]], False, False),
        Case("first-eos-message", required, [EOS[1]], False, False),
        Case("strict-valid", required, city, True, True),
        Case("strict-empty", required, ping, True, True),
        Case("partial", required, city[:-12], True, False),
        Case("required-single-two-calls", required, city + ping, False, True),
        Case("required-parallel-two-calls", parallel, city + ping, True, True),
        Case("required-parallel-tail", parallel, city + "ordinary tail", True, True),
        Case("named-valid", selected, city, True, True),
        Case("named-wrong-tool", selected, ping, False, False),
        Case("named-parallel-offers-one", body("named-parallel"), city + city, False, True),
        Case("required-missing", required, call("weather", ("days", 3)), False, False),
        Case("strict-unknown", required, call("weather", ("city", "Boston"),
                                             ("other", "c")), False, False),
        Case("strict-enum", required, call("weather", ("city", "Boston"),
                                          ("unit", "kelvin")), False, False),
        Case("strict-number", required, call("weather", ("city", "Boston"),
                                             ("days", "3.5")), False, False),
        Case("strict-bound", required, call("weather", ("city", "Boston"),
                                            ("days", 0)), False, False),
        Case("strict-order", required, call("weather", ("days", 3),
                                            ("city", "Boston")), False, False),
        Case("redundant-think-close", selected, "</think>" + city, False, False),
        Case("explicit-think-close", thinking, "hidden</think>" + city, True, True),
        Case("implicit-think-close", thinking, "hidden" + city, True, True),
        Case("think-eos", thinking, "hidden", True, True),
        Case("think-then-call-eos", thinking, "hidden</think>" + city[:-12], True, False),
    ]


def pinned_cases(path):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA
    reference = json.loads(raw)
    assert tuple(reference["eos_ids"]) == EOS
    cases = [Case(f"reference-mask-{index}", body(row["mode"]), row["token_ids"],
                  row["expected_accept"], row["expected_eos"])
             for index, row in enumerate(reference["mask_cases"])
             if row["mode"] in {"required-single", "required-parallel", "named-parallel"}]
    for key, row in reference["reasoning"].items():
        mode = key.split("/")[0]
        cases.append(Case("reference-reasoning-" + key,
                          body(mode, thinking=row["thinking_initially"]),
                          [token["token"] for token in row["tokens"]],
                          all(token["allowed"] for token in row["tokens"]),
                          all(row["eos_allowed"].values())))
    return cases


def probe(binary, checkpoint, cases, out):
    wire = "".join(json.dumps([json.dumps(case.request), case.output]) + "\n" for case in cases)
    result = subprocess.run([binary, checkpoint], input=wire, capture_output=True, text=True,
                            check=True, timeout=120)
    out.mkdir()
    (out / "requests.jsonl").write_text(wire)
    (out / "probe.log").write_text(result.stdout)
    received = []
    lines = []
    for line in result.stdout.splitlines():
        if line == "end":
            received.append(lines)
            lines = []
        else:
            lines.append(line)
    assert len(received) == len(cases) and not lines
    return received


def check(cases, received):
    for case, result in zip(cases, received):
        states = [list(map(int, line.split()[1:])) for line in result if line.startswith("state ")]
        assert states, (case.name, result)
        rejected = [line for line in result if line.startswith("rejected ")]
        assert bool(rejected) != case.accept, (case.name, rejected)
        assert states[-1][2:4] == [int(case.eos)] * 2, (case.name, states[-1])
        prompt = json.loads(result[0].removeprefix("prompt "))
        thinking = case.request["enable_thinking"]
        suffix = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
        assert prompt.endswith("<|im_start|>assistant\n" + suffix), (case.name, prompt[-60:])
        assert '"name": "weather"' in prompt, case.name
        offered = '"name": "ping"' in prompt
        assert offered == (case.request["tool_choice"] == "required"), (case.name, offered)
    assert any(line.startswith("state ") and line.split()[2] == "1"
               for result in received for line in result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    cases = portable_cases()
    if args.reference:
        cases += pinned_cases(args.reference)
    check(cases, probe(args.probe, args.checkpoint, cases, args.evidence))
    print(f"Native tool masks passed {len(cases)} cases")


if __name__ == "__main__":
    main()
