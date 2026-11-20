"""Check Qwen XML string values and strict tool schema capacity."""

import argparse
from pathlib import Path

from tool_mask import CHECKPOINT, Case, body, call, probe, tool

STRING = {"type": "string"}


def schema_case(name, properties, required, parameters, *, accept=True):
    request = body("required-single", tools=[tool("record", properties, required)])
    return Case(name, request, call("record", *parameters), accept, accept)


def cases():
    string_null = {"type": "string", "enum": ["null"]}
    mixed = {"enum": ["null", 7]}
    selected = [
        schema_case("multiline", {"text": STRING}, ["text"], [("text", "line 1\nline 2")]),
        schema_case("less-than", {"text": STRING}, ["text"], [("text", "a<b\n雪 << café")]),
        schema_case("nested-markers", {"text": STRING}, ["text"],
                    [("text", "a<tool_call><think>text</think>")]),
        schema_case("string-null-literal", {"value": string_null}, ["value"], [("value", "null")]),
        schema_case("string-null-other", {"value": string_null}, ["value"], [("value", "NULL")],
                    accept=False),
        schema_case("json-null", {"value": {"type": "null"}}, ["value"], [("value", "null")]),
        schema_case("mixed-enum-string", {"value": mixed}, ["value"], [("value", "null")]),
        schema_case("mixed-enum-integer", {"value": mixed}, ["value"], [("value", "7")]),
        schema_case("number", {"value": {"type": "number"}}, ["value"], [("value", "1.5")]),
        schema_case("array", {"value": {"type": "array", "items": STRING}}, ["value"],
                    [("value", '["a"]')]),
        schema_case("object", {"value": {"type": "object", "properties": {"a": STRING}}},
                    ["value"], [("value", '{"a":"b"}')]),
        schema_case("object-spaces", {"value": {"type": "object",
                                                "properties": {"a": STRING}}},
                    ["value"], [("value", '{"a": "b"}')], accept=False),
    ]
    for count in (3, 8):
        keys = [f"field_{index}" for index in range(count)]
        properties = dict.fromkeys(keys, STRING)
        values = [(key, "one\nx<y") for key in keys]
        selected.extend([
            schema_case(f"{count}-all-required", properties, keys, values),
            schema_case(f"{count}-optional-fields", properties, [keys[-1]],
                        [values[0], values[-1]]),
            schema_case(f"{count}-missing-required", properties, [keys[-1]], values[:-1],
                        accept=False),
        ])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    selected = cases()
    too_many = dict.fromkeys((f"field_{index}" for index in range(9)), STRING)
    over_limit = schema_case("nine-fields", too_many, (), [])
    *results, invalid = probe(args.probe, args.checkpoint, [*selected, over_limit], args.evidence)
    assert len(invalid) == 1 and "supports at most eight properties" in invalid[0], invalid
    for case, result in zip(selected, results):
        states = [line.split() for line in result if line.startswith("state ")]
        rejected = any(line.startswith("rejected ") for line in result)
        assert states and rejected != case.accept, (case.name, result)
        assert states[-1][3:5] == [str(int(case.eos))] * 2, (case.name, states[-1])
    print(f"Strict tool schemas passed {len(selected) + 1} cases")


if __name__ == "__main__":
    main()
