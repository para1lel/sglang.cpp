"""Replay length cuts in Qwen XML through the scripted CPU chat server.

Run with PYTHONPATH=tests/text from the repository root.
"""

import argparse
from pathlib import Path

from check_chat_output_cpu import CHECKPOINT, encode_inputs, request_body
from run_chat_layer1 import server
from test_chat_output import BOTH, COMMON, expect, pair
from tool_mask import WEATHER


def cuts():
    start = "<tool_call>"
    function = start + "\n<function=weather>"
    city = function + "\n<parameter=city>\n"
    city_value = city + "北京"
    city_end = city_value + "\n</parameter>"
    days = city_end + "\n<parameter=days>\n3"
    days_end = days + "\n</parameter>"
    units = days_end + "\n<parameter=unit>\nc"
    unit_end = units + "\n</parameter>"
    city_args = {"city": "北京"}
    days_args = city_args | {"days": 3}
    all_args = days_args | {"unit": "c"}
    return [
        ("wrapper", start + "\n", None),
        ("function-tag", function.removesuffix("=weather>"), None),
        ("function-name", function[:-1], None),
        ("function-open", function, {}),
        ("function-line", function + "\n", {}),
        ("parameter-name", city.removesuffix(">\n"), {}),
        ("empty-city", city, {"city": ""}),
        ("city-value", city_value, city_args),
        ("city-line", city_value + "\n", city_args),
        ("parameter-end-tag", city_end[:-1], city_args),
        ("parameter-end", city_end, city_args),
        ("parameter-line", city_end + "\n", city_args),
        ("days-value", days, days_args),
        ("days-end", days_end + "\n", days_args),
        ("unit-value", units, all_args),
        ("unit-end", unit_end + "\n", all_args),
        ("function-end-tag", unit_end + "\n</function", all_args),
        ("function-end", unit_end + "\n</function>", all_args),
        ("before-wrapper-end", unit_end + "\n</function>\n", all_args),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir()
    cases = [(name, text, arguments, "required", "") for name, text, arguments in cuts()]
    for name, text, arguments in cuts():
        cases.append((f"auto-{name}", "Here is the forecast.\n" + text, arguments, "auto",
                      "Here is the forecast.\n"))
    sequences = encode_inputs(args, {name: (text, False) for name, text, *_ in cases})
    with server(args.server, args.checkpoint, args.evidence / "server", 1,
                flags=COMMON + BOTH) as port:
        for name, _, arguments, choice, prefix in cases:
            seed, ids = sequences[name]
            body = request_body(tools=[WEATHER], tool_choice=choice, seed=seed,
                                max_tokens=len(ids))
            result = pair(port, args.evidence, name, body)
            assert result["ids"] == ids, (name, result)
            calls = [] if arguments is None else [("weather", arguments)]
            expect(result, prefix, calls=calls, finish="length")
    print(f"Tool call replay passed {len(cases)} length cuts in JSON and SSE")


if __name__ == "__main__":
    main()
