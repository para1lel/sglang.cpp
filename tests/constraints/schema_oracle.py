"""Verify bounded native schema masks, JSON semantics and serialization."""

from itertools import combinations, product
import json

from jsonschema import Draft202012Validator

from token_oracle import FiniteLanguage, Probe, arguments


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def object_of(properties, required):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


def array_of(items, minimum, maximum):
    return {"type": "array", "items": items, "minItems": minimum, "maxItems": maximum}


def unique_keys(pairs):
    assert len({key for key, _ in pairs}) == len(pairs), pairs
    return dict(pairs)


def invalid_constant(value):
    raise ValueError(f"Non-JSON number {value}")


def check_order(schema, value):
    if isinstance(value, dict):
        expected = schema["properties"]
        assert list(value) == [key for key in expected if key in value], value
        for key, member in value.items():
            key.encode("utf-8", errors="strict")
            check_order(expected[key], member)
    if isinstance(value, list):
        for member in value:
            check_order(schema["items"], member)
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")


def validate(schema, raw):
    in_string = escaped = False
    for char in raw:
        if escaped:
            escaped = False
        elif in_string and char == "\\":
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            assert char not in " \t\r\n", raw
    value = json.loads(raw, object_pairs_hook=unique_keys, parse_constant=invalid_constant)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(value)
    check_order(schema, value)
    return value


def finite(probe, label, schema, values, target, shortest=False):
    words = [compact(value).encode() for value in values]
    language = FiniteLanguage(probe.by_bytes, words)
    raw = compact(target).encode()
    assert raw in words and compact(validate(schema, raw.decode())) == raw.decode()
    tokens = language.shortest(raw) if shortest else language.byte_tokens(raw)
    probe.finite(label, "schema", compact(schema), words, tokens, len(tokens), True)


def finite_cases(probe):
    enum = [False, 0, True, 1, None, 2**64 - 1, '雪\t\n"\\\x00']
    for index, value in enumerate(enum):
        finite(probe, f"enum-{index}", {"enum": enum}, enum, value, shortest=index < 4)
    keys = {"zulu": 1, 'é"\\\b': None, "alpha": False}
    schema = object_of({key: {"enum": [value]} for key, value in keys.items()}, [list(keys)[1]])
    values = [{key: keys[key] for key in subset} for width in range(1, 4)
              for subset in combinations(keys, width) if list(keys)[1] in subset]
    finite(probe, "ordered-key-fallback", schema, values, values[-1])
    finite(probe, "ordered-key-shortest", schema, values, values[-1], shortest=True)
    items = object_of({"y": {"type": "boolean"}, "x": {"type": "null"}}, ["y", "x"])
    arrays = [[{"y": item, "x": None} for item in choice]
              for length in range(3) for choice in product((True, False), repeat=length)]
    finite(probe, "array-repeated", array_of(items, 0, 2), arrays, arrays[-1], shortest=True)


def walk_bytes(probe, label, schema, raw, valid=True):
    language = FiniteLanguage(probe.by_bytes, ())
    tokens = language.byte_tokens(raw)
    result = probe.run(label, "schema", compact(schema), len(tokens), tokens,
                       status=0 if valid else 1)
    if valid:
        _, masks, stop = result
        assert stop and len(masks) == len(tokens), label
        return validate(schema, raw.decode("utf-8"))


def scalar_cases(probe):
    string = {"type": "string"}
    escaped = rb'"slash\/ tab\t quote\" backslash\\ nul\u0000 pair\uD83D\uDE80"'
    expected = 'slash/ tab\t quote" backslash\\ nul\x00 pair🚀'
    assert walk_bytes(probe, "valid-escapes", string, escaped) == expected
    raw = '"𝄞雪 👩‍🚀"'.encode()
    assert walk_bytes(probe, "utf8-string", string, raw) == "𝄞雪 👩‍🚀"
    for label, invalid in [
        ("bare-control", b'"\x01"'),
        ("bad-escape", rb'"\q"'),
        ("lone-high", rb'"\udbff"'),
        ("lone-low", rb'"\udc00"'),
        ("bad-pair", rb'"\ud800\u0010"'),
        ("bad-utf8", b'"\xed\xa0\x80"'),
    ]:
        walk_bytes(probe, label, string, invalid, valid=False)
    for index, (kind, raw) in enumerate([
        ("integer", b"-123456789012345678"), ("number", b"-0.000012345678"),
        ("integer", b"1234567890123456789"), ("number", b"0.1234567890123"),
        ("number", b"1e1"), ("integer", b"01"),
    ]):
        walk_bytes(probe, f"number-{index}", {"type": kind}, raw, valid=index < 2)


def rejected_cases(probe):
    invalid = [
        {"$ref": "#/$defs/answer"}, {"anyOf": [{"type": "null"}]},
        {"type": ["null", "string"]}, {"type": "string", "pattern": "x"},
        {"type": "integer", "minimum": 0}, {"type": "boolean", "enum": [1]},
        {"enum": [{}]}, {"enum": [0, 0]}, {"enum": [1.5]},
        {"type": "object", "additionalProperties": True},
        object_of({"a": {"type": "string"}}, ["unknown"]),
        array_of({"type": "boolean"}, 3, 2), array_of({"type": "null"}, 0, 9),
    ]
    for index, schema in enumerate(invalid):
        probe.run(f"reject-schema-{index}", "schema", compact(schema), 16, status=2)
    probe.run("reject-duplicate-keyword", "schema", '{"type":"null","type":"null"}', 16, status=2)


def main():
    probe = Probe(arguments())
    finite_cases(probe)
    scalar_cases(probe)
    rejected_cases(probe)
    print("schema finite masks, JSON types, ordered keys, escaping and rejection passed")


if __name__ == "__main__":
    main()
