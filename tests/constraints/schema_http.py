"""Check JSON Schema output and validation through native text and chat HTTP."""

from http_contract import SAMPLED, arguments, assert_piece_text, body, media_guard, run
from schema_oracle import array_of, object_of, validate
from token_oracle import checkpoint_bytes


def response_format(schema):
    return {"type": "json_schema", "json_schema": {"name": "answer", "strict": True,
                                                    "schema": schema}}


def request_for(route, budget, schema, **options):
    return body(route, budget, response_format=response_format(schema), **options)


def cases():
    fields = {'z"\\雪': {"enum": ['a\t"\\', "\U0010fffd\n"]},
              "m": array_of({"type": "boolean"}, 2, 2),
              "a": {"type": "null"}}
    ordered = object_of(fields, fields)
    optional = object_of({"z": {"enum": [1]}, "m": {"type": "boolean"},
                          "a": {"type": "null"}}, ["m"])
    return [
        ("text-greedy", "text", request_for("text", 44, ordered)),
        ("chat-sampled", "chat", request_for("chat", 44, ordered, **SAMPLED)),
        ("text-sampled", "text", request_for("text", 16, optional, **SAMPLED)),
        ("chat-greedy", "chat", request_for("chat", 16, optional)),
        ("scalar-string", "text", request_for("text", 9, {"type": "string"}, **SAMPLED)),
        ("scalar-number", "chat", request_for("chat", 8, {"type": "number"})),
        ("scalar-union-enum", "text", request_for("text", 12, {"enum": [False, 0, True, None]})),
        ("null-exact-budget", "chat", request_for("chat", 1, {"type": "null"})),
    ]


def reject_schema(http, _args):
    null = {"type": "null"}
    invalid = [
        ("alternate-format", "text", body("text", 8, response_format={"type": "json_object"}),
         r"response_format"),
        ("not-strict", "chat", body("chat", 8, response_format={"type": "json_schema",
         "json_schema": {"schema": null, "strict": False}}), r"strict|descriptor"),
        ("missing-schema", "text", body("text", 8, response_format={"type": "json_schema",
         "json_schema": {"name": "missing"}}), r"schema"),
        ("references", "text", request_for("text", 8, {"$ref": "#"}), r"schema"),
        ("combinator", "chat", request_for("chat", 8, {"oneOf": [null]}), r"schema"),
        ("pattern-keyword", "text", request_for("text", 8, {"type": "string", "pattern": "x"}),
         r"pattern"),
        ("required-undeclared", "chat", request_for("chat", 8, object_of({}, ["x"])),
         r"required"),
        ("unbounded-object", "text", request_for("text", 8,
         {"type": "object", "additionalProperties": True}), r"additionalProperties"),
        ("array-cap", "chat", request_for("chat", 8, array_of(null, 0, 9)),
         r"maxItems|array item bounds"),
        ("budget", "chat", request_for("chat", 0, null), r"max_tokens"),
        ("regex-conflict", "text", request_for("text", 8, null, regex="null"), r"regex"),
        ("ebnf-conflict", "chat", request_for("chat", 8, null, ebnf='root ::= "null"'), r"ebnf"),
        ("ignore-eos", "chat", request_for("chat", 8, null, ignore_eos=True), r"EOS"),
        ("stop-override", "text", request_for("text", 8, null, stop_token_ids=[]), r"EOS"),
        ("request-mtp-invalid", "chat", request_for("chat", 8, null, mtp_steps=1), r"mtp_steps"),
    ]
    for label, route, request, error in invalid:
        http.call(label, route, request, stream=True, error=error)


def validate_result(pieces, label, request, result):
    schema = request["response_format"]["json_schema"]["schema"]
    value = validate(schema, result["text"])
    assert_piece_text(pieces, result)
    assert len(result["tokens"]) <= request["max_tokens"]
    assert result["reason"] == "stop", (label, result)
    if label == "null-exact-budget":
        assert value is None and len(result["tokens"]) == 1


def main():
    args = arguments()
    pieces, _ = checkpoint_bytes(args.checkpoint / "tokenizer.json")
    media = {"response_format": response_format({"type": "null"})}
    run(args, pieces, cases(), lambda *fixture: validate_result(pieces, *fixture), reject_schema,
        lambda http, options: media_guard(http, options, media, "response_format|schema"))


if __name__ == "__main__":
    main()
