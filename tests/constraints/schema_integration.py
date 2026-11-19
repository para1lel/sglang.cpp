"""Repeat cross-feature HTTP coverage with the JSON Schema layer enabled."""

from http_contract import arguments
from integration import grammars, run
from schema_http import response_format
from schema_oracle import object_of, validate


def main():
    schema = object_of({"z": {"enum": [True, False]}, "a": {"type": "null"}}, ["z", "a"])
    fields = grammars() | {
        "response_format": (response_format(schema), response_format({"type": "null"}), "null"),
    }
    run(arguments(), fields, validate)


if __name__ == "__main__":
    main()
