"""Compare native constraint masks with independently segmented checkpoint bytes."""

import argparse
from collections import defaultdict
from functools import cache
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import subprocess


EOS = {248044, 248046}
TOKENIZER_HASH = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"


def checkpoint_bytes(path):
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == TOKENIZER_HASH
    tokenizer = json.loads(raw)
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    translated = {chr(byte): byte for byte in direct}
    translated.update({chr(256 + offset): byte for offset, byte in
                       enumerate(sorted(set(range(256)) - direct))})
    special = {entry["id"] for entry in tokenizer["added_tokens"]}
    pieces = {token: bytes(translated[char] for char in encoded)
              for encoded, token in tokenizer["model"]["vocab"].items() if token not in special}
    by_bytes = defaultdict(set)
    for token, piece in pieces.items():
        by_bytes[piece].add(token)
    return pieces, by_bytes


class FiniteLanguage:
    def __init__(self, by_bytes, values):
        self.by_bytes = by_bytes
        self.values = tuple(values)

    @cache
    def shortest(self, raw):
        if not raw:
            return ()
        choices = ((min(self.by_bytes[raw[:end]]), *self.shortest(raw[end:]))
                   for end in range(1, len(raw) + 1) if raw[:end] in self.by_bytes)
        return min(choices, key=lambda ids: (len(ids), ids))

    def mask(self, prefix, remaining):
        allowed = set()
        for value in self.values:
            if not value.startswith(prefix):
                continue
            suffix = value[len(prefix):]
            if not suffix:
                allowed.update(EOS)
            for end in range(1, len(suffix) + 1):
                head = suffix[:end]
                if head in self.by_bytes and len(self.shortest(suffix[end:])) < remaining:
                    allowed.update(self.by_bytes[head])
        return allowed

    def byte_tokens(self, raw):
        return tuple(min(self.by_bytes[bytes([byte])]) for byte in raw)


class Probe:
    def __init__(self, args):
        self.binary = args.probe
        self.tokenizer = args.tokenizer
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=False)
        self.pieces, self.by_bytes = checkpoint_bytes(self.tokenizer)

    def run(self, label, kind, spec, budget, tokens=(), status=0):
        argv = [str(self.binary), str(self.tokenizer), kind, spec, str(budget), *map(str, tokens)]
        result = subprocess.run(argv, capture_output=True, encoding="utf-8", timeout=60)
        (self.output / f"{label}.json").write_text(
            json.dumps({"argv": argv, "status": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr}, ensure_ascii=True))
        assert result.returncode == status, (label, result.returncode, result.stderr)
        if status:
            assert result.stderr.strip(), label
            return None
        lines = result.stdout.splitlines()
        assert lines and lines[0].startswith("minimum "), (label, lines)
        stop = lines[-1] == "stop"
        masks = lines[1:-1] if stop else lines[1:]
        assert all(row == "allowed" or row.startswith("allowed ") for row in masks)
        return int(lines[0].split()[1]), [set(map(int, row.split()[1:])) for row in masks], stop

    def finite(self, label, kind, spec, values, tokens, budget, stop):
        language = FiniteLanguage(self.by_bytes, values)
        minimum, masks, stopped = self.run(label, kind, spec, budget, tokens)
        assert minimum == min(len(language.shortest(value)) for value in values), label
        assert stopped == stop and len(masks) == len(tokens) + (not stop), label
        prefix = b""
        for offset, actual in enumerate(masks):
            expected = language.mask(prefix, budget - offset)
            assert actual == expected, (label, offset, sorted(actual - expected)[:10],
                                        sorted(expected - actual)[:10])
            if offset < len(tokens) and tokens[offset] not in EOS:
                prefix += self.pieces[tokens[offset]]


def finite_regex(probe):
    parts = {("".join(group) + tail).encode() for count in (1, 2)
             for group in product(("ab", "abb"), repeat=count) for tail in ("", "a")}
    pattern = r"^(?:ab|abb){1,2}a?$"
    language = FiniteLanguage(probe.by_bytes, parts)
    shortest = language.shortest(b"abab")
    fallback = language.byte_tokens(b"abab")
    assert len(shortest) < len(fallback)
    probe.finite("overlap-shortest", "regex", pattern, parts, shortest, len(shortest), True)
    probe.finite("overlap-fallback", "regex", pattern, parts, fallback, len(fallback), True)
    stop = language.byte_tokens(b"ab") + (248046,)
    probe.finite("overlap-eos", "regex", pattern, parts, stop, 8, True)
    probe.finite("budget-prunes", "regex", pattern, parts, (), 1, False)


def unicode_regex(probe):
    values = [value.encode() for value in ("👩‍🚀雪", "👩‍🚀!", "\U0010fffd雪", "\U0010fffd!")]
    pattern = r"^(?:👩‍🚀|\U0010FFFD)[雪!]$"
    language = FiniteLanguage(probe.by_bytes, values)
    tokens = language.byte_tokens(values[0])
    assert len(tokens) == 14 and all(len(probe.pieces[token]) == 1 for token in tokens)
    probe.finite("unicode-byte-fallback", "regex", pattern, values, tokens, len(tokens), True)
    tokens = language.shortest(values[0])
    assert len(tokens) < 14
    probe.finite("unicode-alternative", "regex", pattern, values, tokens, len(tokens), True)
    for token in sorted(EOS):
        probe.finite(f"nullable-{token}", "regex", "^(?:)$", [b""], (token,), 1, True)
    probe.finite("nullable-zero", "regex", "^(?:)$", [b""], (), 0, False)


def broad_regex(probe):
    minimum, masks, stopped = probe.run("wide-dot", "regex", r"^.+$", 8)
    allowed = masks[0]
    assert minimum == 1 and not stopped and len(allowed) > 4096 and not (allowed & EOS)
    for token, piece in probe.pieces.items():
        if b"\n" in piece:
            assert token not in allowed
            continue
        try:
            piece.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason != "unexpected end of data":
                assert token not in allowed
        else:
            assert token in allowed


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main():
    probe = Probe(arguments())
    finite_regex(probe)
    unicode_regex(probe)
    broad_regex(probe)
    rejected = (r"(a)\1", r"(?=x)x", "(?i)x", r"\p{L}", "a+?", "[z-a]", "a{65}",
                r"\uD800", "a^b", "(?:a", "", "a" * 1025)
    for index, pattern in enumerate(rejected):
        probe.run(f"reject-{index}", "regex", pattern, 8, status=2)
    print("regex token masks, alternate UTF-8 tokenizations, budget, EOS and rejection passed")


if __name__ == "__main__":
    main()
