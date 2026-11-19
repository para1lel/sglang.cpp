#!/usr/bin/env python3
"""Compare checkpoint mmap views with an independent shard-header reader."""

import argparse
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest


INDEX_NAME = "model.safetensors.index.json"
SHARD_NAME = "fixture.safetensors"
WIDTHS = {"BF16": 2, "F8_E4M3": 1}
MAX_U64 = (1 << 64) - 1
BASE_HEADER = b'{"tensor":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}}'


def unique_object(pairs):
    result = dict(pairs)
    assert len(result) == len(pairs)
    return result


def run_probe(probe, directory, names=()):
    return subprocess.run(
        [probe, directory], input=b"".join(name.encode() + b"\0" for name in names),
        capture_output=True, check=False, timeout=30,
    )


def shard_views(path, weight_map):
    with path.open("rb") as stream:
        stat = path.stat()
        header_size, = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(header_size), object_pairs_hook=unique_object)
    base = 8 + header_size
    expected, coverage = {}, []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        assert weight_map[name] == path.name
        dtype, shape = entry["dtype"], entry["shape"]
        begin, end = entry["data_offsets"]
        size = math.prod(shape) * WIDTHS[dtype]
        assert end - begin == size
        coverage.append((begin, end))
        expected[name] = (dtype, size, base + begin, stat.st_ino, path.name, *shape)
    offset = 0
    for begin, end in sorted(coverage):
        assert begin == offset
        offset = end
    assert base + offset == stat.st_size
    return expected


def verify_views(probe, directory):
    expected = {}
    index = json.loads((directory / INDEX_NAME).read_bytes(), object_pairs_hook=unique_object)
    weight_map = index["weight_map"]
    for shard in sorted(set(weight_map.values())):
        expected.update(shard_views(directory / shard, weight_map))
    assert expected.keys() == weight_map.keys()
    result = run_probe(probe, directory, expected)
    assert result.returncode == 0, result.stderr.decode()
    for name, reported in zip(expected, result.stdout.decode().splitlines(), strict=True):
        dtype, size, offset, inode, shard, *shape = reported.split("\t")
        actual = (dtype, int(size), int(offset), int(inode), shard, *map(int, shape))
        assert actual == expected[name], (name, actual, expected[name])
    elapsed_ns = int(re.search(rb"load_ns=(\d+)", result.stderr)[1])
    print(f"{directory.name}: verified {len(expected)} views, {elapsed_ns} ns")
    return expected


def write_fixture(directory, config, header=BASE_HEADER, size=1, index=None):
    directory.mkdir()
    (directory / "config.json").write_bytes(config)
    if index is None:
        index = json.dumps({"weight_map": {"tensor": SHARD_NAME}}).encode()
    (directory / INDEX_NAME).write_bytes(index)
    with (directory / SHARD_NAME).open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        stream.truncate(8 + len(header) + size)


def malformed_headers():
    header = BASE_HEADER
    yield "duplicate-field", header.replace(b'"shape"', rb'"sh\u0061pe":[1],"shape"'), 1
    for label, shape in {
        "u64-overflow": b"18446744073709551616", "negative": b"-1",
        "fraction": b"1.0", "exponent": b"1e0", "leading-zero": b"01",
        "huge-product": f"{MAX_U64},{MAX_U64}".encode(),
    }.items():
        yield label, header.replace(b'"shape":[1]', b'"shape":[' + shape + b']'), 1
    for label, name in {
        "invalid-utf8": b"t\xc0\xaf", "lone-surrogate": rb"t\ud800",
        "bad-escape": rb"t\q", "raw-control": b"t\x01",
    }.items():
        yield label, header.replace(b"tensor", name), 1
    yield "trailing-data", header + b"null", 1
    yield "offset-overflow", header.replace(b"[0,1]", f"[0,{MAX_U64 + 1}]".encode()), 1
    yield "offset-range", header.replace(b"[0,1]", b"[1,2]"), 1
    yield "coverage-gap", header.replace(b"[0,1]", b"[1,2]"), 2
    yield "coverage-tail", header, 2
    yield "unsupported-dtype", header.replace(b"F8_E4M3", b"F32"), 1


class CheckpointMetadataTest(unittest.TestCase):
    probe = None

    @classmethod
    def setUpClass(cls):
        if cls.probe is None:
            raise unittest.SkipTest("Run this script with the checkpoint metadata probe")

    def setUp(self):
        fixtures = tempfile.TemporaryDirectory(dir=self.fixture_parent)
        self.addCleanup(fixtures.cleanup)
        self.root = Path(fixtures.name)

    def fixture(self, label, config=None, **options):
        path = self.root / label
        write_fixture(path, self.config if config is None else config, **options)
        return path

    def reject(self, path, detail=None):
        result = run_probe(self.probe, path)
        self.assertEqual(result.returncode, 1, (path.name, result))
        self.assertTrue(result.stderr, path.name)
        if detail:
            self.assertIn(detail.encode(), result.stderr, path.name)

    def test_sparse_offsets_and_escaped_names(self):
        large_size = (1 << 32) + 17
        name = 'q"\\😀\u0001'
        header = {
            name: {"dtype": "F8_E4M3", "shape": [large_size], "data_offsets": [0, large_size]},
            "tail": {"dtype": "BF16", "shape": [], "data_offsets": [large_size, large_size + 2]},
            "empty": {"dtype": "BF16", "shape": [MAX_U64, 0], "data_offsets": [0, 0]},
            "__metadata__": {"format": "pt", "escapes": "\r\n\t\b\f/\\\u0000"},
        }
        index = json.dumps({"weight_map": dict.fromkeys((name, "tail", "empty"), SHARD_NAME)})
        path = self.fixture("sparse", header=json.dumps(header).encode() + b" \t\r\n",
                            size=large_size + 2,
                            index=index.replace(r"\ud83d\ude00", "😀").encode())
        expected = verify_views(self.probe, path)
        self.assertGreater(expected["tail"][2], 1 << 32)
        self.assertEqual(expected["empty"][5:], (MAX_U64, 0))

    def test_header_depth_boundary(self):
        extra = BASE_HEADER.replace(b'"dtype"', b'"extra":EXTRA,"dtype"')
        for depth in (32, 33):
            with self.subTest(depth=depth):
                levels = depth - 2
                header = extra.replace(b"EXTRA", b"[" * levels + b"null" + b"]" * levels)
                path = self.fixture(f"depth-{depth}", header=header)
                if depth == 32:
                    verify_views(self.probe, path)
                else:
                    self.reject(path, "nesting exceeds")

    def test_invalid_and_truncated_shards(self):
        for label, header, size in malformed_headers():
            with self.subTest(case=label):
                self.reject(self.fixture(label, header=header, size=size))
        for label, length in (("prefix", 7), ("header", 8 + len(BASE_HEADER) - 1),
                              ("payload", 8 + len(BASE_HEADER))):
            with self.subTest(case=label):
                path = self.fixture(label)
                with (path / SHARD_NAME).open("r+b") as stream:
                    stream.truncate(length)
                self.reject(path)

    def test_invalid_index(self):
        for label, index in (
            ("duplicate", rb'{"weight_map":{"tensor":"fixture.safetensors",'
                          rb'"\u0074ensor":"fixture.safetensors"}}'),
            ("traversal", b'{"weight_map":{"tensor":"../fixture.safetensors"}}'),
            ("mismatch", b'{"weight_map":{"other":"fixture.safetensors"}}'),
        ):
            with self.subTest(case=label):
                self.reject(self.fixture(label, index=index))

    def test_config_comparison(self):
        equivalent = json.loads(self.config)
        text = equivalent["text_config"]
        text["hidden_size"] = 5120.0
        text["attention_dropout"] = 0
        text["rope_parameters"]["mrope_section"] = [11.0, 11.0, 10.0]
        verify_views(self.probe, self.fixture("equivalent", json.dumps(equivalent).encode()))
        mismatches = (("attention_dropout", False), ("layer_types", []), ("hidden_size", 5121))
        for key, value in mismatches:
            with self.subTest(field=key):
                wrong = json.loads(self.config)
                wrong["text_config"][key] = value
                self.reject(self.fixture(key, json.dumps(wrong).encode()),
                            f"text_config.{key} differs")

    def test_historical_number_ranges(self):
        config = json.loads(self.config)
        config["text_config"]["attention_dropout"] = "ROUND_TO_ZERO"
        cases = {
            "required-underflow": {"config": json.dumps(config).encode().replace(
                b'"ROUND_TO_ZERO"', b"1e-400", 1)},
            "ignored-underflow": {"config": self.config.rstrip()[:-1] + b',"extra":1e-400}'},
            "ignored-overflow": {"config": self.config.rstrip()[:-1] + b',"extra":1e400}'},
            "index-underflow": {"index": b'{"weight_map":{"tensor":"fixture.safetensors"},'
                                         b'"extra":-1e-400}'},
        }
        for label, options in cases.items():
            with self.subTest(case=label):
                verify_views(self.probe, self.fixture(label, **options))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--fixtures", type=Path, required=True,
                        help="Existing directory for sparse and malformed fixtures")
    args = parser.parse_args()
    CheckpointMetadataTest.probe = args.probe.resolve()
    CheckpointMetadataTest.config = (args.checkpoint / "config.json").read_bytes()
    CheckpointMetadataTest.fixture_parent = args.fixtures
    verify_views(CheckpointMetadataTest.probe, args.checkpoint)
    unittest.main(argv=[__file__], verbosity=2)


if __name__ == "__main__":
    main()
