"""Materialize or verify TP2 W1–W5 requests. Run with PYTHONPATH=. and python -B.

prepare TOKENIZER NEW_DIR regenerates the measured wire bytes from the original
short texts and archived deterministic inventory generator. verify DIR reads
those bytes offline. Both arms consume load_lanes(DIR, row) in the same order.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.concurrent import freeze_prompts, reference
from benchmarks.streaming import client
from saved_evidence import digest, require
import workload_suite as suite


ADDITIONS = Path(__file__).with_name("tp2-corpus-v1.json")
ADDITIONS_SHA256 = "d65076eb68c7ca81024beb8c3c9e50a14fc5ea476255c8495914cf16aabf901f"
STREAMING = Path(client.__file__).parent


def metadata(row):
    require(digest(ADDITIONS) == ADDITIONS_SHA256, "TP2 corpus manifest digest mismatch")
    shape = suite.geometry(row, 2)
    width = shape["concurrent_requests"]
    if row in client.SHAPES:
        source = STREAMING / "gpu3-reference.json"
        require(digest(source) == client.PUBLISHED_REFERENCE_SHA256, "Short reference mismatch")
        case, fixture_sha = client.load_cases(STREAMING)[row]
        published = json.loads(source.read_bytes())["manifest"]
        require(fixture_sha == published["fixture_sha256"][row], "Original short fixture mismatch")
        lanes = [{"lane": 0, "first_token": case["prompt_token_ids"][0],
                  "request_sha256": published["request_sha256"][row]}]
    else:
        archive = reference.lane_manifest()
        source_row = suite.CONCURRENT_FIXTURES[row]
        lanes = archive["rows"][source_row]["lanes"][:width]
    for lane, title, first, sha in json.loads(ADDITIONS.read_bytes())["rows"].get(row, []):
        item = {"lane": lane, "first_token": first, "request_sha256": sha, "title": title}
        if row not in client.SHAPES:
            item["seed"] = f'{archive["seed"]}/{source_row}/{lane}'
        lanes.append(item)
    require([lane["lane"] for lane in lanes] == list(range(width))
            and len({lane["first_token"] for lane in lanes}) == width,
            f"{row}: expected {width} disjoint, ordered TP2 prompts")
    return lanes


def load_lanes(directory, row):
    shape = suite.geometry(row, 2)
    lanes = []
    for meta in metadata(row):
        path = directory / row / f'{meta["lane"]:02d}.request.json'
        body = path.read_bytes()
        require(reference.digest(body) == meta["request_sha256"], f"{path}: wire digest mismatch")
        prompt = json.loads(body)["prompt"]
        case = {"prompt_token_ids": prompt, "output_tokens": shape["output_tokens"]}
        require(client.token_ids(prompt) and len(prompt) == shape["prompt_tokens"]
                and prompt[0] == meta["first_token"]
                and body == client.json_bytes(client.request_body(case)),
                f"{path}: invalid request")
        lanes.append((meta, body, case))
    return lanes


def token_ids(tokenizer, row, meta, count):
    if row in client.SHAPES:
        if meta["lane"] == 0:
            return client.load_cases(STREAMING)[row][0]["prompt_token_ids"]
        from benchmarks.streaming.freeze_fixtures import W1, long_prompt

        text = W1 if row == "W1" else long_prompt()
        suffix = text[text.index(meta["title"]):]
        return tokenizer.encode(suffix, add_special_tokens=False).ids[:count]
    source = suite.CONCURRENT_FIXTURES[row]
    text, ids = freeze_prompts.exact_prompt(tokenizer, meta["title"], source, meta["lane"],
                                            count, meta["seed"])
    if "text_sha256" in meta:
        require(reference.digest(text.encode()) == meta["text_sha256"], "Archived text mismatch")
    return ids


def prepare(tokenizer_path, destination):
    from tokenizers import Tokenizer

    require(digest(tokenizer_path) == reference.lane_manifest()["tokenizer_sha256"],
            "Checkpoint tokenizer digest mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with TemporaryDirectory(dir=destination.parent) as staging:
        corpus = Path(staging) / ADDITIONS.stem
        corpus.mkdir()
        for row in suite.ROWS:
            (corpus / row).mkdir()
            prompt, output, _ = suite.ROWS[row]
            for lane in metadata(row):
                ids = token_ids(tokenizer, row, lane, prompt)
                case = {"prompt_token_ids": ids, "output_tokens": output}
                path = corpus / row / f'{lane["lane"]:02d}.request.json'
                path.write_bytes(client.json_bytes(client.request_body(case)))
            load_lanes(corpus, row)
        require(not destination.exists(), f"Corpus destination already exists: {destination}")
        corpus.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare")
    freeze.add_argument("tokenizer", type=Path)
    freeze.add_argument("destination", type=Path)
    commands.add_parser("verify").add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.tokenizer, args.destination)
    rows = {}
    for row in suite.ROWS:
        rows[row] = {**suite.geometry(row, 2), "request_sha256": [
            meta["request_sha256"] for meta, _, _ in load_lanes(args.destination, row)]}
    print(json.dumps({"version": ADDITIONS.stem, "manifest_sha256": ADDITIONS_SHA256,
                      "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
