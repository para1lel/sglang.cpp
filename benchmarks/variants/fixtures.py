"""Freeze vLLM variant requests from the published, tokenized W3 corpus."""

import argparse
import json
from pathlib import Path

from benchmarks.concurrent.reference import digest, load_lanes
from benchmarks.streaming.client import json_bytes, request_body, token_ids


ROOT = Path(__file__).resolve().parents[2]
ROWS = ("W3-prefix", "W3-MTP")
PREFIX = 6400
PROMPT = 8192
OUTPUT = 1024
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
PRIOR_BUNDLE_SHA256 = "f6ade173f5f8805202027ec270b85362257cc38f855cb936427bce08147ad374"
POLICY = {
    "rows": list(ROWS),
    "prompt_tokens": PROMPT,
    "output_tokens": OUTPUT,
    "concurrency": 4,
    "warmup_cohorts": 2,
    "measured_cohorts": 5,
    "socket_timeout_seconds": 90,
    "cohort_deadline_seconds": 360,
    "barrier_timeout_seconds": 15,
    "max_submission_span_ms": 50,
    "prefix": {
        "shared_tokens": PREFIX,
        "prime_prompt_tokens": PREFIX + 1,
        "prime_output_tokens": 1,
        "per_cohort": "reset, prime once, then submit the four lanes together",
        "cold_control": "reset once, then four unique per-lane cache salts",
    },
    "mtp": {"method": "mtp", "draft_tokens": 1, "prefix_cache": False},
    "wire": "OpenAI text completions SSE, direct IDs, greedy seed 0, ignore_eos",
    "throughput": "useful output IDs divided by first submit through last DONE",
    "latency": "client monotonic clock, linear interpolation at rank (n-1)*p",
    "throughput_margin": "max(0.01, 2*cohort_sample_stdev/cohort_median)",
    "latency_budget": "request p95 + max(0.01*request p95, 2*request_sample_stdev)",
    "failure": "retain original row and stop timed capture on first contract failure",
    "profiling": "separate first four engine steps and four all-lanes decode steps",
}


def fixture_ids(w3):
    prefix = w3[0][:PREFIX]
    tails = [[prompt[0]] + prompt[-(PROMPT - PREFIX - 1):] for prompt in w3]
    if len(set(tail[0] for tail in tails)) != 4:
        raise ValueError("Prefix continuations must diverge at the first suffix token")
    return [prefix + tail for tail in tails]


def make_request(ids, tokens, salt=None):
    body = request_body({"prompt_token_ids": ids, "output_tokens": tokens})
    if salt is not None:
        body["cache_salt"] = salt
    return json_bytes(body)


def freeze(prior, tokenizer_path, destination):
    import tokenizers
    from tokenizers import Tokenizer

    if digest(tokenizer_path.read_bytes()) != TOKENIZER_SHA256:
        raise ValueError("Checkpoint tokenizer digest mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    specials = {entry["id"]: entry["content"]
                for entry in json.loads(tokenizer_path.read_bytes())["added_tokens"]
                if entry["special"]}
    text_delimiters = {"<|im_start|>", "<|im_end|>"}
    original = load_lanes(prior, "W3")
    w3 = [case["prompt_token_ids"] for _, _, case in original]
    shared = fixture_ids(w3)
    boundary = {ids[PREFIX] for ids in shared}
    prime = next(
        shared[0][:PREFIX] + [token] for token in w3[0][PREFIX:]
        if token not in boundary and token not in specials
    )
    manifest = {
        "format": 1,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizers_version": tokenizers.__version__,
        "prior_bundle_sha256": PRIOR_BUNDLE_SHA256,
        "prior_w3_request_sha256": [lane["request_sha256"] for lane, _, _ in original],
        "policy": POLICY,
        "requests": {},
    }

    def save(key, ids, count, cached, salt=None):
        if any(specials[token] not in text_delimiters for token in set(ids) & specials.keys()):
            raise ValueError(f"{key}: unexpected special token in text-only prompt")
        text = tokenizer.decode(ids, skip_special_tokens=False)
        reencoded = tokenizer.encode(text, add_special_tokens=False).ids
        body = make_request(ids, count, salt)
        (destination / f"{key}.request.json").write_bytes(body)
        raw_text = text.encode()
        (destination / f"{key}.txt").write_bytes(raw_text)
        manifest["requests"][key] = {
            "request_sha256": digest(body), "text_sha256": digest(raw_text),
            "prompt_tokens": len(ids), "output_tokens": count,
            "cached_tokens": cached, "decoded_text_roundtrips": reencoded == ids,
        }

    destination.mkdir()
    save("prime", prime, 1, 0)
    for lane, prompt in enumerate(shared):
        save(f"prefix-{lane}", prompt, OUTPUT, PREFIX)
        save(f"cold-{lane}", prompt, OUTPUT, 0, f"w3-prefix-cold-control-{lane}")
        save(f"mtp-{lane}", w3[lane], OUTPUT, 0)
        if (destination / f"mtp-{lane}.request.json").read_bytes() != original[lane][1]:
            raise ValueError("W3-MTP must use the exact archived W3 request bytes")
    (destination / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def load_requests(fixtures, mode):
    manifest = json.loads((fixtures / "manifest.json").read_bytes())
    if manifest["policy"] != POLICY:
        raise ValueError("Variant fixture policy differs from the pinned protocol")
    keys = ["prime"] if mode == "prime" else [f"{mode}-{lane}" for lane in range(4)]
    lanes = []
    for lane, key in enumerate(keys):
        record = manifest["requests"][key]
        body = (fixtures / f"{key}.request.json").read_bytes()
        payload = json.loads(body)
        ids, count = payload["prompt"], record["output_tokens"]
        if (digest(body) != record["request_sha256"] or not token_ids(ids)
                or len(ids) != record["prompt_tokens"]
                or body != make_request(ids, count, payload.get("cache_salt"))):
            raise ValueError(f"{key}: invalid fixture request")
        case = {"prompt_token_ids": ids, "output_tokens": count,
                "cached_tokens": record["cached_tokens"]}
        lanes.append(({"lane": lane, "request_sha256": digest(body)}, body, case))
    return lanes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior_w3_fixture_directory", type=Path)
    parser.add_argument("tokenizer_json", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    freeze(args.prior_w3_fixture_directory, args.tokenizer_json, args.output_directory)


if __name__ == "__main__":
    main()
