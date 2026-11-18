"""Portable GPU2 reference and exact request identities."""

import hashlib
import json
import struct
import zlib
from base64 import b64decode
from pathlib import Path

from benchmarks.streaming.client import json_bytes, request_body, token_ids


HERE = Path(__file__).resolve().parent
ROWS = {"W3": (8192, 1024, 4), "W4": (32768, 1024, 16),
        "W5": (32768, 4096, 32), "W6": (131072, 1024, 4)}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    return json.loads(path.read_bytes())


def reference():
    data = load_json(HERE / "gpu2-reference.json")
    config = data["observed_vllm_settings"]["config"]
    config["cpu_weight_offload"] = "".join(config.pop("cpu_weight_offload_chunks"))
    arguments = data["launch_python_arguments"]
    profiler = arguments.index("--profiler-config") + 1
    arguments[profiler] = "".join(arguments[profiler])
    kernel_names = ["".join(chunks) for chunks in data.pop("kernel_name_chunks")]
    for row in data["rows"].values():
        golden = row["goldens"]
        golden["zlib_base64"] = "".join(golden.pop("zlib_base64_chunks"))
        for profile in row["route_profiles"].values():
            profile["kernel_signatures"] = {
                kernel_names[index]: count for index, count in profile.pop("kernel_counts")
            }
    return data


def lane_manifest():
    raw = (HERE / "fixture-manifest.json").read_bytes()
    if digest(raw) != reference()["fixture_manifest_sha256"]:
        raise ValueError("Frozen fixture manifest digest mismatch")
    manifest = json.loads(raw)
    first_tokens = []
    for name, (prompt, output, count) in ROWS.items():
        row = manifest["rows"][name]
        if (row["prompt"], row["output"], row["concurrency"]) != (prompt, output, count):
            raise ValueError(f"{name}: wrong fixture shape")
        if [lane["lane"] for lane in row["lanes"]] != list(range(count)):
            raise ValueError(f"{name}: wrong fixture lane sequence")
        first_tokens.extend(lane["first_token"] for lane in row["lanes"])
    if not token_ids(first_tokens) or len(set(first_tokens)) != len(first_tokens):
        raise ValueError("Fixture first tokens must be globally disjoint")
    return manifest


def load_lanes(fixtures, name):
    row = lane_manifest()["rows"][name]
    lanes = []
    for lane in row["lanes"]:
        path = fixtures / name / f'{lane["lane"]:02d}.request.json'
        body = path.read_bytes()
        if digest(body) != lane["request_sha256"]:
            raise ValueError(f"{path}: request digest mismatch")
        prompt = json.loads(body)["prompt"]
        case = {"prompt_token_ids": prompt, "output_tokens": row["output"]}
        if (not token_ids(prompt) or len(prompt) != row["prompt"]
                or prompt[0] != lane["first_token"] or body != json_bytes(request_body(case))):
            raise ValueError(f"{path}: request violates the wire fixture")
        lanes.append((lane, body, case))
    return lanes


def decode_goldens(encoding, tokens, lanes):
    expected_bytes = 4 * tokens * lanes
    inflater = zlib.decompressobj()
    raw = inflater.decompress(b64decode(encoding["zlib_base64"], validate=True), expected_bytes + 1)
    if (len(raw) != expected_bytes or not inflater.eof or inflater.unused_data
            or digest(raw) != encoding["uint32le_sha256"]):
        raise ValueError("Golden output digest or length mismatch")
    return [list(struct.unpack_from(f"<{tokens}I", raw, offset))
            for offset in range(0, expected_bytes, 4 * tokens)]


def goldens(name):
    _, output, lanes = ROWS[name]
    return decode_goldens(reference()["rows"][name]["goldens"], output, lanes)
