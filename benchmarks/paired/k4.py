"""Collect and replay the opt-in matched K4 timing protocol."""

import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from benchmarks.concurrent import client as concurrent
from benchmarks.streaming import check_vllm, client as streaming
from identity import verify_stream_end
from makespan import cohort_makespan_ns, compare, validate_cohort
from saved_evidence import device, digest, fields, load, require, saved_worker


ROWS = (*streaming.SHAPES, *concurrent.ROWS)
PROTOCOL = "matched-k4-v1"
OPT_IN_PROTOCOL = "matched-k4-opt-in-v1"
ARM_ORDER = ("native-k0", "native-k4-trace", "native-k4", "vllm-k4")
EXACT_CONTROL = "native-exact-k0"


def schedule(row, arm):
    if arm in (*ARM_ORDER[:2], EXACT_CONTROL):
        return ["control"]
    if row in streaming.SHAPES:
        return concurrent.tags({"warmup_cohorts": streaming.WARMUPS,
                                "measured_cohorts": streaming.REPEATS})
    return concurrent.tags(concurrent.reference()["policy"])


def save(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def fixture(row, fixtures):
    if row in concurrent.ROWS:
        return concurrent.load_lanes(fixtures, row)
    case, _ = streaming.load_cases(Path(streaming.__file__).parent)[row]
    body = streaming.json_bytes(streaming.request_body(case))
    return [({"lane": 0, "request_sha256": hashlib.sha256(body).hexdigest()}, body, case)]


def verify_request_files(directory, lanes):
    for lane, body, _ in lanes:
        require((directory / f'{lane["lane"]:02d}.request.json').read_bytes() == body,
                f"{directory}: saved request bytes differ from fixture")


def replay_short(record, item):
    lane, _, case = item
    require("error" not in record and record["http_status"] == 200
            and record["request_sha256"] == lane["request_sha256"],
            f"Invalid HTTP outcome or request identity: {record.get('error')}")
    require(any(name.lower() == "content-type" and value.split(";")[0].strip()
                == "text/event-stream" for name, value in record["response_headers"]),
            "Expected text/event-stream")
    replay = {key: record[key] for key in ("submit_ns", "headers_ns")}
    completion = streaming.Completion()
    concurrent.consume_events(iter(record["events"]), replay, case, completion)
    concurrent.finish_record(replay, completion)
    verify_stream_end(record)
    require(replay["submit_ns"] <= replay["headers_ns"] <= replay["first_ns"]
            <= replay["last_ns"] <= replay["done_ns"] <= record["response_end"]["ns"],
            "Request timestamps are out of order")
    require(all(replay[key] == record[key] for key in ("done_ns", "output_ids",
                                                      "finish_reason", "usage"))
            and replay["token_chunks"] == [tuple(chunk) for chunk in record["token_chunks"]]
            and all(replay["metrics"][key] == record["metrics"][key]
                    for key in concurrent.LATENCIES), "Saved stream differs from SSE replay")
    return cohort_makespan_ns([replay]), [replay["output_ids"]]


def replay(raw, row, tag, lanes, policy, binding=None):
    require((raw["row"], raw["tag"]) == (row, tag), "Wrong raw cohort row or tag")
    require(raw.get("binding") == binding, "Wrong raw cohort campaign or arm identity")
    for record in raw["requests"]:
        for _ in streaming.completion_frames(record["events"]):
            pass
    if row in streaming.SHAPES:
        record, = raw["requests"]
        return replay_short(record, lanes[0])
    result = validate_cohort(raw, lanes, policy)
    return result["makespan_ns"], [record["output_ids"] for record in raw["requests"]]


def read_raw(directory, tag):
    with gzip.open(directory / f"{tag}.json.gz", "rt") as stream:
        return json.load(stream)


def campaign(directory):
    path = directory / "identity.json"
    if not path.exists():
        return None
    raw = path.read_bytes()
    identity = json.loads(raw)
    if identity["protocol"] == PROTOCOL:
        return None
    require(identity["protocol"] == OPT_IN_PROTOCOL
            and identity["native_profile"] == "ordered-tc-int8"
            and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
                             identity["campaign_id"])
            and identity["native_config"] == {
                "fp8_prefill": "tensor-core-mtp-target", "mtp_steps": 4, "mtp_int8_head": True},
            "Unknown opt-in native campaign configuration")
    return identity, hashlib.sha256(raw).hexdigest()


def arm_binding(directory, arm, selected):
    if selected is None:
        require(arm != EXACT_CONTROL, "Exact diagnostic belongs to an opt-in campaign")
        return None
    identity, fingerprint = selected
    if arm != "vllm-k4":
        run = directory.parent
        info = load(run / "server-info.json")
        config = dict(identity["native_config"], trace_batches=arm == "native-k4-trace",
                      max_running_requests=32, max_num_batched_tokens=8192, prefix_cache_entries=0,
                      batch_cuda_graphs=True, eager=False, host="127.0.0.1", device=0)
        if arm in ("native-k0", EXACT_CONTROL):
            config.update(mtp_steps=0, mtp_int8_head=False)
        if arm == EXACT_CONTROL:
            config["fp8_prefill"] = "exact"
        require(all(type(info[key]) is type(value) and info[key] == value
                    for key, value in config.items()), f"{arm}: wrong native server configuration")
        require((run / "running-binary.sha256").read_text().split()[0]
                == identity["native_binary_sha256"], f"{arm}: wrong running native binary")
    return {"campaign_sha256": fingerprint, "arm": arm}


def capture(row, arm, endpoint, fixtures, directory, native_control=None):
    url = urlsplit(endpoint)
    require(url.scheme == "http" and url.hostname == "127.0.0.1" and url.port
            and url.path in ("", "/") and not (url.query or url.fragment or url.username),
            "Expected a loopback HTTP origin")
    lanes = fixture(row, fixtures)
    policy = concurrent.reference()["policy"]
    selected = campaign(directory.parent.parent)
    binding = arm_binding(directory, arm, selected)
    expected = None
    if arm in ("native-k4", "native-k4-trace", EXACT_CONTROL):
        require(native_control is not None, "Provide the native K0 control")
        verify_request_files(native_control, lanes)
        control_binding = arm_binding(native_control, "native-k0", selected)
        _, expected = replay(read_raw(native_control, "control"), row, "control", lanes, policy,
                             control_binding)
    directory.mkdir()
    for lane, body, _ in lanes:
        (directory / f'{lane["lane"]:02d}.request.json').write_bytes(body)
    for tag in schedule(row, arm):
        if row in streaming.SHAPES:
            _, body, case = lanes[0]
            raw = {"row": row, "tag": tag, "requests": [streaming.measure(url, case, body)]}
        else:
            with patch.object(concurrent.http.client, "HTTPConnection",
                              streaming.StreamingConnection):
                raw = concurrent.capture_cohort(url, row, lanes, tag, policy)
        if binding is not None:
            raw["binding"] = binding
        with gzip.open(directory / f"{tag}.json.gz", "wt", compresslevel=3) as stream:
            json.dump(raw, stream, separators=(",", ":"), allow_nan=False)
        _, outputs = replay(raw, row, tag, lanes, policy, binding)
        if arm == EXACT_CONTROL:
            diagnostic = {"protocol": OPT_IN_PROTOCOL, "row": row, **binding,
                          **exact_compatibility(expected, outputs)}
            save(directory.parent.parent / "exact-k0-compatibility.json", diagnostic)
        else:
            require(expected is None or outputs == expected,
                    f"{tag}: native K4 differs from native K0")


def exact_compatibility(expected, exact):
    lanes = []
    for policy_ids, exact_ids in zip(expected, exact, strict=True):
        differences = [index for index, (policy_id, exact_id) in
                       enumerate(zip(policy_ids, exact_ids, strict=True)) if policy_id != exact_id]
        first = None
        if differences:
            index = differences[0]
            first = {"output_index": index, "ordered_id": policy_ids[index],
                     "exact_id": exact_ids[index]}
        lanes.append({"different_ids": len(differences), "first_difference": first})
    return {"diagnostic_only": True, "all_ids_match": expected == exact, "lanes": lanes}


def assess(row, fixtures, directory):
    lanes = fixture(row, fixtures)
    policy = concurrent.reference()["policy"]
    selected = campaign(directory)
    samples = {"native-k4": [], "vllm-k4": []}
    expected = None
    exact = None
    arms = (ARM_ORDER[0], EXACT_CONTROL, *ARM_ORDER[1:]) if selected else ARM_ORDER
    for arm in arms:
        if arm == "native-k4-trace" and row != "W1":
            continue
        measurement = directory / arm / "measurement"
        binding = arm_binding(measurement, arm, selected)
        verify_request_files(measurement, lanes)
        tags = schedule(row, arm)
        require({path.name for path in measurement.glob("*.json.gz")}
                == {f"{tag}.json.gz" for tag in tags}, f"{arm}: wrong cohort set")
        for tag in tags:
            duration, outputs = replay(read_raw(measurement, tag), row, tag, lanes, policy, binding)
            if arm == "native-k0":
                expected = outputs
            elif arm == EXACT_CONTROL:
                exact = outputs
            elif arm != "vllm-k4":
                require(outputs == expected, f"{arm} {tag}: native K4 differs from native K0")
            if tag.startswith("measured-"):
                samples[arm].append(duration)
    result = {"protocol": PROTOCOL, "row": row, **compare(
        {row: samples["native-k4"]}, {row: samples["vllm-k4"]}, eligible=True)}
    if selected:
        identity, fingerprint = selected
        result.update(protocol=OPT_IN_PROTOCOL, native_profile=identity["native_profile"],
                      native_config=identity["native_config"], campaign_sha256=fingerprint,
                      native_k0_parity=True,
                      exact_k0_compatibility=exact_compatibility(expected, exact))
    return result


def check_vllm_worker(row, run, reservation, checkpoint):
    cpus = fields(reservation / "cpus.txt")
    group = "W1-W2" if row in streaming.SHAPES else row
    evidence = saved_worker(run, run, cpus, device(reservation)["uuid"], group, speculative=True)
    worker = evidence["effective"]
    reference = concurrent.reference()["observed_vllm_settings"]
    require(all(worker[key] == reference[key] for key in ("vllm", "torch", "torch_cuda"))
            and worker["scheduler"]["max_num_scheduled_tokens_effective"] == 8192,
            "Unexpected vLLM runtime or target row cap")
    check_vllm.audit(load(run / "effective-worker.json"), speculative=True)
    config = load(run / "server-info.json")["vllm_config"]
    spec = config["speculative_config"]
    attention = config["attention_config"]
    require(spec["method"] == "mtp" and spec["num_speculative_tokens"] == 4
            and Path(spec["model"]).resolve() == checkpoint
            and Path(config["model_config"]["model"]).resolve() == checkpoint
            and attention["use_trtllm_attention"] is True
            and attention["disable_flashinfer_q_quantization"] is True,
            "Unexpected resolved vLLM model, MTP, or attention configuration")
    kernels = re.findall(r"FlashInfer resolved query dtypes: ([^\n]+)",
                         (run / "server.log").read_text())
    require(kernels and all(line.startswith(
        "prefill=torch.bfloat16, decode=torch.bfloat16, decode_backend=trtllm-gen")
        and "torch.float8_e4m3fn" in line for line in kernels),
        "Unexpected FlashInfer attention kernel")
    save(run / "attention-kernels.json", kernels)
    save(run / "worker-identity.json", evidence)


def activity(run, engine):
    if engine == "native":
        rounds = re.findall(r"batch mtp slot=\d+ drafted=(\d+) accepted=(\d+) verified=(\d+)",
                            (run / "server.log").read_text())
        evidence = {"rounds": len(rounds), "depth_four_rounds":
                    sum(drafted == "4" and int(verified) >= 4 for drafted, _, verified in rounds)}
        require(evidence["depth_four_rounds"] > 0, "No native depth-four verification")
    else:
        evidence = {}
        for metric, position in (("num_draft_tokens", False),
                                 ("num_verified_tokens_per_pos", True)):
            counts = []
            for moment in ("before", "after"):
                value, = [float(line.split()[-1])
                          for line in (run / f"metrics-{moment}.txt").read_text().splitlines()
                          if line.startswith(f"vllm:spec_decode_{metric}_total{{")
                          and (not position or 'position="3"' in line)]
                require(math.isfinite(value) and value >= 0, f"Invalid vLLM counter {metric}")
                counts.append(value)
            evidence[metric] = counts[1] - counts[0]
            require(evidence[metric] > 0, f"No fresh vLLM K4 activity for {metric}")
    save(run / "mtp-activity.json", evidence)


def cubins(cache, run):
    root = cache / ".cache/flashinfer/cubins"
    verified = {}
    for manifest in root.rglob("checksums.txt"):
        lines = manifest.read_text().splitlines()
        expected = {name: checksum for checksum, name in
                    (line.split(maxsplit=1) for line in lines if line)}
        for binary in manifest.parent.glob("*.cubin"):
            checksum = digest(binary)
            require(expected.get(binary.name) == checksum, f"Invalid cubin {binary}")
            verified[str(binary.relative_to(root))] = checksum
    require(verified, "No pre-downloaded FlashInfer cubins validated")
    save(run / "flashinfer-cubins.json", verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    collect = modes.add_parser("capture")
    collect.add_argument("row", choices=ROWS)
    collect.add_argument("arm", choices=(*ARM_ORDER, EXACT_CONTROL))
    collect.add_argument("endpoint")
    collect.add_argument("fixtures", type=Path)
    collect.add_argument("directory", type=Path)
    collect.add_argument("--native-control", type=Path)
    compare_row = modes.add_parser("assess")
    compare_row.add_argument("row", choices=ROWS)
    compare_row.add_argument("fixtures", type=Path)
    compare_row.add_argument("directory", type=Path)
    worker = modes.add_parser("worker")
    worker.add_argument("row", choices=ROWS)
    worker.add_argument("directory", type=Path)
    worker.add_argument("reservation", type=Path)
    worker.add_argument("checkpoint", type=Path)
    check = modes.add_parser("activity")
    check.add_argument("directory", type=Path)
    check.add_argument("engine", choices=("native", "vllm"))
    cache = modes.add_parser("cubins")
    cache.add_argument("cache", type=Path)
    cache.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "capture":
            capture(args.row, args.arm, args.endpoint, args.fixtures, args.directory,
                    args.native_control)
        elif args.mode == "assess":
            result = assess(args.row, args.fixtures, args.directory)
            save(args.directory / "comparison.json", result)
            print(json.dumps(result, indent=2))
            return int(not result["passed"])
        elif args.mode == "worker":
            check_vllm_worker(args.row, args.directory, args.reservation, args.checkpoint)
        elif args.mode == "cubins":
            cubins(args.cache, args.directory)
        else:
            activity(args.directory, args.engine)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(2, f"{args.mode}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
