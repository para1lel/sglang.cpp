"""Collect and replay matched K4 timing protocols."""

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from benchmarks.concurrent import client as concurrent
from benchmarks.streaming import check_vllm, client as streaming
import cold_cache
import default_evidence as defaults
import workload_suite
from identity import verify_stream_end
from makespan import cohort_makespan_ns, compare, makespan_criterion, validate_cohort
from saved_evidence import check_native_dispatch, command, device, digest, fields, load
from saved_evidence import native_process, read_native_dispatch, require, saved_worker


ROWS = (*streaming.SHAPES, *concurrent.ROWS)
PROTOCOL = "matched-k4-v1"
OPT_IN_PROTOCOL = "matched-k4-opt-in-v1"
ARM_ORDER = ("native-k0", "native-k4-trace", "native-k4", "vllm-k4")
EXACT_CONTROL = "native-exact-k0"
SINGLE_ROUND_SAMPLING = "single-round-v1"
ORDERED_PROFILE = "ordered-tc-int8"
SPLIT_PROFILE = "ordered-tc-int8-split-gqa"
LONG_SPLIT_PROFILE = "ordered-tc-int8-long-split-gqa"
SPLIT_FIELD = "experimental_target_split_gqa"
LONG_SPLIT_FIELD = "experimental_long_target_split_gqa"
TRACE_INTEGER = r"(0|[1-9][0-9]*)"
NATIVE_MTP_LINE = re.compile("batch mtp " + " ".join(
    f"{field}={TRACE_INTEGER}"
    for field in ("slot", "drafted", "accepted", "verified", "replayed", "boundary")))
NATIVE_VERIFY_LINE = re.compile("batch verify " + " ".join(
    f"{field}={TRACE_INTEGER}" for field in ("lanes", "rows", "peak_request_bytes")))


def schedule(row, arm, identity=None):
    if arm in (*ARM_ORDER[:2], EXACT_CONTROL):
        return ["control"]
    if single_round(identity):
        return ["warmup-01", "measured-01"]
    if row in streaming.SHAPES:
        return concurrent.tags({"warmup_cohorts": streaming.ARCHIVED_WARMUPS,
                                "measured_cohorts": streaming.ARCHIVED_REPEATS})
    return concurrent.tags(concurrent.reference()["policy"])


def save(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def fixture(row, fixtures, identity=None):
    if current_campaign(identity):
        require(row in workload_suite.ROWS, f"{row} is unavailable in this workload suite")
        if row in workload_suite.CONCURRENT_FIXTURES:
            gpu_count = len(identity["participating_gpus"])
            return workload_suite.load_concurrent(fixtures, row, gpu_count)
    if row in concurrent.ROWS:
        return concurrent.load_lanes(fixtures, row)
    case, _ = streaming.load_cases(Path(streaming.__file__).parent)[row]
    body = streaming.json_bytes(streaming.request_body(case))
    return [({"lane": 0, "request_sha256": hashlib.sha256(body).hexdigest()}, body, case)]


def single_round(identity):
    if not identity:
        return False
    sampling = identity.get("sampling")
    require("sampling" not in identity or sampling == SINGLE_ROUND_SAMPLING,
            "Unknown K4 sampling policy")
    return (sampling == SINGLE_ROUND_SAMPLING
            or identity.get("protocol") in (workload_suite.PROTOCOL_V5,
                                               *workload_suite.LMCACHE_PROTOCOLS))


def current_campaign(identity):
    return bool(identity and identity["protocol"] in workload_suite.WORKLOAD_PROTOCOLS)


def workload_evidence(row, identity, lanes=None):
    gpu_count = len(identity["participating_gpus"])
    geometry = workload_suite.geometry(row, gpu_count)
    if lanes is not None:
        metadata = [lane for lane, _, _ in lanes]
    elif row in workload_suite.CONCURRENT_FIXTURES:
        metadata = workload_suite.concurrent_metadata(row, gpu_count)
    else:
        metadata = [lane for lane, _, _ in fixture(row, None, identity)]
    hashes = [lane["request_sha256"] for lane in metadata]
    require(len(hashes) == geometry["concurrent_requests"], "Wrong current workload width")
    return {**geometry, "request_sha256": hashes}


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
    duration = cohort_makespan_ns([replay])
    return duration, [replay["output_ids"]], {
        "output_tokens_per_second": len(replay["output_ids"]) * 1e9 / duration,
        "request_latencies": [replay["metrics"]]}


def replay_cohort(raw, row, tag, lanes, policy, binding):
    require((raw["row"], raw["tag"]) == (row, tag), "Wrong raw cohort row or tag")
    require(raw.get("binding") == binding, "Wrong raw cohort campaign or arm identity")
    for record in raw["requests"]:
        for _ in streaming.completion_frames(record["events"]):
            pass
    if row in streaming.SHAPES:
        record, = raw["requests"]
        return replay_short(record, lanes[0])
    result = validate_cohort(raw, lanes, policy)
    return result["makespan_ns"], [record["output_ids"] for record in raw["requests"]], result


def replay(raw, row, tag, lanes, policy, binding=None):
    duration, outputs, _ = replay_cohort(raw, row, tag, lanes, policy, binding)
    return duration, outputs


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
        require("native_profile" not in identity and "native_config" not in identity,
                "Exact campaign cannot select an opt-in native configuration")
        return None
    profile = identity["native_profile"]
    if identity["protocol"] in defaults.DEFAULT_PROTOCOLS:
        require(profile == defaults.PROFILE and receipt_version(identity) == 2
                and "native_config" not in identity
                and Path(identity["checkpoint_path"]).is_absolute()
                and type(identity["port"]) is int and 1 <= identity["port"] <= 65535
                and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
                                 identity["campaign_id"]),
                "Invalid native defaults campaign identity")
        if identity["protocol"] in defaults.CACHE_ON_PROTOCOLS:
            require(identity.get("cache_policy") == defaults.CACHE_POLICY,
                    "Invalid cold-cache campaign policy")
        else:
            require("cache_policy" not in identity, "Historical defaults do not reset the cache")
        if current_campaign(identity):
            workload_suite.validate_identity(identity)
            require(makespan_criterion(identity) is not None, "Missing current makespan criterion")
        return identity, hashlib.sha256(raw).hexdigest()
    config = identity["native_config"]
    expected = {"fp8_prefill": "tensor-core-mtp-target", "mtp_steps": 4, "mtp_int8_head": True}
    if profile == SPLIT_PROFILE:
        expected[SPLIT_FIELD] = True
    if profile == LONG_SPLIT_PROFILE:
        expected[LONG_SPLIT_FIELD] = True
    require(identity["protocol"] == OPT_IN_PROTOCOL
            and profile in (ORDERED_PROFILE, SPLIT_PROFILE, LONG_SPLIT_PROFILE)
            and re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
                             identity["campaign_id"])
            and config.keys() == expected.keys()
            and all(type(config[key]) is type(value) and config[key] == value
                    for key, value in expected.items()),
            "Unknown opt-in native campaign configuration")
    if profile == LONG_SPLIT_PROFILE:
        require(receipt_version(identity) == 2, "Long split requires live dispatch receipts")
    return identity, hashlib.sha256(raw).hexdigest()


def receipt_identity(directory, selected):
    if selected:
        return selected[0]
    path = directory / "identity.json"
    return load(path) if path.exists() else {}


def receipt_version(identity):
    version = identity.get("replay_receipts", 0)
    require(type(version) is int and version in (0, 1, 2), "Unknown K4 replay receipt version")
    require("makespan_criterion" not in identity or version == 2,
            "A K4 makespan criterion requires version 2 replay receipts")
    return version


def arm_binding(directory, arm, selected, version):
    run = directory.parent
    require(selected is not None or arm != EXACT_CONTROL,
            "Exact diagnostic belongs to an opt-in campaign")
    if selected is None and not version:
        return None
    if default_campaign(selected):
        names = {"argv": "live-argv.bin", "server_info": "server-info.json"}
        if arm == "vllm-k4":
            names["worker"] = "effective-worker.json"
        names["environment"] = "live-environment.bin"
        return {"campaign_sha256": selected[1], "arm": arm, **{
            f"{key}_sha256": digest(run / name) for key, name in names.items()}}
    split_mode = ""
    if selected is not None and arm != "vllm-k4":
        _, split_mode = native_info(run, selected[0], opt_in=True)
    fingerprint = selected[1] if selected else digest(run.parent / "identity.json")
    binding = {"campaign_sha256": fingerprint, "arm": arm}
    if version and arm != "vllm-k4":
        binding["native_argv_sha256"] = digest(run / "live-argv.bin")
        if version == 2:
            dispatch = run / "live-dispatch.bin"
            check_native_dispatch(dispatch, split_mode)
            binding["native_dispatch_sha256"] = digest(dispatch)
    return binding


def default_campaign(selected):
    return selected is not None and selected[0]["protocol"] in defaults.DEFAULT_PROTOCOLS


def cache_on_campaign(selected):
    return selected is not None and selected[0]["protocol"] in defaults.CACHE_ON_PROTOCOLS


def native_info(run, identity, *, opt_in):
    arm = run.name
    info = load(run / "server-info.json")
    config = (dict(identity["native_config"]) if opt_in else
              {"fp8_prefill": "exact", "mtp_steps": 4, "mtp_int8_head": False})
    config.update(trace_batches=arm == "native-k4-trace", max_running_requests=32,
                  max_num_batched_tokens=8192, prefix_cache_entries=0, batch_cuda_graphs=True,
                  eager=False, host="127.0.0.1", device=0)
    split = config.pop(SPLIT_FIELD, False) and arm != EXACT_CONTROL
    long_split = config.pop(LONG_SPLIT_FIELD, False) and arm != EXACT_CONTROL
    if arm in ("native-k0", EXACT_CONTROL):
        config.update(mtp_steps=0, mtp_int8_head=False)
    if arm == EXACT_CONTROL:
        config["fp8_prefill"] = "exact"
    if not opt_in:
        info.setdefault("mtp_int8_head", False)
    require(all(type(info.get(key)) is type(value) and info[key] == value
                for key, value in config.items()) and info.get(SPLIT_FIELD, False) is split
            and info.get(LONG_SPLIT_FIELD, False) is long_split,
            f"{arm}: wrong native server configuration")
    require((run / "running-binary.sha256").read_text().split()[0]
            == identity["native_binary_sha256"], f"{arm}: wrong running native binary")
    return info, "long" if long_split else "1" if split else ""


def cache_engine(identity, arm):
    if arm == "vllm-k4":
        return "vllm"
    if identity.get("protocol") == workload_suite.LMCACHE_DRAIN_PROTOCOL:
        return cold_cache.LMCACHE_DRAIN
    if identity.get("protocol") in workload_suite.LMCACHE_PROTOCOLS:
        return "lmcache"
    return "native"


def capture(row, arm, endpoint, fixtures, directory, native_control=None):
    url = urlsplit(endpoint)
    require(url.scheme == "http" and url.hostname == "127.0.0.1" and url.port
            and url.path in ("", "/") and not (url.query or url.fragment or url.username),
            "Expected a loopback HTTP origin")
    policy = concurrent.reference()["policy"]
    row_directory = directory.parent.parent
    selected = campaign(row_directory)
    identity = receipt_identity(row_directory, selected)
    lanes = fixture(row, fixtures, identity)
    version = receipt_version(identity)
    binding = arm_binding(directory, arm, selected, version)
    expected = None
    warmup_outputs = None
    if default_campaign(selected):
        require(arm in ("native-k4", "native-k4-trace", "vllm-k4")
                and native_control is None, "A default launch has no forced K0 control")
    elif arm in ("native-k4", "native-k4-trace", EXACT_CONTROL):
        require(native_control is not None, "Provide the native K0 control")
        verify_request_files(native_control, lanes)
        control_binding = arm_binding(native_control, "native-k0", selected, version)
        _, expected = replay(read_raw(native_control, "control"), row, "control", lanes, policy,
                             control_binding)
    directory.mkdir()
    for lane, body, _ in lanes:
        (directory / f'{lane["lane"]:02d}.request.json').write_bytes(body)
    cache_on = cache_on_campaign(selected)
    engine = cache_engine(identity, arm)
    previous_cache = None
    for tag in schedule(row, arm, identity):
        cache_receipt = cold_cache.before_cohort(url, engine) if cache_on else None
        if row in streaming.SHAPES:
            _, body, case = lanes[0]
            raw = {"row": row, "tag": tag, "requests": [streaming.measure(url, case, body)]}
        else:
            with patch.object(concurrent.http.client, "HTTPConnection",
                              streaming.StreamingConnection):
                raw = concurrent.capture_cohort(url, row, lanes, tag, policy)
        if binding is not None:
            raw["binding"] = binding
        try:
            if cache_receipt is not None:
                raw["cold_cache"] = cache_receipt
                cold_cache.after_cohort(url, engine, cache_receipt)
        finally:
            with gzip.open(directory / f"{tag}.json.gz", "wt", compresslevel=3) as stream:
                json.dump(raw, stream, separators=(",", ":"), allow_nan=False)
        _, outputs = replay(raw, row, tag, lanes, policy, binding)
        if cache_on:
            previous_cache = cold_cache.replay(raw, engine, previous_cache)
        if arm == EXACT_CONTROL:
            identity, _ = selected
            diagnostic = {"protocol": OPT_IN_PROTOCOL, "row": row, **binding,
                          **exact_compatibility(expected, outputs, identity["native_profile"])}
            save(directory.parent.parent / "exact-k0-compatibility.json", diagnostic)
        else:
            require(expected is None or outputs == expected,
                    f"{tag}: native K4 differs from native K0")
        if single_round(identity):
            warmup_outputs = verify_single_round_outputs(arm, tag, outputs, warmup_outputs)


def verify_single_round_outputs(arm, tag, outputs, warmup_outputs):
    if tag == "warmup-01":
        return outputs
    if tag == "measured-01":
        require(outputs == warmup_outputs, f"{arm}: measured output IDs differ from the warmup")
    return warmup_outputs


def exact_compatibility(expected, exact, profile):
    id_key = "ordered_id" if profile == ORDERED_PROFILE else "policy_id"
    lanes = []
    for policy_ids, exact_ids in zip(expected, exact, strict=True):
        differences = [index for index, (policy_id, exact_id) in
                       enumerate(zip(policy_ids, exact_ids, strict=True)) if policy_id != exact_id]
        first = None
        if differences:
            index = differences[0]
            first = {"output_index": index, id_key: policy_ids[index],
                     "exact_id": exact_ids[index]}
        lanes.append({"different_ids": len(differences), "first_difference": first})
    return {"diagnostic_only": True, "all_ids_match": expected == exact, "lanes": lanes}


def verify_arm_receipts(run, identity, *, opt_in, version):
    required = version > 0
    status = run / "client-exit-code.txt"
    if required or status.exists():
        require(status.read_text().strip() == "0", f"{run.name}: client did not succeed")
    if run.name == "vllm-k4":
        saved_activity(run, "vllm", ("metrics-before.txt", "metrics-after.txt"), required)
        return
    checked = run / "server-check.txt"
    if required or checked.exists():
        require(load(checked) is True, f"{run.name}: server configuration check failed")
    have_argv = required or (run / "live-argv.bin").exists()
    have_info = identity and (run / "server-info.json").exists()
    if have_argv or (have_info and not opt_in):
        info, split_mode = native_info(run, identity, opt_in=opt_in)
        if have_argv:
            native_process(run, identity, info, opt_in=opt_in, split_mode=split_mode,
                           dispatch_required=version == 2)
    if run.name == "native-k4-trace":
        saved_activity(run, "native", ("server.log",), required)


def verify_row_status(directory, result, required):
    comparison = directory / "comparison.json"
    if comparison.exists():
        require(json.dumps(load(comparison), sort_keys=True) == json.dumps(result, sort_keys=True),
                "Saved comparison differs from replay")
    status = directory / "exit-code.txt"
    if status.exists() or (required and (directory / "finished-utc.txt").exists()):
        require(status.read_text().strip() == str(int(not result["passed"])),
                "Saved row exit status differs from replay")


def verify_single_round_comparison(row, receipt, criterion):
    workload = receipt.get("workloads", {}).get(row, {})
    samples = [workload.get(arm, {}).get("samples_ns") for arm in ("native", "vllm")]
    require(all(isinstance(values, list) and len(values) == 1
                and type(values[0]) is int and values[0] > 0 for values in samples),
            f"{row}: expected one measured makespan per engine")
    native, vllm = samples
    expected = compare({row: native}, {row: vllm}, eligible=True, criterion=criterion)
    require(expected["passed"] and workload == expected["workloads"][row],
            f"{row}: measured makespan does not certify a pass")


def check_prior_rows(row, root):
    identity_bytes = (root / "identity.json").read_bytes()
    identity = json.loads(identity_bytes)
    criterion = makespan_criterion(identity)
    require(criterion is not None, "Prior rows require the current K4 makespan criterion")
    fingerprint = hashlib.sha256(identity_bytes).hexdigest()
    rows = tuple(workload_suite.ROWS) if current_campaign(identity) else ROWS
    require(row in rows, f"{row} is unavailable in this workload suite")
    if current_campaign(identity):
        workload_suite.validate_identity(identity)
    previous_rows = rows[:rows.index(row)]
    wide_budget = identity["protocol"] in workload_suite.WIDE_BUDGET_PROTOCOLS
    if wide_budget and row in ("W4", "W5"):
        previous_rows = ("W4",) if row == "W5" else ()
    for previous in previous_rows:
        directory = root / previous
        comparison_path = directory / "comparison.json"
        requirement = (f"{previous} must pass under this campaign's makespan "
                       f"criterion before {row}")
        require(all((directory / name).is_file() for name in (
            "comparison.json", "identity.json", "exit-code.txt")), requirement)
        comparison = load(comparison_path)
        require((directory / "exit-code.txt").read_text().strip() == "0"
                and (directory / "identity.json").read_bytes() == identity_bytes
                and comparison.get("row") == previous
                and comparison.get("protocol") == identity["protocol"]
                and comparison.get("campaign_sha256") == fingerprint
                and makespan_criterion(comparison) == criterion
                and comparison.get("passed") is True
                and (not current_campaign(identity) or comparison.get("workload") ==
                     workload_evidence(previous, identity))
                and (not wide_budget or comparison.get("max_num_batched_tokens") ==
                     workload_suite.V4_TOKEN_BUDGET), requirement)
        if single_round(identity):
            require(comparison.get("sampling") == identity.get("sampling"), requirement)
            verify_single_round_comparison(previous, comparison, criterion)


def assess(row, fixtures, directory):
    policy = concurrent.reference()["policy"]
    selected = campaign(directory)
    identity = receipt_identity(directory, selected)
    current = current_campaign(identity)
    lanes = fixture(row, fixtures, identity)
    criterion = makespan_criterion(identity)
    version = receipt_version(identity)
    use_defaults = default_campaign(selected)
    cache_on = cache_on_campaign(selected)
    cache_receipts = {}
    if use_defaults:
        require(digest(directory / "protocol.sha256") == identity["protocol_sha256"],
                "Saved protocol manifest differs from the default campaign")
    samples = {"native-k4": [], "vllm-k4": []}
    diagnostics = {arm: {} for arm in samples}
    expected = None
    exact = None
    if use_defaults:
        arms = ARM_ORDER[1:]
    else:
        arms = (ARM_ORDER[0], EXACT_CONTROL, *ARM_ORDER[1:]) if selected else ARM_ORDER
    arms = tuple(arm for arm in arms if row == "W1" or arm != "native-k4-trace")
    if single_round(identity):
        require(not any((directory / arm).exists()
                        for arm in (*ARM_ORDER, EXACT_CONTROL) if arm not in arms),
                "Unexpected arm for this campaign profile")
    for arm in arms:
        measurement = directory / arm / "measurement"
        if use_defaults:
            verify_default_receipts(row, measurement.parent, identity)
        else:
            verify_arm_receipts(measurement.parent, identity, opt_in=selected is not None,
                                version=version)
        binding = arm_binding(measurement, arm, selected, version)
        verify_request_files(measurement, lanes)
        tags = schedule(row, arm, identity)
        require({path.name for path in measurement.glob("*.json.gz")}
                == {f"{tag}.json.gz" for tag in tags}, f"{arm}: wrong cohort set")
        engine = cache_engine(identity, arm)
        if cache_on:
            cache_receipts[arm] = {}
        warmup_outputs = None
        previous_cache = None
        for tag in tags:
            raw = read_raw(measurement, tag)
            duration, outputs, metrics = replay_cohort(raw, row, tag, lanes, policy, binding)
            if cache_on:
                previous_cache = cold_cache.replay(raw, engine, previous_cache)
                cache_receipts[arm][tag] = previous_cache
            if arm == "native-k0":
                expected = outputs
            elif arm == EXACT_CONTROL:
                exact = outputs
            elif not use_defaults and arm != "vllm-k4":
                require(outputs == expected, f"{arm} {tag}: native K4 differs from native K0")
            if single_round(identity):
                warmup_outputs = verify_single_round_outputs(arm, tag, outputs, warmup_outputs)
            if tag.startswith("measured-"):
                samples[arm].append(duration)
                if current:
                    diagnostics[arm][tag] = {
                        **metrics, "all_lanes_generating_before_first_done":
                        metrics.get("all_generating_overlap_ms", 1) > 0,
                    }
    result = {"protocol": PROTOCOL, "row": row, **compare(
        {row: samples["native-k4"]}, {row: samples["vllm-k4"]},
        eligible=True, criterion=criterion)}
    if criterion is not None and selected is None:
        result["campaign_sha256"] = digest(directory / "identity.json")
    if use_defaults:
        result.update(protocol=identity["protocol"], native_profile=defaults.PROFILE,
                      campaign_sha256=selected[1], vllm_role="timing-only")
        if cache_on:
            result.update(cache_policy=identity["cache_policy"], cache_receipts=cache_receipts)
        if identity["protocol"] in workload_suite.LMCACHE_SLC3_PROTOCOLS:
            result.update(lmcache_version=identity["lmcache_version"],
                          lmcache_wheel_sha256=defaults.lmcache_evidence.WHEEL_SHA256,
                          lmcache_wire=identity["lmcache_wire"],
                          lmcache_ready_version=identity["lmcache_ready_version"])
    elif selected:
        identity, fingerprint = selected
        result.update(protocol=OPT_IN_PROTOCOL, native_profile=identity["native_profile"],
                      native_config=identity["native_config"], campaign_sha256=fingerprint,
                      native_k0_parity=True,
                      exact_k0_compatibility=exact_compatibility(
                          expected, exact, identity["native_profile"]))
    if current:
        result.update(workload=workload_evidence(row, identity, lanes),
                      cohort_metrics=diagnostics)
    if identity.get("protocol") in workload_suite.WIDE_BUDGET_PROTOCOLS:
        result["max_num_batched_tokens"] = defaults.token_budget(identity)
    if "sampling" in identity:
        result["sampling"] = identity["sampling"]
    verify_row_status(directory, result, version > 0)
    return result


def vllm_worker_evidence(row, run, reservation, checkpoint, *, before_measurement=False,
                         prefix_cache=False, max_seqs=32, token_budget=8192, lmcache=False):
    cpus = fields(reservation / "cpus.txt")
    group = "W1-W2" if row in streaming.SHAPES else row
    evidence = saved_worker(run, run, cpus, device(reservation)["uuid"], group, speculative=True,
                            before_measurement=before_measurement, prefix_cache=prefix_cache,
                            max_seqs=max_seqs, paired_budget=token_budget)
    worker = evidence["effective"]
    config = load(run / "server-info.json")["vllm_config"]
    if lmcache:
        defaults.require_vllm_local_apc(config, command(run, "server", cpus))
    verify_vllm_model(worker, config, checkpoint, token_budget, prefix_cache)
    return evidence, attention_kernels(run)


def verify_vllm_model(worker, config, checkpoint, token_budget, prefix_cache, *, tp=1):
    reference = concurrent.reference()["observed_vllm_settings"]
    require(all(worker[key] == reference[key] for key in ("vllm", "torch", "torch_cuda"))
            and worker["scheduler"]["max_num_scheduled_tokens_effective"] == token_budget,
            "Unexpected vLLM runtime or target row cap")
    check_vllm.audit({"results": [worker]}, speculative=True, prefix_cache=prefix_cache)
    if token_budget == workload_suite.V4_TOKEN_BUDGET:
        defaults.vllm_budget(config, worker, token_budget)
    spec = config["speculative_config"]
    attention = config["attention_config"]
    if prefix_cache:
        cache = config["cache_config"]
        keys = ("enable_prefix_caching", "mamba_cache_mode", "block_size")
        require(all(cache[key] == worker["cache"][key] for key in keys),
                "vLLM API and worker disagree about resolved hybrid prefix caching")
        if tp == 2:
            require(cache["mamba_cache_mode"] == "align" and not cache["user_specified_block_size"]
                    and worker["cache"]["mamba_block_size"] == cache["block_size"]
                    and cache["mamba_block_size"] in (64, cache["block_size"]),
                    "Incorrect TP2 worker-local aligned Mamba pages")
        else:
            require(cache["mamba_block_size"] == worker["cache"]["mamba_block_size"],
                    "vLLM API and worker disagree about resolved hybrid prefix caching")
    require(spec["method"] == "mtp" and type(spec["num_speculative_tokens"]) is int
            and spec["num_speculative_tokens"] == 4
            and Path(spec["model"]).resolve() == checkpoint
            and Path(config["model_config"]["model"]).resolve() == checkpoint
            and attention["use_trtllm_attention"] is True
            and attention["disable_flashinfer_q_quantization"] is True,
            "Unexpected resolved vLLM model, MTP, or attention configuration")


def attention_kernels(run):
    kernels = re.findall(r"FlashInfer resolved query dtypes: ([^\n]+)",
                         (run / "server.log").read_text())
    require(kernels and all(line.startswith(
        "prefill=torch.bfloat16, decode=torch.bfloat16, decode_backend=trtllm-gen")
        and "torch.float8_e4m3fn" in line for line in kernels),
        "Unexpected FlashInfer attention kernel")
    return kernels


def check_vllm_worker(row, run, reservation, checkpoint, *, before_measurement=False):
    selected = campaign(run.parent)
    use_defaults = default_campaign(selected)
    max_seqs = defaults.serving_capacity(selected[0]) if use_defaults else 32
    lmcache = use_defaults and selected[0]["protocol"] in workload_suite.LMCACHE_PROTOCOLS
    token_budget = defaults.token_budget(selected[0]) if use_defaults else 8192
    evidence, kernels = vllm_worker_evidence(row, run, reservation, checkpoint,
                                             before_measurement=before_measurement,
                                             prefix_cache=cache_on_campaign(selected),
                                             max_seqs=max_seqs, token_budget=token_budget,
                                             lmcache=lmcache)
    if before_measurement:
        require(default_campaign(selected), "Expected a native defaults campaign")
        port = defaults.vllm_argv(run, fields(reservation / "cpus.txt"),
                                  device(reservation)["uuid"],
                                  wide_budget=token_budget == workload_suite.V4_TOKEN_BUDGET)
        require(port == selected[0]["port"], "vLLM endpoint differs from the default campaign")
        return
    save(run / "attention-kernels.json", kernels)
    save(run / "worker-identity.json", evidence)


def verify_default_receipts(row, run, identity):
    require((run / "client-exit-code.txt").read_text().strip() == "0",
            f"{run.name}: client did not succeed")
    if run.name != "vllm-k4":
        require(load(run / "server-check.txt") is True, f"{run.name}: native check failed")
        defaults.native_completed(run, identity)
        if run.name == "native-k4-trace":
            saved_activity(run, "native", ("server.log",), True)
        return
    reservation = run.parent / "reservation"
    cpus = fields(reservation / "cpus.txt")
    require(cpus == {key: identity[f"{key}_cpus"] for key in ("server", "client")}
            | {"memory": identity["memory"]}
            and device(reservation)["uuid"] == identity["gpu_uuid"].lower(),
            "vLLM reservation differs from the default campaign")
    token_budget = defaults.token_budget(identity)
    port = defaults.vllm_argv(run, cpus, identity["gpu_uuid"].lower(),
                              wide_budget=token_budget == workload_suite.V4_TOKEN_BUDGET)
    require(port == identity["port"], "vLLM endpoint differs from the default campaign")
    require((run / "live-environment-end.bin").read_bytes()
            == (run / "live-environment.bin").read_bytes(),
            "vLLM environment changed during use")
    defaults.client(run, identity, port)
    cache_on = identity["protocol"] in defaults.CACHE_ON_PROTOCOLS
    evidence, kernels = vllm_worker_evidence(row, run, reservation,
                                             Path(identity["checkpoint_path"]),
                                             prefix_cache=cache_on,
                                             max_seqs=defaults.serving_capacity(identity),
                                             token_budget=token_budget,
                                             lmcache=identity["protocol"] in
                                             workload_suite.LMCACHE_PROTOCOLS)
    require(load(run / "worker-identity.json") == evidence
            and load(run / "attention-kernels.json") == kernels,
            "Saved vLLM worker differs from the resolved receipts")
    if identity["protocol"] in workload_suite.LMCACHE_GPU_PROTOCOLS:
        defaults.lmcache_evidence.pinned(run, identity, evidence["worker_pid"])
    saved_activity(run, "vllm", ("metrics-before.txt", "metrics-after.txt"), True)


def native_activity(text, require_groups):
    lines = text.splitlines()
    grouped = require_groups or any(line.startswith("batch verify") for line in lines)
    lanes = rows = 0
    last_slot = None
    evidence = {"rounds": 0, "depth_four_rounds": 0}
    for number, line in enumerate(lines, 1):
        where = f"native MTP trace line {number}"
        if line.startswith(("batch width=", "batch verify")):
            require(lanes == rows == 0, f"Incomplete verification group before {where}")
            last_slot = None
            if line.startswith("batch verify"):
                match = NATIVE_VERIFY_LINE.fullmatch(line)
                require(match is not None, f"Malformed verification header at {where}")
                lanes, rows, _ = map(int, match.groups())
                require(lanes <= 32 and lanes <= rows <= 5 * lanes,
                        f"Invalid verification header counts at {where}")
                last_slot = -1
            continue
        if not line.startswith("batch mtp"):
            continue
        match = NATIVE_MTP_LINE.fullmatch(line)
        require(match is not None, f"Malformed native MTP event at {where}")
        slot, drafted, accepted, verified, replayed, boundary = map(int, match.groups())
        require(slot < 32 and boundary <= 1, f"Invalid native MTP slot or boundary at {where}")
        if boundary:
            require(drafted == accepted == verified == replayed == 0,
                    f"Invalid native MTP boundary counts at {where}")
        else:
            require(0 <= accepted <= drafted <= 4 and 1 <= verified <= 5
                    and drafted <= verified <= drafted + 1
                    and (replayed == 0 or accepted <= replayed <= accepted + 1)
                    and replayed < verified, f"Invalid native MTP round counts at {where}")
        if grouped:
            require(last_slot is not None and slot > last_slot,
                    f"Native MTP event has no verification header or repeats a slot at {where}")
            last_slot = slot
            if verified:
                lanes -= 1
                rows -= verified
                require(lanes >= 0 and rows >= 0, f"Excess verification events at {where}")
        evidence["rounds"] += 1
        evidence["depth_four_rounds"] += drafted == 4 and verified == 5
    require(lanes == rows == 0, "Incomplete verification group at end of native MTP trace")
    require(evidence["depth_four_rounds"] > 0, "No native depth-four verification")
    return evidence


def read_activity(run, engine, *, require_groups=False):
    if engine == "native":
        return native_activity((run / "server.log").read_text(), require_groups)
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
    return evidence


def saved_activity(run, engine, sources, required):
    summary = run / "mtp-activity.json"
    if required or summary.exists() or any((run / name).exists() for name in sources):
        evidence = read_activity(run, engine, require_groups=required)
        if required or summary.exists():
            require(load(summary) == evidence, f"{run.name}: saved MTP activity differs from raw")


def activity(run, engine):
    save(run / "mtp-activity.json", read_activity(run, engine, require_groups=True))


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
    prior = modes.add_parser("check-prior")
    prior.add_argument("row", choices=ROWS)
    prior.add_argument("directory", type=Path)
    worker = modes.add_parser("worker")
    worker.add_argument("row", choices=ROWS)
    worker.add_argument("directory", type=Path)
    worker.add_argument("reservation", type=Path)
    worker.add_argument("checkpoint", type=Path)
    worker.add_argument("--before-measurement", action="store_true")
    check = modes.add_parser("activity")
    check.add_argument("directory", type=Path)
    check.add_argument("engine", choices=("native", "vllm"))
    cache = modes.add_parser("cubins")
    cache.add_argument("cache", type=Path)
    cache.add_argument("directory", type=Path)
    dispatch = modes.add_parser("native-dispatch")
    dispatch.add_argument("pid", type=int)
    environment = modes.add_parser("native-default-environment")
    environment.add_argument("pid", type=int)
    environment.add_argument("--lmcache", action="store_true")
    native_check = modes.add_parser("native-default-check")
    native_check.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "capture":
            directory = args.directory.parent.parent
            identity = receipt_identity(directory, campaign(directory))
            require(single_round(identity), "Live capture requires a single-round campaign")
            capture(args.row, args.arm, args.endpoint, args.fixtures, args.directory,
                    args.native_control)
        elif args.mode == "assess":
            result = assess(args.row, args.fixtures, args.directory)
            comparison = args.directory / "comparison.json"
            if not comparison.exists():
                save(comparison, result)
            print(json.dumps(result, indent=2))
            return int(not result["passed"])
        elif args.mode == "check-prior":
            check_prior_rows(args.row, args.directory)
        elif args.mode == "worker":
            check_vllm_worker(args.row, args.directory, args.reservation, args.checkpoint,
                              before_measurement=args.before_measurement)
        elif args.mode == "cubins":
            cubins(args.cache, args.directory)
        elif args.mode == "native-dispatch":
            sys.stdout.buffer.write(read_native_dispatch(args.pid))
        elif args.mode == "native-default-environment":
            sys.stdout.buffer.write(defaults.read_environment(args.pid, lmcache=args.lmcache))
        elif args.mode == "native-default-check":
            selected = campaign(args.directory.parent)
            require(default_campaign(selected), "Expected a native defaults campaign")
            defaults.native(args.directory, selected[0])
            print("true")
        else:
            activity(args.directory, args.engine)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(2, f"{args.mode}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
