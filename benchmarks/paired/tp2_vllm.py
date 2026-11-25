"""Inspect frozen TP2 W4 requests or verify a pinned two-rank vLLM capture.

Run with PYTHONPATH=. and python -B benchmarks/paired/tp2_vllm.py --help.
The verify command consumes paired run_k4-style command, process, server-info,
compute-ready and compute-measurement-end receipts. Call collective_rpc with
method tp2_snapshot for effective-worker.json. UUID arguments are in rank order.
Verification hashes the launched model's files against the pinned checkpoint manifest.
"""

import argparse
import json
import subprocess
from pathlib import Path

from benchmarks.concurrent import reference
from benchmarks.streaming.client import json_bytes, request_body
import default_evidence as defaults
import k4
import workload_suite as suite
from saved_evidence import UUID, arg, command, compute_owners, digest, fields, load, require
from saved_evidence import worker_settings


def rank_identities(workers, devices, inventories):
    require(len(devices) == 2 and all(UUID.fullmatch(uuid) for uuid in devices)
            and len(set(devices)) == 2, "Expected two distinct GPU UUIDs in rank order")
    require(len(workers) == 2, "Expected two observed worker snapshots")
    ranks = {}
    for worker in workers:
        identity = worker["identity"]
        rank = identity["rank"]
        require(type(rank) is int and rank in (0, 1) and rank not in ranks
                and all(type(identity[key]) is int and identity[key] == rank
                        for key in ("tp_rank", "local_rank", "cuda_device")),
                "Expected distinct matching TP, global, local and CUDA ranks 0 and 1")
        uuid = identity["uuid"].lower().removeprefix("gpu-")
        pid = identity["pid"]
        require(devices[rank] == "gpu-" + uuid and "B200" in identity["device_name"]
                and type(pid) is int and pid > 0, "Worker physical device or PID mismatch")
        for inventory in inventories:
            compute_owners(inventory, devices[rank], [pid])
        ranks[rank] = identity
    require(ranks[0]["pid"] != ranks[1]["pid"], "Expected separate worker processes")
    return [ranks[0], ranks[1]]


def verify_checkpoint(checkpoint, manifest, expected_manifest):
    require(digest(manifest) == expected_manifest, "Checkpoint manifest digest mismatch")
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = checkpoint / name
        require(digest(path) == expected, f"Checkpoint file digest mismatch: {path}")


def verify_dispatch(worker):
    expected = {"prefill_dtype": "torch.bfloat16", "decode_dtype": "torch.bfloat16",
                "kv_dtype": "torch.float8_e4m3fn", "decode_backend": "trtllm-gen"}
    coverage = {}
    for role in ("runner", "speculator"):
        layers = set()
        for entry in worker["flashinfer_dispatch"][role]:
            require(all(entry.get(key) == value for key, value in expected.items()),
                    f"Rank {worker['identity']['rank']}: wrong resolved FlashInfer dispatch")
            layers.update(entry["layers"])
        coverage[role] = layers
    draft = set(worker["draft_attention_layers"])
    require(len(draft) == 1 and coverage["speculator"] == draft and draft <= coverage["runner"]
            and len(coverage["runner"]) == 17,
            "Expected 16 target and one draft GQA layer, including separate speculator dispatch")


def verify(run, reservation, devices, row, checkpoint, vllm_root, ready_only):
    devices = [value.lower() for value in devices]
    workers = load(run / "effective-worker.json")["results"]
    inventories = [run / "compute-ready.csv"]
    if not ready_only:
        end = load(run / "effective-worker-end.json")["results"]
        order = lambda worker: worker["identity"]["rank"]
        require(sorted(workers, key=order) == sorted(end, key=order),
                "Changed TP2 worker identity, dispatch or configuration during measurement")
        inventories.append(run / "compute-measurement-end.csv")
        require((run / "live-environment-end.bin").read_bytes()
                == (run / "live-environment.bin").read_bytes(), "Changed server environment")
    ranks = rank_identities(workers, devices, inventories)
    cpus = fields(reservation / "cpus.txt")
    defaults.vllm_argv(run, cpus, ",".join(devices), wide_budget=True)
    argv = command(run, "server", cpus)
    pin = reference.reference()
    revision = subprocess.check_output(
        ["git", "--no-replace-objects", "-C", vllm_root / "source", "rev-parse", "HEAD"],
        text=True).strip()
    require(revision == pin["vllm_pin"]
            and argv[argv.index("-m") - 2] == str(vllm_root / "venv/bin/python")
            and Path(arg(argv, "--model")).resolve() == checkpoint
            and arg(argv, "--tensor-parallel-size") == "2",
            "Expected pinned vLLM interpreter, source, checkpoint and TP2 launch")
    verify_checkpoint(checkpoint, vllm_root / "checkpoint.sha256",
                      pin["checkpoint_manifest_sha256"])
    config = load(run / "server-info.json")["vllm_config"]
    parallel = config["parallel_config"]
    require(all(parallel[field] == value for field, value in (
        ("tensor_parallel_size", 2), ("data_parallel_size", 1), ("pipeline_parallel_size", 1))),
        "Expected one TP2 group in server config")
    defaults.require_vllm_local_apc(config, argv)
    budget = suite.V4_TOKEN_BUDGET
    group = "W1-W2" if row in ("W1", "W2") else row
    for worker in workers:
        require(Path(worker["vllm_file"]).resolve().is_relative_to(vllm_root / "venv"),
                "Worker vLLM module outside pinned environment")
        worker_settings(run, worker, argv, group, speculative=True, prefix_cache=True,
                        max_seqs=suite.serving_capacity(2), paired_budget=budget, tp=2)
        k4.verify_vllm_model(worker, config, checkpoint, budget, True, tp=2)
        verify_dispatch(worker)
    return {"vllm_commit": revision, "stage": "ready" if ready_only else "completed",
            "ranks": ranks, "workload": suite.geometry(row, 2), "token_budget": budget,
            "flashinfer_dispatch": {worker["identity"]["rank"]: worker["flashinfer_dispatch"]
                                   for worker in workers}}


def w4_fixtures(fixtures):
    lanes = suite.load_concurrent(fixtures, "W4", 2)
    for meta, body, case in lanes:
        require(body == json_bytes(request_body(case)) and case["prompt_token_ids"][0]
                == meta["first_token"], "W4 lane violates frozen wire request")
    return {"workload": suite.geometry("W4", 2), "source_row": suite.CONCURRENT_FIXTURES["W4"],
            "lanes": [meta for meta, _, _ in lanes]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    w4 = commands.add_parser("w4", help="verify original W5/00..15.request.json as TP2 W4")
    w4.add_argument("fixtures", type=Path)
    check = commands.add_parser("verify", help="verify both observed workers")
    check.add_argument("row", choices=suite.ROWS)
    for name in ("run", "reservation", "checkpoint", "vllm_root"):
        check.add_argument(name, type=Path)
    check.add_argument("devices", nargs=2, metavar="GPU-UUID")
    check.add_argument("--ready-only", action="store_true", help="require only ready inventory")
    args = parser.parse_args()
    if args.command == "w4":
        result = w4_fixtures(args.fixtures)
    else:
        result = verify(args.run, args.reservation, args.devices, args.row,
                        args.checkpoint.resolve(), args.vllm_root.resolve(), args.ready_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
