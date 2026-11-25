"""Audit the archived TP2 process groups against the campaign and the selected devices."""

import csv
import json
from pathlib import Path

from benchmarks.concurrent import reference
from saved_evidence import command, compute_owners, digest, fields, hash_manifest, load, require
from tp2_corpus import ADDITIONS_SHA256
import tp2_vllm
import native


def campaign(base):
    identity = load(base / "identity.json")
    sources = base / "protocol.sha256"
    require(identity["protocol"] == "tp2-paired-v1" and identity["mode"] == "measure"
            and identity["protocol_sha256"] == digest(sources)
            and identity["corpus_sha256"] == ADDITIONS_SHA256, "Incorrect campaign protocol")
    project = Path(__file__).resolve().parents[2]
    for name, checksum in hash_manifest(sources).items():
        require(digest(project / name) == checksum, f"Protocol source mismatch: {name}")
    ref = reference.reference()
    require(identity["checkpoint_sha256"] == ref["checkpoint_manifest_sha256"]
            and identity["vllm_pin"] == ref["vllm_pin"], "Incorrect campaign reference")
    reservation = base / "reservation"
    pairs = [line.split() for line in (reservation / "gpu.txt").read_text().splitlines()]
    with (reservation / "devices.csv").open(newline="") as stream:
        inventory = {(d["index"], d["uuid"]) for d in csv.DictReader(stream, skipinitialspace=True)
                     if "B200" in d["name"]}
    devices = identity["devices"]
    require(len(devices) == len(pairs) == len(set(devices)) == 2
            and [gpu for _, gpu in pairs] == devices and all(tuple(p) in inventory for p in pairs),
            "Two reserved B200 devices must match the campaign rank order")
    cpus = fields(reservation / "cpus.txt")
    require(cpus == {"server": identity["server_cpus"], "client": identity["client_cpus"],
                     "memory": "localalloc"}, "Reservation CPU placement mismatch")
    return identity


def verify_arm(run, identity, fixtures):
    base = run.parent.parent
    row = run.parent.name
    devices = identity["devices"]
    ending = load(run / "verified-end.json")
    if run.name == "vllm-k4":
        for gpu in devices:
            compute_owners(run / "compute-before.csv", gpu, ())
        checkpoint, root = Path(identity["checkpoint_path"]), Path(identity["vllm_root"])
        observed = tp2_vllm.verify(run, base / "reservation", devices, row, checkpoint, root, False)
        observed = json.loads(json.dumps(observed))
    else:
        observed = native.verify(run, completed=True)
    require(ending == observed and load(run / "ready.json") == (observed | {"stage": "ready"}),
            "Ready/end evidence differs from the actual worker or process group")
    for gpu in devices:
        compute_owners(run / "compute-after.csv", gpu, ())
    require(not (run / "listeners-after.txt").read_bytes(), "Retired listener remains")
    require((run / "live-argv.bin").read_bytes() == (run / "live-argv-end.bin").read_bytes(),
            "Server command changed during measurement")
    cpus = {"client": identity["client_cpus"], "memory": "localalloc"}
    actual = command(run, "client", cpus)
    script = Path(__file__).with_name("cohorts.py").resolve()
    require(actual == [identity["client_python"], "-B", str(script),
                       "capture", str(run.resolve()), str(fixtures.resolve()), "--endpoint",
                       f"http://127.0.0.1:{identity['port']}"], "Archived capture command mismatch")
