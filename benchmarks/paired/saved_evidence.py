"""Read observed device, placement and engine settings from paired run evidence."""

import csv
import hashlib
import json
import os
import re
import shlex
from pathlib import Path


UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fields(path):
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def device(capture):
    index, uuid = (capture / "gpu.txt").read_text().split()
    require(index.isdecimal() and UUID.fullmatch(uuid), f"{capture}: invalid device receipt")
    uuid = uuid.lower()
    with (capture / "devices.csv").open(newline="") as stream:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(stream, skipinitialspace=True)
        ]
    matches = [row for row in rows if row["uuid"].lower() == uuid]
    require(len(matches) == 1 and matches[0]["index"] == index,
            f"{capture}: resolved device missing from inventory")
    return {"uuid": uuid, **{key: value for key, value in matches[0].items()
                            if key not in {"uuid", "index"}}}


def observed_device(run, pid, expected):
    for name in ("compute-ready.csv", "compute-measurement-end.csv"):
        with (run / name).open(newline="") as stream:
            rows = list(csv.reader(stream, skipinitialspace=True))
        seen = [row[0].strip().lower() for row in rows if row and row[1].strip() == pid]
        other = [row[1].strip() for row in rows if row and row[0].strip().lower() == expected
                 and row[1].strip() != pid]
        require(seen == [expected] and not other,
                f"{run / name}: compute PID {pid} must be alone on {expected}; saw {seen}, {other}")


def command(run, name, cpus):
    args = shlex.split((run / f"{name}-command.sh").read_text())
    placement = ["numactl", "--localalloc", "taskset", "--cpu-list", cpus[name]]
    offset = args.index("numactl")
    require(cpus["memory"] == "localalloc" and args[offset:offset + 5] == placement,
            f"{run}: {name} CPU/NUMA placement disagrees with cpus.txt")
    return args[offset + 5:]


def native_process(run, identity, info, *, opt_in, split):
    cpus = {"server": identity["server_cpus"], "memory": identity["memory"]}
    wrapper = command(run, "server", cpus)
    require(wrapper[0] == "env" and wrapper[1].startswith("CUDA_CACHE_PATH="),
            f"{run.name}: wrong native command wrapper")
    expected_env = ["SGLANG_EXPERIMENTAL_TARGET_SPLIT_GQA=1"] if split else []
    argv = wrapper[2 + len(expected_env):]
    require(wrapper[2:2 + len(expected_env)] == expected_env and Path(argv[0]).is_absolute(),
            f"{run.name}: wrong native dispatch environment")
    port = info["port"]
    require(type(port) is int and 1 <= port <= 65535 and Path(info["model_path"]).is_absolute(),
            f"{run.name}: invalid native endpoint or model")
    flags = ["--checkpoint", info["model_path"], "--device", "0", "--port", str(port),
             "--batch-cuda-graphs", "--prefix-cache-entries", "0",
             "--fp8-prefill", info["fp8_prefill"],
             "--max-active", "32", "--max-num-batched-tokens", "8192"]
    if info["trace_batches"]:
        flags.append("--trace-batches")
    if info["mtp_steps"] == 0 or opt_in:
        flags += ["--mtp-steps", str(info["mtp_steps"])]
    if info.get("mtp_int8_head"):
        flags.append("--mtp-int8-head")
    require(argv[1:] == flags, f"{run.name}: command differs from effective native profile")
    observed = (run / "live-argv.bin").read_bytes()
    require(observed == b"\0".join(map(os.fsencode, argv)) + b"\0",
            f"{run.name}: live process arguments differ from its command")
    pid = (run / "server-pid.txt").read_text().strip()
    binary = (run / "running-binary.sha256").read_text().split()
    require(pid.isdecimal() and int(pid) > 0
            and binary == [identity["native_binary_sha256"], f"/proc/{pid}/exe"],
            f"{run.name}: executable receipt differs from the native server PID")
    client = run / "client-command.sh"
    if client.exists():
        args = shlex.split(client.read_text())
        capture = args.index("capture")
        require(args[capture + 1:capture + 4] ==
                [run.parent.name, run.name, f"http://127.0.0.1:{port}"],
                f"{run.name}: client command differs from native endpoint or arm")


def arg(args, key):
    require(args.count(key) == 1, f"Expected one {key} argument")
    return args[args.index(key) + 1]


def hash_manifest(path):
    values = {}
    for line in path.read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        require(SHA256.fullmatch(sha) and name not in values, f"{path}: invalid manifest")
        values[name] = sha
    return values


def artifact_hashes(path):
    values = {}
    for name, sha in hash_manifest(path).items():
        normalized = "/" + name.lstrip("./")
        key = normalized.partition("/benchmarks/")[2]
        if not key or key.startswith("paired/"):
            key = Path(name).name
        require(key not in values, f"{path}: ambiguous protocol entry {key}")
        values[key] = sha
    return values


def vllm_settings(run, args, group, *, speculative=False):
    workers = load(run / "effective-worker.json")["results"]
    require(len(workers) == 1, f"{run}: expected one observed vLLM worker")
    worker = workers[0]
    scheduler = worker["scheduler"]
    cap = scheduler["max_num_batched_tokens"]
    context = 2560 if group == "W1-W2" else 132096
    configured = scheduler.get("max_num_scheduled_tokens_configured")
    scheduled_flag = "--max-num-scheduled-tokens"
    if scheduled_flag in args:
        require(type(configured) is int and configured == cap
                and arg(args, scheduled_flag) == str(cap),
                f"{run}: {scheduled_flag} differs from worker observation")
    else:
        require(configured is None, f"{run}: unexpected configured scheduled token cap")
    historical = cap == 32768 and "max_num_scheduled_tokens_configured" not in scheduler
    if cap == 8192:
        require(scheduled_flag in args and "--async-scheduling" in args,
                f"{run}: expected explicit 8192 scheduled cap and async scheduling")
    effective = scheduler.get("max_num_scheduled_tokens_effective", cap if historical else None)
    async_scheduling = scheduler.get("async_scheduling", True if historical else None)
    require(type(cap) is int and type(effective) is int and cap == effective
            and scheduler["max_num_seqs"] == 32 and scheduler["max_model_len"] == context
            and scheduler["enable_chunked_prefill"] is True
            and async_scheduling is True
            and not any(item.startswith((scheduled_flag + "=", "--no-async-scheduling",
                                         "--enforce-eager"))
                        for item in args),
            f"{run}: unexpected effective vLLM scheduler settings")
    for key, expected in (("--max-num-batched-tokens", cap), ("--max-num-seqs", 32),
                          ("--max-model-len", context)):
        require(arg(args, key) == str(expected), f"{run}: {key} differs from worker observation")
    cache = worker["cache"]
    attention = worker["attention"]
    compilation = worker["compilation"]
    require(worker["parallel"] == {"tp": 1, "dp": 1, "pp": 1}
            and worker["model_dtype"] == "torch.bfloat16" and worker["quantization"] == "fp8"
            and worker["flashinfer"] == "0.7.4"
            and worker["enforce_eager"] is False and worker["speculative"] is speculative
            and worker["skip_tokenizer_init"] is True and worker["kv_transfer"] is False
            and worker["kv_offloading_size"] is None
            and "cpu_offload_gb=0.0," in worker["cpu_offload"]
            and "offload_group_size=0," in worker["cpu_offload"],
            f"{run}: unexpected effective vLLM model/parallel/offload settings")
    require(cache["enable_prefix_caching"] is False and cache["cache_dtype"] == "fp8_e4m3"
            and cache["mamba_cache_dtype"] == "bfloat16"
            and cache["mamba_ssm_cache_dtype"] == "float32"
            and cache["kv_cache_memory_bytes"] == (10 if group == "W1-W2" else 128) << 30
            and attention == {"backend": "FLASHINFER", "disable_flashinfer_q_quantization": True}
            and compilation["mode"] == "NONE"
            and compilation["cudagraph_mode"] == "FULL_AND_PIECEWISE"
            and compilation["cudagraph_capture_sizes"],
            f"{run}: unexpected effective vLLM cache, attention or graph settings")
    require(arg(args, "--generation-config") == "vllm" and arg(args, "--seed") == "0"
            and arg(args, "--stream-interval") == "1"
            and "--language-model-only" in args and "--no-enable-prefix-caching" in args,
            f"{run}: unexpected vLLM generation or streaming settings")
    return worker


def saved_worker(capture, run, cpus, uuid, group, *, speculative=False):
    args = command(run, "server", cpus)
    log = (run / "server.log").read_text()
    api_pid = (run / "server-pid.txt").read_text().strip()
    require(api_pid.isdecimal() and f"(APIServer pid={api_pid})" in log,
            f"{run}: API server PID missing from log")
    engines = re.findall(r"\(EngineCore pid=(\d+)\).*?Initializing a V1 LLM engine.*?"
                         r" with config: model='([^']+)'", log)
    require(len(engines) == 1 and engines[0][1] == arg(args, "--model"),
            f"{run}: engine log and requested model do not identify one worker")
    pid, model = engines[0]
    observed_device(run, pid, uuid)
    environment = fields(capture / "effective-environment.txt")
    require(environment["CUDA_VISIBLE_DEVICES"].lower() == uuid,
            f"{run}: CUDA visibility conflicts with observed UUID")
    return {"worker_pid": int(pid), "model_path": model,
            "effective": vllm_settings(run, args, group, speculative=speculative)}
