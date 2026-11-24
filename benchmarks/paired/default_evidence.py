"""Validate the live launch of native server defaults and the matched K4 worker."""

import json
import os
from pathlib import Path

from workload_suite import PROTOCOL as CURRENT_PROTOCOL
from saved_evidence import arg, command, fields, load, observed_device, require


PROTOCOL = "matched-k4-native-defaults-v1"
CACHE_ON_PROTOCOL = "matched-k4-native-defaults-cache-on-v2"
CACHE_ON_PROTOCOLS = (CACHE_ON_PROTOCOL, CURRENT_PROTOCOL)
DEFAULT_PROTOCOLS = (PROTOCOL, *CACHE_ON_PROTOCOLS)
CACHE_POLICY = "reset-before-every-cohort"
PROFILE = "server-defaults"
NATIVE_ENVIRONMENT = {
    "PATH", "LANG", "HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
    "CUDA_CACHE_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH",
}


def process_environment(raw):
    entries = raw.rstrip(b"\0").split(b"\0")
    names = [os.fsdecode(entry.split(b"=", 1)[0]) for entry in entries]
    require(raw.endswith(b"\0") and set(names) == NATIVE_ENVIRONMENT
            and len(names) == len(NATIVE_ENVIRONMENT)
            and all(b"=" in entry for entry in entries),
            "Live native default environment contains unexpected or missing settings")
    return dict(os.fsdecode(entry).split("=", 1) for entry in entries)


def read_environment(pid):
    require(pid > 0, "Invalid native server PID")
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    environment = process_environment(raw)
    return b"".join(os.fsencode(f"{name}={value}") + b"\0"
                    for name, value in sorted(environment.items()))


def verify_cpu(run, cpus):
    require((run / "live-cpus.txt").read_text().strip() == cpus,
            f"{run.name}: live API process CPU affinity differs from the reservation")


def client(run, identity, port):
    cpus = {"client": identity["client_cpus"], "memory": identity["memory"]}
    args = command(run, "client", cpus)
    index = args.index("capture")
    require(args[index + 1:index + 4] ==
            [run.parent.name, run.name, f"http://127.0.0.1:{port}"]
            and len(args[index + 1:]) == 5,
            f"{run.name}: client endpoint, arm, or arguments differ")


def verify_argv(run, argv):
    expected = b"".join(os.fsencode(value) + b"\0" for value in argv)
    require((run / "live-argv.bin").read_bytes() == expected,
            f"{run.name}: live process arguments differ from the command")


def serving_capacity(identity):
    return identity["max_running_requests"] if identity["protocol"] == CURRENT_PROTOCOL else 32


def native(run, identity):
    info = load(run / "server-info.json")
    trace = run.name == "native-k4-trace"
    expected = {
        "mtp_steps": 4, "max_running_requests": serving_capacity(identity),
        "max_num_batched_tokens": 8192,
        "prefix_cache_entries": 4 if identity["protocol"] in CACHE_ON_PROTOCOLS else 0,
        "batch_cuda_graphs": True, "eager": False,
        "trace_batches": trace, "host": "127.0.0.1", "device": 0,
        "tp_size": 1, "dp_size": 1, "pp_size": 1, "model_path": identity["checkpoint_path"],
    }
    port = info["port"]
    require(all(type(info.get(key)) is type(value) and info[key] == value
                for key, value in expected.items()) and type(port) is int and 1 <= port <= 65535,
            f"{run.name}: wrong resolved native defaults or shared resource settings")
    flags = ["--port", str(port)]
    if trace:
        flags.append("--trace-batches")
    require(info["launch_command"].split() == ["sglang-server", *flags],
            f"{run.name}: resolved native launch includes unexpected arguments")
    cpus = {"server": identity["server_cpus"], "memory": identity["memory"]}
    wrapper = command(run, "server", cpus)
    begin = (run / "live-environment.bin").read_bytes()
    environment = process_environment(begin)
    boundary = 2 + len(environment)
    require(wrapper[:2] == ["env", "-i"] and
            sorted(wrapper[2:boundary]) == sorted(
                f"{key}={value}" for key, value in environment.items())
            and environment["CUDA_VISIBLE_DEVICES"].lower() == identity["gpu_uuid"].lower()
            and environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and port == identity["port"],
            f"{run.name}: native launch does not use the clean GPU environment")
    argv = wrapper[boundary:]
    require(argv and Path(argv[0]).is_absolute() and argv[1:] == flags,
            f"{run.name}: unexpected native default launch argument")
    verify_argv(run, argv)
    verify_cpu(run, cpus["server"])
    pid = (run / "server-pid.txt").read_text().strip()
    binary = (run / "running-binary.sha256").read_text().split()
    require(pid.isdecimal() and int(pid) > 0
            and binary == [identity["native_binary_sha256"], f"/proc/{pid}/exe"],
            f"{run.name}: executable receipt differs from the native server PID")
    end = run / "live-environment-end.bin"
    if end.exists():
        require(end.read_bytes() == begin, f"{run.name}: native environment changed during use")
    return info


def native_completed(run, identity):
    info = native(run, identity)
    client(run, identity, info["port"])
    pid = (run / "server-pid.txt").read_text().strip()
    observed_device(run, pid, identity["gpu_uuid"].lower())
    require((run / "live-environment-end.bin").read_bytes()
            == (run / "live-environment.bin").read_bytes(),
            f"{run.name}: native environment changed during use")
    return info


def vllm_argv(run, cpus, gpu_uuid):
    wrapper = command(run, "server", cpus)
    module = wrapper.index("-m")
    environment = fields(run / "effective-environment.txt")
    assignments = wrapper[2:module - 2]
    require(wrapper[:2] == ["env", "-i"]
            and sorted(assignments) == sorted(
                f"{key}={value}" for key, value in environment.items())
            and environment["CUDA_VISIBLE_DEVICES"].lower() == gpu_uuid
            and environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID",
            f"{run.name}: vLLM command differs from its clean GPU environment")
    require(Path(wrapper[module - 2]).is_absolute() and wrapper[module - 1] == "-B" and
            wrapper[module + 1] == "vllm.entrypoints.openai.api_server",
            f"{run.name}: unexpected vLLM entry point")
    argv = wrapper[module - 2:]
    verify_argv(run, argv)
    verify_cpu(run, cpus["server"])
    raw_environment = sorted((run / "live-environment.bin").read_bytes().split(b"\0"))
    require(raw_environment == [b"", *sorted(map(os.fsencode, assignments))],
            f"{run.name}: live vLLM process environment differs from the command")
    port = arg(argv, "--port")
    require(port.isdecimal() and 1 <= int(port) <= 65535
            and arg(argv, "--host") == "127.0.0.1"
            and arg(argv, "--served-model-name") == "benchmark"
            and json.loads(arg(argv, "--speculative-config")) ==
            {"method": "mtp", "num_speculative_tokens": 4},
            f"{run.name}: wrong vLLM endpoint, workload alias, or MTP launch")
    return int(port)
