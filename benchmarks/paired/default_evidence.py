"""Validate the live launch of native server defaults and the matched K4 worker."""

import json
import os
from pathlib import Path

import lmcache_evidence
import workload_suite
from saved_evidence import arg, command, fields, load, observed_device, require


PROTOCOL = "matched-k4-native-defaults-v1"
CACHE_ON_PROTOCOL = "matched-k4-native-defaults-cache-on-v2"
CACHE_ON_PROTOCOLS = (CACHE_ON_PROTOCOL, *workload_suite.WORKLOAD_PROTOCOLS)
DEFAULT_PROTOCOLS = (PROTOCOL, *CACHE_ON_PROTOCOLS)
CACHE_POLICY = "reset-before-every-cohort"
PROFILE = "server-defaults"
GRAPH_CONFIG = {"mode": "NONE", "cudagraph_mode": "FULL_AND_PIECEWISE"}
GRAPH_CAPTURE_FIELDS = ("max_cudagraph_capture_size", "max_prefill_cudagraph_capture_tokens",
                        "prefill_cudagraph_capture_token_sizes",
                        "max_decode_cudagraph_capture_requests",
                        "max_decode_cudagraph_capture_tokens",
                        "decode_cudagraph_capture_request_sizes")
VLLM_ENVIRONMENT = {
    "VLLM_HAS_FLASHINFER_CUBIN", "VLLM_NO_USAGE_STATS", "VLLM_CACHE_ROOT",
    "VLLM_CONFIG_ROOT", "VLLM_RPC_BASE_PATH", "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_SERVER_DEV_MODE",
}
NATIVE_ENVIRONMENT = {
    "PATH", "LANG", "HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
    "CUDA_CACHE_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH",
}


def process_environment(raw, *, lmcache=False):
    entries = raw.rstrip(b"\0").split(b"\0")
    names = [os.fsdecode(entry.split(b"=", 1)[0]) for entry in entries]
    expected = NATIVE_ENVIRONMENT | ({"XDG_RUNTIME_DIR"} if lmcache else set())
    require(raw.endswith(b"\0") and set(names) == expected
            and len(names) == len(expected)
            and all(b"=" in entry for entry in entries),
            "Live native default environment contains unexpected or missing settings")
    return dict(os.fsdecode(entry).split("=", 1) for entry in entries)


def read_environment(pid, *, lmcache=False):
    require(pid > 0, "Invalid native server PID")
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    environment = process_environment(raw, lmcache=lmcache)
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
    if identity["protocol"] in workload_suite.WORKLOAD_PROTOCOLS:
        return identity["max_running_requests"]
    return 32


def token_budget(identity):
    if identity["protocol"] in workload_suite.WIDE_BUDGET_PROTOCOLS:
        return workload_suite.V4_TOKEN_BUDGET
    return 8192


def native(run, identity):
    info = load(run / "server-info.json")
    trace = run.name == "native-k4-trace"
    lmcache = identity["protocol"] in workload_suite.LMCACHE_PROTOCOLS
    expected = {
        "mtp_steps": 4, "max_running_requests": serving_capacity(identity),
        "max_num_batched_tokens": token_budget(identity),
        "batch_cuda_graphs": True, "eager": False,
        "trace_batches": trace, "host": "127.0.0.1", "device": 0,
        "tp_size": 1, "dp_size": 1, "pp_size": 1, "model_path": identity["checkpoint_path"],
    }
    if lmcache:
        expected.update(lmcache_enabled=True, lmcache_available=True, prefix_cache_entries=0)
    else:
        expected["prefix_cache_entries"] = 4 if identity["protocol"] in CACHE_ON_PROTOCOLS else 0
    if identity["protocol"] == workload_suite.LMCACHE_POOL_PROTOCOL:
        expected.update(workload_suite.HBM_POOL_DEFAULTS)
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
    environment = process_environment(begin, lmcache=lmcache)
    if lmcache:
        lmcache_evidence.verify(run, environment, identity)
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
    if identity["protocol"] in workload_suite.LMCACHE_GPU_PROTOCOLS:
        lmcache_evidence.completed(run, identity)
    else:
        observed_device(run, pid, identity["gpu_uuid"].lower())
    require((run / "live-environment-end.bin").read_bytes()
            == (run / "live-environment.bin").read_bytes(),
            f"{run.name}: native environment changed during use")
    return info


def vllm_argv(run, cpus, gpu_uuid, *, wide_budget=False):
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
    if wide_budget:
        verify_vllm_graph_launch(argv, environment)
    return int(port)


def verify_vllm_graph_launch(argv, environment):
    graph_flags = ("--cudagraph-", "--max-cudagraph-", "--prefill-cudagraph-",
                   "--max-prefill-cudagraph-", "--decode-cudagraph-",
                   "--max-decode-cudagraph-", "--enforce-eager")
    require(json.loads(arg(argv, "--compilation-config")) == GRAPH_CONFIG
            and not any(flag.startswith(graph_flags) for flag in argv)
            and {key for key in environment if key.startswith("VLLM_")} == VLLM_ENVIRONMENT
            and environment["VLLM_HAS_FLASHINFER_CUBIN"] == "1"
            and not any(key.startswith(("TORCH", "PYTORCH", "CUDAGRAPH"))
                        for key in environment),
            "Unexpected vLLM CUDA graph launch or environment override")


def vllm_budget(config, worker, budget):
    scheduler = config["scheduler_config"]
    expected = {"max_num_batched_tokens": budget, "max_num_scheduled_tokens": budget,
                "max_num_seqs": worker["scheduler"]["max_num_seqs"],
                "async_scheduling": True, "enable_chunked_prefill": True}
    require(all(type(scheduler.get(key)) is type(value) and scheduler[key] == value
                for key, value in expected.items()),
            "vLLM API and worker disagree about resolved v4 scheduler settings")
    graph = config["compilation_config"]
    observed = worker["compilation"]
    sizes = observed["cudagraph_capture_sizes"]
    decode_sizes = graph.get("decode_cudagraph_capture_request_sizes", [])
    prefill_sizes = graph.get("prefill_cudagraph_capture_token_sizes", [])
    require(graph.get("mode") == 0 and graph.get("cudagraph_mode") == [2, 1]
            and sizes == graph.get("cudagraph_capture_sizes")
            and all(key in graph and graph[key] == observed.get(key)
                    for key in GRAPH_CAPTURE_FIELDS)
            and bool(prefill_sizes) and max(prefill_sizes) <= budget
            and expected["max_num_seqs"] in decode_sizes,
            "vLLM API and worker disagree about resolved v4 CUDA graph settings")


def require_vllm_local_apc(config, argv):
    cache = config["cache_config"]
    offload = ("--kv-offloading-size", "--kv-offloading-backend", "--kv-transfer-config")
    require(cache.get("enable_prefix_caching") is True
            and "kv_offloading_size" in cache and cache["kv_offloading_size"] is None
            and "kv_transfer_config" in config and config["kv_transfer_config"] is None
            and not any(item.split("=", 1)[0] in offload for item in argv),
            "Pinned vLLM must use local APC with no KV offload or transfer connector")
