"""Current formal workload geometry and immutable fixture selection."""

import json

from benchmarks.concurrent import reference
from saved_evidence import UUID, require


PROTOCOL = "matched-k4-native-defaults-cache-on-v3"
PROTOCOL_V4 = "matched-k4-native-defaults-cache-on-v4"
PROTOCOL_V5 = "matched-k4-native-defaults-cache-on-v5"
LMCACHE_PROTOCOL = "matched-k4-native-defaults-lmcache-v6"
LMCACHE_GPU_PROTOCOL = "matched-k4-native-defaults-lmcache-slc2"
LMCACHE_PROTOCOLS = (LMCACHE_PROTOCOL, LMCACHE_GPU_PROTOCOL)
WIDE_BUDGET_PROTOCOLS = (PROTOCOL_V4, PROTOCOL_V5, *LMCACHE_PROTOCOLS)
WORKLOAD_PROTOCOLS = (PROTOCOL, *WIDE_BUDGET_PROTOCOLS)
V4_TOKEN_BUDGET = 32768
LMCACHE_VERSION = "0.5.8"
ROWS = {
    "W1": (128, 128, 1),
    "W2": (2048, 512, 1),
    "W3": (8192, 1024, 4),
    "W4": (32768, 4096, 8),
    "W5": (131072, 1024, 4),
}
CONCURRENT_FIXTURES = {"W3": "W3", "W4": "W5", "W5": "W6"}


def serving_capacity(gpu_count):
    require(type(gpu_count) is int and gpu_count > 0, "Expected at least one participating GPU")
    return 8 * gpu_count


def geometry(row, gpu_count):
    serving_capacity(gpu_count)
    prompt, output, per_gpu = ROWS[row]
    return {"prompt_tokens": prompt, "output_tokens": output,
            "requests_per_gpu": per_gpu, "concurrent_requests": per_gpu * gpu_count}


def validate_identity(identity):
    devices = identity.get("participating_gpus")
    require(isinstance(devices, list) and len(devices) == 1
            and type(devices[0]) is str and UUID.fullmatch(devices[0])
            and devices[0] == identity["gpu_uuid"],
            "Current benchmark requires one participating GPU matching the execution device")
    cap = identity.get("max_running_requests")
    require(type(cap) is int and cap == serving_capacity(len(devices)),
            "Wrong current campaign serving concurrency")
    if identity["protocol"] in WIDE_BUDGET_PROTOCOLS:
        budget = identity.get("max_num_batched_tokens")
        require(type(budget) is int and budget == V4_TOKEN_BUDGET,
                f"Wrong {identity['protocol'].rsplit('-', 1)[1]} campaign token budget")
    if identity["protocol"] in LMCACHE_PROTOCOLS:
        port = identity.get("lmcache_port")
        require(type(port) is int and 1 <= port <= 65535 and port != identity.get("port"),
                "LMCache campaign requires a separate loopback port")
        require(identity.get("lmcache_version") == LMCACHE_VERSION
                and identity.get("sampling") == "single-round-v1",
                "LMCache campaign requires the pinned daemon and single-round sampling")
    if identity["protocol"] == LMCACHE_GPU_PROTOCOL:
        require(type(identity.get("lmcache_capacity_bytes")) is int
                and identity["lmcache_capacity_bytes"] == 64 << 30,
                "Direct GPU LMCache campaign requires a 64 GiB daemon")
        uid = identity.get("lmcache_uid")
        python = identity.get("lmcache_python")
        require(type(uid) is int and uid >= 0 and type(python) is str and python.startswith("/")
                and type(identity.get("lmcache_pythonpath")) is str,
                "Direct GPU LMCache campaign requires the daemon UID and interpreter")


def concurrent_metadata(row, gpu_count):
    source = reference.lane_manifest()["rows"][CONCURRENT_FIXTURES[row]]
    count = geometry(row, gpu_count)["concurrent_requests"]
    require(count <= source["concurrency"], "Frozen corpus does not cover the requested devices")
    return source["lanes"][:count]


def load_concurrent(fixtures, row, gpu_count):
    source = fixtures / CONCURRENT_FIXTURES[row]
    shape = geometry(row, gpu_count)
    lanes = []
    for lane in concurrent_metadata(row, gpu_count):
        path = source / f'{lane["lane"]:02d}.request.json'
        body = path.read_bytes()
        require(reference.digest(body) == lane["request_sha256"],
                f"{path}: request digest mismatch")
        request = json.loads(body)
        prompt = request["prompt"]
        require(len(prompt) == shape["prompt_tokens"]
                and request["max_tokens"] == shape["output_tokens"],
                f"{path}: request violates the current workload shape")
        case = {"prompt_token_ids": prompt, "output_tokens": shape["output_tokens"]}
        lanes.append((lane, body, case))
    return lanes
