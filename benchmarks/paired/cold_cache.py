"""Capture and replay cache isolation around one paired cohort."""

import http.client
import json
import re
from decimal import Decimal, InvalidOperation
from time import perf_counter_ns, sleep

from saved_evidence import require


RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
HITS = "vllm:prefix_cache_hits_total"
ENTRIES = "sglang_cpp:prefix_cache_entries"
PAGES = "sglang_cpp:prefix_cache_pages"
IDLE_TIMEOUT = 30
METRIC = re.compile(r"([\w:]+)(?:\{.*\})?\s+(\S+)")
NATIVE_METRICS = (
    ENTRIES, PAGES, "sglang_cpp:prefix_cache_device_bytes",
    "sglang_cpp:prefix_cache_hits_total", "sglang_cpp:prefix_cache_stores_total",
    "sglang_cpp:prefix_cache_evictions_total", "sglang_cpp:prefix_cache_draft_device_bytes",
    "sglang_cpp:prefix_cache_draft_cached_rows_total",
    "sglang_cpp:prefix_cache_draft_prefill_rows_total",
    "sglang_cpp:prefix_cache_draft_misses_total",
)


def metrics(text, engine):
    names = (RUNNING, WAITING, HITS)
    if engine == "native":
        names += NATIVE_METRICS
    result = {}
    for line in text.splitlines():
        match = METRIC.fullmatch(line)
        if match is None or match[1] not in names:
            continue
        name, number = match.groups()
        try:
            value = Decimal(number)
        except InvalidOperation as error:
            raise ValueError(f"Invalid cache metric {name}: {number}") from error
        require(name not in result and value.is_finite() and value >= 0
                and value == value.to_integral_value(), f"Invalid or duplicate cache metric {name}")
        result[name] = int(value)
    require(result.keys() == set(names), "Missing cache metrics from the single-engine endpoint")
    return result


def request(url, method, route, body=None):
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=IDLE_TIMEOUT)
    try:
        headers = {"Connection": "close"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, route, body=body, headers=headers)
        response = connection.getresponse()
        return {"status": response.status, "body": response.read().decode()}
    finally:
        connection.close()


def idle_metrics(url, engine):
    deadline = perf_counter_ns() + IDLE_TIMEOUT * 10**9
    while True:
        response = request(url, "GET", "/metrics")
        require(response["status"] == 200, "Cache metrics endpoint failed")
        values = metrics(response["body"], engine)
        if values[RUNNING] == values[WAITING] == 0:
            return response["body"]
        require(perf_counter_ns() < deadline, "Server did not drain before cache control")
        sleep(0.05)


def verify_reset(response):
    body = json.loads(response["body"])
    require(type(response["status"]) is int and response["status"] == 200
            and isinstance(body, dict) and body.keys() == {"success"}
            and body["success"] is True,
            f"Prefix cache reset was not acknowledged: {response}")


def before_cohort(url, engine):
    before = idle_metrics(url, engine)
    reset_started_ns = perf_counter_ns()
    reset = request(url, "POST", "/reset_prefix_cache", b"{}")
    reset_completed_ns = perf_counter_ns()
    verify_reset(reset)
    cleared = idle_metrics(url, engine)
    if engine == "native":
        values = metrics(cleared, engine)
        require(values[ENTRIES] == values[PAGES] == 0, "Native cache is occupied after reset")
    return {"before": before, "reset": reset, "cleared": cleared,
            "reset_started_ns": reset_started_ns, "reset_completed_ns": reset_completed_ns,
            "ready_ns": perf_counter_ns()}


def after_cohort(url, engine, receipt):
    receipt["capture_finished_ns"] = perf_counter_ns()
    receipt["after"] = idle_metrics(url, engine)


def replay(raw, engine):
    receipt = raw.get("cold_cache")
    require(isinstance(receipt, dict), "Missing cold cache control receipt")
    verify_reset(receipt["reset"])
    snapshots = {key: metrics(receipt[key], engine) for key in ("before", "cleared", "after")}
    require(all(values[RUNNING] == values[WAITING] == 0 for values in snapshots.values()),
            "Cache control receipt includes an active or queued request")
    requests = raw["requests"]
    first = min(record["submit_ns"] for record in requests)
    last = max(record["response_end"]["ns"] if "response_end" in record else record["eof_ns"]
               for record in requests)
    times = (receipt["reset_started_ns"], receipt["reset_completed_ns"], receipt["ready_ns"],
             first, last, receipt["capture_finished_ns"])
    require(all(type(value) is int and value > 0 for value in times)
            and list(times) == sorted(times) and times[2] < first,
            "Reset or metrics snapshot overlaps the timed cohort")
    require(snapshots["cleared"][HITS] == snapshots["after"][HITS],
            "Prefix cache hit counter advanced during a cold cohort")
    if engine == "native":
        require(snapshots["cleared"][ENTRIES] == snapshots["cleared"][PAGES] == 0,
                "Native cache is occupied after reset")
    return snapshots
