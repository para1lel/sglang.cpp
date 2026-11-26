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
STORES = "sglang_cpp:prefix_cache_stores_total"
LMCACHE_DRAIN = "lmcache-drain"
PENDING_STORES = "sglang_cpp:lmcache_pending_stores"
CHECKPOINT_INTERVAL = 8192
IDLE_TIMEOUT = 30
METRIC = re.compile(r"([\w:]+)(?:\{.*\})?\s+(\S+)")
LMCACHE_REUSE = tuple("sglang_cpp:lmcache_" + name + "_total" for name in (
    "io_errors", "gqa_restored_tokens", "gqa_read_only_tokens", "gdn_restored_tokens",
    "draft_restored_rows",
))
LMCACHE_METRICS = LMCACHE_REUSE + tuple("sglang_cpp:lmcache_" + name + "_total" for name in (
    "lookup_misses", "gqa_computed_tokens", "gqa_written_tokens", "gdn_computed_tokens",
))
LMCACHE_STORES = "sglang_cpp:lmcache_stores_total"
DECLINED_STORES = "sglang_cpp:lmcache_declined_stores_total"
GQA_WRITTEN_TOKENS = "sglang_cpp:lmcache_gqa_written_tokens_total"
LMCACHE_WRITTEN_BYTES = tuple(f"sglang_cpp:lmcache_{family}_written_bytes_total"
                              for family in ("gqa", "gdn", "draft"))
LMCACHE_WRITE_NS = "sglang_cpp:lmcache_write_nanoseconds_total"
LMCACHE_READ_ACTIVITY = tuple(f"sglang_cpp:lmcache_{name}_total" for name in (
    "gqa_read_bytes", "gdn_read_bytes", "draft_read_bytes", "restore_nanoseconds"))
LMCACHE_DRAIN_METRICS = (PENDING_STORES, LMCACHE_STORES, DECLINED_STORES,
                         *LMCACHE_WRITTEN_BYTES, LMCACHE_WRITE_NS, *LMCACHE_READ_ACTIVITY)
NATIVE_METRICS = (
    ENTRIES, PAGES, "sglang_cpp:prefix_cache_device_bytes",
    "sglang_cpp:prefix_cache_hits_total", STORES,
    "sglang_cpp:prefix_cache_evictions_total", "sglang_cpp:prefix_cache_draft_device_bytes",
    "sglang_cpp:prefix_cache_draft_cached_rows_total",
    "sglang_cpp:prefix_cache_draft_prefill_rows_total",
    "sglang_cpp:prefix_cache_draft_misses_total",
)


def metrics(text, engine):
    names = (RUNNING, WAITING, HITS)
    if engine == "native":
        names += NATIVE_METRICS
    elif engine in ("lmcache", LMCACHE_DRAIN):
        names += (*LMCACHE_METRICS, STORES)
        if engine == LMCACHE_DRAIN:
            names += LMCACHE_DRAIN_METRICS
    return parse_metrics(text, names)


def parse_metrics(text, names):
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
        if values[RUNNING] == values[WAITING] == values.get(PENDING_STORES, 0) == 0:
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
    if engine == LMCACHE_DRAIN:
        receipt["settled_ns"] = perf_counter_ns()


def verify_lmcache_drain(snapshots, requests, previous):
    before, cleared, after = (snapshots[key] for key in ("before", "cleared", "after"))
    require(all(sample[PENDING_STORES] == 0 for sample in snapshots.values()),
            "Cache control receipt includes pending LMCache stores")
    if previous is not None:
        require(before == previous["after"],
                "LMCache counters changed between settled cohorts")
    require(before == cleared, "Cache counters changed during reset")
    require(after[DECLINED_STORES] == cleared[DECLINED_STORES],
            "LMCache declined a checkpoint store during the cold cohort")
    require(all(after[name] == cleared[name] for name in LMCACHE_READ_ACTIVITY),
            "LMCache issued a restore during the cold cohort")
    ends = [request["usage"]["prompt_tokens"] + request["usage"]["completion_tokens"] - 1
            for request in requests]
    expected = sum((end + CHECKPOINT_INTERVAL - 1) // CHECKPOINT_INTERVAL for end in ends)
    require(after[LMCACHE_STORES] - cleared[LMCACHE_STORES] == expected
            and after[STORES] == after[LMCACHE_STORES],
            "LMCache did not acknowledge the expected number of checkpoint stores")
    require(after[GQA_WRITTEN_TOKENS] - cleared[GQA_WRITTEN_TOKENS] == sum(ends),
            "LMCache checkpoint stores did not cover every completed token position")
    require(all(after[name] > cleared[name] for name in (*LMCACHE_WRITTEN_BYTES, LMCACHE_WRITE_NS)),
            "LMCache checkpoint stores did not report all three transfer families")


def replay(raw, engine, previous=None):
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
    if engine == LMCACHE_DRAIN:
        times += (receipt.get("settled_ns"),)
    require(all(type(value) is int and value > 0 for value in times)
            and list(times) == sorted(times) and times[2] < first,
            "Reset or metrics snapshot overlaps the timed cohort")
    require(snapshots["cleared"][HITS] == snapshots["after"][HITS],
            "Prefix cache hit counter advanced during a cold cohort")
    if engine in ("lmcache", LMCACHE_DRAIN):
        require(all(snapshots["cleared"][name] == snapshots["after"][name]
                    for name in LMCACHE_REUSE),
                "LMCache reused state or reported an I/O error during a cold cohort")
        require(snapshots["cleared"][STORES] == snapshots["before"][STORES]
                and snapshots["after"][STORES] > snapshots["cleared"][STORES],
                "LMCache did not acknowledge any checkpoint stores during the cold cohort")
    if engine == LMCACHE_DRAIN:
        verify_lmcache_drain(snapshots, requests, previous)
    if engine == "native":
        require(snapshots["cleared"][ENTRIES] == snapshots["cleared"][PAGES] == 0,
                "Native cache is occupied after reset")
    return snapshots
