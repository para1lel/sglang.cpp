"""Metrics and reset controls shared by the sampled HTTP regression."""

from __future__ import annotations

import http.client
import json
import re
import time
from contextlib import closing
from decimal import Decimal, InvalidOperation
from http import HTTPStatus
from typing import TYPE_CHECKING

from routes import HEADERS

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path

RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
GENERATED = "vllm:generation_tokens_total"
GDN_ENTRIES = "sglang_cpp:prefix_cache_entries"
STORES = "sglang_cpp:lmcache_stores_total"
PENDING_STORES = "sglang_cpp:lmcache_pending_stores"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
METRIC = re.compile(r"([\w:]+)(?:\{.*\})?\s+(\S+)")
PREFILL_CHECKPOINTS = tuple(
    "sglang_cpp:lmcache_" + name + "_total"
    for name in (
        "prefill_checkpoints_planned",
        "prefill_checkpoints_resident",
        "prefill_checkpoints_published",
        "prefill_checkpoints_declined",
        "prefill_checkpoint_stores",
        "prefill_checkpoint_failures",
    )
)


def parse_metrics(text: str, names: Collection[str]) -> dict[str, int]:
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
        if (
            name in result
            or not value.is_finite()
            or value < 0
            or value != value.to_integral_value()
        ):
            raise ValueError(f"Invalid or duplicate cache metric {name}")
        result[name] = int(value)
    if result.keys() != set(names):
        raise ValueError("Missing cache metrics from the single-engine endpoint")
    return result


class Controls:
    def __init__(self, port: int, directory: Path) -> None:
        self.port = port
        self.directory = directory

    def control(
        self,
        name: str,
        route: str,
        body: bytes | None = None,
        status: int = HTTPStatus.OK,
    ) -> bytes:
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)) as client:
            client.request("GET" if body is None else "POST", route, body, HEADERS)
            with client.getresponse() as response:
                raw = response.read()
                observed = response.status
                content = response.headers.get_content_type()
        (self.directory / f"{name}.http").write_bytes(
            f"{observed} {content}\n".encode() + raw
        )
        assert observed == status, (name, observed, raw)
        expected = "text/plain" if route == "/metrics" else "application/json"
        assert content == expected, (name, content)
        return raw

    def metrics(self, name: str) -> dict[str, float]:
        raw = self.control(name, "/metrics")
        assert raw.startswith(b"# Native sglang.cpp metrics"), raw
        metrics = {}
        for line in raw.decode().splitlines()[1:]:
            key, separator, value = line.partition('{engine="0"} ')
            assert separator, line
            assert key not in metrics, line
            metrics[key] = float(value)
        return metrics

    def wait_for(
        self,
        name: str,
        running: int,
        waiting: int,
        generated: int = 0,
        *,
        drain_backups: bool = False,
    ) -> dict[str, float]:
        deadline = time.monotonic() + 15
        while True:
            metrics = self.metrics(name)
            if (
                (metrics[RUNNING], metrics[WAITING]) == (running, waiting)
                and metrics[GENERATED] >= generated
                and (not drain_backups or metrics[PENDING_STORES] == 0)
            ):
                return metrics
            if time.monotonic() >= deadline:
                raise AssertionError((name, metrics))
            time.sleep(0.02)

    def idle(self, name: str) -> dict[str, float]:
        return self.wait_for(name, 0, 0, drain_backups=True)

    def reset(self, name: str, status: int = HTTPStatus.OK) -> None:
        result = self.control(name, "/reset_prefix_cache", b"{}", status)
        if status == HTTPStatus.OK:
            assert json.loads(result) == {"success": True}, result

