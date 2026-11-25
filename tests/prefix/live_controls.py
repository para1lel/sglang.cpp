"""Metrics and reset controls shared by the sampled HTTP regression."""

from contextlib import closing
import http.client
import json
import time

from routes import HEADERS


RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
GENERATED = "vllm:generation_tokens_total"
STORES = "sglang_cpp:lmcache_stores_total"
PENDING_STORES = "sglang_cpp:lmcache_pending_stores"


class Controls:
    def __init__(self, port, directory):
        self.port = port
        self.directory = directory

    def control(self, name, route, body=None, status=200):
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

    def metrics(self, name):
        raw = self.control(name, "/metrics")
        assert raw.startswith(b"# Native sglang.cpp metrics"), raw
        metrics = {}
        for line in raw.decode().splitlines()[1:]:
            key, separator, value = line.partition('{engine="0"} ')
            assert separator and key not in metrics, line
            metrics[key] = float(value)
        return metrics

    def wait_for(self, name, running, waiting, generated=0, *, stores=None):
        deadline = time.monotonic() + 15
        while True:
            metrics = self.metrics(name)
            if ((metrics[RUNNING], metrics[WAITING]) == (running, waiting)
                    and metrics[GENERATED] >= generated
                    and (stores is None or
                         (metrics[STORES] >= stores and metrics[PENDING_STORES] == 0))):
                return metrics
            if time.monotonic() >= deadline:
                raise AssertionError((name, metrics))
            time.sleep(0.02)

    def reset(self, name, status=200):
        result = self.control(name, "/reset_prefix_cache", b"{}", status)
        if status == 200:
            assert json.loads(result) == {"success": True}, result

