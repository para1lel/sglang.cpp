"""HTTP and wire witnesses for LMCache route fixtures."""

from __future__ import annotations

import json
import logging
import queue
import struct
import traceback
from contextlib import closing
from http import HTTPStatus
from http.client import HTTPConnection
from threading import Thread
from time import monotonic, perf_counter_ns, sleep
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeVar, cast

from cold_cache import LMCACHE_DRAIN, LMCACHE_METRICS, PENDING_STORES, metrics
from lmcache_relay import DRAFT, ERASE, GDN, GQA, LOOKUP, RANK_PUT, RESET, control, key_body

from media import http_routes as http

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from lmcache_relay import Relay

Key = tuple[int, bytes]
NativeResult = tuple[list[int], str, dict[str, object]]
_T = TypeVar("_T")
NAMESPACE = "sglang"
LOGGER = logging.getLogger(__name__)


class Witness:
    def __init__(self, port: int, evidence: Path, daemon: Path, relay: Relay) -> None:
        self.port = port
        self.evidence = evidence
        self.daemon = daemon
        self.relay = relay

    def action(self, opcode: int, payload: bytes = b"") -> tuple[int, bytes]:
        return control(self.daemon, NAMESPACE, opcode, payload)

    def get(self, route: str) -> bytes:
        status, _, raw = http.exchange(self.port, route)
        assert status == HTTPStatus.OK, (route, status, raw)
        return raw

    def pending_stores(self) -> int:
        return metrics(self.get("/metrics").decode(), LMCACHE_DRAIN)[PENDING_STORES]

    def drain(self) -> None:
        wait_until(lambda: self.pending_stores() == 0, "LMCache stores did not settle")

    def snapshot(self, label: str) -> dict[str, int]:
        self.drain()
        raw = self.get("/metrics")
        (self.evidence / f"{label}.prom").write_bytes(raw)
        summary = metrics(raw.decode(), "lmcache")
        return {
            name.removeprefix("sglang_cpp:lmcache_").removesuffix("_total"): summary[name]
            for name in LMCACHE_METRICS
        }

    def reset(self) -> None:
        self.drain()
        count = len(self.relay.since())
        status, _, body = http.exchange(self.port, "/reset_prefix_cache", {})
        assert status == HTTPStatus.OK, (status, body)
        assert json.loads(body) == {"success": True}, (status, body)
        resets = [entry for entry in self.relay.since(count) if entry["opcode"] == RESET]
        assert len(resets) == 1, resets
        assert resets[0]["status"] == 0, resets

    def generate(
        self, label: str, prompt: list[int], *, count: int = 1, **options: object
    ) -> dict[str, object]:
        body = http.native_text(prompt, count) | options
        return http.capture(
            SimpleNamespace(evidence=self.evidence), label, self.port, http.GENERATE, body
        )

    def stores(self, label: str, prompt: list[int]) -> tuple[dict[str, object], set[Key]]:
        begin = len(self.relay.since())
        response = self.generate(label, prompt)
        assert len(response["output_ids"]) == 1
        keys = self.published_keys(begin)
        assert {family for family, _ in keys} == {GQA, GDN, DRAFT}, (label, keys)
        return response, keys

    def published_keys(self, begin: int) -> set[Key]:
        self.drain()
        entries = [entry for entry in self.relay.since(begin) if entry["opcode"] == RANK_PUT]
        assert all(entry["status"] == 0 for entry in entries), entries
        return {key for entry in entries for key in entry["keys"]}

    def changes(
        self,
        label: str,
        prompt: list[int],
        cached: int,
        expected: dict[str, int],
        *,
        count: int = 1,
        **options: object,
    ) -> tuple[NativeResult, dict[str, int]]:
        before = self.snapshot(f"{label}-before")
        result = self.generate(label, prompt, count=count, **options)
        outputs = http.native_result(result, prompt, count, cached)
        after = self.snapshot(f"{label}-after")
        delta = {name: after[name] - before[name] for name in before}
        assert all(amount >= 0 for amount in delta.values()), delta
        for name, amount in expected.items():
            assert delta[name] == amount, (label, name, amount, delta)
        return outputs, delta

    def lookup(self, keys: Sequence[Key], expected: Sequence[int]) -> None:
        payload = struct.pack("!I", len(keys)) + b"".join(map(key_body, keys))
        assert self.action(LOOKUP, payload) == (0, bytes(expected)), (keys, expected)

    def erase(self, keys: Sequence[Key] | set[Key]) -> None:
        for key in keys:
            assert self.action(ERASE, key_body(key)) == (0, b""), key
        self.lookup(list(keys), [0] * len(keys))


def transfer_records(
    entries: list[dict[str, object]], opcode: int, family: int
) -> Iterator[tuple[str, Key, dict[str, object]]]:
    for entry in entries:
        if entry["opcode"] == opcode and entry["status"] == 0 and entry.get("forwarded"):
            for key, record in zip(entry["keys"], entry["records"], strict=True):
                if key[0] == family:
                    yield entry["device"], key, record


def tokens(result: NativeResult) -> tuple[list[int], str]:
    ids, text, _ = result
    return ids, text


class Stream:
    def __init__(
        self, witness: Witness, label: str, prompt: list[int], count: int, **sampling: object
    ) -> None:
        evidence_label = cast("str | None", sampling.pop("evidence_label", None))
        cache_salt = sampling.pop("cache_salt", None)
        self.witness = witness
        self.label = label
        self.evidence_label = evidence_label or label
        self.body = http.native_text(prompt, count) | {"rid": label, "stream": True}
        if cache_salt is not None:
            self.body["cache_salt"] = cache_salt
        self.body["sampling_params"].update(sampling)
        self.events = queue.Queue()
        self.thread = Thread(target=self.read)
        self.thread.start()

    def read(self) -> None:
        try:
            self.capture()
        except Exception as error:
            LOGGER.debug("Could not read %s stream", self.label, exc_info=error)
            self.events.put(error)

    def capture(self) -> None:
        path = self.witness.evidence / f"{self.evidence_label}.sse"
        with (
            path.open("wb") as raw,
            closing(HTTPConnection("127.0.0.1", self.witness.port, timeout=90)) as session,
        ):
            session.request("POST", http.GENERATE, json.dumps(self.body), http.HEADERS)
            with session.getresponse() as response:
                assert response.status == HTTPStatus.OK, (response.status, response.read())
                assert response.headers.get_content_type() == "text/event-stream"
                for line in iter(response.readline, b""):
                    raw.write(line)
                    if line.startswith(b"data: "):
                        body = line[6:].strip()
                        value = "[DONE]" if body == b"[DONE]" else json.loads(body)
                        self.events.put((perf_counter_ns(), value))
        self.events.put(None)

    def event(self, timeout: float = 10) -> tuple[int, str | dict[str, object]]:
        record = self.events.get(timeout=timeout)
        if isinstance(record, Exception):
            raise record
        assert record is not None, (self.label, "unexpected EOF")
        return record

    def progress(self, after: int = 0) -> tuple[int, int]:
        deadline = monotonic() + 5
        while (remaining := deadline - monotonic()) > 0:
            instant, frame = self.event(remaining)
            if frame == "[DONE]":
                raise AssertionError((self.label, "completed before demonstrating progress"))
            assert "meta_info" in frame, frame
            if instant > after and frame["output_ids"] and not frame["meta_info"]["finish_reason"]:
                return instant, len(frame["output_ids"])
        raise TimeoutError(f"No stream progress from {self.label}")

    def finish(self, terminal_after: int = 0) -> dict[str, object]:
        deadline = monotonic() + 90
        while True:
            instant, frame = self.event(max(0.001, deadline - monotonic()))
            if frame == "[DONE]" or frame["meta_info"]["finish_reason"]:
                assert instant > terminal_after, (
                    self.label,
                    "terminal before the held lease was released",
                )
            if frame == "[DONE]":
                break
        self.thread.join(5)
        assert not self.thread.is_alive()
        assert self.events.get(timeout=1) is None
        raw = (self.witness.evidence / f"{self.evidence_label}.sse").read_bytes()
        frames = http.stream_frames(raw)
        last = frames[-1]
        assert all(
            last["output_ids"][: len(frame["output_ids"])] == frame["output_ids"]
            and frame["meta_info"]["finish_reason"] is None
            for frame in frames[:-1]
        )
        return last


def wait_until(predicate: Callable[[], _T], message: str) -> _T:
    deadline = monotonic() + 5
    while monotonic() < deadline:
        result = predicate()
        if result:
            return result
        sleep(0.01)
    raise AssertionError(message)


def capture_case(
    report: dict[str, object],
    failures: dict[str, str],
    name: str,
    check: Callable[..., object],
    *args: object,
) -> None:
    try:
        report[name] = check(*args)
    except Exception:
        failures[name] = traceback.format_exc()
        LOGGER.exception("Route check failed: %s", name)
