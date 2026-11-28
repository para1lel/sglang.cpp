"""Observe and control the real LMCache Unix transport for live prefix tests."""

from __future__ import annotations

import os
import select
import socket
import socketserver
import struct
import threading
from contextlib import ExitStack
from ctypes import addressof, create_string_buffer, string_at
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from io import BytesIO
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic, perf_counter_ns, sleep, time_ns
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

HEADER = struct.Struct("!4sIQ")
MAGIC = b"SLC3"
READY, LOOKUP, RESET, ERASE, LEASE = 1, 2, 5, 6, 8
PREFIX_DETAIL, RANK_GET, RANK_PUT = range(9, 12)
TRANSFERS = (RANK_GET, RANK_PUT)
WIRE_VERSION = b"0.5.8+native-v5"
ADMISSION = struct.Struct("!16s16sI")
RECEIPT = struct.Struct("!QQQI")
GQA, GDN, DRAFT = range(1, 4)
TIMEOUT = 30
BLOCK_TOKENS = 256
PAGE_TOKENS = 128
PAGE_MAP_FLAG = 0x80000000


def receive(sock, size):
    payload = bytearray(size)
    pending = memoryview(payload)
    while pending:
        count = sock.recv_into(pending)
        if not count:
            raise EOFError("Incomplete LMCache frame")
        pending = pending[count:]
    return bytes(payload)


def header(sock):
    raw = receive(sock, HEADER.size)
    magic, code, size = HEADER.unpack(raw)
    if magic != MAGIC:
        raise ValueError("Unexpected LMCache frame magic")
    return raw, code, size


def field(value):
    return struct.pack("!H", len(value)) + value


def key_body(key):
    family, identity = key
    return bytes((family,)) + field(identity)


def send(sock, namespace, opcode, payload):
    body = field(namespace.encode("utf-8")) + payload
    sock.sendall(HEADER.pack(MAGIC, opcode, len(body)) + body)
    _, status, size = header(sock)
    return status, size


def control(path, namespace, opcode, payload=b""):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(TIMEOUT)
        sock.connect(str(path))
        status, size = send(sock, namespace, opcode, payload)
        if size > 4096:
            raise ValueError("Unexpected LMCache control reply size")
        return status, receive(sock, size)


def _receive_gdn(path: Path, namespace: str, payload: bytes, size: int) -> None:
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(TIMEOUT)
        sock.connect(str(path))
        assert send(sock, namespace, RANK_GET, payload) == (4, ADMISSION.size)
        admission = receive(sock, ADMISSION.size)
        _, _, pid = ADMISSION.unpack(admission)
        pidfd = os.pidfd_open(pid)
        try:
            sock.sendall(b"\x01")
            _, status, length = header(sock)
            assert (status, length) == (0, RECEIPT.size)
            assert RECEIPT.unpack(receive(sock, length)) == (0, size, 0, 0)
        finally:
            sock.close()
            try:
                while not select.select([pidfd], [], [], 0.1)[0]:
                    try:
                        if control(path, namespace, LEASE, admission[:32]) == (0, b""):
                            break
                    except (OSError, EOFError):
                        continue
            finally:
                os.close(pidfd)


def restore_gdn(
    path: Path,
    namespace: str,
    device: tuple[int, str],
    key: tuple[int, bytes],
    record: dict[str, int | list[int]],
) -> str:
    torch = import_module("torch")
    cuda = import_module("cuda.bindings.runtime")
    ordinal, uuid = device

    def checked(result: tuple) -> list:
        status, *values = result
        if int(status):
            raise RuntimeError(f"CUDA GDN-only test transfer failed: {status}")
        return values

    size = record["bytes"]
    received = []
    with torch.cuda.device(ordinal), ExitStack() as cleanup:
        (address,) = checked(cuda.cudaMalloc(size))
        cleanup.callback(lambda: checked(cuda.cudaFree(address)))
        (memory,) = checked(cuda.cudaIpcGetMemHandle(address))
        allocation = struct.pack("!I64sQ", 1, string_at(memory.getPtr(), 64), size)
        gdn = struct.pack("!I64s", record["endpoint"], key[1])
        offset = 0
        for width in record["widths"]:
            gdn += struct.pack("!IQQ", 0, offset, width)
            offset += width
        assert offset == size
        host = create_string_buffer(size)
        stream = torch.cuda.current_stream()
        for fill in (0x2B, 0xAB):
            checked(cuda.cudaMemsetAsync(address, fill, size, stream.cuda_stream))
            event = torch.cuda.Event(interprocess=True)
            event.record(stream)
            payload = bytes.fromhex(uuid) + event.ipc_handle() + allocation + bytes(16) + gdn
            _receive_gdn(path, namespace, payload, size)
            checked(
                cuda.cudaMemcpy(
                    addressof(host), address, size, cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost
                )
            )
            received.append(sha256(host).hexdigest())
        assert received[0] == received[1], "GDN-only DMA did not fully overwrite its destination"
    return received[0]


def _prior_inventory(read: Callable[[int], bytes], records: list[dict[str, int]]) -> dict:
    count, draft_required = struct.unpack("!BB", read(2))
    anchors = [struct.unpack("!I64s", read(68)) for _ in range(count)]
    highest = anchors[-1][0] if anchors else 0
    complete = {
        (record["family"], record["end"] + (record["family"] == DRAFT))
        for record in records
        if record["tokens"] == BLOCK_TOKENS
    }
    prerequisites = [
        (GQA, end, read(64))
        for end in range(BLOCK_TOKENS, highest, BLOCK_TOKENS)
        if (GQA, end) not in complete
    ]
    if draft_required:
        prerequisites.extend(
            (DRAFT, end, read(64))
            for end in range(BLOCK_TOKENS + 1, highest, BLOCK_TOKENS)
            if (DRAFT, end) not in complete
        )
    return {
        "prior_gdn": {"draft_required": bool(draft_required), "anchors": anchors},
        "prior_prerequisites": prerequisites,
    }


def _prefix_inventory(body: bytes) -> dict:
    gqa, draft, count = struct.unpack("!III", body[:12])
    checkpoints = list(struct.iter_unpack("!IB", body[12:]))
    if len(checkpoints) != count:
        raise ValueError("Invalid prefix inventory response length")
    return {"gqa": gqa, "draft": draft, "checkpoints": checkpoints}


def _transfer_metadata(
    entry: dict, read: Callable[[int], bytes], number: Callable[[int], int]
) -> None:
    entry["device"] = read(16).hex()
    read(64)
    entry["allocations"] = number(4)
    read(72 * entry["allocations"])
    records = []
    for family, layers in ((GQA, 16), (DRAFT, 1)):
        begin, end = number(4), number(4)
        if begin == end == 0:
            continue
        kv_heads = number(4)
        pages = number(4)
        read(4 * 12)
        if pages & PAGE_MAP_FLAG:
            count = (end + PAGE_TOKENS - 1) // PAGE_TOKENS - begin // PAGE_TOKENS
            if count < 0:
                raise ValueError("Invalid mapped LMCache extent")
            entry.setdefault("page_maps", {})[family] = tuple(
                page for (page,) in struct.iter_unpack("!I", read(4 * count))
            )
        for start in range(begin, end, BLOCK_TOKENS):
            entry["keys"].append((family, read(64)))
            tokens = min(BLOCK_TOKENS, end - start)
            records.append(
                {
                    "bytes": tokens * layers * kv_heads * 528,
                    "tokens": tokens,
                    "family": family,
                    "begin": start,
                    "end": start + tokens,
                }
            )
    endpoint = number(4)
    if endpoint:
        entry["keys"].append((GDN, read(64)))
        widths = []
        for _ in range(4):
            read(12)
            widths.append(number(8))
        records.append(
            {
                "bytes": sum(widths),
                "tokens": 0,
                "family": GDN,
                "endpoint": endpoint,
                "widths": widths,
            }
        )
    entry["records"] = records
    if entry["opcode"] == RANK_PUT:
        entry.update(_prior_inventory(read, records))


def frame_metadata(opcode: int, payload: bytes) -> dict:
    view = BytesIO(payload)

    def read(size: int) -> bytes:
        part = view.read(size)
        if len(part) != size:
            raise ValueError("Incomplete LMCache request metadata")
        return part

    def number(size: int) -> int:
        return int.from_bytes(read(size), "big")

    entry = {"opcode": opcode, "namespace": read(number(2)).decode(), "keys": []}
    if opcode in TRANSFERS:
        _transfer_metadata(entry, read, number)
    elif opcode == READY:
        entry["devices"] = [read(16).hex() for _ in range((len(payload) - view.tell()) // 16)]
    elif opcode == PREFIX_DETAIL:
        count, draft = struct.unpack("!IB", read(5))
        digests = read(64 * count)
        entry["coverage_query"] = {
            "count": count,
            "draft": draft,
            "rank_key": digests[:64].decode("ascii"),
        }
    elif opcode == LEASE:
        entry["incarnation"], entry["lease"] = read(16).hex(), read(16).hex()
    elif opcode in (LOOKUP, ERASE):
        count = number(4) if opcode == LOOKUP else 1
        entry["keys"] = [(number(1), read(number(2))) for _ in range(count)]
    if view.read(1):
        raise ValueError("Trailing LMCache request metadata")
    return entry


@dataclass(frozen=True)
class BarrierSelection:
    skip: int = 0
    endpoint: int | None = None


class Barrier:
    def __init__(
        self,
        opcode: int,
        target: int | str | None,
        *,
        phase: str,
        drop: bool,
        selection: BarrierSelection,
    ) -> None:
        self.opcode = opcode
        self.family = target if isinstance(target, int) else None
        self.phase = phase
        self.drop = drop
        self.skip = selection.skip
        self.endpoint = selection.endpoint
        self.device = target if isinstance(target, str) else None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.entry = None

    def wait(self):
        if not self.entered.wait(TIMEOUT):
            raise TimeoutError("Expected real LMCache reply did not reach the relay")
        return self.entry


class Request(socketserver.BaseRequestHandler):
    def forward_arm(self, remote, entry):
        raw, status, size = header(remote)
        body = receive(remote, size)
        if status != 4:
            return raw, status, body
        incarnation, token, pid = ADMISSION.unpack(body)
        peer, _, _ = struct.unpack(
            "3i", remote.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        if pid != peer:
            raise ValueError("Lease admission does not name the upstream daemon")
        self.request.sendall(raw + body)
        arm = receive(self.request, 1)
        if arm != b"\x01":
            raise ValueError("Invalid native lease arm")
        remote.sendall(arm)
        entry.update(
            incarnation=incarnation.hex(),
            lease=token.hex(),
            daemon_pid=pid,
            status=4,
            armed_ns=perf_counter_ns(),
        )
        if not self.deliver(entry, "arm"):
            return None
        raw, status, size = header(remote)
        return raw, status, receive(remote, size)

    def deliver(self, entry, phase):
        barrier = self.server.observe(entry, phase)
        if barrier is None:
            return True
        if not barrier.release.wait(TIMEOUT):
            raise TimeoutError("LMCache relay barrier was not released")
        if phase == "reply":
            entry["barrier_open_unix_ns"] = time_ns()
        if barrier.drop:
            entry["forwarded"] = False
        return not barrier.drop

    def handle(self):
        self.request.settimeout(TIMEOUT)
        entry = None
        phase = "client header"
        try:
            raw, opcode, size = header(self.request)
            phase = f"opcode {opcode} client payload ({size} bytes)"
            payload = receive(self.request, size)
            phase = f"opcode {opcode} metadata ({size} bytes)"
            entry = frame_metadata(opcode, payload)
            phase = f"opcode {opcode} daemon request"
            with socket.socket(socket.AF_UNIX) as remote:
                remote.settimeout(TIMEOUT)
                remote.connect(str(self.server.upstream))
                remote.sendall(raw + payload)
                phase = f"opcode {opcode} daemon reply"
                if opcode in TRANSFERS:
                    result = self.forward_arm(remote, entry)
                    if result is None:
                        return
                    reply, status, body = result
                    if status == 0:
                        entry["receipt"] = RECEIPT.unpack(body)
                else:
                    reply, status, length = header(remote)
                    body = receive(remote, length)
                    if opcode == PREFIX_DETAIL and status == 0:
                        entry["coverage"] = _prefix_inventory(body)
                entry.update(status=status, response_bytes=len(body), reply_ns=perf_counter_ns())
                phase = f"opcode {opcode} forwarding reply"
                if self.deliver(entry, "reply"):
                    self.request.sendall(reply + body)
                    entry["forwarded"] = True
        except (ConnectionError, EOFError, TimeoutError, ValueError, struct.error) as error:
            with self.server.guard:
                self.server.errors.append(f"{phase}: {error}")
        finally:
            self.server.completed(entry)


class Relay(socketserver.ThreadingUnixStreamServer):
    request_queue_size = 64

    def __init__(self, path, upstream):
        self.upstream = upstream
        self.guard = threading.Lock()
        self.entries = []
        self.errors = []
        self.barrier = None
        super().__init__(str(path), Request)

    def __enter__(self):
        self.thread = threading.Thread(target=self.serve_forever)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        if self.barrier is not None:
            self.barrier.release.set()
        self.shutdown()
        self.thread.join()
        self.server_close()

    def observe(self, entry, phase):
        with self.guard:
            if phase == "reply":
                self.entries.append(entry)
            barrier = self.barrier
            if (
                barrier is None
                or barrier.release.is_set()
                or barrier.entered.is_set()
                or entry["opcode"] != barrier.opcode
                or phase != barrier.phase
                or entry["status"] != (4 if phase == "arm" else 0)
                or (barrier.device is not None and entry.get("device") != barrier.device)
                or (
                    barrier.endpoint is not None
                    and not any(
                        record.get("endpoint") == barrier.endpoint
                        for record in entry.get("records", ())
                    )
                )
                or (
                    barrier.family is not None
                    and not any(family == barrier.family for family, _ in entry["keys"])
                )
            ):
                return None
            if barrier.skip:
                barrier.skip -= 1
                return None
            if phase == "arm":
                self.entries.append(entry)
            barrier.entry = entry
            barrier.entered.set()
            return barrier

    def completed(self, entry):
        with self.guard:
            if entry is not None and self.barrier is not None and self.barrier.entry is entry:
                self.barrier.finished.set()

    def arm(
        self,
        opcode: int,
        target: int | str | None = None,
        *,
        phase: str = "reply",
        drop: bool = False,
        **selection: int | None,
    ) -> Barrier:
        with self.guard:
            previous = self.barrier
            if (
                previous
                and not previous.finished.is_set()
                and (previous.entered.is_set() or not previous.release.is_set())
            ):
                raise ValueError("LMCache relay already has an outstanding barrier")
            self.barrier = Barrier(
                opcode, target, phase=phase, drop=drop, selection=BarrierSelection(**selection)
            )
            return self.barrier

    def since(self, index=0):
        with self.guard:
            return [dict(entry) for entry in self.entries[index:]]


def arm_device_fence(directory):
    attempt = Path(mkdtemp(prefix="get-", dir=directory))
    marker = attempt / "arm"
    marker.symlink_to(attempt.resolve())
    marker.replace(directory / "arm")
    return attempt


def install_device_fence(directory):
    """Hold one real H2D operation before its stream fence and CUDA IPC unmap."""
    from sglang_gpu_transfer import DeviceCopies, device_ops, runtime

    path = Path(directory)
    local = threading.local()
    run = DeviceCopies.run
    batch = runtime.cudaMemcpy3DBatchAsync
    serial = device_ops.execute_object_group_transfer
    claim = threading.Lock()

    def gated_run(self, transfer, records, objects, store):
        local.retrieve = not store
        try:
            return run(self, transfer, records, objects, store)
        finally:
            local.retrieve = False

    def hold_retrieve():
        if not getattr(local, "retrieve", False):
            return
        with claim:
            armed = path / "arm"
            try:
                claimed = armed.replace(path / f"claimed-{os.getpid()}")
            except FileNotFoundError:
                return
            attempt = claimed.readlink()
            claimed.unlink()
        (attempt / "entered").touch()
        deadline = monotonic() + TIMEOUT
        while not (attempt / "release").exists() and monotonic() < deadline:
            sleep(0.01)
        if not (attempt / "release").exists():
            raise TimeoutError("Real LMCache stream fence was not released")

    def gated_batch(*args):
        result = batch(*args)
        if int(result[0]) == 0:
            hold_retrieve()
        return result

    def gated_serial(*args):
        result = serial(*args)
        hold_retrieve()
        return result

    DeviceCopies.run = gated_run
    runtime.cudaMemcpy3DBatchAsync = gated_batch
    device_ops.execute_object_group_transfer = gated_serial


if __name__ == "__main__":
    import runpy

    install_device_fence(os.environ["SGLANG_TEST_FENCE_DIR"])
    runpy.run_module("lmcache.v1.multiprocess.server", run_name="__main__", alter_sys=True)
