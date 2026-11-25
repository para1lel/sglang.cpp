"""Observe and control the real LMCache Unix transport for live prefix tests."""

from io import BytesIO
import os
from pathlib import Path
import socket
import socketserver
import struct
from tempfile import mkdtemp
import threading
from time import monotonic, perf_counter_ns, sleep, time_ns


HEADER = struct.Struct("!4sIQ")
MAGIC = b"SLC3"
READY, LOOKUP, GET, PUT, RESET, ERASE, PREFIX, LEASE = range(1, 9)
PREFIX_DETAIL, RANK_GET, RANK_PUT = range(9, 12)
TRANSFERS = (GET, PUT, RANK_GET, RANK_PUT)
ADMISSION = struct.Struct("!16s16sI")
RECEIPT = struct.Struct("!QQQI")
GQA, GDN, DRAFT = range(1, 4)
TIMEOUT = 30


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


def frame_metadata(opcode, payload):
    view = BytesIO(payload)

    def read(size):
        part = view.read(size)
        if len(part) != size:
            raise ValueError("Incomplete LMCache request metadata")
        return part

    def number(size):
        return int.from_bytes(read(size), "big")

    def read_field():
        return read(number(2))

    def read_key():
        return number(1), read_field()

    entry = {"opcode": opcode, "namespace": read_field().decode(), "keys": []}
    if opcode in TRANSFERS:
        entry["device"] = read(16).hex()
        read(64)
        entry["allocations"] = number(4)
        read(72 * entry["allocations"])
        records = []
        for family, layers in ((GQA, 16), (DRAFT, 1)):
            begin, end = number(4), number(4)
            if begin == end == 0:
                continue
            kv_heads = number(4) if opcode in (RANK_GET, RANK_PUT) else 4
            read(4 + 4 * 12)
            for start in range(begin, end, 256):
                entry["keys"].append((family, read(64)))
                tokens = min(256, end - start)
                records.append({"bytes": tokens * layers * kv_heads * 528, "tokens": tokens})
        endpoint = number(4)
        if endpoint:
            entry["keys"].append((GDN, read(64)))
            size = 0
            for _ in range(4):
                read(12)
                size += number(8)
            records.append({"bytes": size, "tokens": 0, "endpoint": endpoint})
        entry["records"] = records
    elif opcode in (PREFIX, PREFIX_DETAIL):
        count, limit, draft = struct.unpack("!IIB", read(9))
        read(64 * count)
        entry["coverage_query"] = {"count": count, "limit": limit, "draft": draft}
    elif opcode == LEASE:
        entry["incarnation"], entry["lease"] = read(16).hex(), read(16).hex()
    elif opcode in (LOOKUP, ERASE):
        count = number(4) if opcode == LOOKUP else 1
        entry["keys"] = [read_key() for _ in range(count)]
    if view.read(1):
        raise ValueError("Trailing LMCache request metadata")
    return entry


class Barrier:
    def __init__(self, opcode, family, *, phase, drop, skip):
        self.opcode = opcode
        self.family = family
        self.phase = phase
        self.drop = drop
        self.skip = skip
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
            "3i", remote.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if pid != peer:
            raise ValueError("Lease admission does not name the upstream daemon")
        self.request.sendall(raw + body)
        arm = receive(self.request, 1)
        if arm != b"\x01":
            raise ValueError("Invalid native lease arm")
        remote.sendall(arm)
        entry.update(incarnation=incarnation.hex(), lease=token.hex(), daemon_pid=pid,
                     status=4, armed_ns=perf_counter_ns())
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
        try:
            raw, opcode, size = header(self.request)
            payload = receive(self.request, size)
            entry = frame_metadata(opcode, payload)
            with socket.socket(socket.AF_UNIX) as remote:
                remote.settimeout(TIMEOUT)
                remote.connect(str(self.server.upstream))
                remote.sendall(raw + payload)
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
                    if opcode == PREFIX and status == 0:
                        entry["coverage"] = struct.unpack("!IIII", body)
                    elif opcode == PREFIX_DETAIL and status == 0:
                        gqa, matched, total, listed = struct.unpack("!IIII", body[:16])
                        ends = list(struct.iter_unpack("!IB", body[16:]))
                        if len(ends) != listed:
                            raise ValueError("Invalid ranked prefix response length")
                        entry["coverage"] = (gqa, matched, total, ends)
                entry.update(status=status, response_bytes=len(body), reply_ns=perf_counter_ns())
                if self.deliver(entry, "reply"):
                    self.request.sendall(reply + body)
                    entry["forwarded"] = True
        except (ConnectionError, EOFError, TimeoutError, ValueError, struct.error) as error:
            with self.server.guard:
                self.server.errors.append(str(error))
        finally:
            self.server.completed(entry)


class Relay(socketserver.ThreadingUnixStreamServer):
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
            if (barrier is None or barrier.release.is_set() or barrier.entered.is_set()
                    or entry["opcode"] != barrier.opcode
                    or phase != barrier.phase or entry["status"] != (4 if phase == "arm" else 0)
                    or (barrier.family is not None and
                        not any(family == barrier.family for family, _ in entry["keys"]))):
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

    def arm(self, opcode, family=None, *, phase="reply", drop=False, skip=0):
        with self.guard:
            previous = self.barrier
            if previous and not previous.finished.is_set() and (
                    previous.entered.is_set() or not previous.release.is_set()):
                raise ValueError("LMCache relay already has an outstanding barrier")
            self.barrier = Barrier(opcode, family, phase=phase, drop=drop, skip=skip)
            return self.barrier

    def since(self, index=0):
        with self.guard:
            return [dict(entry) for entry in self.entries[index:]]


def arm_device_fence(directory):
    # Give each GET its own signals, including when reusing a daemon after a failed run.
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
                # Rename consumes exactly the published marker, even if another process rearms.
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
