import concurrent.futures
import contextlib
import os
import re
import select
import socket
import socketserver
import stat
import struct
import threading
import time
from importlib.metadata import version

import torch
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.storage_manager import l1_owners_of
from lmcache.v1.multiprocess.engine_module import DiscoverableModule


VERSION = "0.5.8"
MAGIC = b"SLC1"
HEADER = struct.Struct("!4sIQ")
MAX_RECORD = 1 << 30
MAX_METADATA = 64 << 20
MAX_KEYS = 1 << 20
LOOKUP_BATCH = 4096
WORKERS = 16
REQUEST_TIMEOUT = 10
LOG = init_logger(__name__)


class Request:
    def __init__(self, peer, deadline):
        self.peer = peer
        self.deadline = deadline
        self.remaining = HEADER.size
        magic, self.op, self.remaining = HEADER.unpack(self.read(HEADER.size))
        if magic != MAGIC:
            raise ValueError("invalid frame magic")
        limit = MAX_RECORD + MAX_METADATA if self.op == 4 else MAX_METADATA
        if self.remaining > limit:
            raise ValueError("request too large")

    def time_left(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("LMCache request deadline exceeded")
        return left

    def check_mutation(self):
        self.time_left()
        hangup = select.POLLRDHUP | select.POLLHUP | select.POLLERR
        poller = select.poll()
        poller.register(self.peer, hangup)
        if poller.poll(0):
            raise ConnectionError("LMCache caller disconnected while waiting")

    def receive(self, data):
        if len(data) > self.remaining:
            raise ValueError("truncated request")
        self.remaining -= len(data)
        while data:
            self.peer.settimeout(self.time_left())
            count = self.peer.recv_into(data)
            if not count:
                raise EOFError("incomplete LMCache request")
            data = data[count:]

    def read(self, size):
        if size > self.remaining:
            raise ValueError("truncated request")
        data = bytearray(size)
        self.receive(memoryview(data))
        return data

    def number(self, width):
        return int.from_bytes(self.read(width), "big")

    def key(self, namespace):
        family = self.number(1)
        identity = bytes(self.read(self.number(2)))
        if family not in (1, 2, 3) or not identity:
            raise ValueError("invalid record key")
        return ObjectKey(
            chunk_hash=identity, model_name="sglang.cpp.v1", kv_rank=0,
            object_group_id=family, cache_salt=namespace,
        )

    def end(self):
        if self.remaining:
            raise ValueError("unexpected request bytes")

    def discard(self):
        buffer = memoryview(bytearray(min(self.remaining, 65536)))
        while self.remaining:
            self.receive(buffer[:self.remaining])

    def reply(self, status, data=b""):
        self.peer.settimeout(self.time_left())
        self.peer.sendall(HEADER.pack(MAGIC, status, len(data)))
        if data:
            self.peer.settimeout(self.time_left())
            self.peer.sendall(data)


class Records:
    def __init__(self, storage):
        self.storage = storage
        (self.l1,) = storage.l1_managers()
        if storage.l2_adapters():
            raise ValueError("sglang LMCache requires a dedicated L1 daemon")

    @contextlib.contextmanager
    def read(self, keys):
        results = self.l1.reserve_read(keys)
        locked = {key: obj for key, (error, obj) in results.items() if error == L1Error.SUCCESS}
        try:
            yield locked
        finally:
            if locked:
                self.l1.finish_read(list(locked))

    def lookup(self, request, namespace):
        count = request.number(4)
        if count > MAX_KEYS or count * 4 > request.remaining:
            raise ValueError("invalid lookup size")
        found = bytearray(count)
        for start in range(0, count, LOOKUP_BATCH):
            end = min(start + LOOKUP_BATCH, count)
            keys = [request.key(namespace) for _ in range(start, end)]
            with self.read(list(dict.fromkeys(keys))) as resident:
                found[start:end] = bytes(key in resident for key in keys)
        request.end()
        request.reply(0, found)

    def get(self, request, key):
        size = request.number(8)
        request.end()
        if not 0 < size <= MAX_RECORD:
            raise ValueError("invalid record size")
        with self.read([key]) as resident:
            obj = resident.get(key)
            if obj is None or obj.get_size() != size:
                request.reply(1)
            else:
                request.reply(0, obj.byte_array.cast("B"))
                self.l1.touch_keys([key])

    def put(self, request, key):
        size = request.remaining
        if not 0 < size <= MAX_RECORD:
            raise ValueError("invalid record size")
        with self.read([key]) as resident:
            exists = key in resident
        if exists:
            request.discard()
            self.acknowledge(request, key, size)
            return
        layout = MemoryLayoutDesc([torch.Size([size])], [torch.uint8])
        reserved = self.storage.reserve_write([key], layout)
        if key not in reserved:
            request.discard()
            self.acknowledge(request, key, size)
            return
        owners = l1_owners_of(reserved)
        try:
            request.receive(reserved[key].byte_array.cast("B"))
            request.end()
        except BaseException:
            self.storage.abort_write([key], owners)
            raise
        self.storage.finish_write([key], owners)
        self.acknowledge(request, key, size)

    def acknowledge(self, request, key, size):
        with self.read([key]) as resident:
            obj = resident.get(key)
            if obj is None or obj.get_size() != size:
                request.reply(1)
            else:
                request.reply(0)
                self.l1.touch_keys([key])


class SocketServer(socketserver.UnixStreamServer):
    request_queue_size = 64

    def __init__(self, endpoint, namespace, records):
        self.namespace = namespace.encode("ascii")
        self.records = records
        self.condition = threading.Condition()
        self.operations = 0
        self.pending_mutations = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS)
        self.slots = threading.BoundedSemaphore(WORKERS)
        super().__init__(endpoint, Handler)
        info = os.stat(endpoint, follow_symlinks=False)
        self.socket_identity = (info.st_dev, info.st_ino)
        os.chmod(endpoint, 0o600, follow_symlinks=False)

    def verify_request(self, request, _address):
        credentials = request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        return struct.unpack("3i", credentials)[1] == os.geteuid()

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        self.pool.submit(self.run_request, request, address)

    def run_request(self, request, address):
        try:
            self.finish_request(request, address)
        finally:
            self.shutdown_request(request)
            self.slots.release()

    @contextlib.contextmanager
    def operation(self, request, exclusive):
        with self.condition:
            if exclusive:
                self.pending_mutations += 1
            try:
                if exclusive:
                    ready = self.condition.wait_for(lambda: self.operations == 0,
                                                    timeout=request.time_left())
                else:
                    ready = self.condition.wait_for(lambda: self.pending_mutations == 0,
                                                    timeout=request.time_left())
                if not ready:
                    raise TimeoutError("LMCache operation deadline exceeded")
            except BaseException:
                if exclusive:
                    self.pending_mutations -= 1
                    self.condition.notify_all()
                raise
            self.operations += 1
        try:
            yield
        finally:
            with self.condition:
                self.operations -= 1
                if exclusive:
                    self.pending_mutations -= 1
                self.condition.notify_all()

    def close(self):
        self.shutdown()
        self.server_close()
        self.pool.shutdown()
        try:
            info = os.stat(self.server_address, follow_symlinks=False)
            if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == self.socket_identity:
                os.unlink(self.server_address)
        except FileNotFoundError:
            pass


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        deadline = time.monotonic() + REQUEST_TIMEOUT
        try:
            request = Request(self.request, deadline)
            namespace = request.read(request.number(2))
            if namespace != self.server.namespace:
                raise ValueError("wrong LMCache namespace")
            if request.op == 1:
                request.end()
                request.reply(0, VERSION.encode("ascii"))
                return
            with self.server.operation(request, request.op in (5, 6)):
                self.dispatch(request, namespace.decode("ascii"))
        except (EOFError, OSError):
            return
        except Exception as error:
            if not isinstance(error, ValueError):
                LOG.exception("LMCache storage operation failed")
            detail = str(error).encode("utf-8", errors="replace")[:1024]
            with contextlib.suppress(OSError):
                left = deadline - time.monotonic()
                if left <= 0:
                    return
                self.request.settimeout(left)
                self.request.sendall(HEADER.pack(MAGIC, 2, len(detail)) + detail)

    def dispatch(self, request, namespace):
        records = self.server.records
        if request.op == 2:
            records.lookup(request, namespace)
        elif request.op == 3:
            records.get(request, request.key(namespace))
        elif request.op == 4:
            records.put(request, request.key(namespace))
        elif request.op == 5:
            request.end()
            request.check_mutation()
            records.storage.clear()
            request.reply(0)
        elif request.op == 6:
            key = request.key(namespace)
            request.end()
            request.check_mutation()
            _, skipped = records.storage.delete_l1_keys([key])
            request.reply(1 if skipped else 0)
        else:
            raise ValueError("unknown LMCache operation")


class SglangSnapshotModule(DiscoverableModule):
    module_name = "sglang_snapshots_v1"

    @classmethod
    def create(cls, build_ctx):
        config = build_ctx.config_for(cls)
        if version("lmcache") != VERSION:
            raise RuntimeError(f"Requires LMCache {VERSION}")
        if not os.environ.get("LMCACHE_API_KEY"):
            raise RuntimeError("sglang LMCache requires the dedicated authenticated launcher")
        endpoint = config["socket"]
        namespace = config["namespace"]
        if not os.path.isabs(endpoint) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", namespace):
            raise ValueError("invalid LMCache endpoint or namespace")
        return cls(build_ctx.engine_context, endpoint, namespace)

    def __init__(self, context, endpoint, namespace):
        records = Records(context.storage_manager)
        self.server = SocketServer(endpoint, namespace, records)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        LOG.info("sglang LMCache at %s, L1 capacity %d bytes", endpoint,
                 records.l1.get_capacity_bytes())

    def get_handlers(self):
        return {}

    def report_status(self):
        return {"capacity_bytes": self.server.records.l1.get_capacity_bytes()}

    def close(self):
        self.server.close()
        self.thread.join()
