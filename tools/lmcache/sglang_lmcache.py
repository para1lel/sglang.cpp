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

from sglang_gpu_transfer import DeviceCopies, Transfer


VERSION = "0.5.8"
MAGIC = b"SLC2"
HEADER = struct.Struct("!4sIQ")
MAX_METADATA = 64 << 20
MAX_KEYS = 1 << 20
LOOKUP_BATCH = 4096
PREFIX_CHUNK = 256
DIGEST_BYTES = 64
WORKERS = 16
REQUEST_TIMEOUT = 10
LOG = init_logger(__name__)


def object_key(namespace, family, identity):
    return ObjectKey(chunk_hash=identity, model_name="sglang.cpp.gpu.v2", kv_rank=0,
                     object_group_id=family, cache_salt=namespace)


class PrefixLookup:
    def __init__(self, request, namespace, records):
        self.count = request.number(4)
        self.limit = request.number(4)
        self.draft = request.number(1)
        if self.count > MAX_KEYS or self.limit > self.count or self.draft > 1 or \
                self.count * DIGEST_BYTES != request.remaining:
            raise ValueError("invalid prefix lookup size")
        self.digests = request.read(request.remaining)
        self.namespace = namespace
        self.records = records
        self.request = request

    def key(self, family, end):
        index = (end - 1 + (family == 3)) * DIGEST_BYTES
        return object_key(self.namespace, family, bytes(self.digests[index:index + DIGEST_BYTES]))

    def present(self, family, ends):
        self.request.time_left()
        keys = [self.key(family, end) for end in ends]
        with self.records.read(keys) as resident:
            return [key in resident for key in keys]

    def full_coverage(self, family, limit):
        ends = range(PREFIX_CHUNK, limit + 1, PREFIX_CHUNK)
        for end, found in zip(ends, self.present(family, ends)):
            if not found:
                return end - PREFIX_CHUNK
        return limit // PREFIX_CHUNK * PREFIX_CHUNK

    def matched_gdn(self):
        for upper in range(self.count, 0, -LOOKUP_BATCH):
            ends = range(upper, max(0, upper - LOOKUP_BATCH), -1)
            match = next((end for end, found in zip(ends, self.present(2, ends)) if found), 0)
            if match:
                return match
        return 0

    def resolve(self):
        full = self.full_coverage(1, self.count)
        tail = range(min(self.count, full + PREFIX_CHUNK - 1), full, -1)
        gqa = next((end for end, found in zip(tail, self.present(1, tail)) if found), full)
        maximum = min(self.limit, gqa)
        draft_full = self.full_coverage(3, maximum - 1) if self.draft and maximum else 0
        if self.draft:
            maximum = min(maximum, draft_full + PREFIX_CHUNK)
        for upper in range(maximum, 0, -LOOKUP_BATCH):
            ends = range(upper, max(0, upper - LOOKUP_BATCH), -1)
            states = [end for end, found in zip(ends, self.present(2, ends)) if found]
            if not self.draft:
                if states:
                    return gqa, states[0], 0
                continue
            tails = [end - 1 for end in states if (end - 1) % PREFIX_CHUNK]
            usable = {end for end, found in zip(tails, self.present(3, tails)) if found}
            for end in states:
                if (end - 1) % PREFIX_CHUNK == 0 or end - 1 in usable:
                    return gqa, end, end - 1
        return gqa, 0, 0


class Request:
    def __init__(self, peer, deadline):
        self.peer = peer
        self.deadline = deadline
        self.armed = False
        self.remaining = HEADER.size
        magic, self.op, self.remaining = HEADER.unpack(self.read(HEADER.size))
        if magic != MAGIC:
            raise ValueError("invalid frame magic")
        if self.remaining > MAX_METADATA:
            raise ValueError("request too large")

    def time_left(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("LMCache request deadline exceeded")
        return left

    def check_mutation(self):
        if not self.armed:
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
        return object_key(namespace, family, identity)

    def end(self):
        if self.remaining:
            raise ValueError("unexpected request bytes")

    def reply(self, status, data=b""):
        self.peer.settimeout(self.time_left())
        self.peer.sendall(HEADER.pack(MAGIC, status, len(data)))
        if data:
            self.peer.settimeout(self.time_left())
            self.peer.sendall(data)

    def arm(self, admission):
        self.reply(4, admission)
        self.peer.settimeout(self.time_left())
        if self.peer.recv(1) != b"\x01":
            raise EOFError("GPU operation was not armed")
        self.armed = True

    def release_reply(self, status, data=b""):
        # GPU IPC leases end after device synchronization and unmapping, even after a deadline.
        self.peer.settimeout(REQUEST_TIMEOUT)
        self.peer.sendall(HEADER.pack(MAGIC, status, len(data)) + data)


class Records:
    def __init__(self, storage):
        self.storage = storage
        (self.l1,) = storage.l1_managers()
        if storage.l2_adapters():
            raise ValueError("sglang LMCache requires a dedicated L1 daemon")
        self.device = DeviceCopies()
        self.writes = threading.Lock()

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

    def retrieve(self, request, transfer):
        keys = [record.key for record in transfer.records]
        with self.read(keys) as resident:
            if any(not self.fits(resident, record) for record in transfer.records):
                return 1, b""
            request.check_mutation()
            self.device.run(transfer, transfer.records, resident, store=False)
            self.l1.touch_keys(keys)
            return self.receipt(transfer.records, store=False)

    def store(self, request, transfer):
        keys = [record.key for record in transfer.records]
        # Serialize writers to keep immutable records and their state marker coherent.
        with self.writes, self.read(keys) as resident:
            pending = [record for record in transfer.records if record.key not in resident]
            if any(not self.fits(resident, record) for record in transfer.records
                   if record.key in resident):
                return 3, b""
            result = self.write(request, transfer, pending)
            if result[0] == 0:
                self.l1.touch_keys(keys)
            return result

    def write(self, request, transfer, pending):
        reserved = {}
        try:
            for record in pending:
                request.check_mutation()
                layout = MemoryLayoutDesc([torch.Size([record.size])], [torch.uint8])
                reserved.update(self.storage.reserve_write([record.key], layout))
                if record.key not in reserved:
                    return 3, b""
            if pending:
                self.device.run(transfer, pending, reserved, store=True)
            request.check_mutation()
            for record in pending:
                key = record.key
                self.storage.finish_write([key], l1_owners_of({key: reserved[key]}))
                del reserved[key]
            return self.receipt(pending, store=True)
        finally:
            if reserved:
                self.storage.abort_write(list(reserved), l1_owners_of(reserved))

    @staticmethod
    def fits(resident, record):
        obj = resident.get(record.key)
        return obj is not None and obj.get_size() == record.size

    @staticmethod
    def receipt(records, store):
        tally = [0, 0, 0]
        tokens = 0
        for record in records:
            family = record.key.object_group_id
            tally[family - 1] += record.size
            if store and family == 1:
                tokens += record.tokens
        return 0, struct.pack("!QQQI", *tally, tokens)


class SocketServer(socketserver.UnixStreamServer):
    request_queue_size = 64

    def __init__(self, endpoint, namespace, records):
        self.namespace = namespace.encode("ascii")
        self.records = records
        self.incarnation = os.urandom(16)
        self.leases = set()
        self.condition = threading.Condition()
        self.operations = 0
        self.pending_mutations = 0
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS)
        self.slots = threading.BoundedSemaphore(WORKERS)
        # Armed GPU operations cannot occupy every worker needed to report lease status.
        self.gpu_slots = threading.BoundedSemaphore(WORKERS - 4)
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
    def lease(self, request):
        token = os.urandom(16)
        with self.condition:
            self.leases.add(token)
        try:
            request.arm(struct.pack("!16s16sI", self.incarnation, token, os.getpid()))
            yield
        finally:
            with self.condition:
                self.leases.remove(token)

    def lease_status(self, request):
        incarnation = request.read(16)
        token = bytes(request.read(16))
        request.end()
        with self.condition:
            status = 3 if incarnation != self.incarnation else (4 if token in self.leases else 0)
        request.reply(status)

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
            if request.op == 8:
                self.server.lease_status(request)
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
        elif request.op in (3, 4):
            if not self.server.gpu_slots.acquire(blocking=False):
                request.reply(3)
                return
            try:
                transfer = Transfer.parse(request, namespace)
                operation = records.retrieve if request.op == 3 else records.store
                with self.server.lease(request):
                    result = operation(request, transfer)
                request.release_reply(*result)
            finally:
                self.server.gpu_slots.release()
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
        elif request.op == 7:
            query = PrefixLookup(request, namespace, records)
            coverage = query.resolve()
            # The second probe can race eviction of an already-observed usable checkpoint.
            matched_gdn = max(coverage[1], query.matched_gdn())
            request.reply(0, struct.pack("!IIII", *coverage, matched_gdn))
        else:
            raise ValueError("unknown LMCache operation")


class SglangSnapshotModule(DiscoverableModule):
    module_name = "sglang_gpu_snapshots_v2"

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
        LOG.info("sglang GPU LMCache at %s, device %s, L1 capacity %d bytes", endpoint,
                 records.device.uuid.hex(), records.l1.get_capacity_bytes())

    def get_handlers(self):
        return {}

    def report_status(self):
        return {"capacity_bytes": self.server.records.l1.get_capacity_bytes()}

    def close(self):
        self.server.close()
        self.thread.join()
