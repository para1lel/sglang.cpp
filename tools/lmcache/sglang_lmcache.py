from __future__ import annotations

import concurrent.futures
import contextlib
import os
import re
import select
import socket
import socketserver
import struct
import threading
import time
from collections import Counter
from importlib.metadata import version
from io import BytesIO
from typing import TYPE_CHECKING

import torch
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.l1_manager.dram import DramL1Manager
from lmcache.v1.distributed.storage_manager import _L1_WRITE_TAG, l1_owners_of
from lmcache.v1.multiprocess.engine_module import DiscoverableModule
from sglang_gpu_transfer import (
    DIGEST_BYTES,
    DRAFT_FAMILY,
    GDN_END,
    MAX_TOKENS,
    PREFIX_CHUNK,
    DeviceCopies,
    Transfer,
)
from sglang_socket import OwnedUnixStreamServer

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from lmcache.v1.distributed.storage_manager import StorageManager
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from sglang_gpu_transfer import Retention


VERSION = "0.5.8"
WIRE_VERSION = VERSION + "+native-v4"
MAGIC = b"SLC3"
HEADER = struct.Struct("!4sIQ")
MAX_METADATA = 64 << 20
MAX_KEYS = 1 << 20
LOOKUP_BATCH = 4096
LOOKUP_PREFIX_OP = 9
RETRIEVE_OP = 10
STORE_OP = 11
WORKERS = 16
REQUEST_TIMEOUT = 10
LOG = init_logger(__name__)


def object_key(namespace, family, identity, endpoint=0):
    key = ObjectKey(chunk_hash=identity, model_name="sglang.cpp.gpu.v2", kv_rank=0,
                    object_group_id=family, cache_salt=namespace)
    if endpoint:
        return ObjectKey(identity=key.identity, extra_attributes={GDN_END: endpoint})
    return key


class GdnEndpoints(L1ManagerListener):
    def __init__(self):
        self.lock = threading.Lock()
        self.by_key = {}
        self.counts = Counter()

    def positions(self) -> set[int]:
        with self.lock:
            return set(self.counts)

    def snapshot(self, limit: int) -> list[int]:
        return sorted((end for end in self.positions() if end <= limit), reverse=True)

    def on_l1_keys_write_finished(self, keys):
        with self.lock:
            for key in keys:
                end = key.extra_attributes.get(GDN_END)
                if end is not None and 0 < end <= MAX_TOKENS and key not in self.by_key:
                    self.by_key[key] = end
                    self.counts[end] += 1

    def on_l1_keys_deleted_by_manager(self, keys):
        with self.lock:
            for key in keys:
                end = self.by_key.pop(key, None)
                if end is not None:
                    self.counts[end] -= 1
                    if not self.counts[end]:
                        del self.counts[end]

    def on_l1_keys_reserved_read(self, keys):
        pass

    def on_l1_keys_read_finished(self, keys):
        pass

    def on_l1_keys_reserved_write(self, keys):
        pass

    def on_l1_keys_finish_write_and_reserve_read(self, keys):
        self.on_l1_keys_write_finished(keys)

    def on_l1_keys_accessed(self, keys):
        pass


class TransferMetadata:
    def __init__(self, request):
        self.body = BytesIO(request.read(request.remaining))

    def read(self, size):
        data = self.body.read(size)
        if len(data) != size:
            raise ValueError("truncated native GPU transfer")
        return data

    def number(self, width):
        return int.from_bytes(self.read(width), "big")

    def end(self):
        if self.body.read(1):
            raise ValueError("unexpected native GPU transfer bytes")


class PrefixLookup:
    def __init__(self, request, namespace, records):
        self.count = request.number(4)
        self.draft = request.number(1)
        if (
            self.count > MAX_TOKENS
            or self.draft > 1
            or self.count * DIGEST_BYTES != request.remaining
        ):
            raise ValueError("invalid prefix lookup size")
        self.digests = request.read(request.remaining)
        self.namespace = namespace
        self.records = records
        self.request = request

    def key(self, family, end):
        index = (end - 1 + (family == DRAFT_FAMILY)) * DIGEST_BYTES
        return object_key(self.namespace, family, bytes(self.digests[index : index + DIGEST_BYTES]))

    def present(self, family, ends):
        for start in range(0, len(ends), LOOKUP_BATCH):
            self.request.time_left()
            keys = [self.key(family, end) for end in ends[start : start + LOOKUP_BATCH]]
            with self.records.read(keys) as resident:
                found = [key in resident for key in keys]
            yield from found

    def full_coverage(self, family, limit):
        ends = range(PREFIX_CHUNK, limit + 1, PREFIX_CHUNK)
        for end, found in zip(ends, self.present(family, ends)):
            if not found:
                return end - PREFIX_CHUNK
        return limit // PREFIX_CHUNK * PREFIX_CHUNK

    def coverage(self):
        full = self.full_coverage(1, self.count)
        tail = range(min(self.count, full + PREFIX_CHUNK - 1), full, -1)
        gqa = next((end for end, found in zip(tail, self.present(1, tail)) if found), full)
        ends = self.records.gdn_endpoints.snapshot(self.count)
        states = [end for end, found in zip(ends, self.present(2, ends)) if found]
        if not self.draft or not states:
            return gqa, [(end, False) for end in states]

        draft_full = self.full_coverage(DRAFT_FAMILY, states[0] - 1)
        tails = [
            end - 1
            for end in states
            if end <= draft_full + PREFIX_CHUNK and (end - 1) % PREFIX_CHUNK
        ]
        resident_tails = {
            end
            for end, found in zip(tails, self.present(DRAFT_FAMILY, tails), strict=True)
            if found
        }
        return gqa, [
            (
                end,
                end <= draft_full + PREFIX_CHUNK
                and ((end - 1) % PREFIX_CHUNK == 0 or end - 1 in resident_tails),
            )
            for end in states
        ]


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
        self.peer.settimeout(REQUEST_TIMEOUT)
        self.peer.sendall(HEADER.pack(MAGIC, status, len(data)) + data)


class DmaLock:
    def __init__(self, ttl, finishing):
        self.ttl = ttl
        self.finishing = finishing
        self.pins = 0

    def is_locked(self):
        return self.pins > 0 or self.ttl.is_locked()

    def lock(self):
        self.ttl.lock()

    def unlock(self):
        locks = getattr(self.finishing, "locks", None)
        if locks is not None and self in locks:
            locks.remove(self)
            self.pins -= 1
        else:
            self.ttl.unlock()


class DmaReservations(L1ManagerListener):
    """Keep eviction and reservation under the pinned manager's own mutex."""

    def __init__(self, manager: DramL1Manager) -> None:
        match manager:
            case DramL1Manager(_lock=lock, _objects=objects, _registered_listeners=listeners):
                self.lock = lock
                self.objects = objects
                self.listeners = listeners
            case _:
                raise RuntimeError("Incompatible LMCache 0.5.8 DRAM manager state")
        self.manager = manager
        self.finishing = threading.local()
        self.writing = threading.local()
        self.activity_lock = threading.Lock()
        self.active = 0
        self.clear_l1 = manager.clear
        self.delete_l1 = manager.delete
        manager.clear = self.clear
        manager.delete = self.delete
        with self.lock:
            self.listeners.insert(0, self)

    @contextlib.contextmanager
    def activity(self):
        with self.activity_lock:
            self.active += 1
        try:
            yield
        finally:
            with self.activity_lock:
                self.active -= 1

    def clear(self, force=False):
        return self.reclaim(self.clear_l1, force)

    def delete(self, keys, force=False):
        return self.reclaim(self.delete_l1, force, keys)

    def reclaim(self, operation, force, *args):
        if not force:
            return operation(*args)
        with self.activity_lock:
            if self.active:
                raise RuntimeError("Cannot force reclaim during an active sglang DMA reservation")
            return operation(*args, force=True)

    def pin(self, entry, field):
        lock = getattr(entry, field)
        if not isinstance(lock, DmaLock):
            lock = DmaLock(lock, self.finishing)
            setattr(entry, field, lock)
        lock.pins += 1
        return lock

    @contextlib.contextmanager
    def read(self, keys):
        with self.activity(), self.read_pinned(keys) as entries:
            yield entries

    @contextlib.contextmanager
    def read_pinned(self, keys):
        entries = {}
        try:
            with self.lock:
                for key in dict.fromkeys(keys):
                    entry = self.objects.get(key)
                    if entry is not None:
                        entries[key] = (entry, self.pin(entry, "read_lock"))
                self.manager._report_read_reserved(list(entries))
            yield {key: entry.memory_obj for key, (entry, _) in entries.items()}
        finally:
            if entries:
                self.finish_read(entries)

    def finish_read(self, entries):
        with self.lock:
            pending = {
                key: lock
                for key, (entry, lock) in entries.items()
                if self.objects.get(key) is entry and entry.read_lock is lock
            }
        self.finishing.locks = set(pending.values())
        try:
            if pending:
                self.manager.finish_read(list(pending))
        finally:
            unconsumed = self.finishing.locks
            del self.finishing.locks
            with self.lock:
                for key, (entry, lock) in entries.items():
                    if key not in pending or lock in unconsumed:
                        lock.pins -= 1
                    if lock.pins == 0 and entry.read_lock is lock:
                        entry.read_lock = lock.ttl

    @contextlib.contextmanager
    def write(self, storage, reserved):
        with self.activity(), self.write_pinned(storage, reserved):
            yield

    @contextlib.contextmanager
    def write_pinned(self, storage, reserved):
        self.writing.reserved = reserved
        self.writing.entries = {}
        self.writing.published = set()
        self.writing.key = None
        try:
            yield
        finally:
            entries = self.writing.entries
            try:
                with self.lock:
                    aborted = {key: obj for key, obj in reserved.items()
                               if self.manager._get_staging(key, _L1_WRITE_TAG) is entries[key][0]}
                if aborted:
                    storage.abort_write(list(aborted), l1_owners_of(aborted))
            finally:
                with self.lock:
                    for entry, lock in entries.values():
                        if entry.write_lock is lock:
                            entry.write_lock = lock.ttl
                del self.writing.entries, self.writing.published, self.writing.key
                del self.writing.reserved

    def reserve_write(self, storage, key, layout, reserved):
        with self.lock:
            if self.manager._get_staging(key, _L1_WRITE_TAG) is not None:
                return False
        self.writing.key = key
        try:
            objects = storage.reserve_write([key], layout)
            return key in self.writing.entries and objects.get(key) is reserved[key]
        finally:
            self.writing.key = None

    def finish_write(self, storage, key, reserved):
        storage.finish_write([key], l1_owners_of({key: reserved[key]}))
        return key in self.writing.published

    def on_l1_keys_reserved_write(self, keys):
        key = getattr(self.writing, "key", None)
        if key is not None and key in keys:
            entry = self.manager._get_staging(key, _L1_WRITE_TAG)
            if entry is not None:
                self.writing.entries[key] = (entry, self.pin(entry, "write_lock"))
                self.writing.reserved[key] = entry.memory_obj

    def on_l1_keys_write_finished(self, keys):
        entries = getattr(self.writing, "entries", {})
        for key in keys:
            if key in entries and self.objects.get(key) is entries[key][0]:
                self.writing.published.add(key)

    def on_l1_keys_reserved_read(self, keys):
        pass

    def on_l1_keys_read_finished(self, keys):
        pass

    def on_l1_keys_finish_write_and_reserve_read(self, keys):
        pass

    def on_l1_keys_deleted_by_manager(self, keys):
        pass

    def on_l1_keys_accessed(self, keys):
        pass


class Records:
    def __init__(self, storage: StorageManager, *, single_device: bool = False) -> None:
        self.storage = storage
        (self.l1,) = storage.l1_managers()
        if not isinstance(self.l1, DramL1Manager) or storage.l2_adapters():
            raise ValueError("sglang LMCache requires a dedicated DRAM L1 daemon")
        self.device = DeviceCopies(single_device=single_device)
        self.writes = threading.Condition()
        self.writing_keys = set()
        self.dma = DmaReservations(self.l1)
        self.gdn_endpoints = GdnEndpoints()
        with self.dma.lock:
            self.gdn_endpoints.on_l1_keys_write_finished(self.dma.objects)
            self.dma.listeners.append(self.gdn_endpoints)

    @contextlib.contextmanager
    def read(self, keys):
        with self.dma.read(keys) as resident:
            yield resident

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

    @contextlib.contextmanager
    def claim_write(self, request: Request, keys: list[ObjectKey]) -> Generator[None, None, None]:
        with self.writes:
            while not self.writing_keys.isdisjoint(keys):
                request.check_mutation()
                self.writes.wait(timeout=0.25)
            request.check_mutation()
            self.writing_keys.update(keys)
        try:
            yield
        finally:
            with self.writes:
                self.writing_keys.difference_update(keys)
                self.writes.notify_all()

    def touch_prefix(self, transfer: Transfer, *, stored: bool = False) -> None:
        ordered = [(record.prefix_end, record.key) for record in transfer.records]
        with self.dma.lock:
            if stored and transfer.retention is not None:
                ordered.extend(self.resident_anchors(transfer.retention, self.dma.objects))
            ordered.sort(key=lambda item: item[0])
            self.l1.touch_keys(
                list(dict.fromkeys(key for _, key in ordered if key in self.dma.objects))
            )

    @staticmethod
    def resident_anchors(
        retention: Retention, objects: Mapping[ObjectKey, object]
    ) -> Generator[tuple[int, ObjectKey], None, None]:
        def consecutive(keys: list[ObjectKey]) -> int:
            return next((index for index, key in enumerate(keys) if key not in objects), len(keys))

        target_ready = consecutive(retention.gqa_chunks)
        draft_ready = consecutive(retention.draft_chunks)
        target_needed = draft_needed = 0
        for end, target, anchor, shifted in retention.anchors:
            target_chunks = (end - 1) // PREFIX_CHUNK
            draft_chunks = (end - 2) // PREFIX_CHUNK if shifted is not None else 0
            if (
                target_chunks <= target_ready
                and draft_chunks <= draft_ready
                and target in objects
                and anchor in objects
                and (shifted is None or shifted in objects)
            ):
                yield end, target
                yield end, anchor
                if shifted is not None:
                    yield end, shifted
                target_needed = target_chunks
                draft_needed = draft_chunks
        for index in range(target_needed):
            yield (index + 1) * PREFIX_CHUNK, retention.gqa_chunks[index]
        for index in range(draft_needed):
            yield (index + 1) * PREFIX_CHUNK + 1, retention.draft_chunks[index]

    def retrieve(self, request, transfer):
        keys = [record.key for record in transfer.records]
        with self.read(keys) as resident:
            if any(not self.fits(resident, record) for record in transfer.records):
                return 1, b""
            request.check_mutation()
            self.device.run(transfer, transfer.records, resident, store=False)
            self.touch_prefix(transfer)
            return self.receipt(transfer.records, store=False)

    def store(self, request, transfer):
        keys = [record.key for record in transfer.records]
        with self.claim_write(request, keys), self.read(keys) as resident:
            pending = [record for record in transfer.records if record.key not in resident]
            if any(not self.fits(resident, record) for record in transfer.records
                   if record.key in resident):
                return 3, b""
            result = self.write(request, transfer, pending)
            if result[0] == 0:
                self.touch_prefix(transfer, stored=True)
            return result

    def write(self, request, transfer, pending):
        reserved = {}
        with self.dma.write(self.storage, reserved):
            for record in pending:
                request.check_mutation()
                layout = MemoryLayoutDesc([torch.Size([record.size])], [torch.uint8])
                if not self.dma.reserve_write(self.storage, record.key, layout, reserved):
                    return 3, b""
            if pending:
                self.device.run(transfer, pending, reserved, store=True)
            request.check_mutation()
            for record in pending:
                if not self.dma.finish_write(self.storage, record.key, reserved):
                    return 3, b""
                del reserved[record.key]
            return self.receipt(pending, store=True)

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


class SocketServer(OwnedUnixStreamServer):
    request_queue_size = 64

    def __init__(
        self,
        endpoint: str,
        namespace: str,
        records: Records,
        device_copies: DeviceCopies | None = None,
    ) -> None:
        self.namespace = namespace.encode("ascii")
        self.records = records
        self.device_copies = device_copies
        self.incarnation = os.urandom(16)
        self.leases = set()
        self.condition = threading.Condition()
        self.operations = 0
        self.pending_mutations = 0
        device_count = 1 if device_copies is None else len(device_copies.devices)
        workers = WORKERS * device_count
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.slots = threading.BoundedSemaphore(workers)
        self.gpu_slots = threading.BoundedSemaphore((WORKERS - 4) * device_count)
        super().__init__(endpoint, Handler)

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
        self.pool.shutdown()
        try:
            self.close_endpoint()
        except FileExistsError:
            LOG.warning("LMCache endpoint was replaced: %s", self.server_address)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        deadline = time.monotonic() + REQUEST_TIMEOUT
        try:
            request = Request(self.request, deadline)
            namespace = request.read(request.number(2))
            if namespace != self.server.namespace:
                raise ValueError("wrong LMCache namespace")
            if request.op == 1:
                self.ready(request)
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

    def ready(self, request: Request) -> None:
        uuid_bytes = 16
        if request.remaining % uuid_bytes:
            raise ValueError("invalid readiness GPU UUID list")
        uuids = request.read(request.remaining)
        copies = self.server.device_copies
        if any(
            copies is None or bytes(uuids[index : index + uuid_bytes]) not in copies.devices
            for index in range(0, len(uuids), uuid_bytes)
        ):
            request.reply(3)
        else:
            request.reply(0, WIRE_VERSION.encode("ascii"))

    def dispatch(self, request, namespace):
        records = self.server.records
        if request.op == 2:
            records.lookup(request, namespace)
        elif request.op in (RETRIEVE_OP, STORE_OP):
            if not self.server.gpu_slots.acquire(blocking=False):
                request.reply(3)
                return
            try:
                metadata = TransferMetadata(request)
                transfer = Transfer.parse(
                    metadata,
                    lambda family, identity, endpoint: object_key(
                        namespace, family, identity, endpoint
                    ),
                    store=request.op == STORE_OP,
                )
                if self.server.device_copies is not None:
                    self.server.device_copies.device_for(transfer.device)
                operation = records.retrieve if request.op == RETRIEVE_OP else records.store
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
        elif request.op == LOOKUP_PREFIX_OP:
            gqa, entries = PrefixLookup(request, namespace, records).coverage()
            request.reply(
                0,
                struct.pack("!II", gqa, len(entries))
                + b"".join(struct.pack("!IB", end, draft_ok) for end, draft_ok in entries),
            )
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
        single_device = config.get("single_device", False)
        return cls(build_ctx.engine_context, endpoint, namespace, single_device=single_device)

    def __init__(
        self,
        context: MPCacheServerContext,
        endpoint: str,
        namespace: str,
        *,
        single_device: bool = False,
    ) -> None:
        records = Records(context.storage_manager, single_device=single_device)
        self.server = SocketServer(endpoint, namespace, records, records.device)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        devices = records.device.devices
        LOG.info(
            "sglang GPU LMCache at %s, %s %s, L1 capacity %d bytes",
            endpoint,
            "device" if len(devices) == 1 else "devices",
            ",".join(uuid.hex() for uuid in devices),
            records.l1.get_capacity_bytes(),
        )

    def report_status(self):
        return {"capacity_bytes": self.server.records.l1.get_capacity_bytes()}

    def close(self):
        self.server.close()
        self.thread.join()
