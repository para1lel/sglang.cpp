from __future__ import annotations

import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import pairwise
from threading import Lock
from typing import TYPE_CHECKING, Protocol

import torch
from cuda.bindings import driver, runtime
from lmcache import device_ops
from lmcache.lmcache_native import TransferDirection
from lmcache.logging import init_logger
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.devices.cuda.utils import cuda_ipc_handle_from_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from lmcache.v1.distributed.api import ObjectKey

    Allocation = tuple[bytes, int]
    Span = tuple[int, int, int]

    RecordKey = Callable[[int, bytes, int], ObjectKey]


MAX_RECORD = 1 << 30
MAX_ALLOCATION = 1 << 40
MAX_TOKENS = 1 << 18
PAGE_TOKENS = 128
MAX_KV_HEADS = 128
PAYLOAD_PLANES = 2
MAPPED_PAGES_FLAG = 1 << 31
PREFIX_CHUNK = 256
DIGEST_BYTES = 64
DRAFT_FAMILY = 3
RETENTION_LIMIT = 12
LOWER_DIGEST = re.compile(rb"[0-9a-f]{64}")
GDN_END = "sglang_gdn_end"
LOG = init_logger(__name__)


class TransferReader(Protocol):
    def read(self, size: int) -> bytes: ...

    def number(self, width: int) -> int: ...

    def end(self) -> None: ...


def plane_span(allocations: list[Allocation], allocation: int, offset: int, size: int) -> Span:
    if allocation >= len(allocations) or not size or offset + size > allocations[allocation][1]:
        raise ValueError("native CUDA plane exceeds allocation")
    return allocation, offset, offset + size


def validate_disjoint(spans: list[Span]) -> None:
    spans.sort()
    for (allocation, _, end), (next_allocation, start, _) in pairwise(spans):
        if allocation == next_allocation and end > start:
            raise ValueError("overlapping native CUDA operands")


@dataclass(frozen=True)
class Region:
    allocation: int
    device_offset: int
    device_pitch: int
    record_offset: int
    record_pitch: int
    width: int
    rows: int


@dataclass(frozen=True)
class GdnRecord:
    key: ObjectKey
    size: int
    regions: tuple
    tokens: int = 0

    @property
    def prefix_end(self) -> int:
        return self.key.extra_attributes[GDN_END]

    @classmethod
    def parse(
        cls,
        request: TransferReader,
        allocations: list[Allocation],
        key: RecordKey,
        spans: list[Span],
    ) -> GdnRecord | None:
        endpoint = request.number(4)
        if not endpoint:
            return None
        if endpoint > MAX_TOKENS:
            raise ValueError("invalid native GDN endpoint")
        record_key = key(2, request.read(DIGEST_BYTES), endpoint)
        size = 0
        regions = []
        for _ in range(4):
            allocation, offset, width = (request.number(4), request.number(8), request.number(8))
            spans.append(plane_span(allocations, allocation, offset, width))
            regions.append(Region(allocation, offset, width, size, width, width, 1))
            size += width
        if size > MAX_RECORD:
            raise ValueError("native GDN record exceeds size limit")
        return cls(record_key, size, tuple(regions))


@dataclass(frozen=True)
class NativeMxfp8:
    layers: int
    pages: int
    planes: tuple
    kv_heads: int
    physical_pages: tuple[int, ...] = ()
    first_page: int = 0

    @classmethod
    def parse_records(
        cls,
        request: TransferReader,
        allocations: list[Allocation],
        family: int,
        key: RecordKey,
        spans: list[Span],
    ) -> list[NativeRecord]:
        begin, end = request.number(4), request.number(4)
        if begin == end == 0:
            return []
        kv_heads = request.number(4)
        encoded_pages = request.number(4)
        mapped = bool(encoded_pages & MAPPED_PAGES_FLAG)
        pages = encoded_pages & (MAPPED_PAGES_FLAG - 1)
        planes = tuple((request.number(4), request.number(8)) for _ in range(4))
        layers = 16 if family == 1 else 1
        if (
            not begin < end <= MAX_TOKENS - (family == DRAFT_FAMILY)
            or begin % PREFIX_CHUNK
            or (not mapped and end > pages * PAGE_TOKENS)
        ):
            raise ValueError("invalid native MXFP8 extent")
        if (
            not pages
            or (not mapped and pages > MAX_TOKENS // PAGE_TOKENS)
            or not 0 < kv_heads <= MAX_KV_HEADS
        ):
            raise ValueError("invalid native MXFP8 geometry")
        row_bytes = (kv_heads * 256, kv_heads * 256, kv_heads * 8, kv_heads * 8)
        for (allocation, offset), plane_bytes in zip(planes, row_bytes, strict=True):
            spans.append(
                plane_span(
                    allocations, allocation, offset, layers * pages * PAGE_TOKENS * plane_bytes
                )
            )
        first_page = begin // PAGE_TOKENS
        count = (end + PAGE_TOKENS - 1) // PAGE_TOKENS - first_page
        if mapped and count > pages:
            raise ValueError("native MXFP8 physical page map exceeds allocation")
        physical = tuple(request.number(4) for _ in range(count)) if mapped else ()
        if mapped and (max(physical) >= pages or len(set(physical)) != count):
            raise ValueError("invalid native MXFP8 physical page map")
        view = cls(layers, pages, planes, kv_heads, physical, first_page)
        return [
            NativeRecord(
                key(family, request.read(DIGEST_BYTES), 0),
                min(PREFIX_CHUNK, end - start),
                start,
                view,
            )
            for start in range(begin, end, PREFIX_CHUNK)
        ]

    def page_runs(self, begin: int, rows: int) -> tuple[tuple[int, int, int], ...]:
        if not self.physical_pages:
            return ((0, begin, rows),)
        page = begin // PAGE_TOKENS - self.first_page
        first = self.physical_pages[page]
        if rows <= PAGE_TOKENS or self.physical_pages[page + 1] == first + 1:
            return ((0, first * PAGE_TOKENS, rows),)
        second = self.physical_pages[page + 1]
        return (
            (0, first * PAGE_TOKENS, PAGE_TOKENS),
            (PAGE_TOKENS, second * PAGE_TOKENS, rows - PAGE_TOKENS),
        )

    def regions(self, begin: int, rows: int) -> Iterator[Region]:
        offset = 0
        runs = self.page_runs(begin, rows)
        for index, (allocation, address) in enumerate(self.planes):
            payload = index < PAYLOAD_PLANES
            row_bytes = self.kv_heads * (256 if payload else 8)
            width = rows * row_bytes
            pitch = self.pages * PAGE_TOKENS * row_bytes
            for logical, physical, count in runs:
                origin = address + physical * row_bytes
                target = offset + logical * row_bytes
                full = count if payload else count // PAGE_TOKENS * PAGE_TOKENS
                if full:
                    yield Region(
                        allocation, origin, pitch, target, width, full * row_bytes, self.layers
                    )
                if full < count:
                    shift = full * row_bytes
                    tail = Region(
                        allocation,
                        origin + shift,
                        pitch,
                        target + shift,
                        width,
                        (count - full) * row_bytes,
                        self.layers,
                    )
                    yield from self.tail_regions(tail, count - full)
            offset += width * self.layers

    def tail_regions(self, region: Region, rows: int) -> Iterator[Region]:
        complete = rows // 32 * 8
        tail = rows % 32
        lanes = self.kv_heads * 32
        for layer in range(region.rows):
            device = region.device_offset + layer * region.device_pitch
            record = region.record_offset + layer * region.record_pitch
            if complete:
                yield Region(region.allocation, device, 32, record, complete, complete, lanes)
                record += complete * lanes
            if tail:
                for head in range(self.kv_heads):
                    yield Region(
                        region.allocation, device + head * 1024 + complete, 32, record, 8, 8, tail
                    )
                    record += tail * 8


@dataclass(frozen=True)
class NativeRecord:
    key: ObjectKey
    tokens: int
    begin: int
    view: NativeMxfp8

    @property
    def prefix_end(self) -> int:
        return self.begin + self.tokens + (self.key.object_group_id == DRAFT_FAMILY)

    @property
    def size(self):
        return self.tokens * self.view.layers * 2 * self.view.kv_heads * (256 + 8)

    @property
    def regions(self):
        return self.view.regions(self.begin, self.tokens)


@dataclass(frozen=True)
class Retention:
    anchors: list[tuple[int, ObjectKey, ObjectKey, ObjectKey | None]]
    gqa_chunks: list[ObjectKey]
    draft_chunks: list[ObjectKey]

    @classmethod
    def parse(
        cls,
        request: TransferReader,
        key: RecordKey,
        current: list[NativeRecord | GdnRecord],
        gdn: GdnRecord | None,
    ) -> Retention | None:
        count, draft = request.number(1), request.number(1)
        if count > RETENTION_LIMIT or draft > 1 or (draft and not count):
            raise ValueError("invalid native retention header")
        if not count:
            return None
        if gdn is None:
            raise ValueError("native retention requires a current GDN endpoint")

        current_keys = {
            (record.key.object_group_id, record.prefix_end): record.key
            for record in current
            if record.tokens
        }
        anchors = []
        last = 0
        for _ in range(count):
            end = request.number(4)
            if not last < end < gdn.prefix_end:
                raise ValueError("invalid native retention endpoint")
            identity = retention_digest(request)
            target = key(1, identity, 0)
            shifted = key(DRAFT_FAMILY, identity, 0) if draft and end > 1 else None
            anchors.append((end, target, key(2, identity, end), shifted))
            last = end
        check_retention_anchors(anchors, current_keys)

        def chunks(family: int, shift: int) -> list[ObjectKey]:
            result = []
            for end in range(PREFIX_CHUNK + shift, last, PREFIX_CHUNK):
                record = current_keys.get((family, end))
                result.append(
                    record if record is not None else key(family, retention_digest(request), 0)
                )
            return result

        return cls(anchors, chunks(1, 0), chunks(DRAFT_FAMILY, 1) if draft else [])


def retention_digest(request: TransferReader) -> bytes:
    identity = request.read(DIGEST_BYTES)
    if LOWER_DIGEST.fullmatch(identity) is None:
        raise ValueError("invalid native retention digest")
    return identity


def check_retention_anchors(
    anchors: list[tuple[int, ObjectKey, ObjectKey, ObjectKey | None]],
    current: dict[tuple[int, int], ObjectKey],
) -> None:
    for end, target, _, shifted in anchors:
        if current.get((1, end), target) != target or (
            shifted is not None and current.get((DRAFT_FAMILY, end), shifted) != shifted
        ):
            raise ValueError("native retention digest disagrees with transfer")


@dataclass(frozen=True)
class Transfer:
    device: bytes
    event: bytes
    allocations: list[tuple[bytes, int]]
    records: list[NativeRecord | GdnRecord]
    retention: Retention | None = None

    @classmethod
    def parse(cls, request: TransferReader, key: RecordKey, *, store: bool = False) -> Transfer:
        device = bytes(request.read(16))
        event = bytes(request.read(64))
        count = request.number(4)
        if not 0 < count <= 64:
            raise ValueError("invalid CUDA IPC allocation count")
        allocations = [(bytes(request.read(64)), request.number(8)) for _ in range(count)]
        if any(not 0 < size <= MAX_ALLOCATION for _, size in allocations):
            raise ValueError("invalid CUDA IPC allocation size")
        if len({handle for handle, _ in allocations}) != count:
            raise ValueError("duplicate CUDA IPC allocation")
        records: list[NativeRecord | GdnRecord] = []
        spans: list[tuple[int, int, int]] = []
        records.extend(NativeMxfp8.parse_records(request, allocations, 1, key, spans))
        records.extend(NativeMxfp8.parse_records(request, allocations, DRAFT_FAMILY, key, spans))
        gdn = GdnRecord.parse(request, allocations, key, spans)
        if gdn:
            records.append(gdn)
        validate_disjoint(spans)
        retention = Retention.parse(request, key, records, gdn) if store else None
        request.end()
        if not records or len({record.key for record in records}) != len(records):
            raise ValueError("empty or duplicate native GPU records")
        return cls(device, event, allocations, records, retention)


def cuda_check(result, operation):
    status, *values = result
    if int(status) != 0:
        raise RuntimeError(f"{operation}: {status}")
    return values


def build_copies(records, objects, pointers, store):
    copies = []
    for record in records:
        obj = objects[record.key]
        if not isinstance(obj.parent(), LazyMemoryAllocator) or obj.get_size() != record.size:
            raise ValueError("LMCache allocation is not a compatible lazy pinned record")
        for region in record.regions:
            packed = region.width == region.device_pitch == region.record_pitch
            rows = 1 if packed else region.rows
            size = region.width * region.rows if packed else region.width
            for row in range(rows):
                offset = region.record_offset + row * region.record_pitch
                device = (
                    pointers[region.allocation] + region.device_offset + row * region.device_pitch
                )
                host = obj.data_ptr + offset
                dest, src = (host, device) if store else (device, host)
                copies.append(device_ops.StagingCopy(dest, src, size, obj.meta.address + offset))
    return copies


def pin_rectangles(virtual, width, rows, pitch, chunk):
    row = 0
    while row < rows:
        position = virtual + row * pitch
        room = chunk - position % chunk
        if room >= width:
            count = min(rows - row, 1 + (room - width) // pitch)
            yield row, 0, width, count
            row += count
            continue
        column = 0
        while column < width:
            size = min(width - column, chunk - (position + column) % chunk)
            yield row, column, size, 1
            column += size
        row += 1


def make_copy(src, dst, width, rows):
    op = runtime.cudaMemcpy3DBatchOp()
    op.srcAccessOrder = runtime.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    for operand, (address, pitch) in ((op.src, src), (op.dst, dst)):
        operand.type = runtime.cudaMemcpy3DOperandType.cudaMemcpyOperandTypePointer
        operand.op.ptr.ptr = address
        operand.op.ptr.rowLength = pitch if rows > 1 else width
        operand.op.ptr.layerHeight = rows
    op.extent.width = width
    op.extent.height = rows
    op.extent.depth = 1
    return op


def build_rectangles(records, objects, pointers, store):
    copies = []
    for record in records:
        obj = objects[record.key]
        if not isinstance(obj.parent(), LazyMemoryAllocator) or obj.get_size() != record.size:
            raise ValueError("LMCache allocation is not a compatible lazy pinned record")
        data, address = obj.data_ptr, obj.meta.address
        for region in record.regions:
            width, rows = region.width, region.rows
            host_pitch, device_pitch = region.record_pitch, region.device_pitch
            if width == host_pitch == device_pitch:
                width, rows = width * rows, 1
            rects = pin_rectangles(address + region.record_offset, width, rows, host_pitch,
                                   LazyMemoryAllocator.PIN_CHUNK_SIZE)
            for row, col, size, height in rects:
                host = (data + region.record_offset + row * host_pitch + col, host_pitch)
                device = (pointers[region.allocation] + region.device_offset +
                          row * device_pitch + col, device_pitch)
                src, dst = (device, host) if store else (host, device)
                copies.append(make_copy(src, dst, size, height))
    return copies


@dataclass
class MappedAllocation:
    address: int
    span: int
    users: int = 0


class DeviceCopies:
    def __init__(self, *, single_device: bool = False) -> None:
        count = torch.cuda.device_count()
        if count == 0 or (single_device and count != 1):
            required = "exactly one" if single_device else "at least one"
            raise RuntimeError(f"sglang LMCache requires {required} visible CUDA device")
        torch.cuda.init()
        device_ops.BatchStep([], [])
        self.batched = device_ops.batch_memcpy_supported()
        self.devices = {}
        self.mapping_lock = Lock()
        self.mappings: dict[tuple[int, bytes], MappedAllocation] = {}
        for ordinal in range(count):
            (uuid,) = cuda_check(driver.cuDeviceGetUuid(ordinal), "read daemon CUDA UUID")
            self.devices[bytes(uuid.bytes)] = torch.device("cuda", ordinal)
        if len(self.devices) != count:
            raise RuntimeError("sglang LMCache requires distinct visible CUDA device UUIDs")
        for device in self.devices.values():
            with torch.cuda.device(device):
                torch.cuda.Stream(device=device)

    def device_for(self, uuid: bytes) -> torch.device:
        device = self.devices.get(uuid)
        if device is None:
            raise ValueError("GPU UUID is not visible to the daemon")
        return device

    def run(self, transfer, records, objects, store):
        device = self.device_for(transfer.device)
        backend = get_event_ipc_backend(device)
        with torch.cuda.device(device), ExitStack() as cleanup:
            pointers = self.map_allocations(cleanup, transfer.allocations, device.index)
            event = backend.import_event(transfer.event, device)
            stream = torch.cuda.Stream(device=device)
            batched = self.batched
            build = build_rectangles if batched else build_copies
            copies = build(records, objects, pointers, store)
            direction = TransferDirection.D2H if store else TransferDirection.H2D
            try:
                backend.wait_event(event, stream)
                with torch.cuda.stream(stream):
                    if batched:
                        cuda_check(runtime.cudaMemcpy3DBatchAsync(
                            len(copies), copies, 0, stream.cuda_stream), "submit batched CUDA copy")
                    else:
                        device_ops.execute_object_group_transfer(
                            direction,
                            device,
                            LazyMemoryAllocator.PIN_CHUNK_SIZE,
                            [],
                            [device_ops.BatchStep(copies, [])],
                        )
            finally:
                try:
                    stream.synchronize()
                except Exception:
                    LOG.exception(
                        "Unable to drain daemon CUDA transfer; stopping before lease release")
                    os._exit(1)

    def map_allocations(
        self, cleanup: ExitStack, allocations: list[tuple[bytes, int]], ordinal: int
    ) -> list[int]:
        pointers = []
        for handle, size in allocations:
            key = ordinal, handle
            mapping = self.retain_mapping(key, size)
            cleanup.callback(self.release_mapping, key)
            pointers.append(mapping.address)
        validate_disjoint(
            [
                (0, address, address + size)
                for address, (_, size) in zip(pointers, allocations, strict=True)
            ]
        )
        return pointers

    def retain_mapping(self, key: tuple[int, bytes], size: int) -> MappedAllocation:
        with self.mapping_lock:
            mapping = self.mappings.get(key)
            if mapping is None:
                mapping = self.import_allocation(key, size)
                self.mappings[key] = mapping
            if size > mapping.span:
                raise ValueError("CUDA IPC view exceeds mapped allocation")
            mapping.users += 1
            return mapping

    def release_mapping(self, key: tuple[int, bytes]) -> None:
        with self.mapping_lock:
            mapping = self.mappings[key]
            mapping.users -= 1
            if mapping.users == 0:
                self.unmap_allocation(mapping.address)
                del self.mappings[key]

    def import_allocation(self, key: tuple[int, bytes], size: int) -> MappedAllocation:
        ordinal, handle = key
        (address,) = cuda_check(
            runtime.cudaIpcOpenMemHandle(
                cuda_ipc_handle_from_bytes(handle), runtime.cudaIpcMemLazyEnablePeerAccess
            ),
            "import CUDA IPC allocation",
        )
        try:
            return self.validate_mapping(address, ordinal, size)
        except BaseException:
            self.unmap_allocation(address)
            raise

    @staticmethod
    def validate_mapping(address: int, ordinal: int, size: int) -> MappedAllocation:
        (attributes,) = cuda_check(
            runtime.cudaPointerGetAttributes(address), "inspect CUDA IPC allocation device"
        )
        if attributes.device != ordinal:
            raise ValueError("CUDA IPC allocation does not belong to the requested GPU UUID")
        base, span = cuda_check(driver.cuMemGetAddressRange(address), "validate CUDA IPC span")
        if int(base) != address or span < size:
            raise ValueError("CUDA IPC view exceeds mapped allocation")
        return MappedAllocation(address, span)

    @staticmethod
    def unmap_allocation(address: int) -> None:
        try:
            cuda_check(runtime.cudaIpcCloseMemHandle(address), "unmap CUDA IPC allocation")
        except Exception:
            LOG.exception("Unable to unmap CUDA IPC allocation; stopping before lease release")
            os._exit(1)
