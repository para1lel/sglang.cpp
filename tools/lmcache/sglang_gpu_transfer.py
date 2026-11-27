from contextlib import ExitStack
from dataclasses import dataclass
import os

import torch
from cuda.bindings import driver, runtime
from lmcache import device_ops
from lmcache.lmcache_native import TransferDirection
from lmcache.logging import init_logger
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.devices.cuda.utils import cuda_ipc_handle_from_bytes


MAX_RECORD = 1 << 30
MAX_ALLOCATION = 1 << 40
MAX_TOKENS = 1 << 18
PREFIX_CHUNK = 256
DIGEST_BYTES = 64
GDN_END = "sglang_gdn_end"
LOG = init_logger(__name__)


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
    key: object
    size: int
    regions: tuple
    tokens: int = 0

    @classmethod
    def parse(cls, request, allocations, key):
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
            if allocation >= len(allocations) or not width or \
                    offset + width > allocations[allocation][1]:
                raise ValueError("native GDN plane exceeds allocation")
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

    @classmethod
    def parse_records(cls, request, allocations, family, key, legacy=True):
        begin, end = request.number(4), request.number(4)
        if begin == end == 0:
            return []
        kv_heads = 4 if legacy else request.number(4)
        pages = request.number(4)
        planes = tuple((request.number(4), request.number(8)) for _ in range(4))
        layers = 16 if family == 1 else 1
        if not begin < end <= MAX_TOKENS or begin % PREFIX_CHUNK or not end <= pages * 128:
            raise ValueError("invalid native MXFP8 extent")
        if not 0 < pages <= MAX_TOKENS // 128 or not 0 < kv_heads <= 128:
            raise ValueError("invalid native MXFP8 geometry")
        row_bytes = (kv_heads * 256, kv_heads * 256, kv_heads * 8, kv_heads * 8)
        for (allocation, offset), plane_bytes in zip(planes, row_bytes, strict=True):
            if allocation >= len(allocations) or \
                    offset + layers * pages * 128 * plane_bytes > allocations[allocation][1]:
                raise ValueError("native MXFP8 plane exceeds allocation")
        view = cls(layers, pages, planes, kv_heads)
        return [NativeRecord(key(family, request.read(DIGEST_BYTES)),
                             min(PREFIX_CHUNK, end - start), start, view)
                for start in range(begin, end, PREFIX_CHUNK)]

    def regions(self, begin, rows):
        offset = 0
        for index, (allocation, address) in enumerate(self.planes):
            row_bytes = self.kv_heads * 256 if index < 2 else self.kv_heads * 8
            width = rows * row_bytes
            pitch = self.pages * 128 * row_bytes
            origin = address + begin * row_bytes
            full = width if index < 2 else rows // 128 * 128 * self.kv_heads * 8
            if full:
                yield Region(allocation, origin, pitch, offset, width, full, self.layers)
            if full < width:
                for layer in range(self.layers):
                    device = origin + layer * pitch + full
                    record = offset + layer * width + full
                    complete = (rows % 128) // 32 * 8
                    tail = rows % 32
                    if complete:
                        yield Region(allocation, device, 32, record, complete, complete,
                                     self.kv_heads * 32)
                        record += complete * self.kv_heads * 32
                    if tail:
                        for head in range(self.kv_heads):
                            yield Region(allocation, device + head * 1024 + complete,
                                         32, record, 8, 8, tail)
                            record += tail * 8
            offset += width * self.layers


@dataclass(frozen=True)
class NativeRecord:
    key: object
    tokens: int
    begin: int
    view: NativeMxfp8

    @property
    def size(self):
        return self.tokens * self.view.layers * 2 * self.view.kv_heads * (256 + 8)

    @property
    def regions(self):
        return self.view.regions(self.begin, self.tokens)


@dataclass(frozen=True)
class Transfer:
    device: bytes
    event: bytes
    allocations: list
    records: list

    @classmethod
    def parse(cls, request, key, legacy=True):
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
        records = NativeMxfp8.parse_records(request, allocations, 1, key, legacy)
        records += NativeMxfp8.parse_records(request, allocations, 3, key, legacy)
        gdn = GdnRecord.parse(request, allocations, key)
        if gdn:
            records.append(gdn)
        request.end()
        if not records or len({record.key for record in records}) != len(records):
            raise ValueError("empty or duplicate native GPU records")
        return cls(device, event, allocations, records)


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


def independent_destinations(records, pointers):
    spans = sorted((pointers[region.allocation] + region.device_offset + row * region.device_pitch,
                    region.width) for record in records for region in record.regions
                   for row in range(region.rows))
    return all(start + size <= next_start for (start, size), (next_start, _) in
               zip(spans, spans[1:]))


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
            batched = self.batched and (store or independent_destinations(records, pointers))
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
            (address,) = cuda_check(runtime.cudaIpcOpenMemHandle(
                cuda_ipc_handle_from_bytes(handle), runtime.cudaIpcMemLazyEnablePeerAccess),
                "import CUDA IPC allocation")
            cleanup.callback(self.unmap_allocation, address)
            (attributes,) = cuda_check(
                runtime.cudaPointerGetAttributes(address), "inspect CUDA IPC allocation device"
            )
            if attributes.device != ordinal:
                raise ValueError("CUDA IPC allocation does not belong to the requested GPU UUID")
            base, span = cuda_check(driver.cuMemGetAddressRange(address), "validate CUDA IPC span")
            if int(base) != address or span < size:
                raise ValueError("CUDA IPC view exceeds mapped allocation")
            pointers.append(address)
        return pointers

    @staticmethod
    def unmap_allocation(address):
        try:
            cuda_check(runtime.cudaIpcCloseMemHandle(address), "unmap CUDA IPC allocation")
        except Exception:
            LOG.exception("Unable to unmap CUDA IPC allocation; stopping before lease release")
            os._exit(1)
