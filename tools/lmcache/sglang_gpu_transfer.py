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
MAX_REGIONS = 1 << 20
MAX_ALLOCATION = 1 << 40
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

    @classmethod
    def parse(cls, request):
        return cls(request.number(4), *(request.number(8) for _ in range(5)), request.number(4))

    def validate(self, allocations, size):
        if not (self.allocation < len(allocations) and 0 < self.rows <= MAX_REGIONS and
                0 < self.width <= min(self.device_pitch, self.record_pitch)):
            raise ValueError("invalid GPU region")
        if self.device_offset + (self.rows - 1) * self.device_pitch + self.width > \
                allocations[self.allocation][1]:
            raise ValueError("GPU region exceeds allocation")
        if self.record_offset + (self.rows - 1) * self.record_pitch + self.width > size:
            raise ValueError("region exceeds record")


@dataclass(frozen=True)
class Record:
    key: object
    size: int
    tokens: int
    regions: list

    @classmethod
    def parse(cls, request, namespace, allocations):
        key = request.key(namespace)
        size = request.number(8)
        tokens = request.number(4)
        count = request.number(4)
        if not 0 < size <= MAX_RECORD or not 0 < count <= min(MAX_REGIONS, request.remaining // 48):
            raise ValueError("invalid GPU record")
        regions = [Region.parse(request) for _ in range(count)]
        if sum(region.rows for region in regions) > MAX_REGIONS:
            raise ValueError("too many GPU region rows")
        for region in regions:
            region.validate(allocations, size)
        intervals = sorted((region.record_offset + row * region.record_pitch, region.width)
                           for region in regions for row in range(region.rows))
        end = 0
        for start, width in intervals:
            if start != end:
                raise ValueError("overlapping or incomplete GPU record")
            end += width
        if end != size:
            raise ValueError("incomplete GPU record")
        return cls(key, size, tokens, regions)


@dataclass(frozen=True)
class Transfer:
    device: bytes
    event: bytes
    allocations: list
    records: list

    @classmethod
    def parse(cls, request, namespace):
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
        count = request.number(4)
        if not 0 < count <= request.remaining // 68:
            raise ValueError("invalid GPU record count")
        records = [Record.parse(request, namespace, allocations) for _ in range(count)]
        request.end()
        if len({record.key for record in records}) != count:
            raise ValueError("duplicate GPU record")
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


class DeviceCopies:
    def __init__(self):
        if torch.cuda.device_count() != 1:
            raise RuntimeError("sglang LMCache requires exactly one visible CUDA device")
        torch.cuda.init()
        device_ops.BatchStep([], [])
        self.device = torch.device("cuda:0")
        (uuid,) = cuda_check(driver.cuDeviceGetUuid(0), "read daemon CUDA UUID")
        self.uuid = bytes(uuid.bytes)
        self.backend = get_event_ipc_backend(self.device)

    def run(self, transfer, records, objects, store):
        if transfer.device != self.uuid:
            raise ValueError("GPU UUID differs from daemon device")
        with torch.cuda.device(self.device), ExitStack() as cleanup:
            event = self.backend.import_event(transfer.event, self.device)
            stream = torch.cuda.Stream(device=self.device)
            pointers = self.map_allocations(cleanup, transfer.allocations)
            copies = build_copies(records, objects, pointers, store)
            direction = TransferDirection.D2H if store else TransferDirection.H2D
            try:
                self.backend.wait_event(event, stream)
                with torch.cuda.stream(stream):
                    device_ops.execute_object_group_transfer(
                        direction, self.device, LazyMemoryAllocator.PIN_CHUNK_SIZE,
                        [], [device_ops.BatchStep(copies, [])])
            finally:
                try:
                    stream.synchronize()
                except Exception:
                    LOG.exception(
                        "Unable to drain daemon CUDA transfer; stopping before lease release")
                    os._exit(1)

    def map_allocations(self, cleanup, allocations):
        pointers = []
        for handle, size in allocations:
            (address,) = cuda_check(runtime.cudaIpcOpenMemHandle(
                cuda_ipc_handle_from_bytes(handle), runtime.cudaIpcMemLazyEnablePeerAccess),
                "import CUDA IPC allocation")
            cleanup.callback(self.unmap_allocation, address)
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
