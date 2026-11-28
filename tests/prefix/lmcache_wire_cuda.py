"""Check native-v5 CUDA IPC transfers against independent byte offsets."""

from __future__ import annotations

import os
import socket
import struct
import sys
from contextlib import ExitStack
from ctypes import addressof, create_string_buffer, string_at
from dataclasses import dataclass, replace
from hashlib import shake_256
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from typing import TYPE_CHECKING

from lmcache_relay import (
    ADMISSION,
    BLOCK_TOKENS,
    DRAFT,
    GDN,
    GQA,
    LEASE,
    PAGE_MAP_FLAG,
    PAGE_TOKENS,
    RANK_GET,
    RANK_PUT,
    READY,
    RECEIPT,
    TIMEOUT,
    WIRE_VERSION,
    Relay,
    control,
    header,
    receive,
    send,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

cuda = import_module("cuda.bindings.runtime")
driver = import_module("cuda.bindings.driver")
torch = import_module("torch")
NAMESPACE = "sglang"
WORLD = 2
SCALE_PLANE = 2
ARMED = 4
REJECTED = 2
SLAB_COUNT = 2
Handles = list[tuple[bytes, int]]
GdnPlanes = tuple[tuple[int, int, int], ...]


def checked(result: tuple) -> list:
    status, *values = result
    if int(status):
        raise RuntimeError(f"CUDA wire fixture failed: {status}")
    return values


@dataclass(frozen=True)
class Cache:
    family: int
    layers: int
    heads: int
    pages: int
    physical: tuple[int, ...]
    begin: int
    end: int
    planes: tuple[tuple[int, int], ...]

    def wire(self, identity: int) -> bytes:
        if not self.end:
            return bytes(8)
        return (
            struct.pack("!4I", self.begin, self.end, self.heads, self.pages | PAGE_MAP_FLAG)
            + b"".join(struct.pack("!IQ", *plane) for plane in self.planes)
            + b"".join(struct.pack("!I", page) for page in self.physical)
            + b"".join(
                digest(identity, self.family, start)
                for start in range(self.begin, self.end, BLOCK_TOKENS)
            )
        )

    def offset(self, plane: int, layer: int, token: int, head: int) -> tuple[int, int, int]:
        page, slot = divmod(token - self.begin, PAGE_TOKENS)
        scale = plane >= SCALE_PLANE
        width = 8 if scale else 256
        owner, base = self.planes[plane]
        base += (layer * self.pages + self.physical[page]) * PAGE_TOKENS * self.heads * width
        if scale:
            return owner, base + head * 1024 + (slot % 32) * 32 + (slot // 32) * 8, width
        return owner, base + (slot * self.heads + head) * width, width

    def copy_reference(
        self, source: Cache, expected: list[bytearray], source_bytes: list[bytes]
    ) -> None:
        for plane in range(4):
            for layer in range(self.layers):
                for token in range(self.begin, self.end):
                    for head in range(self.heads):
                        src, offset, width = source.offset(plane, layer, token, head)
                        dst, where, _ = self.offset(plane, layer, token, head)
                        expected[dst][where : where + width] = source_bytes[src][
                            offset : offset + width
                        ]


def digest(identity: int, family: int, start: int) -> bytes:
    return f"{identity:016x}{family:016x}{start:032x}".encode("ascii")


def layout(
    rank: int, *, source: bool, tail: bool
) -> tuple[tuple[Cache, Cache], GdnPlanes, list[int]]:
    offsets = [41, 109]

    def plane(owner: int, span: int) -> tuple[int, int]:
        base = offsets[owner]
        offsets[owner] += span + 71
        return owner, base

    def view(family: int, layers: int, heads: int, end: int) -> Cache:
        if not end:
            return Cache(family, layers, heads, 0, (), 0, 0, ())
        pages = (2 if source else 1) if tail else (6 if source else 5)
        physical = ((1,) if source else (0,)) if tail else ((4, 1, 5) if source else (0, 1, 3))
        planes = tuple(
            plane(0, layers * pages * PAGE_TOKENS * heads * width) for width in (256, 256, 8, 8)
        )
        return Cache(family, layers, heads, pages, physical, 256 if tail else 0, end, planes)

    target = view(GQA, 16, 4 // WORLD, 321)
    draft = view(DRAFT, 1, 4, 320 if rank == 0 else 0)
    gdn = tuple((1, plane(1, width)[1], width) for width in (19, 103, 67, 31))
    return (target, draft), gdn, offsets


class Slabs:
    def __init__(self, cleanup: ExitStack, sizes: Sequence[int], seed: int) -> None:
        self.contents = [
            shake_256(f"{seed}:{index}".encode()).digest(size) for index, size in enumerate(sizes)
        ]
        self.pointers = []
        self.handles = []
        for size, data in zip(sizes, self.contents, strict=True):
            (pointer,) = checked(cuda.cudaMalloc(size))
            cleanup.callback(lambda ptr=pointer: checked(cuda.cudaFree(ptr)))
            (handle,) = checked(cuda.cudaIpcGetMemHandle(pointer))
            host = create_string_buffer(data)
            checked(
                cuda.cudaMemcpy(
                    pointer, addressof(host), size, cuda.cudaMemcpyKind.cudaMemcpyHostToDevice
                )
            )
            self.pointers.append(pointer)
            self.handles.append((string_at(handle.getPtr(), 64), size))

    def download(self) -> list[bytes]:
        actual = []
        for pointer, (_, size) in zip(self.pointers, self.handles, strict=True):
            host = create_string_buffer(size)
            checked(
                cuda.cudaMemcpy(
                    addressof(host), pointer, size, cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost
                )
            )
            actual.append(host.raw)
        return actual


@dataclass(frozen=True)
class Wire:
    upstream: Path
    pid: int
    uuid: bytes
    event: bytes
    identity: int

    def payload(
        self, handles: Handles, views: tuple[Cache, Cache], gdn: GdnPlanes, *, store: bool
    ) -> bytes:
        return (
            self.uuid
            + self.event
            + struct.pack("!I", len(handles))
            + b"".join(struct.pack("!64sQ", *handle) for handle in handles)
            + b"".join(view.wire(self.identity) for view in views)
            + struct.pack("!I64s", 321, digest(self.identity, GDN, 321))
            + b"".join(struct.pack("!IQQ", *plane) for plane in gdn)
            + (bytes(2) if store else b"")
        )

    def transfer(self, path: Path, opcode: int, body: bytes) -> tuple[int, bytes]:
        with socket.socket(socket.AF_UNIX) as sock:
            sock.settimeout(TIMEOUT)
            sock.connect(str(path))
            assert send(sock, NAMESPACE, opcode, body) == (ARMED, ADMISSION.size)
            admission = receive(sock, ADMISSION.size)
            _, _, pid = ADMISSION.unpack(admission)
            assert pid == self.pid
            assert control(self.upstream, NAMESPACE, LEASE, admission[:32]) == (ARMED, b"")
            sock.sendall(b"\x01")
            _, status, size = header(sock)
            response = receive(sock, size)
            assert control(self.upstream, NAMESPACE, LEASE, admission[:32]) == (0, b"")
            return status, response


def rejected(path: Path, body: bytes, detail: bytes) -> None:
    status, reply = control(path, NAMESPACE, RANK_GET, body)
    assert status == REJECTED, (status, reply)
    assert detail in reply, reply


def invalid_views(wire: Wire, slabs: Slabs, views: tuple[Cache, Cache], gdn: GdnPlanes) -> None:
    target, draft = views
    for physical in ((0, 0, 3), (0, target.pages, 3), (0, 1)):
        body = wire.payload(
            slabs.handles, (replace(target, physical=physical), draft), gdn, store=False
        )
        rejected(wire.upstream, body, b"physical page map")
    overlap = replace(target, planes=(target.planes[0], *target.planes[:3]))
    body = wire.payload(slabs.handles, (overlap, draft), gdn, store=False)
    rejected(wire.upstream, body, b"overlapping")
    (handle, _), aux = slabs.handles
    body = wire.payload([(handle, 1), aux], views, gdn, store=False)
    rejected(wire.upstream, body, b"allocation")
    body = wire.payload([(handle, 1 << 30), aux], views, gdn, store=False)
    status, detail = wire.transfer(wire.upstream, RANK_GET, body)
    assert status == REJECTED, (status, detail)
    assert b"exceeds mapped allocation" in detail, detail


def roundtrip(relay: Relay, uuid: bytes, rank: int, pid: int, *, tail: bool) -> None:
    identity = 100 + rank + 10 * tail
    source, src_gdn, source_sizes = layout(rank, source=True, tail=tail)
    target, dst_gdn, target_sizes = layout(rank, source=False, tail=tail)
    with torch.cuda.device(rank), ExitStack() as cleanup:
        src = Slabs(cleanup, source_sizes, identity)
        dst = Slabs(cleanup, target_sizes, identity + 100)
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream())
        wire = Wire(relay.upstream, pid, uuid, event.ipc_handle(), identity)
        counts = [(v.end - v.begin) * v.heads * 528 * v.layers for v in target]
        gdn_bytes = sum(width for _, _, width in dst_gdn)
        expected = [bytearray(data) for data in dst.contents]
        for view, old in zip(target, source, strict=True):
            view.copy_reference(old, expected, src.contents)
        for (src_owner, src_base, width), (dst_owner, dst_base, _) in zip(
            src_gdn, dst_gdn, strict=True
        ):
            expected[dst_owner][dst_base : dst_base + width] = src.contents[src_owner][
                src_base : src_base + width
            ]
        for opcode, slabs, views, gdn in (
            (RANK_PUT, src, source, src_gdn),
            (RANK_GET, dst, target, dst_gdn),
        ):
            store = opcode == RANK_PUT
            body = wire.payload(slabs.handles, views, gdn, store=store)
            rows = target[0].end - target[0].begin if store else 0
            receipt = (counts[0], gdn_bytes, counts[1], rows)
            status, reply = wire.transfer(Path(relay.server_address), opcode, body)
            assert status == 0, (status, reply)
            assert RECEIPT.unpack(reply) == receipt
            entry = relay.since()[-1]
            assert entry["opcode"] == opcode
            assert entry["device"] == uuid.hex()
            assert entry["status"] == 0
            assert entry["receipt"] == receipt
            assert entry["allocations"] == SLAB_COUNT
            assert entry["page_maps"] == {v.family: v.physical for v in views if v.end}
            assert entry["armed_ns"] <= entry["reply_ns"]
        assert src.download() == src.contents, (rank, tail, "store changed source bytes")
        if not tail:
            invalid_views(wire, dst, target, dst_gdn)
        assert dst.download() == expected, (rank, tail, "restored bytes or untouched guard differ")
    print(
        f"rank={rank} tail={tail} GPU={uuid.hex()} "
        f"GQA={counts[0]} GDN={gdn_bytes} draft={counts[1]}"
    )


def main() -> None:
    upstream = Path(sys.argv[1])
    pid = int(sys.argv[2])
    assert torch.cuda.device_count() == WORLD
    torch.cuda.init()
    uuids = [bytes(checked(driver.cuDeviceGetUuid(rank))[0].bytes) for rank in range(WORLD)]
    deadline = monotonic() + TIMEOUT
    while not upstream.exists() and monotonic() < deadline:
        sleep(0.05)
    assert control(upstream, NAMESPACE, READY, b"".join(uuids)) == (0, WIRE_VERSION)
    assert os.getpid() != pid
    with TemporaryDirectory(prefix="sglang-wire-") as work:
        path = Path(work) / "relay.sock"
        with Relay(path, upstream) as relay:
            for rank, uuid in enumerate(uuids):
                for tail in (False, True):
                    roundtrip(relay, uuid, rank, pid, tail=tail)
            assert not relay.errors, relay.errors
    print("native-v5 CUDA offset, bounds and acknowledgement checks passed")


if __name__ == "__main__":
    main()
