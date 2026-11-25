"""Run the real Unix daemon handler and native rank client with CPU-backed record inventories."""

from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread

from lmcache_relay import PREFIX_DETAIL, RANK_GET, RANK_PUT, RECEIPT, Relay

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/lmcache"))
from sglang_lmcache import GdnEndpoints, SocketServer
from sglang_gpu_transfer import NativeMxfp8


class HostRecords:
    """Keep the parsed record descriptors in place of a CUDA-backed DRAM L1."""

    def __init__(self, world, rank):
        self.world = world
        self.rank = rank
        self.records = {}
        self.gdn_endpoints = GdnEndpoints()

    @contextmanager
    def read(self, keys):
        yield {key: self.records[key] for key in keys if key in self.records}

    def receipt(self, transfer):
        assert transfer.device[0] == self.rank + 1
        tally = [0, 0, 0]
        for record in transfer.records:
            family = record.key.object_group_id
            tally[family - 1] += record.size
            if family == 2:
                widths = [region.width for region in record.regions]
                assert widths == [2949120 // self.world, 150994944 // self.world, 10240, 10240]
            else:
                heads = 4 // self.world if family == 1 else 4
                assert family == 1 or self.rank == 0
                assert record.view.kv_heads == heads
                assert record.size == record.tokens * heads * 528 * (16 if family == 1 else 1)
        tokens = sum(r.tokens for r in transfer.records if r.key.object_group_id == 1)
        return 0, RECEIPT.pack(*tally, tokens)

    def store(self, request, transfer):
        response = self.receipt(transfer)
        for record in transfer.records:
            self.records[record.key] = record
        self.gdn_endpoints.on_l1_keys_write_finished([r.key for r in transfer.records])
        return response

    def retrieve(self, request, transfer):
        response = self.receipt(transfer)
        if any(self.records.get(r.key) != r for r in transfer.records):
            return 1, b""
        return response


def check_scales():
    for heads in (2, 4):
        for count in (1, 31, 32, 33, 65, 96, 127, 128, 129, 225, 256):
            view = NativeMxfp8(layers=2, pages=4, planes=tuple((i, 0) for i in range(4)),
                               kv_heads=heads)
            for begin in (0, 256):
                regions = [r for r in view.regions(begin, count) if r.allocation == 2]
                actual, packed = set(), set()
                for r in regions:
                    for row in range(r.rows):
                        src = r.device_offset + row * r.device_pitch
                        dst = r.record_offset + row * r.record_pitch
                        actual.update(range(src, src + r.width))
                        packed.update(range(dst, dst + r.width))
                expected = set()
                for layer in range(2):
                    for token in range(begin, begin + count):
                        page, slot = divmod(token, 128)
                        for head in range(heads):
                            lane = ((layer * 4 + page) * heads + head) * 32 + slot % 32
                            start = (lane * 4 + slot // 32) * 8
                            expected.update(range(start, start + 8))
                assert actual == expected, (heads, count, begin)
                offset = 2 * 2 * count * heads * 256
                assert packed == set(range(offset, offset + 2 * count * heads * 8))
                assert sum(r.rows * r.width for r in regions) == len(expected)


def exercise(binary, runtime, world):
    def call(op, end=225, rank=0, draft=True, swap=False):
        args = [binary, op, world, rank, end, int(draft), int(swap)]
        result = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=8,
                                env={**os.environ, "XDG_RUNTIME_DIR": str(runtime)})
        assert result.returncode == 0, (args, result.stderr)
        return result.stdout.strip()

    stores = []
    relays = []
    with ExitStack() as stack:
        for rank in range(world):
            suffix = f"-r{rank}" if world > 1 else ""
            name = f"sglang{suffix}"
            records = HostRecords(world, rank)
            stores.append(records)
            endpoint = runtime / f"sglang-lmcache{suffix}.sock"
            upstream = runtime / f"daemon{suffix}.sock"
            server = SocketServer(str(upstream), name, records)
            thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
            thread.start()
            stack.callback(thread.join)
            stack.callback(server.close)
            relays.append(stack.enter_context(Relay(endpoint, upstream)))
        call("ready")
        assert call("lookup") == "0 0 0 0 0"
        call("store", 65)
        if world == 1:
            assert call("lookup") == "65 65 65 65 64"
            assert call("get", 65) == f"{65 * 33792} 153964544 {64 * 2112} 65"
            return
        assert call("lookup") == "0 0 0 0 0"
        assert call("get", 65) == "miss"
        call("store", 161, rank=1)
        assert call("lookup") == "65 65 65 0 0"
        call("store", 65, rank=1)
        call("store", 193, draft=False)
        call("store", 225, rank=1)
        assert call("lookup") == "193 193 193 65 64"
        assert call("lookup", swap=True) == "0 0 0 0 0"
        assert call("get", 65) == f"{65 * 33792} {2 * 76992512} {64 * 2112} 65"
        call("store", 193, rank=1)
        assert call("lookup", draft=False) == "193 193 193 193 0"
        assert call("lookup") == "193 193 193 65 64"
        call("store", 193)
        assert call("lookup") == "193 193 193 193 192"
        assert call("lookup", 160) == "193 193 193 65 64"
        key, = [k for k in stores[1].records if k.extra_attributes.get("sglang_gdn_end") == 193]
        del stores[1].records[key]
        stores[1].gdn_endpoints.on_l1_keys_deleted_by_manager([key])
        assert call("get", 193) == "miss"
        assert call("lookup") == "193 193 193 65 64"
        for rank in range(world):
            call("store", 1, rank=rank, draft=False)
        assert call("lookup", 1) == "193 193 193 1 0"
        assert call("get", 1) == f"33792 {2 * 76992512} 0 1"
        assert call("lookup", 0) == "193 193 193 0 0"
        for rank, relay in enumerate(relays):
            assert not relay.errors
            trace = relay.since()
            assert {PREFIX_DETAIL, RANK_GET, RANK_PUT} <= {event["opcode"] for event in trace}
            assert all(event["namespace"] == f"sglang-r{rank}" for event in trace)
            if rank == 1:
                assert not any(family == 3 for event in trace for family, _ in event["keys"])


if __name__ == "__main__":
    check_scales()
    with TemporaryDirectory(prefix="sglang-ranks-") as work:
        for world_size in (1, 2):
            exercise(sys.argv[1], Path(work), world_size)
    print("TP1 and TP2 daemon framing, rank-local geometry, common endpoints and joint GET passed")
