"""Run the real Unix daemon handler and native rank client with CPU-backed record inventories."""

import os
import subprocess
import sys
import unittest
from contextlib import ExitStack, contextmanager
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from lmcache_relay import DRAFT, PREFIX_DETAIL, RANK_GET, RANK_PUT, RECEIPT, Relay

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/lmcache"))
gpu_transfer = import_module("sglang_gpu_transfer")
NativeMxfp8 = gpu_transfer.NativeMxfp8
lmcache = import_module("sglang_lmcache")


class DeviceStartupTest(unittest.TestCase):
    uuids = (bytes([2]) + bytes(15), bytes([1]) + bytes(15))

    def setUp(self) -> None:
        patches = ExitStack()
        self.addCleanup(patches.close)
        self.cuda = patches.enter_context(patch.object(gpu_transfer.torch, "cuda", MagicMock()))
        self.cuda.device_count.return_value = len(self.uuids)
        self.uuid = patches.enter_context(patch.object(gpu_transfer.driver, "cuDeviceGetUuid"))
        self.uuid.side_effect = lambda ordinal: (0, SimpleNamespace(bytes=self.uuids[ordinal]))
        ops = Mock(spec=["BatchStep", "batch_memcpy_supported"])
        patches.enter_context(patch.object(gpu_transfer, "device_ops", ops))

    def test_visible_order_and_unknown_uuid(self) -> None:
        copies = gpu_transfer.DeviceCopies()
        devices = [gpu_transfer.torch.device("cuda", ordinal) for ordinal in range(len(self.uuids))]
        assert self.uuid.call_args_list == [call(ordinal) for ordinal in range(len(self.uuids))]
        expected = [call.device_count(), call.init()]
        for device in devices:
            expected.extend(
                [
                    call.device(device),
                    call.device().__enter__(),
                    call.Stream(device=device),
                    call.device().__exit__(None, None, None),
                ]
            )
        assert self.cuda.mock_calls == expected
        assert [copies.device_for(uuid) for uuid in self.uuids] == devices
        with self.assertRaisesRegex(ValueError, "GPU UUID is not visible"):
            copies.device_for(bytes(16))
        assert self.cuda.mock_calls == expected

    def test_rank_mode_validates_before_streams(self) -> None:
        for count in (0, len(self.uuids)):
            with self.subTest(count=count):
                self.cuda.device_count.return_value = count
                with self.assertRaisesRegex(RuntimeError, "requires exactly one"):
                    gpu_transfer.DeviceCopies(single_device=True)
        self.cuda.Stream.assert_not_called()
        self.cuda.init.assert_not_called()
        self.uuid.assert_not_called()
        self.cuda.device_count.return_value = 1
        copies = gpu_transfer.DeviceCopies(single_device=True)
        self.cuda.Stream.assert_called_once_with(device=copies.device_for(self.uuids[0]))

    def test_duplicate_uuid_fails_before_streams(self) -> None:
        self.uuid.side_effect = None
        self.uuid.return_value = (0, SimpleNamespace(bytes=self.uuids[0]))
        with self.assertRaisesRegex(RuntimeError, "distinct visible CUDA device UUIDs"):
            gpu_transfer.DeviceCopies()
        self.cuda.Stream.assert_not_called()

    def test_second_device_failure_prevents_listener(self) -> None:
        self.cuda.Stream.side_effect = [Mock(), RuntimeError("second CUDA device failed")]
        manager = Mock(spec=lmcache.DramL1Manager, _registered_listeners=[])
        storage = Mock(l1_managers=Mock(return_value=[manager]), l2_adapters=Mock(return_value=[]))
        with (
            patch.object(
                lmcache, "SocketServer", side_effect=AssertionError("listener opened")
            ) as listener,
            self.assertRaisesRegex(RuntimeError, "second CUDA device failed"),
        ):
            lmcache.SglangSnapshotModule(
                SimpleNamespace(storage_manager=storage), "/unused", "sglang"
            )
        listener.assert_not_called()
        assert self.cuda.Stream.call_count == len(self.uuids)


class HostRecords:
    """Keep the parsed record descriptors in place of a CUDA-backed DRAM L1."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.records = {}
        self.devices = tuple(bytes([rank + 1]) + bytes(15) for rank in range(world))
        self.owners = {}
        self.gdn_endpoints = lmcache.GdnEndpoints()

    @contextmanager
    def read(self, keys):
        yield {key: self.records[key] for key in keys if key in self.records}

    def receipt(self, transfer):
        rank = self.devices.index(bytes(transfer.device))
        tally = [0, 0, 0]
        for record in transfer.records:
            family = record.key.object_group_id
            tally[family - 1] += record.size
            if family == 2:
                widths = [region.width for region in record.regions]
                assert widths == [2949120 // self.world, 150994944 // self.world, 10240, 10240]
            else:
                heads = 4 // self.world if family == 1 else 4
                assert family == 1 or rank == 0
                assert record.view.kv_heads == heads
                assert record.size == record.tokens * heads * 528 * (16 if family == 1 else 1)
        tokens = sum(r.tokens for r in transfer.records if r.key.object_group_id == 1)
        return 0, RECEIPT.pack(*tally, tokens)

    def store(self, request, transfer):
        response = self.receipt(transfer)
        for record in transfer.records:
            self.owners[record.key] = bytes(transfer.device)
            self.records[record.key] = record
        self.gdn_endpoints.on_l1_keys_write_finished([r.key for r in transfer.records])
        return response

    def retrieve(self, request, transfer):
        response = self.receipt(transfer)
        if any(
            self.records.get(r.key) != r or self.owners[r.key] != bytes(transfer.device)
            for r in transfer.records
        ):
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

    records = HostRecords(world)
    endpoint = runtime / "sglang-lmcache.sock"
    upstream = runtime / "daemon.sock"
    with ExitStack() as stack:
        server = lmcache.SocketServer(str(upstream), "sglang", records)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
        thread.start()
        stack.callback(thread.join)
        stack.callback(server.close)
        relay = stack.enter_context(Relay(endpoint, upstream))
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
        assert call("get", 65, swap=True) == "miss"
        assert call("get", 65) == f"{65 * 33792} {2 * 76992512} {64 * 2112} 65"
        call("store", 193, rank=1)
        assert call("lookup", draft=False) == "193 193 193 193 0"
        assert call("lookup") == "193 193 193 65 64"
        terminal = 193
        call("store", terminal)
        assert call("lookup") == "193 193 193 193 192"
        assert call("lookup", 160) == "193 193 193 65 64"
        (key,) = (
            k
            for k, device in records.owners.items()
            if device == records.devices[1] and k.extra_attributes.get("sglang_gdn_end") == terminal
        )
        del records.records[key]
        del records.owners[key]
        records.gdn_endpoints.on_l1_keys_deleted_by_manager([key])
        assert call("get", 193) == "miss"
        assert call("lookup") == "193 193 193 65 64"
        for rank in range(world):
            call("store", 1, rank=rank, draft=False)
        assert call("lookup", 1) == "193 193 193 1 0"
        assert call("get", 1) == f"33792 {2 * 76992512} 0 1"
        assert call("lookup", 0) == "193 193 193 0 0"
        assert not relay.errors
        trace = relay.since()
        assert all(event["namespace"] == "sglang" for event in trace)
        assert sum(event["opcode"] == PREFIX_DETAIL for event in trace) > 1
        for rank, device in enumerate(records.devices):
            transfers = [event for event in trace if event.get("device") == device.hex()]
            assert {RANK_GET, RANK_PUT} <= {event["opcode"] for event in transfers}
            if rank == 1:
                assert not any(
                    family == DRAFT for event in transfers for family, _ in event["keys"]
                )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DeviceStartupTest)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        raise SystemExit(1)
    check_scales()
    for world_size in (1, 2):
        with TemporaryDirectory(prefix="sglang-ranks-") as work:
            exercise(sys.argv[1], Path(work), world_size)
    print("TP1 and TP2 daemon framing, rank-local geometry, common endpoints and joint GET passed")
