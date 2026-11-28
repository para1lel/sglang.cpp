"""Run the real Unix daemon handler and native rank client with CPU-backed record inventories."""

import os
import struct
import subprocess
import sys
import unittest
from collections.abc import Callable, Generator, Iterable
from contextlib import ExitStack, contextmanager
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from lmcache_relay import (
    DRAFT,
    GDN,
    GQA,
    LEASE,
    PAGE_TOKENS,
    PREFIX_DETAIL,
    RANK_GET,
    RANK_PUT,
    READY,
    RECEIPT,
    WIRE_VERSION,
    Relay,
    control,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/lmcache"))
gpu_transfer = import_module("sglang_gpu_transfer")
NativeMxfp8 = gpu_transfer.NativeMxfp8
lmcache = import_module("sglang_lmcache")
ObjectKey = lmcache.ObjectKey
Request = lmcache.Request
Transfer = gpu_transfer.Transfer
Record = gpu_transfer.NativeRecord | gpu_transfer.GdnRecord

RankCall = Callable[..., str]
REJECTED = 2
RANK0_ANCHOR, RANK_SHARED_ANCHOR = 65, 193
SHARED_DRAFT_ROWS = RANK0_ANCHOR - 1
DRAFT_ROW_BYTES = 2112
PLANE_OFFSET = 128
ACTIVE_LEASE = 4
CLIENT_ERROR = 1


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

    def test_ready_checks_visible_uuids(self) -> None:
        copies = gpu_transfer.DeviceCopies()
        with TemporaryDirectory(prefix="sglang-ready-") as work, ExitStack() as stack:
            endpoint = Path(work) / "ready.sock"
            server = lmcache.SocketServer(str(endpoint), "sglang", HostRecords(2), copies)
            worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
            worker.start()
            stack.callback(worker.join)
            stack.callback(server.close)
            success = (0, WIRE_VERSION)
            first, second = self.uuids
            for payload in (b"", first, second, first + second, second + first):
                with self.subTest(payload=payload.hex()):
                    self.assertEqual(control(endpoint, "sglang", READY, payload), success)
            self.assertEqual(control(endpoint, "sglang", READY, second + bytes(16)), (3, b""))
            self.assertEqual(control(endpoint, "sglang", READY, first + b"\x00")[0], 2)

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
        manager = Mock(
            spec=lmcache.DramL1Manager, _registered_listeners=[], _objects={}, _lock=ExitStack()
        )
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

    lookup = lmcache.Records.lookup

    def __init__(self, world: int) -> None:
        self.world = world
        self.records = {}
        self.devices = tuple(bytes([rank + 1]) + bytes(15) for rank in range(world))
        self.owners = {}
        self.gdn_endpoints = lmcache.GdnEndpoints()
        self.legacy = False
        self.hold_retrieve = None
        self.release_retrieve = Event()

    @contextmanager
    def read(self, keys: Iterable[ObjectKey]) -> Generator[dict[ObjectKey, Record], None, None]:
        yield {key: self.records[key] for key in keys if key in self.records}

    def receipt(self, transfer: Transfer, *, store: bool) -> tuple[int, bytes]:
        tally = [0, 0, 0]
        for record in transfer.records:
            family = record.key.object_group_id
            tally[family - 1] += record.size
            if family == GDN:
                widths = [region.width for region in record.regions]
                assert widths == [2949120 // self.world, 150994944 // self.world, 10240, 10240]
            else:
                heads = 4 if self.legacy and family == DRAFT else 4 // self.world
                layers = 16 if family == GQA else 1
                assert record.view.kv_heads == heads
                assert record.size == record.tokens * heads * 528 * layers
                assert len({allocation for allocation, _ in record.view.planes}) == 1
                assert record.view.planes[0][1] == PLANE_OFFSET
        tokens = sum(r.tokens for r in transfer.records if store and r.key.object_group_id == GQA)
        return 0, RECEIPT.pack(*tally, tokens)

    def store(self, _request: Request, transfer: Transfer) -> tuple[int, bytes]:
        response = self.receipt(transfer, store=True)
        for record in transfer.records:
            self.owners[record.key] = bytes(transfer.device)
            self.records[record.key] = record
        self.gdn_endpoints.on_l1_keys_write_finished([r.key for r in transfer.records])
        return response

    def retrieve(self, _request: Request, transfer: Transfer) -> tuple[int, bytes]:
        if bytes(transfer.device) == self.hold_retrieve and not self.release_retrieve.wait(5):
            raise TimeoutError("ranked fixture GET was held too long")
        response = self.receipt(transfer, store=False)
        if any(
            r.key not in self.records
            or self.records[r.key].size != r.size
            or self.owners[r.key] != bytes(transfer.device)
            for r in transfer.records
        ):
            return 1, b""
        return response


def check_scales() -> None:
    key_scale_plane = 2
    for heads in (2, 4):
        for count in (1, 31, 32, 33, 65, 96, 127, 128, 129, 225, 256):
            view = NativeMxfp8(
                layers=2, pages=4, planes=tuple((i, 0) for i in range(4)), kv_heads=heads
            )
            for begin in (0, 256):
                regions = [r for r in view.regions(begin, count) if r.allocation == key_scale_plane]
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


def reject_old_put(path: Path, uuid: bytes) -> None:
    payload = (
        uuid
        + bytes(64)
        + struct.pack("!I64sQ", 1, bytes(64), 4)
        + bytes(16)
        + struct.pack("!I64s", 65, b"a" * 64)
        + b"".join(struct.pack("!IQQ", 0, offset, 1) for offset in range(4))
    )
    status, reply = control(path, "sglang", RANK_PUT, payload)
    assert status == REJECTED, (status, reply)
    assert reply


def cache_digest(world: int, rank: int, rows: int, *, legacy: bool = False) -> bytes:
    base = "sglang/qwen3_5/split-prefix-mxfp8-"
    if world == 1:
        domain = f"{base}v2/"
    elif legacy:
        family = "draft" if rank == 0 else "target"
        domain = f"{base}v3/tp2-r{rank}/kv2/gdn8x24/{family}/"
    else:
        domain = f"{base}v4/tp2-r{rank}/kv2/gdn8x24/draft2/"
    digest = sha256(f"{domain}cpu-rank-test".encode()).digest()
    for row in range(rows):
        digest = sha256(digest + struct.pack("<5i", 42, row, row, row, -1)).digest()
    return digest.hex().encode()


def check_rank_draft_keys(
    events: list[dict], records: HostRecords, *, legacy: bool = False
) -> None:
    for rank, device in enumerate(records.devices):
        (event,) = [entry for entry in events if entry["device"] == device.hex()]
        expected = (
            []
            if legacy and rank
            else [(DRAFT, cache_digest(records.world, rank, RANK0_ANCHOR, legacy=legacy))]
        )
        assert [(family, key) for family, key in event["keys"] if family == DRAFT] == expected


def check_draft_only(call: RankCall, relay: Relay, records: HostRecords) -> None:
    offset = len(relay.since())
    call("store-draft", RANK0_ANCHOR)
    if records.world > 1:
        assert call("draft-coverage") == "0 0 0"
        assert call("get-draft", RANK0_ANCHOR) == "miss"
        before = len(relay.since())
        call("get-no-peer", RANK0_ANCHOR)
        assert len(relay.since()) == before
    size = SHARED_DRAFT_ROWS * DRAFT_ROW_BYTES
    assert call("store-joint-draft", RANK0_ANCHOR) == f"0 0 {size} 0"
    assert call("draft-coverage") == f"0 0 {SHARED_DRAFT_ROWS}"
    assert call("draft-coverage", draft=False) == "0 0 0"
    assert call("get-draft", RANK0_ANCHOR) == f"0 0 {size} 0"
    trace = relay.since(offset)
    puts = [event for event in trace if event["opcode"] == RANK_PUT]
    gets = [event for event in trace if event["opcode"] == RANK_GET]
    check_rank_draft_keys(puts[-records.world :], records)
    check_rank_draft_keys(gets[-records.world :], records)
    for event in gets[-records.world :]:
        assert event["receipt"] == (0, 0, size // records.world, 0)
    for event in trace:
        if event["opcode"] == PREFIX_DETAIL:
            rows = event["coverage"]["draft"]
            assert rows in (0, SHARED_DRAFT_ROWS)
            if not event["coverage_query"]["draft"]:
                assert rows == 0


def check_draft_tail(call: RankCall, relay: Relay, records: HostRecords) -> None:
    tails = [
        key
        for key, record in records.records.items()
        if key.object_group_id == DRAFT and record.tokens == SHARED_DRAFT_ROWS
    ]
    assert len(tails) == records.world
    for tail in tails:
        del records.records[tail]
        del records.owners[tail]
    draft_rows = 2 * PAGE_TOKENS
    prompt_end = draft_rows + 1
    for rank in range(records.world):
        call("store-draft", prompt_end, rank=rank)
    for stored in range(records.world + 1):
        if stored:
            call("store-draft", RANK0_ANCHOR, rank=stored - 1)
        offset = len(relay.since())
        assert call("draft-coverage", prompt_end).split()[-1] == str(draft_rows)
        inventory = call("draft-inventory", prompt_end).split()
        assert (str(RANK0_ANCHOR) in inventory) == (stored == records.world)
        queries = [event for event in relay.since(offset) if event["opcode"] == PREFIX_DETAIL]
        assert len(queries) == 2 * records.world
        for rank, event in enumerate(queries[: records.world]):
            assert event["coverage"]["draft"] == draft_rows
            assert dict(event["coverage"]["checkpoints"])[RANK0_ANCHOR] == (rank < stored)


def check_tp2_inventory(call: RankCall, relay: Relay, records: HostRecords) -> None:
    before_lookup = len(relay.since())
    assert call("lookup", 160) == "193 211 193 65 64 65 193 211"
    inventory = [event for event in relay.since(before_lookup) if event["opcode"] == PREFIX_DETAIL]
    assert [event["coverage_query"]["count"] for event in inventory] == [257, 257]
    assert [event["coverage_query"]["draft"] for event in inventory] == [1, 1]
    assert [event["coverage"] for event in inventory] == [
        {"gqa": 193, "draft": 192, "checkpoints": [(211, 0), (193, 1), (65, 1)]},
        {
            "gqa": 225,
            "draft": 224,
            "checkpoints": [(225, 1), (211, 0), (193, 1), (161, 1), (65, 1)],
        },
    ]
    assert call("draft-inventory", 160) == "65 193"
    (key,) = (
        k
        for k, device in records.owners.items()
        if device == records.devices[1]
        and k.extra_attributes.get("sglang_gdn_end") == RANK_SHARED_ANCHOR
    )
    del records.records[key]
    del records.owners[key]
    records.gdn_endpoints.on_l1_keys_deleted_by_manager([key])
    assert call("get", 193) == "miss"
    assert call("lookup") == "193 211 193 65 64 65 211"
    assert call("draft-inventory") == "65"


def check_tp2(call: RankCall, relay: Relay, records: HostRecords, gdn_bytes: int) -> None:
    assert call("get", 65) == "miss"
    call("store", 161, rank=1)
    assert call("lookup") == "0 0 0 0 0"
    call("store", 65, rank=1)
    call("store", 193, draft=False)
    call("store", 225, rank=1)
    assert call("lookup") == "65 65 65 65 64 65"
    assert call("lookup", swap=True) == "0 0 0 0 0"
    assert call("get", 65, swap=True) == "miss"
    assert call("get", 65) == f"{65 * 33792} {gdn_bytes} {64 * 2112} 0"
    call("store", 193, rank=1)
    assert call("lookup", draft=False) == "193 193 193 193 0 65 193"
    assert call("lookup") == "193 193 193 65 64 65 193"
    assert call("draft-inventory") == "65"
    call("store", 193)
    assert call("lookup") == "193 193 193 193 192 65 193"
    call("store-gdn", 211)
    assert call("lookup") == "193 193 193 193 192 65 193"
    call("store-gdn", 211, rank=1)
    check_tp2_inventory(call, relay, records)
    for rank in range(records.world):
        call("store", 1, rank=rank, draft=False)
    assert call("lookup", 1) == "193 211 193 1 0 1 65 211"
    assert call("get", 1) == f"33792 {gdn_bytes} 0 0"
    assert call("lookup", 0) == "193 211 193 0 0 1 65 211"


def check_retention(relay: Relay, call: RankCall, records: HostRecords) -> None:
    for rank in range(records.world):
        call("store", 193, rank=rank)
    prior_puts = [event for event in relay.since() if event["opcode"] == RANK_PUT]
    assert prior_puts
    assert all(
        event["prior_gdn"] == {"draft_required": False, "anchors": []} for event in prior_puts
    )
    offset = len(relay.since())
    for rank in range(records.world):
        call("store-retaining", 225, rank=rank)
    hinted = relay.since(offset)
    assert len(hinted) == records.world
    for rank, event in enumerate(hinted):
        device = records.devices[rank].hex()
        assert event["opcode"] == RANK_PUT
        assert event["device"] == device
        expected = []
        for end in (RANK0_ANCHOR, RANK_SHARED_ANCHOR):
            (key,) = {
                key
                for put in prior_puts
                if put["device"] == device
                for (family, key), record in zip(put["keys"], put["records"], strict=True)
                if family == GDN and record["endpoint"] == end
            }
            expected.append((end, key))
        assert event["prior_gdn"] == {"draft_required": True, "anchors": expected}


def check_transfers(relay: Relay, records: HostRecords, gdn_bytes: int) -> None:
    assert not relay.errors
    trace = relay.since()
    assert all(event["namespace"] == "sglang" for event in trace)
    assert sum(event["opcode"] == PREFIX_DETAIL for event in trace) > 1
    gdn_gets = [
        event
        for event in trace
        if event["opcode"] == RANK_GET
        and event["keys"]
        and all(family == GDN for family, _ in event["keys"])
    ]
    assert len(gdn_gets) == records.world
    assert all(event["receipt"] == (0, gdn_bytes // records.world, 0, 0) for event in gdn_gets)
    for device in records.devices:
        transfers = [event for event in trace if event.get("device") == device.hex()]
        assert {RANK_GET, RANK_PUT} <= {event["opcode"] for event in transfers}
        assert any(family == DRAFT for event in transfers for family, _ in event["keys"])


def check_mapped(relay: Relay, call: RankCall, records: HostRecords) -> None:
    prior = relay.since()
    for event in prior:
        if event["opcode"] in (RANK_GET, RANK_PUT):
            families = {r["family"] for r in event["records"] if r.get("end", 0) > PAGE_TOKENS}
            assert event.get("page_maps", {}) == dict.fromkeys(families, (2, 0))
    for rank in range(records.world):
        call("store-tail", rank=rank, draft=False)
    tails = relay.since(len(prior))
    assert len(tails) == records.world
    for event, device in zip(tails, records.devices, strict=True):
        assert event["device"] == device.hex()
        assert event["opcode"] == RANK_PUT
        assert event["status"] == 0
        assert event["page_maps"] == {GQA: (1,)}
        assert event["allocations"] == 1
        assert [(r["begin"], r["end"]) for r in event["records"]] == [(256, 321)]
        assert event["receipt"] == (65 * 33792 // records.world, 0, 0, 65)
        assert event["armed_ns"] <= event["reply_ns"]


def check_cancel(
    command: list[str], env: dict[str, str], relay: Relay, records: HostRecords
) -> None:
    rank = records.world - 1
    records.hold_retrieve = records.devices[rank]
    barrier = relay.arm(RANK_GET, records.devices[rank].hex(), phase="arm", drop=True)
    with subprocess.Popen(
        command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as child:
        try:
            entry = barrier.wait()
            lease = bytes.fromhex(entry["incarnation"] + entry["lease"])
            assert control(relay.upstream, "sglang", LEASE, lease) == (ACTIVE_LEASE, b"")
            assert child.stdin is not None
            child.stdin.write(b"stop")
            child.stdin.flush()
            deadline = monotonic() + 2
            while not any(e["opcode"] == LEASE for e in relay.since()) and monotonic() < deadline:
                sleep(0.01)
            assert any(e["opcode"] == LEASE for e in relay.since())
            assert child.poll() is None
            assert control(relay.upstream, "sglang", LEASE, lease) == (ACTIVE_LEASE, b"")
        finally:
            records.release_retrieve.set()
            barrier.release.set()
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == CLIENT_ERROR
        assert not stdout
        assert b"restore stopped" in stderr
    assert control(relay.upstream, "sglang", LEASE, lease) == (0, b"")
    assert barrier.finished.wait(2)
    records.hold_retrieve = None


def check_old_keys(call: RankCall, old: RankCall, relay: Relay, records: HostRecords) -> None:
    records.gdn_endpoints.on_l1_keys_deleted_by_manager(list(records.records))
    records.records.clear()
    records.owners.clear()
    offset = len(relay.since())
    records.legacy = True
    for rank in range(records.world):
        old("store", RANK0_ANCHOR, rank=rank)
    assert old("lookup") == "65 65 65 65 64 65"
    puts = [e for e in relay.since(offset) if e["opcode"] == RANK_PUT]
    check_rank_draft_keys(puts, records, legacy=True)
    records.legacy = False
    if records.world == 1:
        assert call("lookup") == "65 65 65 65 64 65"
        assert call("get-draft", RANK0_ANCHOR) == f"0 0 {SHARED_DRAFT_ROWS * DRAFT_ROW_BYTES} 0"
    else:
        assert call("lookup") == "0 0 0 0 0"
        assert call("get", RANK0_ANCHOR) == "miss"
        assert call("get-draft", RANK0_ANCHOR) == "miss"


def rank_call(binary: str, env: dict[str, str], world: int) -> RankCall:
    def call(
        op: str, end: int = 225, *, rank: int = 0, draft: bool = True, swap: bool = False
    ) -> str:
        operation = "ready" if op == "old-ready" else op
        expected = {"old-ready": REJECTED, "get-no-peer": CLIENT_ERROR}.get(op, 0)
        args = [binary, operation, world, rank, end, int(draft), int(swap)]
        result = subprocess.run(
            list(map(str, args)),
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
            env=env,
        )
        assert result.returncode == expected, (result.args, result.stderr)
        if op == "get-no-peer":
            assert "extent or draft geometry mismatch" in result.stderr
        return result.stdout.strip()

    return call


def exercise(binary: str, runtime: Path, world: int, baseline: str | None) -> None:
    env = {**os.environ, "XDG_RUNTIME_DIR": str(runtime)}
    call = rank_call(binary, env, world)

    records = HostRecords(world)
    endpoint = runtime / "sglang-lmcache.sock"
    upstream = runtime / "daemon.sock"
    copies = SimpleNamespace(
        devices=dict.fromkeys(reversed(records.devices)), device_for=records.devices.index
    )
    with ExitStack() as stack:
        server = lmcache.SocketServer(str(upstream), "sglang", records, copies)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
        thread.start()
        stack.callback(thread.join)
        stack.callback(server.close)
        relay = stack.enter_context(Relay(endpoint, upstream))
        call("ready")
        (readiness,) = relay.since()
        assert readiness["opcode"] == READY
        assert readiness["devices"] == (
            [device.hex() for device in records.devices] if world > 1 else []
        )
        old_version = "0.5.8+native-v4"
        with patch.object(lmcache, "WIRE_VERSION", old_version):
            devices = b"" if world == 1 else b"".join(records.devices)
            assert control(upstream, "sglang", READY, devices) == (0, old_version.encode())
            call("old-ready")
        call("ready")
        for obsolete in (3, 4, 7):
            assert control(upstream, "sglang", obsolete) == (2, b"unknown LMCache operation")
        reject_old_put(upstream, records.devices[0])
        assert call("lookup") == "0 0 0 0 0"
        check_draft_only(call, relay, records)
        call("store", 65)
        gdn_bytes = 153964544 if world == 1 else 2 * 76992512
        if world == 1:
            assert call("lookup") == "65 65 65 65 64 65"
            assert call("get", 65) == f"{65 * 33792} {gdn_bytes} {64 * 2112} 0"
            call("store-gdn", 161)
            assert call("lookup", 65) == "65 161 65 65 64 65 161"
            gdn_end = 161
        else:
            check_tp2(call, relay, records, gdn_bytes)
            gdn_end = 211
        assert call("get-gdn", gdn_end) == f"0 {gdn_bytes} 0 0"
        check_retention(relay, call, records)
        check_transfers(relay, records, gdn_bytes)
        check_mapped(relay, call, records)
        check_draft_tail(call, relay, records)
        if world > 1:
            command = [binary, "get-cancel", str(world), "0", str(RANK0_ANCHOR), "1", "0"]
            check_cancel(command, env, relay, records)
            size = SHARED_DRAFT_ROWS * DRAFT_ROW_BYTES
            assert call("get-draft", RANK0_ANCHOR) == f"0 0 {size} 0"
        if baseline:
            check_old_keys(call, rank_call(baseline, env, world), relay, records)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DeviceStartupTest)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        raise SystemExit(1)
    check_scales()
    _, candidate_binary, *comparison = sys.argv
    legacy_binary = comparison[0] if comparison else None
    for world_size in (1, 2):
        with TemporaryDirectory(prefix="sglang-ranks-") as work:
            exercise(candidate_binary, Path(work), world_size, legacy_binary)
    print("TP1 and TP2 daemon framing, rank-local geometry, draft coverage and joint GET passed")
