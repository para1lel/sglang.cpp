"""Exercise ranked BatchCache workers through the Unix daemon with controlled CPU receipts."""

import select
import subprocess
import sys
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace

from lmcache_relay import GDN, RECEIPT

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/lmcache"))
from sglang_lmcache import GdnEndpoints, SocketServer

RANK_DEVICES = (bytes([1]) + bytes(15), bytes([2]) + bytes(15))
GDN_FIXTURE_BYTES = 4


class ControlledRecords:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.entered = (Event(), Event())
        self.release = Event()
        self.storage = self
        self.resets = 0
        self.gdn_endpoints = GdnEndpoints()
        self.digests = []

    @contextmanager
    def read(self, keys):
        self.digests.extend(key.chunk_hash for key in keys)
        yield {}

    def clear(self):
        self.resets += 1
        if self.mode == "reset-error":
            raise ValueError("injected shared reset error")

    def transfer(self, request, view, store):
        rank = RANK_DEVICES.index(bytes(view.device))
        (checkpoint,) = view.records
        assert checkpoint.key.object_group_id == GDN
        assert checkpoint.size == GDN_FIXTURE_BYTES
        self.entered[rank].set()
        if rank == 1:
            if not self.release.wait(5):
                raise ValueError("test failed to release rank 1")
            if self.mode == "error":
                raise ValueError("injected rank 1 receipt error")
            if self.mode == "decline":
                return 3, b""
            if self.mode == "miss":
                return 1, b""
        copied = GDN_FIXTURE_BYTES if store or rank == 0 else 1
        return 0, RECEIPT.pack(0, copied, 0, 3 if store else 0)

    def store(self, request, view):
        return self.transfer(request, view, True)

    def retrieve(self, request, view):
        return self.transfer(request, view, False)


@contextmanager
def daemon(
    endpoint: Path, mode: str, devices: tuple[bytes, ...]
) -> Generator[ControlledRecords, None, None]:
    records = ControlledRecords(mode)
    copies = SimpleNamespace(devices=dict.fromkeys(devices), device_for=devices.index)
    server = SocketServer(str(endpoint), "service", records, copies)
    worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    worker.start()
    try:
        yield records
    finally:
        records.release.set()
        server.close()
        worker.join()


def exercise(
    binary: str, work: Path, action: str, mode: str, devices: tuple[bytes, ...] = RANK_DEVICES[::-1]
) -> None:
    endpoint = work / "service.sock"
    with ExitStack() as startup:
        records = startup.enter_context(daemon(endpoint, mode, devices))
        args = [binary, endpoint]
        if action != "lookup":
            args.extend([action, mode])
        with subprocess.Popen(args, text=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE) as child:
            try:
                if action in ("store", "restore"):
                    assert all(entered.wait(3) for entered in records.entered), (action, mode)
                    child.stdin.write("held\n")
                    child.stdin.flush()
                    assert select.select([child.stdout], [], [], 3)[0], (action, mode)
                    assert child.stdout.readline() == "release\n", (action, mode)
                    records.release.set()
                if action == "readiness":
                    assert select.select([child.stdout], [], [], 3)[0], "readiness timeout"
                    assert child.stdout.readline() == "restart\n"
                    assert records.resets == 1
                    assert not records.digests, records.digests
                    startup.close()
                    records = startup.enter_context(daemon(endpoint, mode, RANK_DEVICES[::-1]))
                    child.stdin.write("held\n")
                    child.stdin.flush()
                assert child.wait(4) == 0, (action, mode)
            finally:
                records.release.set()
                if child.poll() is None:
                    child.kill()
        if action in ("register", "restore", "reset"):
            assert records.resets == 1, (action, mode)
        if action == "lookup" and len(devices) == 1:
            assert not records.digests, records.digests
        elif action in ("lookup", "readiness"):
            assert len(records.digests) == len(RANK_DEVICES), records.digests
            assert len(set(records.digests)) == len(RANK_DEVICES), records.digests


if __name__ == "__main__":
    cases = [("register", "cancel"), ("reset", "reset-error")]
    cases.extend(("store", mode) for mode in ("success", "decline", "error"))
    cases.extend(("restore", mode) for mode in ("success", "miss", "error", "stop"))
    with TemporaryDirectory(prefix="sglang-service-") as directory:
        exercise(sys.argv[2], Path(directory), "lookup", "cold")
        print("public service ranked key factory and cold generation: passed", flush=True)
        exercise(sys.argv[2], Path(directory), "lookup", "cold", RANK_DEVICES[:1])
        print("public service one-device daemon cold fallback: passed", flush=True)
        wrong_inventory = (RANK_DEVICES[0], RANK_DEVICES[1][::-1])
        exercise(sys.argv[1], Path(directory), "readiness", "ready", wrong_inventory)
        print("ranked cache wrong UUID, successful reset, and daemon recovery: passed", flush=True)
        for action, mode in cases:
            exercise(sys.argv[1], Path(directory), action, mode)
            print(f"{action} {mode}: passed", flush=True)
