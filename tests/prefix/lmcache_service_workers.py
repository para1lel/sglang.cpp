"""Exercise ranked BatchCache workers through the Unix daemon with controlled CPU receipts."""

from contextlib import contextmanager, ExitStack
from pathlib import Path
import select
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread

from lmcache_relay import RECEIPT

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/lmcache"))
from sglang_lmcache import GdnEndpoints, SocketServer


class ControlledRecords:
    def __init__(self, rank, mode):
        self.rank = rank
        self.mode = mode
        self.entered = Event()
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
        if self.mode == "reset-error" and self.rank == 0:
            raise ValueError("injected rank 0 reset error")

    def transfer(self, request, view, store):
        assert view.device[0] == self.rank + 1
        assert len(view.records) == 1 and view.records[0].size == 4
        self.entered.set()
        if self.rank == 1:
            if not self.release.wait(5):
                raise ValueError("test failed to release rank 1")
            if self.mode == "error":
                raise ValueError("injected rank 1 receipt error")
            if self.mode == "decline":
                return 3, b""
            if self.mode == "miss":
                return 1, b""
        copied = 4 if store or self.rank == 0 else 1
        return 0, RECEIPT.pack(0, copied, 0, 3 if store else 0)

    def store(self, request, view):
        return self.transfer(request, view, True)

    def retrieve(self, request, view):
        return self.transfer(request, view, False)


def exercise(binary, work, action, mode):
    with ExitStack() as stack:
        ranks = []
        for rank in range(2):
            records = ControlledRecords(rank, mode)
            ranks.append(records)
            endpoint = work / f"service-r{rank}.sock"
            server = SocketServer(str(endpoint), f"service-r{rank}", records)
            worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
            worker.start()
            stack.callback(worker.join)
            stack.callback(server.close)
        args = [binary, work / "service.sock"]
        if action != "lookup":
            args.extend([action, mode])
        with subprocess.Popen(args, text=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE) as child:
            try:
                if action in ("store", "restore"):
                    assert all(rank.entered.wait(3) for rank in ranks), (action, mode)
                    child.stdin.write("held\n")
                    child.stdin.flush()
                    assert select.select([child.stdout], [], [], 3)[0], (action, mode)
                    assert child.stdout.readline() == "release\n", (action, mode)
                    ranks[1].release.set()
                assert child.wait(4) == 0, (action, mode)
            finally:
                ranks[1].release.set()
                if child.poll() is None:
                    child.kill()
        if action in ("register", "restore", "reset"):
            assert [r.resets for r in ranks] == [1, 1], (action, mode)
        if action == "lookup":
            assert all(len(r.digests) == 1 for r in ranks)
            assert ranks[0].digests != ranks[1].digests, "both ranks used the same key domain"


if __name__ == "__main__":
    cases = [("register", "cancel"), ("reset", "reset-error")]
    cases.extend(("store", mode) for mode in ("success", "decline", "error"))
    cases.extend(("restore", mode) for mode in ("success", "miss", "error", "stop"))
    with TemporaryDirectory(prefix="sglang-service-") as directory:
        exercise(sys.argv[2], Path(directory), "lookup", "cold")
        print("public service ranked key factory and cold generation: passed", flush=True)
        for action, mode in cases:
            exercise(sys.argv[1], Path(directory), action, mode)
            print(f"{action} {mode}: passed", flush=True)
