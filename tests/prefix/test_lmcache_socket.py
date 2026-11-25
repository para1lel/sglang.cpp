import errno
import os
import socket
import socketserver
import stat
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/lmcache"))
from sglang_socket import OwnedUnixStreamServer


def start(path):
    return OwnedUnixStreamServer(str(path), socketserver.BaseRequestHandler)


def test_concurrent_restart_recovers_one_stale_socket(tmp_path):
    path = tmp_path / "daemon.sock"
    with socket.socket(socket.AF_UNIX) as stale:
        stale.bind(str(path))
    barrier = threading.Barrier(2)

    def restart():
        barrier.wait(timeout=5)
        try:
            return start(path)
        except OSError as error:
            assert error.errno == errno.EADDRINUSE
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        attempts = [workers.submit(restart) for _ in range(2)]
        owner, = [server for attempt in attempts if (server := attempt.result())]
    try:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(path))
            connection, _ = owner.socket.accept()
            connection.close()
    finally:
        owner.close_endpoint()
    assert not path.exists()


@pytest.mark.parametrize("listen", [False, True])
def test_preserves_live_socket_during_bind_and_listen(tmp_path, listen):
    path = tmp_path / "daemon.sock"
    with socket.socket(socket.AF_UNIX) as live:
        live.bind(str(path))
        if listen:
            live.listen()
        initial = path.stat()
        with pytest.raises(OSError) as failure:
            start(path)
        assert failure.value.errno == errno.EADDRINUSE
        assert os.path.samestat(initial, path.stat())


@pytest.mark.parametrize("symlink", [False, True])
def test_refuses_other_endpoint_types(tmp_path, symlink):
    path = tmp_path / "daemon.sock"
    if symlink:
        path.symlink_to(tmp_path / "missing")
    else:
        path.write_text("retain")
    initial = path.lstat()
    with pytest.raises(FileExistsError):
        start(path)
    assert os.path.samestat(initial, path.lstat())


def test_close_leaves_replacement_listener(tmp_path):
    path = tmp_path / "daemon.sock"
    original = start(path)
    path.unlink()
    successor = start(path)
    try:
        with pytest.raises(FileExistsError):
            original.close_endpoint()
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(path))
            connection, _ = successor.socket.accept()
            connection.close()
    finally:
        successor.close_endpoint()


def test_refuses_replacement_after_stale_probe(tmp_path, monkeypatch):
    path = tmp_path / "daemon.sock"
    with socket.socket(socket.AF_UNIX) as old:
        old.bind(str(path))
    connect = socket.socket.connect

    def replace_during_probe(probe, address):
        try:
            return connect(probe, address)
        except ConnectionRefusedError:
            path.unlink()
            path.write_text("replacement")
            raise

    monkeypatch.setattr(socket.socket, "connect", replace_during_probe)
    with pytest.raises(FileExistsError):
        start(path)
    assert path.read_text() == "replacement"
