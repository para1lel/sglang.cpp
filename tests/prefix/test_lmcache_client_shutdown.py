"""Drive the native GET client through shutdown while withholding its receipt."""

import multiprocessing
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory

from lmcache_relay import ADMISSION, GET, HEADER, LEASE, MAGIC, frame_metadata, header, receive


def accept_request(listener, opcode):
    connection, _ = listener.accept()
    try:
        connection.settimeout(5)
        _, code, size = header(connection)
        if code != opcode or size > 4096:
            raise AssertionError(f"Unexpected LMCache request {code=} {size=}")
        entry = frame_metadata(code, receive(connection, size))
        if entry["namespace"] != "shutdown":
            raise AssertionError("Request used the wrong daemon namespace")
        return connection, entry
    except BaseException:
        connection.close()
        raise


def serve_lease(listener, stop_write, final_probe, release, finished, kill_daemon):
    identity, token = os.urandom(16), os.urandom(16)
    get, entry = accept_request(listener, GET)
    with get:
        if entry["records"] != [{"bytes": 4, "tokens": 0, "endpoint": 1}]:
            raise AssertionError("Expected one GDN destination")
        admission = ADMISSION.pack(identity, token, os.getpid())
        get.sendall(HEADER.pack(MAGIC, 4, len(admission)) + admission)
        if receive(get, 1) != b"\x01":
            raise AssertionError("GET failed to arm")
        os.write(stop_write, b"\x01")
        for state in (4, 3, 0):  # active, wrong daemon incarnation, released
            status, entry = accept_request(listener, LEASE)
            with status:
                if (entry["incarnation"], entry["lease"]) != (identity.hex(), token.hex()):
                    raise AssertionError("Status probe did not identify the original lease")
                if state == 0:
                    final_probe.set()
                    if not release.wait(5) or kill_daemon:
                        raise AssertionError("Daemon failed to reach its expected terminal state")
                status.sendall(HEADER.pack(MAGIC, state, 0))
        # Keep the original connection and process alive until the client has returned.
        if not finished.wait(5):
            raise AssertionError("GET failed to return after confirmed lease release")


def check_shutdown(binary, listener, path, kill_daemon):
    ctx = multiprocessing.get_context("fork")
    final_probe, release, finished = (ctx.Event() for _ in range(3))
    stop_read, stop_write = os.pipe()
    daemon = ctx.Process(target=serve_lease, args=(listener, stop_write, final_probe,
                                                   release, finished, kill_daemon))
    daemon.start()
    try:
        child = subprocess.Popen([binary, path, str(stop_read)], pass_fds=(stop_read,))
        try:
            if not final_probe.wait(5) or child.poll() is not None:
                raise AssertionError("GET did not wait through active and stale lease replies")
            if kill_daemon:
                daemon.kill()
            else:
                release.set()
            if child.wait(timeout=5) != 0:
                raise AssertionError("Quiescent GET did not report shutdown")
            finished.set()
            daemon.join(5)
            if daemon.exitcode != (-signal.SIGKILL if kill_daemon else 0):
                raise AssertionError(f"Unexpected daemon exit code {daemon.exitcode}")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
    finally:
        if daemon.is_alive():
            daemon.kill()
        daemon.join()
        os.close(stop_read)
        os.close(stop_write)


if __name__ == "__main__":
    with TemporaryDirectory() as directory, socket.socket(socket.AF_UNIX) as listener:
        path = str(Path(directory) / "daemon.sock")
        listener.bind(path)
        listener.listen()
        listener.settimeout(5)
        for kill_daemon in (False, True):
            check_shutdown(sys.argv[1], listener, path, kill_daemon)
    print("Armed GET shutdown passed for a live daemon and a killed daemon")
