import os
import socket
import struct
from pathlib import Path

VERSION = "0.5.8"
WIRE_VERSION = VERSION + "+native-v5"
MAGIC = b"SLC3"
FRAME = struct.Struct("!4sIQ")


def ready(path: Path, pid: int, namespace: str = "sglang") -> dict[str, str | int]:
    expected = WIRE_VERSION.encode("ascii")
    before = os.lstat(path)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        connection.connect(str(path))
        peer, uid, _ = struct.unpack(
            "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
        )
        if uid != os.geteuid() or peer != pid:
            raise ValueError("LMCache Unix socket peer differs from the launched daemon")
        encoded = namespace.encode("ascii")
        body = struct.pack("!H", len(encoded)) + encoded
        connection.sendall(FRAME.pack(MAGIC, 1, len(body)) + body)
        with connection.makefile("rb") as reply:
            header = reply.read(FRAME.size)
            if len(header) != FRAME.size:
                raise ValueError("Incomplete LMCache readiness header")
            if FRAME.unpack(header) != (MAGIC, 0, len(expected)):
                raise ValueError("Unexpected LMCache readiness response")
            payload = reply.read(len(expected))
            if payload != expected or reply.read(1):
                raise ValueError("Unexpected LMCache readiness response")
    if not os.path.samestat(before, os.lstat(path)):
        raise ValueError("LMCache readiness endpoint was replaced")
    stat = (Path("/proc") / str(peer) / "stat").read_text().rsplit(")", 1)[1].split()
    return {
        "socket": str(path),
        "version": VERSION,
        "namespace": namespace,
        "group_pid": pid,
        "peer_pid": peer,
        "peer_uid": uid,
        "peer_start_ticks": int(stat[19]),
        "frame": (header + payload).hex(),
        "wire": MAGIC.decode("ascii"),
        "ready_version": WIRE_VERSION,
        "socket_id": f"{before.st_dev}:{before.st_ino}",
    }
