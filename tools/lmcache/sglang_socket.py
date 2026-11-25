"""Serialize pathname changes for the Linux LMCache Unix socket."""

import contextlib
import errno
import fcntl
import os
import socket
import socketserver
import stat
import sys
import time


LOCK_TIMEOUT_SECONDS = 2.0


@contextlib.contextmanager
def locked_directory(endpoint):
    flags = os.O_RDONLY | os.O_DIRECTORY
    uid = os.geteuid()
    with contextlib.ExitStack() as handles:
        directory = os.open(os.path.dirname(endpoint), flags)
        handles.callback(os.close, directory)
        parent = os.fstat(directory)
        if (parent.st_uid not in (0, uid) or
                (parent.st_mode & 0o022 and not parent.st_mode & stat.S_ISVTX)):
            raise PermissionError(f"Unsafe LMCache socket directory permissions: {endpoint}")

        name = f".sglang-lmcache-locks-{uid}"
        try:
            os.mkdir(name, 0o700, dir_fd=directory)
        except FileExistsError:
            pass
        private = os.open(name, flags | os.O_NOFOLLOW, dir_fd=directory)
        handles.callback(os.close, private)
        info = os.fstat(private)
        if info.st_uid != uid or info.st_mode & 0o777 != 0o700:
            raise PermissionError(f"Unsafe LMCache lock directory: {name}")

        # Keep the inode across restarts so all waiters use the same lock.
        lock_name = os.path.basename(endpoint) + ".lock"
        lock = os.open(lock_name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                       0o600, dir_fd=private)
        handles.callback(os.close, lock)
        info = os.fstat(lock)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != uid
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise PermissionError(f"Unsafe LMCache lock file: {lock_name}")
        deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(errno.ETIMEDOUT, "LMCache socket lock timed out", endpoint)
                time.sleep(min(0.02, remaining))
        yield directory


def remove_stale(directory, endpoint, expected=None):
    name = os.path.basename(endpoint)
    try:
        node = os.open(name, os.O_PATH | os.O_NOFOLLOW, dir_fd=directory)
    except FileNotFoundError:
        return
    try:
        info = os.fstat(node)
        if (not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid()
                or (expected is not None and (info.st_dev, info.st_ino) != expected)):
            raise FileExistsError(f"LMCache endpoint is not the expected owned socket: {endpoint}")
        # Linux rejects datagram connects to a live stream socket with EPROTOTYPE,
        # even between bind and listen. Only an unbound inode yields ECONNREFUSED.
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as probe:
            try:
                probe.connect(f"/proc/self/fd/{node}")
            except ConnectionRefusedError:
                pass
            except OSError as error:
                if error.errno != errno.EPROTOTYPE:
                    raise
                raise OSError(errno.EADDRINUSE, "LMCache socket is active", endpoint) from error
            else:
                raise OSError(errno.EADDRINUSE, "LMCache socket is active", endpoint)
        if not os.path.samestat(info, os.stat(name, dir_fd=directory, follow_symlinks=False)):
            raise FileExistsError(f"LMCache endpoint changed during recovery: {endpoint}")
        os.unlink(name, dir_fd=directory)
    finally:
        os.close(node)


class OwnedUnixStreamServer(socketserver.UnixStreamServer):
    def __init__(self, endpoint, handler):
        with locked_directory(endpoint) as directory:
            remove_stale(directory, endpoint)
            super().__init__(endpoint, handler)
            info = os.stat(endpoint, follow_symlinks=False)
            self.socket_identity = (info.st_dev, info.st_ino)
            try:
                os.chmod(endpoint, 0o600, follow_symlinks=False)
            except BaseException:
                self.server_close()
                remove_stale(directory, endpoint, self.socket_identity)
                raise

    def close_endpoint(self):
        with contextlib.ExitStack() as handles:
            try:
                directory = handles.enter_context(locked_directory(self.server_address))
            finally:
                self.server_close()
            remove_stale(directory, self.server_address, self.socket_identity)


if __name__ == "__main__":
    path, identity = sys.argv[1:]
    device, inode = map(int, identity.split(":"))
    with locked_directory(path) as directory:
        remove_stale(directory, path, (device, inode))
