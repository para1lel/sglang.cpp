"""Serialize pathname changes for the Linux LMCache Unix socket."""

import contextlib
import errno
import fcntl
import os
import socket
import socketserver
import stat
import sys


@contextlib.contextmanager
def locked_directory(endpoint):
    directory = os.open(os.path.dirname(endpoint), os.O_DIRECTORY | os.O_RDONLY)
    try:
        # Flock the directory inode so concurrent starts and closes share one lock.
        fcntl.flock(directory, fcntl.LOCK_EX)
        yield directory
    finally:
        os.close(directory)


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
        parent = os.fstat(directory)
        if (parent.st_uid not in (0, os.geteuid()) or
                (parent.st_mode & 0o022 and not parent.st_mode & stat.S_ISVTX)):
            raise PermissionError(f"Unsafe LMCache socket directory permissions: {endpoint}")
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
        with locked_directory(self.server_address) as directory:
            self.server_close()
            remove_stale(directory, self.server_address, self.socket_identity)


if __name__ == "__main__":
    path, identity = sys.argv[1:]
    device, inode = map(int, identity.split(":"))
    with locked_directory(path) as directory:
        remove_stale(directory, path, (device, inode))
