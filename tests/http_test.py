"""Raw loopback HTTP helpers for native serving tests."""

from __future__ import annotations

import json
import socket
import struct
from contextlib import contextmanager
from http.client import HTTPResponse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

LOOPBACK = "127.0.0.1"


def json_body(body: object) -> bytes:
    return body if isinstance(body, bytes) else json.dumps(body).encode()


def raw_post(port: int, route: str, body: object, *, timeout: float = 5) -> socket.socket:
    wire = json_body(body)
    headers = (
        f"POST {route} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(wire)}\r\n\r\n"
    )
    client = socket.create_connection((LOOPBACK, port), timeout=timeout)
    try:
        client.sendall(headers.encode("ascii") + wire)
    except OSError:
        client.close()
        raise
    return client


@contextmanager
def raw_response(client: socket.socket) -> Generator[HTTPResponse, None, None]:
    with HTTPResponse(client) as response:
        response.begin()
        yield response


def reset_on_close(client: socket.socket) -> None:
    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))


def batch_frame(response: HTTPResponse) -> object:
    line = response.readline()
    if not line:
        return None
    if line == b":\n":
        assert response.readline() == b"\n"
        return "keepalive"
    assert line.startswith(b"data: "), line
    assert line.endswith(b"\n"), line
    assert response.readline() == b"\n"
    return "[DONE]" if line == b"data: [DONE]\n" else json.loads(line[6:])


def batch_frames(response: HTTPResponse, eof_message: str) -> Iterator[object]:
    while (frame := batch_frame(response)) != "[DONE]":
        if frame is None:
            raise AssertionError(eof_message)
        if frame != "keepalive":
            yield frame
