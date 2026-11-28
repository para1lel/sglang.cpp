"""Decode fixed codec requests to RGB8 using OpenCV as an independent reference."""

import argparse
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from codec_routes import MEDIA_KINDS, decoded_request, read_request


def decode_video(encoded: bytes, indices: list[int], source: Path) -> bytes:
    source.write_bytes(encoded)
    decoder = cv2.VideoCapture(str(source))
    try:
        frames = []
        for index in indices:
            assert decoder.set(cv2.CAP_PROP_POS_FRAMES, index), index
            ok, frame = decoder.read()
            assert ok, index
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
        return b"".join(frames)
    finally:
        decoder.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rgb", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as scratch:
        for name in MEDIA_KINDS:
            case, frozen = read_request(name)
            part = json.loads(frozen)["messages"][0]["content"][0]
            encoded = base64.b64decode(part[part["type"]]["url"].split(",", 1)[1], validate=True)
            if name == "image":
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()
            else:
                indices = case["media"]["sample_indices"]
                rgb = decode_video(encoded, indices, Path(scratch) / "clip.mp4")
            output = args.rgb / name / "input/frames.rgb8"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(rgb)
            decoded_request(case, frozen, output)


if __name__ == "__main__":
    main()
