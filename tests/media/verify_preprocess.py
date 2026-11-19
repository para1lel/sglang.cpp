import argparse
import ast
import hashlib
import json
import logging
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
import transformers
from transformers import AutoProcessor


SGLANG_MEDIA_SHA256 = "359eb630b837c158f43462dcd6eb7620d5a3b7fd063a3be48206f321881fe3bc"


@dataclass(frozen=True)
class Case:
    name: str
    mode: str
    shape: tuple
    fps: float = 0
    source_frames: int = 0
    indices: tuple = ()


CASES = (
    Case("image-min-wide", "image", (1, 79, 225)),
    Case("image-min-tall", "image", (1, 225, 79)),
    Case("image-ties-down", "image", (1, 272, 304)),
    Case("image-no-resize", "image", (1, 256, 288)),
    Case("image-horizontal-only", "image", (1, 256, 275)),
    Case("image-mixed-resize", "image", (1, 249, 289)),
    Case("video-two-tiny", "checkpoint", (2, 11, 17), 7, 2, (0, 1)),
    Case("video-three-odd", "checkpoint", (3, 45, 69), 29.97, 11, (1, 5, 10)),
    Case("video-five-odd", "checkpoint", (5, 72, 55), 3.25, 8, (0, 0, 3, 5, 7)),
    Case("video-default-full", "sglang", (8, 131, 189), 4, 8),
    Case("video-default-sparse", "sglang", (4, 131, 189), 4, 8, (0, 2, 4, 7)),
    Case("video-default-two", "sglang", (3, 32, 55), 24, 3),
    Case("video-default-horizontal-only", "sglang", (2, 336, 351), 1, 2),
    Case("video-default-downsize", "sglang", (2, 769, 1053), 1, 2),
)

SEEDED_CASES = (
    Case("seeded-image-aspect-limit", "image", (1, 32, 6400)),
    Case("seeded-image-adjacent-to-ties", "image", (1, 271, 305)),
    Case("seeded-video-checkpoint-seven", "checkpoint", (7, 33, 65), 0.001, 1000000000,
         (0, 0, 17, 10389, 7456789, 985000000, 999999999)),
    Case("seeded-video-legacy-upsize", "sglang", (4, 19, 251), 50, 17, (0, 5, 10, 16)),
    Case("seeded-video-legacy-downsize", "sglang", (2, 1137, 947), 3, 2),
)


class DecodedVideo:
    def __init__(self, frames, fps, source_count, indices):
        self.frames = frames
        self.avg_fps = fps
        self.source_count = source_count
        self.decoded_indices = indices or tuple(range(source_count))

    def __len__(self):
        return self.source_count

    def get_frames_as_tensor(self, indices):
        decoded = [self.decoded_indices.index(index) for index in indices]
        return torch.from_numpy(self.frames[decoded])


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_sglang(path):
    source = path.read_bytes()
    assert sha256(source) == SGLANG_MEDIA_SHA256, path
    names = {
        "IMAGE_FACTOR", "MIN_PIXELS", "MAX_PIXELS", "MAX_RATIO", "VIDEO_TOTAL_PIXELS",
        "VIDEO_MIN_PIXELS", "VIDEO_MAX_PIXELS", "FRAME_FACTOR", "FPS",
        "FPS_MIN_FRAMES", "FPS_MAX_FRAMES", "logger",
    }
    functions = {
        "smart_resize", "round_by_factor", "ceil_by_factor", "floor_by_factor",
        "smart_nframes", "preprocess_video_sync",
    }
    nodes = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
    scope = dict(
        np=np, torch=torch, torchvision=torchvision, math=math, logging=logging,
        os=SimpleNamespace(environ={}), time=time, is_cpu=lambda: True,
        InterpolationMode=torchvision.transforms.InterpolationMode,
        VideoDecoderWrapper=DecodedVideo,
        envs=SimpleNamespace(SGLANG_IMAGE_MAX_PIXELS=SimpleNamespace(get=lambda: 784 * 16384)),
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def fixture(shape):
    frame, row, column, channel = np.ogrid[:shape[0], :shape[1], :shape[2], :3]
    return ((17 * row + 53 * column + 67 * frame + 97 * channel
             + (row * (column + 2)) % 71 + (column * frame) % 37) % 256).astype(np.uint8)


def oracle(case, frames, processor, sglang):
    if case.mode == "image":
        prepared = processor.image_processor(
            images=torch.from_numpy(frames[0]).permute(2, 0, 1), return_tensors="pt",
        )
        return prepared.image_grid_thw[0].tolist(), [], [], prepared.pixel_values
    if case.mode == "sglang":
        source = DecodedVideo(frames, case.fps, case.source_frames, case.indices)
        sampled, metadata = sglang["preprocess_video_sync"](source)
        indices = metadata["frames_indices"].tolist()
    else:
        sampled = torch.from_numpy(frames).permute(0, 3, 1, 2)
        indices = list(case.indices)
    prepared = processor.video_processor(
        videos=[sampled], do_sample_frames=False, cap_pixels_per_frame=False,
        return_tensors="pt",
    )
    timestamps = processor._calculate_timestamps(indices.copy(), case.fps, merge_size=2)
    return (
        prepared.video_grid_thw[0].tolist(), indices, timestamps, prepared.pixel_values_videos,
    )


def native_args(case, binary, rgb_path, output_path):
    arguments = [binary, case.mode, rgb_path, output_path, *map(str, case.shape[1:])]
    arguments.append(str(case.shape[0]))
    if case.mode != "image":
        arguments.extend(map(str, (case.source_frames, case.fps, *case.indices)))
    return arguments


def verify_case(case, args, processor, sglang, frames=None):
    if frames is None:
        frames = fixture(case.shape)
    rgb_path = args.output / f"{case.name}.rgb"
    output_path = args.output / f"{case.name}.f32"
    raw = frames.tobytes()
    rgb_path.write_bytes(raw)
    grid, indices, timestamps, reference = oracle(case, frames, processor, sglang)
    observed = subprocess.check_output(
        native_args(case, args.binary, rgb_path, output_path), text=True,
    ).splitlines()
    assert len(observed) == 3, (case, observed)
    assert list(map(int, observed[0].split())) == grid, (case, observed, grid)
    assert list(map(int, observed[1].split())) == indices, (case, observed, indices)
    assert list(map(float, observed[2].split())) == timestamps, (case, observed, timestamps)
    values = output_path.read_bytes()
    assert reference.dtype == torch.float32 and reference.shape == (math.prod(grid), 1536)
    if values != reference.numpy().tobytes():
        actual = np.frombuffer(values, dtype="<f4").reshape(-1, 1536)
        expected = reference.numpy()
        mismatch = np.argwhere(actual != expected)
        location = tuple(mismatch[0])
        raise AssertionError(
            (case.name, len(mismatch), location, actual[location], expected[location]),
        )
    return {
        "case": case.name, "rgb_sha256": sha256(raw), "patch_sha256": sha256(values),
        "grid_thw": grid, "patch_values": reference.numel(), "source_indices": indices,
        "timestamps_seconds": timestamps,
    }


def require_failure(command, code, diagnostic):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == code and diagnostic in result.stderr, (command, result)


def require_reference_failure(action, exception, diagnostic=""):
    try:
        action()
    except exception as error:
        assert diagnostic in str(error), error
    else:
        raise AssertionError((exception, diagnostic))


def verify_rejections(args):
    rgb_path = args.output / "video-two-tiny.rgb"
    truncated = args.output / "truncated.rgb"
    truncated.write_bytes(rgb_path.read_bytes()[:-1])
    output_path = args.output / "rejected.f32"
    video = [args.binary, "checkpoint", rgb_path, output_path, "11", "17", "2", "2"]
    require_failure(video + ["nan", "0", "1"], 2, "source video")
    require_failure(video + ["0", "0", "1"], 2, "source video")
    require_failure(video + ["1", "-1", "1"], 2, "frame indices")
    require_failure(video + ["1", "1", "0"], 2, "frame indices")
    require_failure(video + ["1", "0", "2"], 2, "frame indices")
    require_failure(video + ["1", "0"], 2, "frame indices")
    require_failure(video + ["1"], 2, "frame indices")
    require_failure(
        [args.binary, "checkpoint", truncated, *video[3:], "1", "0", "1"], 2, "byte length",
    )
    require_failure(
        [args.binary, "sglang", *video[2:-1], "8", "4", "0", "7"], 2, "default sample",
    )
    require_failure(
        [args.binary, "sglang", *video[2:-1], "8", "4"], 2, "differs from the source count",
    )
    require_failure([args.binary, "sample", "1", "1"], 2, "source video")
    require_failure([args.binary, "image", *video[2:7]], 2, "exactly one")
    require_failure(
        [args.binary, "image", rgb_path, output_path, "0", "17", "1"], 2, "dimensions",
    )
    require_failure(
        [args.binary, "image", rgb_path, output_path, "16384", "16384", "1"], 3, "byte limit",
    )
    tall = args.output / "invalid-aspect.rgb"
    tall.write_bytes(fixture((1, 201, 1)).tobytes())
    require_failure(
        [args.binary, "image", tall, output_path, "201", "1", "1"], 2, "aspect ratio",
    )
    require_failure(
        [args.binary, "checkpoint", tall, output_path, "201", "1", "1", "2", "1", "0"],
        2, "frame indices",
    )
    many = args.output / "too-many-patches.rgb"
    many.write_bytes(fixture((768, 1, 100)).tobytes())
    require_failure(
        [args.binary, "sglang", many, output_path, "1", "100", "768", "768", "1"],
        3, "Normalized patches exceed",
    )
    assert not output_path.exists()


def verify_reference_rejections(processor, scope):
    require_reference_failure(
        lambda: oracle(Case("bad-aspect", "image", (1, 201, 1)),
                       fixture((1, 201, 1)), processor, scope),
        ValueError, "aspect ratio",
    )
    require_reference_failure(
        lambda: oracle(Case("one-frame", "checkpoint", (1, 11, 17), 1, 2, (0,)),
                       fixture((1, 11, 17)), processor, scope),
        ValueError, "temporal_factor",
    )
    require_reference_failure(lambda: scope["smart_nframes"]({}, 1, 1), ValueError, "nframes")
    require_reference_failure(lambda: scope["smart_nframes"]({}, 2, 0), ZeroDivisionError)
    require_reference_failure(lambda: scope["smart_nframes"]({}, 2, math.nan), ValueError)
    unchecked_centers = processor._calculate_timestamps([1, 0, 4], 1, merge_size=2)
    assert unchecked_centers == [0.5, 4.0]
    return 5


def verify_sampler(args, scope):
    sources = ((2, 180.0), (3, 180.0), (7, 3.0), (39, 3.0), (117, 29.97),
               (767, 1.0), (999, 0.5), (1000000000, 1000000.0))
    for count, fps in sources:
        frames = scope["smart_nframes"]({}, count, fps)
        expected = np.unique(np.linspace(0, count - 1, num=frames, dtype=np.int64)).tolist()
        observed = subprocess.check_output(
            [args.binary, "sample", str(count), str(fps)], text=True,
        )
        assert list(map(int, observed.split())) == expected, (count, fps)
    return len(sources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("sglang_media_source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_default_device("cpu")
    assert transformers.__version__ == "5.19.0", transformers.__version__
    sglang = load_sglang(args.sglang_media_source)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    records = [verify_case(case, args, processor, sglang) for case in CASES]
    random = np.random.default_rng(813642)
    for case in SEEDED_CASES:
        frames = random.integers(0, 256, size=(*case.shape, 3), dtype=np.uint8)
        records.append(verify_case(case, args, processor, sglang, frames))
    subprocess.run([args.binary, "contract"], check=True)
    verify_rejections(args)
    reference_rejections = verify_reference_rejections(processor, sglang)
    print(json.dumps({
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__,
        "sglang_media_source_sha256": SGLANG_MEDIA_SHA256,
        "sampler_cases": verify_sampler(args, sglang),
        "native_rejections": 23, "reference_rejections": reference_rejections,
        "cases": records,
    }, indent=2))


if __name__ == "__main__":
    main()
