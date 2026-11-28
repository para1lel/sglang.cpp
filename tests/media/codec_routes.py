"""Compare pinned decoded RGB8 and inline PNG/MP4 through native HTTP."""

import argparse
import base64
import http.client
import json
import re
import subprocess
from contextlib import closing
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from http_routes import CHAT, HEADERS, MODEL, RGB8, Json, received, serve, stream_result

FIXTURES = Path(__file__).resolve().with_name("fixtures")
MEDIA_KINDS = ("image", "video")
CODECS = {"png16", "avformat", "avcodec", "swscale", "avutil"}


def check_linkage(args):
    for label, binary, expected in (("default", args.default, set()),
                                    ("enabled", args.enabled, CODECS)):
        elf = subprocess.check_output(["readelf", "-d", str(binary)], text=True)
        linked = subprocess.check_output(["ldd", str(binary)], text=True)
        (args.evidence / f"{label}.readelf").write_text(elf)
        (args.evidence / f"{label}.ldd").write_text(linked)
        direct = set(re.findall(r"\(NEEDED\).*?\[lib([^.\]]+)\.so", elf))
        resolved = dict(re.findall(r"lib([^.\s]+)\.so\S*\s+=>\s+(\S+)", linked))
        assert "not found" not in linked, linked
        assert direct & CODECS == resolved.keys() & CODECS == expected, label
        prefix = (args.codec_prefix / "lib").resolve()
        for name in expected:
            assert Path(resolved[name]).resolve().parent == prefix, (name, resolved[name])


def read_request(name: str) -> tuple[Json, bytes]:
    case = json.loads((FIXTURES / f"{name}.case.json").read_bytes())
    data = (FIXTURES / f"{name}.http-body").read_bytes()
    assert sha256(data).hexdigest() == case["request_sha256"]
    assert len(data) == case["request_bytes"]
    return case, data


def decoded_request(case, frozen, path):
    body = json.loads(frozen)
    part = body["messages"][0]["content"][0]
    kind = part["type"]
    media = case["media"]
    fields = {}
    if kind == "video_url":
        height, width, channels = media["decoded_frame_shape"]
        indices = media["sample_indices"]
        hashes = [media["decoded_rgb8_sha256"][index] for index in indices]
        fields = {"source_fps": media["fps"], "frame_indices": indices,
                  "source_frame_count": media["loader_metadata"]["total_num_frames"]}
        del body["media_io_kwargs"], body["mm_processor_kwargs"]
    else:
        height, width, channels = media["decoded_shape"]
        hashes = [media["decoded_rgb8_sha256"]]
    pixels = path.read_bytes()
    size = height * width * channels
    assert channels == 3 and len(pixels) == size * len(hashes), path
    for index, expected in enumerate(hashes):
        assert sha256(pixels[index * size:(index + 1) * size]).hexdigest() == expected, path
    part[kind] = {"url": RGB8 + base64.b64encode(pixels).decode(),
                  "height": height, "width": width, **fields}
    body["model"] = MODEL
    return json.dumps(body).encode()


def post(args, port, label, request, status, mime):
    (args.evidence / f"{label}.request").write_bytes(request)
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=90)) as client:
        client.request("POST", CHAT, request, HEADERS)
        with client.getresponse() as response:
            raw = response.read()
            kind = response.headers.get_content_type()
            (args.evidence / f"{label}.body").write_bytes(raw)
            (args.evidence / f"{label}.http").write_text(f"{response.status} {kind}\n")
            assert (response.status, kind) == (status, mime), (label, response.status, raw[:300])
    return raw


def validate(args, port, label, request, case):
    body = json.loads(request)
    raw = post(args, port, f"{label}-sse", request, 200, "text/event-stream")
    budget = body["max_tokens"]
    ids, usage, prompt = stream_result(raw, budget)
    frames = raw.rstrip(b"\n").split(b"\n\n")
    events = [json.loads(frame[6:]) for frame in frames[:-1]]
    assert frames[-1] == b"data: [DONE]"
    expected_prompt = case["effective_input_ids"]
    assert prompt == expected_prompt
    assert usage["prompt_tokens"] == len(expected_prompt)
    assert events[-1]["choices"] == []
    assert events[-1].get("prompt_token_ids") is None
    for index, event in enumerate(events[:-1]):
        assert event.get("prompt_token_ids") == (expected_prompt if index == 0 else None)
        (choice,) = event["choices"]
        assert choice["index"] == 0
        assert choice.get("stop_reason") is None
        tokens = choice.get("token_ids") or []
        assert type(tokens) is list
        assert all(type(token) is int and token >= 0 for token in tokens)
    text = "".join(event["choices"][0]["delta"].get("content") or "" for event in events[:-1])

    body["stream"] = False
    del body["stream_options"]
    reply = json.loads(post(args, port, f"{label}-json", json.dumps(body).encode(),
                            200, "application/json"))
    assert received(reply, budget) == (ids, usage), label
    assert reply["prompt_token_ids"] == prompt == case["effective_input_ids"], label
    choice, = reply["choices"]
    assert choice["message"] == {"role": "assistant", "content": text}, label
    return {"prompt_ids": prompt, "output_ids": ids, "usage": usage,
            "finish": choice["finish_reason"], "text": text}


def reject(args, port, label, request, reason):
    reply = json.loads(post(args, port, label, request, 400, "application/json"))
    assert reason in reply["error"]["message"].lower(), (label, reply)


def invalid_requests(name, request):
    body = json.loads(request)
    part = body["messages"][0]["content"][0]
    payload = part[part["type"]]
    prefix, _, encoded = payload["url"].partition(",")
    payload["url"] = prefix + ",AB=="
    yield "base64", json.dumps(body).encode(), "canonical base64"
    payload["url"] = prefix + ",AAAA"
    reason = "png" if name == "image" else "mp4"
    yield "container", json.dumps(body).encode(), reason
    if name == "image":
        png = bytearray(base64.b64decode(encoded))
        png[16:20] = (1025).to_bytes(4, "big")
        payload["url"] = prefix + "," + base64.b64encode(png).decode()
        yield "geometry", json.dumps(body).encode(), "png dimensions"


def run(args):
    args.evidence.mkdir()
    check_linkage(args)
    pairs = {}
    for name in MEDIA_KINDS:
        case, frozen = read_request(name)
        pixels = args.rgb / name / "input" / "frames.rgb8"
        pairs[name] = (case, frozen, decoded_request(case, frozen, pixels))

    outcomes = {}
    for mode, binary, options in (("default", args.default, ()), ("enabled", args.enabled, ()),
                                  ("eager", args.enabled, ("--eager",))):
        current = SimpleNamespace(server=binary, checkpoint=args.checkpoint, evidence=args.evidence)
        with serve(current, mode, "--max-active", "1", "--mtp-steps", "0", *options) as (port, _):
            for name, (case, frozen, decoded) in pairs.items():
                label = f"{mode}-{name}"
                if mode == "default":
                    reject(args, port, f"{label}-disabled", frozen, "enabled inline media")
                if mode == "enabled":
                    for suffix, request, reason in invalid_requests(name, frozen):
                        reject(args, port, f"{label}-{suffix}", request, reason)
                requests = {"default": (("decoded", decoded),),
                            "enabled": (("decoded", decoded), ("compressed", frozen)),
                            "eager": (("compressed", frozen),)}[mode]
                for encoding, request in requests:
                    result = validate(args, port, f"{label}-{encoding}", request, case)
                    assert outcomes.setdefault(name, result) == result, (label, encoding)
    (args.evidence / "parity.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    print("Default linkage, PNG/MP4 rejection, and all frozen decoded/encoded responses match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("default", "enabled", "codec_prefix", "checkpoint", "rgb", "evidence"):
        parser.add_argument(argument, type=Path)
    run(parser.parse_args())
