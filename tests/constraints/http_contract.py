"""Exercise real HTTP request and streaming paths and preserve their wire responses."""

import argparse
import codecs
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from hashlib import sha256
import http.client
import json
from pathlib import Path
import re
import socket
import subprocess
from threading import Barrier
from time import monotonic, sleep

from token_oracle import EOS, checkpoint_bytes


MODEL = "Qwen3.8-27B-FP8"
SAMPLED = {"temperature": 0.81, "top_k": 17, "top_p": 0.94, "seed": -123019}
CONSTRAINT_FIELDS = {"ebnf", "regex", "response_format"}


def body(route, budget, **options):
    prompt = "Complete the requested output."
    base = {"model": MODEL, "max_tokens": budget, "return_token_ids": True,
            "temperature": 0.0, "seed": 0}
    if route == "text":
        base["prompt"] = prompt
    else:
        base["messages"] = [{"role": "user", "content": prompt}]
        base["enable_thinking"] = False
    return base | options


def parse_frames(raw):
    paragraphs = raw.decode("utf-8", errors="strict").split("\n\n")
    assert paragraphs[-2:] == ["data: [DONE]", ""], paragraphs[-3:]
    assert all(frame.startswith("data: {") for frame in paragraphs[:-2])
    return [json.loads(frame[6:]) for frame in paragraphs[:-2]]


def transcript(frames, stream, route, media=False):
    choices = [choice for frame in frames for choice in frame["choices"]]
    assert all(choice["index"] == 0 for choice in choices)
    usages = [frame["usage"] for frame in frames if frame.get("usage") is not None]
    usage, = usages
    finished, = [choice["finish_reason"] for choice in choices if choice["finish_reason"]]
    contents = []
    tokens = []
    for choice in choices:
        tokens.extend(choice.get("token_ids", ()))
        if route == "text":
            contents.append(choice.get("text", ""))
        elif stream:
            contents.append(choice["delta"].get("content", ""))
        else:
            assert choice["message"]["role"] == "assistant"
            contents.append(choice["message"]["content"])
    if stream:
        assert frames[-1]["choices"] == [] and "usage" in frames[-1]
        assert choices[-1]["finish_reason"] == finished
        if route == "chat":
            assert choices[0]["delta"] == {"role": "assistant"}
    assert usage["completion_tokens"] == len(tokens)
    assert usage["total_tokens"] == usage["prompt_tokens"] + len(tokens)
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    assert 0 <= cached <= usage["prompt_tokens"]
    prompts = [choice["prompt_token_ids"] for choice in choices if "prompt_token_ids" in choice]
    if media:
        assert not prompts
        prompt = None
    else:
        prompt, = prompts
        assert len(prompt) == usage["prompt_tokens"]
    return {"text": "".join(contents), "tokens": tokens, "prompt": prompt,
            "reason": finished, "usage": usage}


def verify_stream_bytes(frames, route, pieces):
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    for frame in frames:
        for choice in frame["choices"]:
            ids = choice.get("token_ids", [])
            if not ids:
                continue
            token, = ids
            expected = decoder.decode(b"" if token in EOS else pieces[token])
            text = choice["text"] if route == "text" else choice["delta"]["content"]
            assert text == expected, (token, expected, text)
    assert decoder.decode(b"", final=True) == ""


class HTTP:
    def __init__(self, port, output, pieces):
        self.port = port
        self.output = output
        self.pieces = pieces
        self.summary = {}

    def call(self, label, route, request, stream=False, error=None, barrier=None):
        if stream:
            request = request | {"stream": True, "stream_options": {"include_usage": True}}
        wire = json.dumps(request, ensure_ascii=True, separators=(",", ":")).encode()
        stem = self.output / label
        stem.with_suffix(".request.json").write_bytes(wire)
        endpoint = "/v1/completions" if route == "text" else "/v1/chat/completions"
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)) as connection:
            if barrier:
                barrier.wait(timeout=15)
            headers = {"Content-Type": "application/json", "Connection": "close"}
            connection.request("POST", endpoint, wire, headers)
            with connection.getresponse() as response:
                content_type = response.headers.get_content_type()
                status, headers, raw = response.status, response.getheaders(), response.read()
        stem.with_suffix(".response").write_bytes(raw)
        stem.with_suffix(".headers.json").write_text(json.dumps([status, headers]))
        if error:
            assert status == 400, (label, status, raw[:500])
            assert re.search(error, raw.decode("utf-8"), re.I), (label, error, raw[:500])
            assert "error" in json.loads(raw)
            return None
        assert status == 200, (label, status, raw[:500])
        assert content_type == ("text/event-stream" if stream else "application/json")
        frames = parse_frames(raw) if stream else [json.loads(raw)]
        if stream and CONSTRAINT_FIELDS.intersection(request):
            verify_stream_bytes(frames, route, self.pieces)
        media = any(isinstance(message["content"], list)
                    for message in request.get("messages", ()))
        return transcript(frames, stream, route, media)

    def pairs(self, cases, width, validate):
        for start in range(0, len(cases), width):
            group = cases[start:start + width]
            responses = []
            for stream in (False, True):
                barrier = Barrier(len(group))
                extension = "-sse" if stream else "-json"
                with ThreadPoolExecutor(len(group)) as pool:
                    pending = [pool.submit(self.call, label + extension, route, request,
                                           stream, barrier=barrier)
                               for label, route, request in group]
                    responses.append([job.result() for job in pending])
            assert responses[0] == responses[1], group
            for (label, _, request), result in zip(group, responses[0], strict=True):
                assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
                validate(label, request, result)
                self.summary[label] = result


def assert_piece_text(pieces, result):
    ids = result["tokens"]
    assert not (set(ids[:-1]) & EOS)
    raw = b"".join(pieces[token] for token in ids if token not in EOS)
    assert raw.decode("utf-8", errors="strict") == result["text"], result


@contextmanager
def serve(args, pieces):
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [str(args.server), "--checkpoint", str(args.checkpoint), "--device", "0",
               "--port", str(port), "--max-active", str(args.width),
               "--mtp-steps", str(args.mtp), "--trace-batches"]
    if args.graph:
        command.append("--batch-cuda-graphs")
    if args.eager:
        command.append("--eager")
    if args.cache_entries:
        command += ["--prefix-cache-entries", str(args.cache_entries)]
    (args.output / "command.json").write_text(json.dumps(command))
    with (args.output / "server.log").open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            deadline = monotonic() + 150
            while True:
                with socket.socket() as probe:
                    if probe.connect_ex(("127.0.0.1", port)) == 0:
                        break
                assert process.poll() is None and monotonic() < deadline, command
                sleep(0.05)
            yield HTTP(port, args.output, pieces)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def trace_slice(path, offset=0):
    with path.open() as log:
        log.seek(offset)
        return log.read()


def verify_verifier(path, offset, expect_work):
    trace = trace_slice(path, offset)
    rows = re.findall(r"batch mtp slot=\d+ drafted=(\d+) accepted=\d+ verified=(\d+)", trace)
    assert any(int(drafted) > 0 and int(verified) > 0
               for drafted, verified in rows) == expect_work, (path, trace)


def media_guard(http, args, constraint, field, label="media"):
    if args.media_body is None:
        return
    assert args.width == 1
    raw = args.media_body.read_bytes()
    assert sha256(raw).hexdigest() == args.media_sha256
    request = json.loads(raw) | {"max_tokens": 1, "return_token_ids": True}
    before = http.call(f"{label}-before", "chat", request)
    offset = (args.output / "server.log").stat().st_size
    http.call(f"{label}-blocked", "chat", request | constraint, stream=True,
              error=rf"(media.*(?:{field})|(?:{field}).*media)")
    assert not trace_slice(args.output / "server.log", offset)
    after = http.call(f"{label}-after", "chat", request, stream=True)
    assert before == after


def regex_cases():
    rare = "🧑🏾‍🔬 \U0010fffd"
    return [
        ("text-greedy", "text", body("text", 8, regex=r"^(?:K1|K12){1,2}$")),
        ("chat-sampled", "chat", body("chat", 16, regex=f"^{rare}$", **SAMPLED)),
        ("text-sampled", "text", body("text", 4, regex=r"^.+$", **SAMPLED)),
        ("chat-greedy", "chat", body("chat", 8, regex=r"^(?:map|maps){1,2}$")),
        ("nullable-zero", "text", body("text", 0, regex="^(?:)$")),
        ("nullable-eos", "chat", body("chat", 1, regex="^(?:)$")),
        ("exact-budget", "text", body("text", 1, regex="^A$")),
    ]


def reject_regex(http, integrated):
    invalid = [
        ("backref", "chat", body("chat", 8, regex=r"(ab)\1"), r"regex"),
        ("lookaround", "text", body("text", 8, regex=r"(?=x)x"), r"regex"),
        ("budget", "text", body("text", 0, regex="x"), r"max_tokens"),
        ("invalid-type", "text", body("text", 8, regex=[]), r"regex"),
        ("ebnf-conflict", "chat", body("chat", 8, regex="x", ebnf='root ::= "x"'),
         r"regex.*ebnf|ebnf.*regex"),
        ("ignore-eos", "chat", body("chat", 8, regex="x", ignore_eos=True), r"EOS"),
        ("stop-override", "text", body("text", 8, regex="x", stop_token_ids=[]), r"EOS"),
    ]
    mtp_depth = 7 if integrated else 1
    invalid.append(("request-mtp-invalid", "text", body("text", 8, regex="x", mtp_steps=mtp_depth),
                    r"mtp_steps"))
    for label, route, request, error in invalid:
        http.call(label, route, request, stream=True, error=error)


def validate_regex(pieces, label, request, result):
    assert re.fullmatch(request["regex"], result["text"], re.ASCII), (label, result)
    assert_piece_text(pieces, result)
    budget = request["max_tokens"]
    assert len(result["tokens"]) <= budget
    assert result["reason"] == ("stop" if budget else "length"), (label, result)
    if label == "nullable-eos":
        assert len(result["tokens"]) == 1 and result["tokens"][0] in EOS
    if label == "exact-budget":
        assert len(result["tokens"]) == 1 and result["tokens"][0] not in EOS


def run(args, pieces, cases, validate, reject, media):
    with serve(args, pieces) as http:
        log = args.output / "server.log"
        if args.mtp and not args.standin:
            request = body("text", 16, ignore_eos=True)
            offset = log.stat().st_size
            http.call("ordinary-mtp-control", "text", request)
            verify_verifier(log, offset, True)
        offset = log.stat().st_size
        http.pairs(cases, args.width, validate)
        if not args.standin:
            verify_verifier(log, offset, bool(args.mtp))
            if args.graph and not args.mtp:
                assert re.search(r"batch graph_slots=[1-9]\d*", trace_slice(log, offset))
        offset = log.stat().st_size
        reject(http, args)
        assert not trace_slice(log, offset), "Rejected requests reached traced model work"
        http.call("ordinary-after-errors", "chat", body("chat", 3), stream=True)
        media(http, args)
        (args.output / "summary.json").write_text(json.dumps(http.summary, ensure_ascii=True))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, choices=(1, 4), default=1)
    parser.add_argument("--mtp", type=int, choices=range(5), default=0)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--standin", action="store_true")
    parser.add_argument("--integrated", action="store_true")
    parser.add_argument("--cache-entries", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--media-body", type=Path)
    parser.add_argument("--media-sha256")
    return parser.parse_args()


def main():
    args = arguments()
    pieces, _ = checkpoint_bytes(args.checkpoint / "tokenizer.json")
    run(args, pieces, regex_cases(), lambda *fixture: validate_regex(pieces, *fixture),
        lambda http, options: reject_regex(http, options.integrated),
        lambda http, options: media_guard(http, options, {"regex": "x"}, "regex"))


if __name__ == "__main__":
    main()
