"""Verify SSE deadlines across stop-buffered steps with the scripted CPU server."""

import argparse
from contextlib import closing
import http.client
import json
import os
from pathlib import Path

from run_chat_layer1 import save, server
from test_native_chat import CHAT, HEADERS, MODEL, tokenizer_pieces


def frames(port, path, body):
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=10)) as connection:
        connection.request("POST", path, json.dumps(body), HEADERS)
        with connection.getresponse() as response:
            assert response.status == 200, response.status
            content_type = response.headers.get_content_type()
            assert content_type == "text/event-stream", content_type
            wire = response.read()
    result = []
    for frame in wire.removesuffix(b"\n\n").split(b"\n\n"):
        if frame == b":":
            result.append("keepalive")
        else:
            assert frame.startswith(b"data: "), frame
            result.append("[DONE]" if frame == b"data: [DONE]" else json.loads(frame[6:]))
    return result


def check(port, route, body, content, evidence):
    packets = frames(port, route, body)
    save(evidence, packets)
    assert packets[-1] == "[DONE]", packets
    usage = packets[-2]["usage"]
    assert usage["completion_tokens"] == len(content) and usage["prompt_tokens"] > 0, usage
    assert packets[-3]["choices"][0]["finish_reason"] == "length", packets[-3]
    observed = []
    roles = []
    for packet in packets[:-3]:
        if packet == "keepalive":
            observed.append(packet)
            continue
        choice = packet["choices"][0]
        if route == CHAT:
            delta = choice["delta"]
            if "role" in delta:
                roles.append(delta["role"])
            if delta.get("content"):
                observed.append(delta["content"])
        elif choice["text"]:
            observed.append(choice["text"])
    assert roles == (["assistant"] if route == CHAT else []), packets
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True)
    content = "!!!!!!!A!!!B!!!!!!C"
    singles = {piece[0]: token for token, piece in tokenizer_pieces(args.checkpoint).items()
               if len(piece) == 1}
    script = args.evidence / "token-ids.txt"
    script.write_text("13 " + " ".join(str(singles[byte]) for byte in content.encode()) + "\n")
    os.environ["SGLANG_TEST_CHAT_OUTPUT_IDS"] = str(script.resolve())
    os.environ["SGLANG_TEST_STREAM_CLOCK"] = "1"
    body = {"model": MODEL, "max_tokens": len(content), "seed": 13, "stop": "!" * 20,
            "stream": True, "stream_options": {"include_usage": True}}
    with server(args.binary, args.checkpoint, args.evidence / "server", 1) as port:
        text = check(port, "/v1/completions", body | {"prompt": "Hello"}, content,
                     args.evidence / "text.json")
        chat = check(port, CHAT, body | {"messages": [{"role": "user", "content": "Hello"}],
                                        "enable_thinking": False}, content,
                     args.evidence / "chat.json")
    expected = ["keepalive", "!!!!!!!A", "!!!B", "keepalive", "!!!!!!C"]
    assert text == chat == expected, {"text": text, "chat": chat}
    print("text and chat stop-buffered keepalive deadlines passed")


if __name__ == "__main__":
    main()
