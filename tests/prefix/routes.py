"""HTTP evidence helpers for prefix route fixtures."""

from contextlib import closing
import gzip
import http.client
import json


MODEL = "Qwen3.8-27B-FP8"
HEADERS = {"Content-Type": "application/json", "Connection": "close"}


class Routes:
    def __init__(self, port, directory):
        self.port = port
        self.directory = directory

    def raw(self, name, body, code=200, route="/v1/completions"):
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        with closing(http.client.HTTPConnection("127.0.0.1", self.port, timeout=90)) as connection:
            connection.request("POST", route, encoded, HEADERS)
            with connection.getresponse() as response:
                raw = response.read()
                status = response.status
                content = response.headers.get_content_type()
        (self.directory / f"{name}.request.json").write_bytes(encoded)
        with gzip.open(self.directory / f"{name}.response.gz", "wb") as stream:
            stream.write(raw)
        assert status == code, (name, status, raw[:240])
        if code != 200:
            assert content == "application/json" and "error" in json.loads(raw), name
        return raw, content

    @staticmethod
    def usage(name, usage, prompt, completion, cached):
        assert usage["prompt_tokens"] == prompt and usage["completion_tokens"] == completion, name
        assert usage["prompt_tokens_details"] == {"cached_tokens": cached}, (name, usage)
        if "total_tokens" in usage:
            assert usage["total_tokens"] == prompt + completion, name

