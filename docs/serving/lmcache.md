# LMCache daemon

The launcher runs LMCache 0.5.8 with a dedicated CPU L1 store for sglang prefix payloads. It exposes a Unix socket for the native client. Use a separate Python environment for the daemon. [Dependency pin](../../tools/lmcache/requirements.txt#L1)<!-- prefix: "lmcache==0.5" --> · [Upstream v0.5.8](https://github.com/LMCache/LMCache/commit/651617efcd556edf27387b646997869a2805fad2)

## Start the daemon

Install `tools/lmcache/requirements.txt` in an environment that can import the LMCache multiprocess server. Set `LMCACHE_PYTHON` to that environment's interpreter. The launcher uses `python3` if the variable is unset. It checks the installed version before starting the daemon. Run these commands from the repository root under the same Linux user as the client. [Version check](../../tools/lmcache/serve.sh#L43)<!-- prefix: "path, namesp" -->

```bash
export LMCACHE_PYTHON=/path/to/lmcache-environment/bin/python
"$LMCACHE_PYTHON" -c 'from importlib.metadata import version; import lmcache.v1.multiprocess.server; assert version("lmcache") == "0.5.8"'
bash tools/lmcache/serve.sh
```

| Option | Default and behavior |
| --- | --- |
| `--socket PATH` | `$XDG_RUNTIME_DIR/sglang-lmcache.sock` if that variable is absolute, otherwise `/tmp/sglang-lmcache-<euid>.sock`. Use an absolute path shorter than 108 bytes under an existing directory. |
| `--namespace NAME` | `sglang`. The module serves one namespace. The client and daemon must use the same value. |
| `--memory-gib N` | `16`. Set a finite positive CPU L1 limit in GiB. Use `64` for the formal benchmark daemon configuration. |
| `--port N` | `17655`. This is the separate upstream LMCache TCP listener on `127.0.0.1`. |

The launcher hides GPUs, starts with zero allocated L1 bytes, and selects LRU eviction. It authenticates the upstream TCP listener with a generated key. The Unix socket has mode `0600` and checks the client's effective user ID. [Defaults](../../tools/lmcache/serve.sh#L4)<!-- prefix: "case ${XDG_R" --> · [Daemon invocation](../../tools/lmcache/serve.sh#L74)<!-- prefix: "exec \"$pytho" --> · [Socket access](../../tools/lmcache/sglang_lmcache.py#L203)<!-- prefix: "def verify_r" -->

## Native transport

The client checks the daemon version with `ready()`, checks keys with `lookup()`, and transfers opaque bytes with `get()` and `put()`. Keys distinguish GQA, GDN, and draft payloads. A successful `put()` means LMCache committed the bytes and a read reservation found the expected record. `reset()` clears the dedicated L1 store. The module requires a dedicated store without L2 adapters. [Native client](../../include/sglang/runtime/cache/lmcache_client.h#L25)<!-- prefix: "class LmCach" --> · [Storage binding](../../tools/lmcache/sglang_lmcache.py#L108)<!-- prefix: "class Record" --> · [Store acknowledgement](../../tools/lmcache/sglang_lmcache.py#L177)<!-- prefix: "def acknowle" --> · [Reset](../../tools/lmcache/sglang_lmcache.py#L299)<!-- prefix: "records.stor" -->
