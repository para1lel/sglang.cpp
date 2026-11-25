# LMCache daemon

Run a dedicated LMCache 0.5.8 daemon for the native server's GPU. The daemon moves GQA, GDN, and draft checkpoint data between native GPU allocations and its CPU L1 store. [Dependency pin](../../tools/lmcache/requirements.txt#L1)<!-- prefix: "lmcache==0.5" --> · [Upstream v0.5.8](https://github.com/LMCache/LMCache/commit/651617efcd556edf27387b646997869a2805fad2)

## Start the daemon

Install `tools/lmcache/requirements.txt` in a CUDA-capable Python environment. Set `LMCACHE_PYTHON` to its interpreter, or leave it unset to use `python3`. Run from the repository root under the same Linux user as the native server. Set `GPU_UUID` to the reserved physical GPU UUID and `LMCACHE_SOCKET` to a private absolute socket path. [Version check](../../tools/lmcache/serve.sh#L44)<!-- prefix: "path, namesp" -->

```bash
export LMCACHE_PYTHON=/path/to/lmcache-environment/bin/python
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved GPU UUID}" \
    bash tools/lmcache/serve.sh --socket "${LMCACHE_SOCKET:?set an absolute socket path}"
```

In the native server's shell, set `CUDA_VISIBLE_DEVICES` to the same physical `GPU_UUID` and add `--lmcache-socket "$LMCACHE_SOCKET"` to the usual server command. Expose exactly one GPU to the daemon. Keep the daemon running while restarting a native worker to preserve reusable checkpoints. Start a fresh daemon when you need an empty L1 store or change the checkpoint. Stop native workers before shutting down their daemon. [GPU validation](../../tools/lmcache/sglang_gpu_transfer.py#L137)<!-- prefix: "if torch.cud" -->

| Option | Default and behavior |
| --- | --- |
| `--socket PATH` | `$XDG_RUNTIME_DIR/sglang-lmcache.sock` if that variable is absolute, otherwise `/tmp/sglang-lmcache-<euid>.sock`. Use an absolute path shorter than 108 bytes under an existing directory. |
| `--namespace NAME` | `sglang`. The module serves one namespace. The client and daemon must use the same value. |
| `--memory-gib N` | `16`. Sets the initial and maximum CPU L1 capacity in GiB. Use a finite positive value. |
| `--port N` | `17655`. This is the separate upstream LMCache TCP listener on `127.0.0.1`. |

The launcher selects `--supported-transfer-mode lmcache_driven` and configures a dedicated DRAM L1 with LRU eviction. It sets `--l1-init-size-gb` and `--l1-size-gb` to the configured capacity. LMCache allocates and registers that memory before opening the Unix socket. Allow startup to finish before starting native workers. The upstream listener uses a generated API key. The Unix socket has mode `0600` and checks the peer's effective user ID. [Daemon invocation](../../tools/lmcache/serve.sh#L75)<!-- prefix: "exec \"$pytho" --> · [Storage binding](../../tools/lmcache/sglang_lmcache.py#L182)<!-- prefix: "class Record" --> · [Socket access](../../tools/lmcache/sglang_lmcache.py#L293)<!-- prefix: "os.chmod(end" -->

## Native transport

The application submits GPU allocation views and a CUDA readiness event. It keeps both alive until the daemon releases the views. The daemon imports the views and waits for the event. LMCache moves the payload between GPU memory and its CPU L1 records and closes the imported mappings before replying. Native sends transfer descriptions and receives receipts over the socket. The application allocates no host buffer for the cache payload. [GPU transfer](../../tools/lmcache/sglang_gpu_transfer.py#L146)<!-- prefix: "def run(self" --> · [Release reply](../../tools/lmcache/sglang_lmcache.py#L176)<!-- prefix: "def release_" -->

The daemon deadline covers command receipt and the client's ARM acknowledgement. It still applies to unarmed reset and delete requests. An armed GPU transfer can finish beyond that deadline. The daemon still checks for disconnects before changing records and releases the views after device synchronization and unmapping. [Deadline check](../../tools/lmcache/sglang_lmcache.py#L121)<!-- prefix: "def check_mu" --> · [ARM acknowledgement](../../tools/lmcache/sglang_lmcache.py#L169)<!-- prefix: "def arm(self" -->

## Standalone verification

The B200 standalone proof stored and retrieved all three payload families into separate GPU allocations. It compared the region contents and untouched guards byte for byte. The GQA record held 459 rows across 16 layers, the draft record held 458 rows, and a single GDN span crossed a 64 MiB pinning boundary. This proof covers the daemon transport. Check worker restart, serving correctness, and request performance separately with the integrated implementation.
