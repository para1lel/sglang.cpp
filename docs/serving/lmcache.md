# LMCache daemon

Run a dedicated LMCache 0.5.8 daemon for the native server's GPU. The daemon moves GQA, GDN, and draft checkpoint data between native GPU allocations and its CPU L1 store. [Dependency pin](../../tools/lmcache/requirements.txt#L1)<!-- prefix: "lmcache==0.5" --> · [Upstream v0.5.8](https://github.com/LMCache/LMCache/commit/651617efcd556edf27387b646997869a2805fad2)

## Start the daemon

Install `tools/lmcache/requirements.txt` in a CUDA-capable Python environment. Set `LMCACHE_PYTHON` to its interpreter, or leave it unset to use `python3`. Run from the repository root under the same Linux user as the native server. Set `GPU_UUID` to the reserved physical GPU UUID and `LMCACHE_SOCKET` to a private absolute socket path. [Version check](../../tools/lmcache/serve.sh#L44)<!-- prefix: "path, namesp" -->

```bash
export LMCACHE_PYTHON=/path/to/lmcache-environment/bin/python
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved GPU UUID}" \
    bash tools/lmcache/serve.sh --socket "${LMCACHE_SOCKET:?set an absolute socket path}"
```

In the native server's shell, set `CUDA_VISIBLE_DEVICES` to the same physical `GPU_UUID` and add `--lmcache-socket "$LMCACHE_SOCKET"` to the usual server command. Expose exactly one GPU to the daemon. Keep the daemon running while restarting a compatible native worker to preserve reusable checkpoints. Start a fresh daemon when you need an empty L1 store or change the checkpoint. Stop native workers before shutting down their daemon. [GPU validation](../../tools/lmcache/sglang_gpu_transfer.py#L257)<!-- prefix: "if torch.cud" -->

| Option | Default and behavior |
| --- | --- |
| `--socket PATH` | `$XDG_RUNTIME_DIR/sglang-lmcache.sock` if that variable is absolute, otherwise `/tmp/sglang-lmcache-<euid>.sock`. Use an absolute path shorter than 108 bytes under an existing directory. |
| `--namespace NAME` | `sglang`. The module serves one namespace. The client and daemon must use the same value. |
| `--memory-gib N` | `16`. Sets the initial and maximum CPU L1 capacity in GiB. Use a finite positive value. |
| `--port N` | `17655`. This is the separate upstream LMCache TCP listener on `127.0.0.1`. |

The launcher selects `--supported-transfer-mode lmcache_driven` and configures a dedicated DRAM L1 with LRU eviction. It sets `--l1-init-size-gb` and `--l1-size-gb` to the configured capacity. LMCache allocates and registers that memory before opening the Unix socket. Allow startup to finish before starting native workers. The upstream listener uses a generated API key. The Unix socket has mode `0600` and checks the peer's effective user ID. [Daemon invocation](../../tools/lmcache/serve.sh#L75)<!-- prefix: "exec \"$pytho" --> · [Storage binding](../../tools/lmcache/sglang_lmcache.py#L430)<!-- prefix: "class Record" --> · [Socket access](../../tools/lmcache/sglang_lmcache.py#L535)<!-- prefix: "os.chmod(end" -->

## Native transport

Run the daemon and native server from the same checkout. The native handshake requires `0.5.8+native-v3` for the compact SLC3 protocol. The application sends one four-plane MXFP8 view per GQA or draft family, with row bounds and chunk digests. A GDN record identifies one exact endpoint and describes four state spans. Native keeps the GPU allocations and readiness event alive until the daemon releases them. The application allocates no host buffer for the cache payload. [Handshake](../../src/runtime/cache/lmcache_client.cpp#L27)<!-- prefix: "constexpr st" --> · [MXFP8 view](../../src/runtime/cache/lmcache_client.cpp#L257)<!-- prefix: "void append_" --> · [GDN view](../../src/runtime/cache/lmcache_client.cpp#L278)<!-- prefix: "void append_" --> · [Release reply](../../tools/lmcache/sglang_lmcache.py#L237)<!-- prefix: "def release_" -->

The daemon waits for GPU readiness. When CUDA supports batched copies, the daemon uses a pitched copy batch for stores and for restores with disjoint destinations. Otherwise it uses LMCache's object-group transfer. A batched submission error propagates. The daemon synchronizes the transfer stream and closes imported mappings before releasing the views. [Copy dispatch](../../tools/lmcache/sglang_gpu_transfer.py#L267)<!-- prefix: "def run(self" -->

The daemon deadline covers command receipt and the client's ARM acknowledgement. It still applies to unarmed reset and delete requests. An armed GPU transfer can finish beyond that deadline. The daemon still checks for disconnects before changing records and releases the views after device synchronization and unmapping. [Deadline check](../../tools/lmcache/sglang_lmcache.py#L182)<!-- prefix: "def check_mu" --> · [ARM acknowledgement](../../tools/lmcache/sglang_lmcache.py#L230)<!-- prefix: "def arm(self" -->

The daemon pins L1 read and write reservations through GPU transfer and reservation cleanup. These pins protect the records from eviction after the ordinary reservation timeout expires. The daemon refuses forced clear or delete while a reservation is active. [Reservation pins](../../tools/lmcache/sglang_lmcache.py#L264)<!-- prefix: "class DmaRes" --> · [Forced reclaim](../../tools/lmcache/sglang_lmcache.py#L298)<!-- prefix: "def reclaim(" -->

## Standalone verification

The B200 standalone proof stored and retrieved all three payload families into separate GPU allocations. It compared the region contents and untouched guards byte for byte. The GQA record held 459 rows across 16 layers, the draft record held 458 rows, and a single GDN span crossed a 64 MiB pinning boundary. This proof covers the daemon transport. Check worker restart, serving correctness, and request performance separately with the integrated implementation.
