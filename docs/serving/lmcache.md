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

The public Qwen3.8-27B B200 recipe specifies MTP3 and LMCache 0.5.1. With the repository's pinned vLLM and LMCache 0.5.8, the resolver selects the wheel's `LMCacheMPConnector` when `LMCACHE_USE_UPSTREAM_MP` is unset and the import succeeds. Its `wait_for_save()` records a CUDA event and submits stores asynchronously. vLLM can finish the request while a store is pending and retains its blocks until `finished_sending`. The stock defaults store complete prompt chunks. Native runs MTP4 in the matched benchmark and submits a checkpoint every 8,192 processed target tokens and at successful request end. Native lookup uses [exact GDN endpoints and independent GQA coverage](../prefix-cache.md#checkpoint-compatibility-and-resume). [B200 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B.json?hardware=b200&features=tool_calling%2Creasoning%2Cencoder_parallel%2Cspec_decoding&kv_offload=lmcache&variant=fp8) · [Pinned resolver](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py#L1192)<!-- prefix: "def _resolve" --> · [Store submission](https://github.com/LMCache/LMCache/blob/651617efcd556edf27387b646997869a2805fad2/lmcache/integration/vllm/lmcache_mp_connector.py#L1193)<!-- prefix: "event = self" --> · [Retention](https://github.com/LMCache/LMCache/blob/651617efcd556edf27387b646997869a2805fad2/lmcache/integration/vllm/lmcache_mp_connector.py#L1791)<!-- prefix: "# Keep the b" --> · [Release](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/v1/core/sched/scheduler.py#L3684)<!-- prefix: "for req_id i" --> · [Chunk defaults](https://github.com/LMCache/LMCache/blob/651617efcd556edf27387b646997869a2805fad2/lmcache/integration/vllm/lmcache_mp_connector.py#L612)<!-- prefix: "- lmcache.mp" -->

## Standalone verification

The B200 standalone proof stored and retrieved all three payload families into separate GPU allocations. It compared the region contents and untouched guards byte for byte. The GQA record held 459 rows across 16 layers, the draft record held 458 rows, and a single GDN span crossed a 64 MiB pinning boundary. This proof covers the daemon transport. Check worker restart, serving correctness, and request performance separately with the integrated implementation.
