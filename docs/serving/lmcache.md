# LMCache daemon

Run a dedicated LMCache 0.5.8 daemon for the native server's GPU. The daemon moves GQA, GDN, and draft checkpoint data between native GPU allocations and its CPU L1 store. [Dependency pin](../../tools/lmcache/requirements.txt#L1)<!-- prefix: "lmcache==0.5" --> · [Upstream v0.5.8](https://github.com/LMCache/LMCache/commit/651617efcd556edf27387b646997869a2805fad2)

## Start the daemon

Install `tools/lmcache/requirements.txt` in a CUDA-capable Python environment. Set `LMCACHE_PYTHON` to its interpreter, or leave it unset to use `python3`. Run from the repository root under the same Linux user as the native server. Set `GPU_UUID` to the reserved physical GPU UUID and `LMCACHE_SOCKET` to a private absolute socket path. [Version check](../../tools/lmcache/serve.sh#L67)<!-- prefix: "path, namesp" -->

```bash
export LMCACHE_PYTHON=/path/to/lmcache-environment/bin/python
CUDA_VISIBLE_DEVICES="${GPU_UUID:?set the reserved GPU UUID}" \
    bash tools/lmcache/serve.sh --socket "${LMCACHE_SOCKET:?set an absolute socket path}"
```

In the native server's shell, set `CUDA_VISIBLE_DEVICES` to the same physical `GPU_UUID` and add `--lmcache-socket "$LMCACHE_SOCKET"` to the usual server command. Expose exactly one GPU to the daemon. Keep the daemon running while restarting a compatible native worker to preserve reusable checkpoints. Start a fresh daemon when you need an empty L1 store or change the checkpoint. Stop native workers before shutting down their daemon. [GPU validation](../../tools/lmcache/sglang_gpu_transfer.py#L261)<!-- prefix: "if torch.cud" -->

Use a socket parent owned by root or the service UID. A group- or world-writable parent must have the sticky bit. Startup and shutdown take a per-endpoint lock for up to two seconds. The service keeps the `0600` lock file in a `0700` directory named `.sglang-lmcache-locks-<euid>` under the socket parent. Leave it in place across restarts. If shutdown times out acquiring the lock, it closes the listener and leaves the socket path. On restart, the daemon can reclaim a stale socket owned by its UID. A live socket or a different type of file at that path prevents startup. [Lock acquisition](../../tools/lmcache/sglang_socket.py#L18)<!-- prefix: "def locked_d" --> · [Socket recovery](../../tools/lmcache/sglang_socket.py#L62)<!-- prefix: "def remove_s" -->

| Option | Default and behavior |
| --- | --- |
| `--socket PATH` | `$XDG_RUNTIME_DIR/sglang-lmcache.sock` if that variable is absolute, otherwise `/tmp/sglang-lmcache-<euid>.sock`. Use an absolute path shorter than 108 bytes under an existing directory. |
| `--namespace NAME` | `sglang`. The module serves one namespace. The client and daemon must use the same value. |
| `--memory-gib N` | `16`. Sets the initial and maximum CPU L1 capacity in GiB. Use a finite positive value. |
| `--port N` | `17655`. This is the separate upstream LMCache TCP listener on `127.0.0.1`. |
| `--rank N` | Unset. Rank `N` from 0 to 15 for a two-GPU native group. It suffixes the default socket with `-rN` and sets the default namespace to `sglang-rN` and the default port to `17655+N`. Explicit options override these defaults. |

The launcher selects `--supported-transfer-mode lmcache_driven` and configures a dedicated DRAM L1 with LRU eviction. It sets `--l1-init-size-gb` and `--l1-size-gb` to the configured capacity. LMCache allocates and registers that memory before opening the Unix socket. Allow startup to finish before starting native workers. [Daemon invocation](../../tools/lmcache/serve.sh#L100)<!-- prefix: "exec \"$pytho" --> · [Storage binding](../../tools/lmcache/sglang_lmcache.py#L433)<!-- prefix: "class Record" -->

The upstream ZeroMQ listener on loopback TCP checks the generated API key. The native Unix socket uses mode `0600` and accepts peers with the daemon's effective UID. Same-UID peers with socket access and the configured namespace can look up, transfer, and delete records or clear the entire store. Use a separate service UID, private socket directory, and dedicated daemon for each trust domain. [API key](../../tools/lmcache/serve.sh#L98)<!-- prefix: "LMCACHE_API_" --> · [Socket mode](../../tools/lmcache/sglang_socket.py#L101)<!-- prefix: "os.chmod(end" --> · [Peer credentials](../../tools/lmcache/sglang_lmcache.py#L536)<!-- prefix: "credentials " --> · [Native HTTP reset and flush](../prefix-cache.md#native-http-controls)

## Native transport

Run the daemon and native server from the same checkout. The native handshake requires `0.5.8+native-v3` for the compact SLC3 protocol. The application sends one four-plane MXFP8 view per GQA or draft family, with row bounds and chunk digests. A GDN record identifies one exact endpoint and describes four state spans. Native keeps the GPU allocations and readiness event alive until the daemon releases them. The application allocates no host buffer for the cache payload. [Handshake](../../src/runtime/cache/lmcache_client.cpp#L27)<!-- prefix: "constexpr st" --> · [MXFP8 view](../../src/runtime/cache/lmcache_client.cpp#L267)<!-- prefix: "void append_" --> · [GDN view](../../src/runtime/cache/lmcache_client.cpp#L290)<!-- prefix: "void append_" --> · [Release reply](../../tools/lmcache/sglang_lmcache.py#L247)<!-- prefix: "def release_" -->

The daemon waits for GPU readiness. When CUDA supports batched copies, the daemon uses a pitched copy batch for stores and for restores with disjoint destinations. Otherwise it uses LMCache's object-group transfer. A batched submission error propagates. The daemon synchronizes the transfer stream and closes imported mappings before releasing the views. [Copy dispatch](../../tools/lmcache/sglang_gpu_transfer.py#L271)<!-- prefix: "def run(self" -->

The daemon deadline covers command receipt and the client's ARM acknowledgement. It still applies to unarmed reset and delete requests. An armed GPU transfer can finish beyond that deadline. The daemon still checks for disconnects before changing records and releases the views after device synchronization and unmapping. [Deadline check](../../tools/lmcache/sglang_lmcache.py#L192)<!-- prefix: "def check_mu" --> · [ARM acknowledgement](../../tools/lmcache/sglang_lmcache.py#L240)<!-- prefix: "def arm(self" -->

The daemon pins L1 read and write reservations through GPU transfer and reservation cleanup. These pins protect the records from eviction after the ordinary reservation timeout expires. The daemon refuses forced clear or delete while a reservation is active. [Reservation pins](../../tools/lmcache/sglang_lmcache.py#L273)<!-- prefix: "class DmaRes" --> · [Forced reclaim](../../tools/lmcache/sglang_lmcache.py#L305)<!-- prefix: "def reclaim(" -->

The public Qwen3.8-27B B200 recipe specifies MTP3 and LMCache 0.5.1. With the repository's pinned vLLM and LMCache 0.5.8, the resolver selects the wheel's `LMCacheMPConnector` when `LMCACHE_USE_UPSTREAM_MP` is unset and the import succeeds. Its `wait_for_save()` records a CUDA event and submits stores asynchronously. vLLM can finish the request while a store is pending and retains its blocks until `finished_sending`. The stock defaults store complete prompt chunks. Native runs MTP4 in the matched benchmark and submits a checkpoint every 8,192 processed target tokens and at successful request end. Native lookup uses [exact GDN endpoints and independent GQA coverage](../prefix-cache.md#checkpoint-compatibility-and-resume). [B200 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B.json?hardware=b200&features=tool_calling%2Creasoning%2Cencoder_parallel%2Cspec_decoding&kv_offload=lmcache&variant=fp8) · [Pinned resolver](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py#L1192)<!-- prefix: "def _resolve" --> · [Store submission](https://github.com/LMCache/LMCache/blob/651617efcd556edf27387b646997869a2805fad2/lmcache/integration/vllm/lmcache_mp_connector.py#L1193)<!-- prefix: "event = self" --> · [Retention](https://github.com/LMCache/LMCache/blob/651617efcd556edf27387b646997869a2805fad2/lmcache/integration/vllm/lmcache_mp_connector.py#L1791)<!-- prefix: "# Keep the b" --> · [Release](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/v1/core/sched/scheduler.py#L3684)<!-- prefix: "for req_id i" --> · [Chunk defaults](https://github.com/LMCache/LMCache/blob/651617efcd556edf27387b646997869a2805fad2/lmcache/integration/vllm/lmcache_mp_connector.py#L612)<!-- prefix: "- lmcache.mp" -->

## Ranked TP2 transport

Each rank in a two-GPU native group runs its own daemon on its own GPU. Pass `--rank N` to select the rank defaults. The client configuration names one daemon per rank. [Rank defaults](../../tools/lmcache/serve.sh#L47)<!-- prefix: "if (( rank >" --> · [Ranked configuration](../../src/runtime/cache/lmcache_group.cpp#L18)<!-- prefix: "std::vector<" -->

Ranked clients select the operations `lookup_prefix_detail`, `retrieve_v4`, and `store_v4`. A `retrieve_v4` or `store_v4` frame matches the SLC3 frame and inserts the KV head count after the end row bound of each nonempty MXFP8 section. A daemon without these operations rejects the request, so the client reports an error instead of a partial restore. [Operations](../../src/runtime/cache/lmcache_client.cpp#L46)<!-- prefix: "lookup_prefi" --> · [KV head count](../../src/runtime/cache/lmcache_client.cpp#L282)<!-- prefix: "if (v4) appe" -->

Ranked records live in per-rank key domains that name the world size, rank, KV head counts, and lane role. The TP1 domain stays unchanged, so TP1 and TP2 records do not mix. [Key domain](../../src/runtime/cache/prefix_payload.cpp#L61)<!-- prefix: "std::string " -->

A ranked lookup asks every rank for its matched lengths and endpoint inventory. The group takes the minimum matched GQA and GDN lengths. It picks the largest GDN endpoint present on every rank, draft-eligible on the draft lane when a draft is requested. A truncated inventory reply is refused. [Composite lookup](../../src/runtime/cache/lmcache_group.cpp#L87)<!-- prefix: "PrefixCacheR" -->

A ranked restore drives one transfer per rank and reports completion only when every rank completes. Byte receipts sum across ranks, and restored tokens are counted once. Native serving does not construct the ranked client yet. The standalone rank-client test exercises it. [Ranked restore](../../src/runtime/cache/lmcache_group.cpp#L152)<!-- prefix: "LmCacheStatu" --> · [Rank client test](../../tests/prefix/lmcache_rank_client.cpp#L58)<!-- prefix: "LmCacheGroup" -->

## Standalone verification

The B200 standalone proof stored and retrieved all three payload families into separate GPU allocations. It compared the region contents and untouched guards byte for byte. The GQA record held 459 rows across 16 layers, the draft record held 458 rows, and a single GDN span crossed a 64 MiB pinning boundary. This proof covers the daemon transport. Check worker restart, serving correctness, and request performance separately with the integrated implementation.
