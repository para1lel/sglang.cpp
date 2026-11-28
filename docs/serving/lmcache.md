# LMCache daemon

Run a dedicated LMCache 0.5.8 daemon when using CPU backup for native prefix reuse. The daemon transfers GQA, GDN, and draft state between rank-local GPU pools and a shared CPU L1 store. [Dependency pin](../../tools/lmcache/requirements.txt#L1)<!-- prefix: "lmcache==0.5" --> · [Upstream v0.5.8](https://github.com/LMCache/LMCache/commit/651617efcd556edf27387b646997869a2805fad2)

## Start the daemon

Install `tools/lmcache/requirements.txt` in a CUDA-capable Python environment. Set `LMCACHE_PYTHON` to its interpreter, or leave it unset to use `python3`. Run from the repository root under the same Linux user as the native server. Set `GPU_UUIDS` to one reserved physical GPU UUID for TP1 or a comma-separated pair for TP2 and `LMCACHE_SOCKET` to a private absolute socket path. [Version check](../../tools/lmcache/serve.sh#L69)<!-- prefix: "installed = " -->

```bash
export LMCACHE_PYTHON=/path/to/lmcache-environment/bin/python
CUDA_VISIBLE_DEVICES="${GPU_UUIDS:?set the reserved GPU UUIDs}" \
    bash tools/lmcache/serve.sh --socket "${LMCACHE_SOCKET:?set an absolute socket path}"
```

Expose the same physical UUID set to the daemon and native server. Set the server’s `CUDA_VISIBLE_DEVICES` in model rank order and use `--lmcache-socket "$LMCACHE_SOCKET"` with namespace `sglang`. Keep the daemon running across compatible worker restarts to retain checkpoints. Start a fresh store when needed and stop native workers before the daemon.

Use a socket parent owned by root or the service UID. A group- or world-writable parent must have the sticky bit. Startup and shutdown take a per-endpoint lock for up to two seconds. The service keeps the `0600` lock file in a `0700` directory named `.sglang-lmcache-locks-<euid>` under the socket parent. Leave it in place across restarts. If shutdown times out acquiring the lock, it closes the listener and leaves the socket path. On restart, the daemon can reclaim a stale socket owned by its UID. A live socket or a different type of file at that path prevents startup. [Lock acquisition](../../tools/lmcache/sglang_socket.py#L18)<!-- prefix: "def locked_d" --> · [Socket recovery](../../tools/lmcache/sglang_socket.py#L62)<!-- prefix: "def remove_s" -->

| Option | Default and behavior |
| --- | --- |
| `--socket PATH` | `$XDG_RUNTIME_DIR/sglang-lmcache.sock` if that variable is absolute, otherwise `/tmp/sglang-lmcache-<euid>.sock`. Use an absolute path shorter than 108 bytes under an existing directory. |
| `--namespace NAME` | `sglang`. The module serves one namespace. The client and daemon must use the same value. |
| `--memory-gib N` | `16`. Sets the initial and maximum CPU L1 capacity in GiB. Use a finite positive value. The paired drivers use 64 GiB for TP1 and 128 GiB for TP2. |
| `--port N` | `17655`. This is the separate upstream LMCache TCP listener on `127.0.0.1`. |
| `--rank N` | Unset. Legacy naming override for a single visible GPU, with `N` from 0 to 15. Omit it for the shared daemon. It suffixes the default socket with `-rN` and sets the default namespace to `sglang-rN` and the default port to `17655+N`. Explicit options override these defaults. [Naming defaults](../../tools/lmcache/serve.sh#L47)<!-- prefix: "if (( rank >" --> |

The launcher selects `--supported-transfer-mode lmcache_driven` and sets `--l1-init-size-gb` and `--l1-size-gb` for the dedicated DRAM L1 with LRU eviction. LMCache registers L1 memory and initializes the CUDA context and PyTorch stream pool before the socket opens. Wait for the ready log before starting native workers. [Daemon invocation](../../tools/lmcache/serve.sh#L101)<!-- prefix: "exec \"$pytho" --> · [Storage binding](../../tools/lmcache/sglang_lmcache.py#L469)<!-- prefix: "class Record" --> · [CUDA initialization](../../tools/lmcache/sglang_gpu_transfer.py#L473)<!-- prefix: "for device i" --> · [Ready log](../../tools/lmcache/sglang_lmcache.py#L834)<!-- prefix: "LOG.info(" -->

The upstream ZeroMQ listener on loopback TCP checks the generated API key. The Unix socket uses mode `0600` and accepts peers with the daemon's effective UID. Peers with access and the configured namespace can transfer, delete, or clear records. Use a separate service UID, private socket directory, and dedicated daemon for each trust domain. [API key](../../tools/lmcache/serve.sh#L99)<!-- prefix: "LMCACHE_API_" --> · [Socket mode](../../tools/lmcache/sglang_socket.py#L101)<!-- prefix: "os.chmod(end" --> · [Peer credentials](../../tools/lmcache/sglang_lmcache.py#L640)<!-- prefix: "credentials " --> · [Native HTTP reset and flush](../prefix-cache.md#native-http-controls)

## Native transport

Run the daemon and native server from the same checkout. Both widths require SLC3 `READY` version `0.5.8+native-v5`. Op 9 returns independent contiguous target GQA and shifted-draft lengths plus descending exact GDN endpoints, each with draft eligibility. The draft length is zero when draft lookup is disabled. A partial-length match is specific to its exact key. Ops 10 and 11 transfer four-plane GQA and draft views with explicit KV heads and optional physical page maps. A GDN record names an exact endpoint and four state spans. [Handshake](../../src/runtime/cache/lmcache_client.cpp#L27)<!-- prefix: "constexpr st" --> · [Lookup](../../src/runtime/cache/lmcache_client.cpp#L537)<!-- prefix: "LmCachePrefi" --> · [Transfer metadata](../../src/runtime/cache/lmcache_client.cpp#L276)<!-- prefix: "void append_" --> · [Daemon operations](../../tools/lmcache/sglang_lmcache.py#L757)<!-- prefix: "def dispatch" -->

The page map translates logical rows to shared slab slots. Allocation offsets locate each plane within the exported backing. Each map must contain distinct in-range slots. The native transfer retains the referenced pages, allocation and event through lease release. The daemon splits a chunk across discontinuous physical pages. [Page validation](../../tools/lmcache/sglang_gpu_transfer.py#L162)<!-- prefix: "first_page =" --> · [Physical spans](../../tools/lmcache/sglang_gpu_transfer.py#L180)<!-- prefix: "def page_run" -->

The daemon uses the request’s physical GPU UUID for its context, stream, event and imported mappings. It validates the UUID before L1 access or ARM and waits for GPU readiness before copying. It uses pitched copy batches or LMCache object-group transfer. Batched submission errors propagate. It synchronizes the stream before releasing each transfer's mapping references. [Copy dispatch](../../tools/lmcache/sglang_gpu_transfer.py#L483)<!-- prefix: "def run(self" --> · [UUID validation](../../tools/lmcache/sglang_lmcache.py#L775)<!-- prefix: "self.server." --> · [Allocation ownership](../../tools/lmcache/sglang_gpu_transfer.py#L516)<!-- prefix: "def map_allo" -->

The daemon deadline covers command receipt and ARM acknowledgement. It still applies to unarmed reset and delete requests. Armed GPU transfers can finish beyond that deadline. The daemon checks disconnects before mutation and releases views after synchronization and unmapping. [Deadline check](../../tools/lmcache/sglang_lmcache.py#L218)<!-- prefix: "def check_mu" --> · [ARM acknowledgement](../../tools/lmcache/sglang_lmcache.py#L266)<!-- prefix: "def arm(self" -->

The daemon pins L1 read and write reservations through GPU transfer and cleanup, including past the ordinary reservation timeout. It refuses forced clear or delete while a reservation is active. [Reservation pins](../../tools/lmcache/sglang_lmcache.py#L299)<!-- prefix: "class DmaRes" --> · [Forced reclaim](../../tools/lmcache/sglang_lmcache.py#L338)<!-- prefix: "def reclaim(" -->

After successful STORE or GET, the daemon refreshes resident transferred keys in increasing endpoint order. A STORE can also refresh earlier endpoints named by the client. A listed endpoint needs its exact GDN record and matching GQA and required draft coverage in L1. The wire carries up to 12 endpoint digests. GQA uses chunk ends, GDN exact endpoints, and draft chunk ends plus one. [Recency](../../tools/lmcache/sglang_lmcache.py#L517)<!-- prefix: "def touch_pr" --> · [Matching residents](../../tools/lmcache/sglang_lmcache.py#L528)<!-- prefix: "def resident" --> · [Wire encoding](../../src/runtime/cache/lmcache_client.cpp#L369)<!-- prefix: "void append_" -->

The public Qwen3.8-27B B200 recipe specifies MTP3 and LMCache 0.5.1. With the repository's pinned vLLM and LMCache 0.5.8, the resolver selects the wheel's `LMCacheMPConnector` when `LMCACHE_USE_UPSTREAM_MP` is unset and the import succeeds. Its `wait_for_save()` records a CUDA event and submits stores asynchronously. vLLM can finish the request while a store is pending and retains its blocks until `finished_sending`. The stock defaults store complete prompt chunks. Native runs MTP4 in the matched benchmark. See [native checkpoint boundaries](../prefix-cache.md#checkpoint-boundaries) and [lookup coverage](../prefix-cache.md#checkpoint-compatibility-and-resume). [B200 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B.json?hardware=b200&features=tool_calling%2Creasoning%2Cencoder_parallel%2Cspec_decoding&kv_offload=lmcache&variant=fp8) · [Pinned resolver](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py#L1192)<!-- prefix: "def _resolve" --> · [Store submission](https://github.com/LMCache/LMCache/blob/651617efcd556edf27387b646997869a2805fad2/lmcache/integration/vllm/lmcache_mp_connector.py#L1193)<!-- prefix: "event = self" --> · [Retention](https://github.com/LMCache/LMCache/blob/651617efcd556edf27387b646997869a2805fad2/lmcache/integration/vllm/lmcache_mp_connector.py#L1791)<!-- prefix: "# Keep the b" --> · [Release](https://github.com/vllm-project/vllm/blob/81a8cb063e6247fe254d7caf7e98a6691d71022a/vllm/v1/core/sched/scheduler.py#L3684)<!-- prefix: "for req_id i" --> · [Chunk defaults](https://github.com/LMCache/LMCache/blob/651617efcd556edf27387b646997869a2805fad2/lmcache/integration/vllm/lmcache_mp_connector.py#L612)<!-- prefix: "- lmcache.mp" -->

## Ranked TP2 transport

Both ranks share one absolute socket, the `sglang` namespace, and one CPU L1. Check readiness and reset through the shared connection. Preserve both physical UUIDs in the receipts. [Shared client](../../src/runtime/cache/lmcache_group.cpp#L114)<!-- prefix: "LmCacheGroup" --> · [Single reset](../../src/runtime/cache/lmcache_group.cpp#L320)<!-- prefix: "void LmCache" -->

Ranked `READY` appends the raw 16-byte physical UUID for every rank. An unknown UUID returns status `3` and leaves LMCache unavailable. An empty list still returns the wire version. [Wire request](../../src/runtime/cache/lmcache_client.cpp#L501)<!-- prefix: "bool LmCache" --> · [Daemon validation](../../tools/lmcache/sglang_lmcache.py#L743)<!-- prefix: "def ready(se" -->

Ranked records live in per-rank key domains that name the world size, rank, KV and GDN head counts, and draft or target role. The TP1 domain stays unchanged, so TP1 and TP2 records do not mix. [Key domain](../../src/runtime/cache/prefix_payload.cpp#L63)<!-- prefix: "std::string " -->

Lookup takes the common contiguous GQA extent and intersects exact GDN endpoint inventories across ranks. Rank 0 also returns independent shifted draft coverage. The lookup selects the highest common endpoint within the CPU GQA extent and request limit, with draft eligibility when required. Native can combine CPU and resident coverage before choosing what to restore. [Composite lookup](../../src/runtime/cache/lmcache_group.cpp#L130)<!-- prefix: "PrefixCacheR" --> · [Combined coverage](../../src/runtime/execution/decoder_batch_prefix.cpp#L382)<!-- prefix: "bool Decoder" -->

A restore commits only after every required rank finishes. A partial failure clears staged extents before retrying or going cold. L1 keeps immutable rank shards for each independently backed family. The GPU record becomes backed after all required rank acknowledgements and releases. Physical bytes sum across ranks and restored tokens count once. [Ranked restore](../../src/runtime/cache/lmcache_group.cpp#L251)<!-- prefix: "LmCacheStatu" -->

## Cache identity and vendor isolation

Each request's key domain embeds a checkpoint identity: a SHA-256 over `config.json`, the safetensors index, and every safetensors file's stat data. [Checkpoint identity](../../src/runtime/server/cache_identity.cpp#L38)<!-- prefix: "std::string " --> · [Request domain](../../apps/server/detail/batch_service_cache.cpp#L14)<!-- prefix: "std::string " -->

Current native ranked keys use the checkpoint identity without a recurrence tag. [Frozen vendor recurrence tag behavior](https://github.com/para1lel/sglang.cpp/blob/a5974ba4eb99bd8e8f890c2e7262a27dbc3aace1/docs/serving/lmcache.md#cache-identity-and-vendor-isolation)

If an older vendor opt-in wrote untagged records to persistent daemons, drain and reset the old store or start with a fresh namespace before reusing native keys. A per-request cache salt leaves the namespace unchanged. [Reset controls](../prefix-cache.md#native-http-controls)

## Standalone verification

To verify native-v5 transfers, start a fresh daemon with two reserved B200 UUIDs in rank order. Use the same UUID order and the daemon's Python environment for the fixture. Set `LMCACHE_PID` to the daemon process ID.

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUIDS" "$LMCACHE_PYTHON" -B \
    tests/prefix/lmcache_wire_cuda.py "$LMCACHE_SOCKET" "$LMCACHE_PID"
```

The fixture stores and restores mapped GQA and draft pages and exact GDN state on both ranks. It checks transfer bytes, untouched guards, short final pages, invalid maps, allocation bounds, and lease acknowledgement. [Round trip](../../tests/prefix/lmcache_wire_cuda.py#L238)<!-- prefix: "def roundtri" --> · [Rank control](../../tests/prefix/lmcache_wire_cuda.py#L288)<!-- prefix: "def main() -" --> · [Serving pool verification](../prefix-cache.md#verification)

The earlier standalone proof used separate GPU allocations. It compared 459 GQA rows across 16 layers, 458 draft rows, and one GDN span crossing a 64 MiB pinning boundary. It compared transferred data and untouched guards byte for byte.
