# Request memory and GPU pools

Each rank starts one device arena. The default `gpu_memory_utilization` is `0.88`. The arena caps the startup budget at that fraction of total device memory and leaves up to 2 GiB of free memory for the CUDA driver. The budget includes startup weights and other framework-owned device storage. The decoder commits separate general and exportable cache backing before serving. [Budget constant](../../include/sglang/runtime/backend/device_arena.h#L10)<!-- prefix: "inline const" --> · [Driver headroom](../../src/runtime/backend/device_arena.cpp#L21)<!-- prefix: "constexpr st" --> · [Rank startup](../../src/decoder.cpp#L492)<!-- prefix: "Decoder::Dec" --> · [Backing commit](../../src/runtime/backend/device_arena.cpp#L376)<!-- prefix: "void DeviceA" -->

After startup, framework allocations lease ranges from these backings. Releasing a request returns ranges after stream and export ownership clear. Each exported cache lease carries the backing address and byte offset and keeps the range alive through unmapping. [Range allocation](../../src/runtime/backend/device_arena.cpp#L433)<!-- prefix: "void* Device" --> · [Stream-ordered return](../../src/runtime/backend/device_arena.cpp#L460)<!-- prefix: "cudaError_t " --> · [IPC address](../../src/runtime/backend/device_arena.cpp#L525)<!-- prefix: "ArenaAddress" --> · [Export ownership](../../src/runtime/backend/cuda_resources.cpp#L59)<!-- prefix: "struct Expor" -->

The [native prefix cache](../prefix-cache.md#runtime-path) uses rank-local storage for GQA pages and GDN checkpoints. Shared GQA pages carry immutable complete prefixes. When a hit ends inside a page, its valid rows are copied to a private writable page before appending the suffix. [Slab layout](../../src/runtime/cache/prefix_slab.cpp#L41)<!-- prefix: "PrefixSlab::" --> · [Request page table](../../src/runtime/model/qwen_request.cpp#L342)<!-- prefix: "void Request" -->

Each resident MTP slot reuses pinned host storage for up to five verifier winners. The batch keeps the buffer through its execution stream drain. Direct generation and a batch that cannot allocate the shared pinned buffer use the existing request-owned readback. [Winner buffer](../../src/decoder_verify.cpp#L504)<!-- prefix: "std::shared_" -->

HTTP shutdown stops restores and drains LMCache transfers before the final request release drain. Execution streams drain before their request storage is destroyed. Outstanding IPC owners retain the referenced backing until the lease clears. [HTTP shutdown](../../apps/server/batch_service.cpp#L54)<!-- prefix: "if (lmcache)" --> · [Exported ranges](../../src/runtime/backend/cuda_resources.cpp#L65)<!-- prefix: "~ExportedDev" -->

<a id="peer-draft-repair"></a>

## Rank-local draft storage

Each TP2 rank holds its own draft shard, KV page map and request workspace. Draft seed, proposal and repair use device-local state. The peer gets the recurrent input needed for its shard as part of the ordered rank wave. Target-forward replay leaves these draft stages outside capture. [Ranked proposal](../../src/decoder_tp.cpp#L244)<!-- prefix: "bool Decoder" --> · [Ranked repair](../../src/decoder_tp.cpp#L305)<!-- prefix: "void Decoder" --> · [Ranked seeding](../../src/decoder_tp.cpp#L341)<!-- prefix: "void Decoder" -->

The rank-local resident state remains owned until its stream and transfer leases complete. Check the [ranked restart contract](lmcache.md#standalone-verification) before treating a reused page or checkpoint as ready on both ranks.

Check greedy and sampled continuation for both a singleton and mixed draft depths. Record rank-local KV and actual proposal and repair work. Include sparse slot replacement, cancellation and cleanup on both ranks.
