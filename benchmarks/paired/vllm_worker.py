"""Observe the paired vLLM worker's resolved engine configuration."""

from importlib.metadata import version

from benchmarks.streaming.vllm_observer import StreamingBaselineObserver
from default_evidence import GRAPH_CAPTURE_FIELDS
from workload_suite import V4_TOKEN_BUDGET


class PairedBaselineObserver(StreamingBaselineObserver):
    def baseline_snapshot(self):
        state = super().baseline_snapshot()
        config = self.vllm_config
        cache = config.cache_config
        if cache.enable_prefix_caching:
            state["cache"].update(block_size=cache.block_size,
                                  mamba_block_size=cache.mamba_block_size,
                                  mamba_cache_mode=cache.mamba_cache_mode)
        scheduler = config.scheduler_config
        scheduled = scheduler.max_num_scheduled_tokens
        state["flashinfer"] = version("flashinfer-python")
        state["scheduler"] = {
            "max_model_len": config.model_config.max_model_len,
            "max_num_seqs": scheduler.max_num_seqs,
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "max_num_scheduled_tokens_configured": scheduled,
            "max_num_scheduled_tokens_effective": (
                scheduler.max_num_batched_tokens if scheduled is None else scheduled),
            "async_scheduling": scheduler.async_scheduling,
            "enable_chunked_prefill": scheduler.enable_chunked_prefill,
        }
        if scheduler.max_num_batched_tokens == V4_TOKEN_BUDGET:
            compilation = config.compilation_config
            state["compilation"].update(
                (key, getattr(compilation, key)) for key in GRAPH_CAPTURE_FIELDS)
        state["parallel"] = {
            "tp": config.parallel_config.tensor_parallel_size,
            "dp": config.parallel_config.data_parallel_size,
            "pp": config.parallel_config.pipeline_parallel_size,
        }
        state["kv_offloading_size"] = config.cache_config.kv_offloading_size
        state["kv_transfer"] = config.kv_transfer_config is not None
        state["cpu_offload"] = repr(config.offload_config)
        state["skip_tokenizer_init"] = config.model_config.skip_tokenizer_init
        state["quantization"] = config.model_config.quantization
        return state

    def tp2_snapshot(self):
        import os

        import torch
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        state = self.baseline_snapshot()
        runner = self.model_runner
        state["draft_attention_layers"] = sorted(runner.speculator.draft_attn_layer_names)
        state["flashinfer_dispatch"] = {
            "runner": flashinfer_dispatch(runner.attn_groups),
            "speculator": flashinfer_dispatch(runner.speculator.attn_groups),
        }
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        state["identity"] = {
            "rank": self.rank, "tp_rank": get_tensor_model_parallel_rank(),
            "local_rank": self.local_rank, "cuda_device": device,
            "uuid": str(properties.uuid), "device_name": properties.name, "pid": os.getpid(),
        }
        return state


def flashinfer_dispatch(attention_groups):
    dispatch = []
    for groups in attention_groups:
        for group in groups:
            if group.backend.get_name() != "FLASHINFER":
                continue
            for builder in group.metadata_builders:
                kernel = builder.flashinfer_trtllm_api_decode_kernel
                dispatch.append({
                    "layers": group.layer_names,
                    "prefill_dtype": str(builder.q_data_type_prefill),
                    "decode_dtype": str(builder.q_data_type_decode),
                    "kv_dtype": str(builder.kv_cache_dtype),
                    "decode_backend": kernel.value if kernel else "flashinfer-native",
                })
    return dispatch
