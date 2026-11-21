"""Observe the paired vLLM worker's resolved engine configuration."""

from importlib.metadata import version

from benchmarks.streaming.vllm_observer import StreamingBaselineObserver


class PairedBaselineObserver(StreamingBaselineObserver):
    def baseline_snapshot(self):
        state = super().baseline_snapshot()
        config = self.vllm_config
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
