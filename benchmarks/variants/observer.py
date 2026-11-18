"""Read the live variant configuration from the isolated vLLM worker."""

from concurrent_observer import ConcurrentObserver


class VariantObserver(ConcurrentObserver):
    def variant_snapshot(self):
        state = self.concurrent_snapshot()
        config = self.vllm_config
        cache = config.cache_config
        speculative = config.speculative_config
        state["variant"] = {
            "block_size": cache.block_size,
            "mamba_block_size": cache.mamba_block_size,
            "mamba_cache_mode": cache.mamba_cache_mode,
            "speculative_method": speculative.method if speculative else None,
            "speculative_tokens": speculative.num_speculative_tokens if speculative else 0,
            "draft_architectures": (
                speculative.draft_model_config.hf_config.architectures if speculative else []
            ),
        }
        return state
