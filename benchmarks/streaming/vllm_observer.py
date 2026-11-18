class StreamingBaselineObserver:
    def baseline_snapshot(self):
        import torch
        import vllm

        config = self.vllm_config
        layers = self.get_model().language_model.model.layers
        gqa = {}
        gdn = {}
        for index, layer in enumerate(layers):
            if layer.layer_type == "full_attention":
                attention = layer.self_attn.attn
                backend = attention.get_attn_backend()
                scales = {}
                for name in ("q", "k", "v"):
                    scale = getattr(attention, f"_{name}_scale")
                    scales[name] = {
                        "tensor": scale.tolist(),
                        "dtype": str(scale.dtype),
                        "host": getattr(attention, f"_{name}_scale_float"),
                    }
                gqa[index] = {
                    "backend": backend.get_name(),
                    "cache_storage_dtype": str(attention.kv_cache.dtype),
                    "kernel_cache_dtype": str(
                        backend.get_dtype_for_flashinfer(attention.impl.kv_cache_dtype)
                    ),
                    "scales": scales,
                }
            else:
                attention = layer.linear_attn
                gdn[index] = {
                    "convolution_dtype": str(attention.kv_cache[0].dtype),
                    "recurrence_dtype": str(attention.kv_cache[1].dtype),
                    "prefill_backend": attention.gdn_prefill_backend,
                    "packed_recurrent_decode": attention.enable_packed_recurrent_decode,
                }
        return {
            "vllm": vllm.__version__,
            "vllm_file": vllm.__file__,
            "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda,
            "model_dtype": str(config.model_config.dtype),
            "cache": {
                "cache_dtype": config.cache_config.cache_dtype,
                "enable_prefix_caching": config.cache_config.enable_prefix_caching,
                "kv_cache_memory_bytes": config.cache_config.kv_cache_memory_bytes,
                "mamba_cache_dtype": config.cache_config.mamba_cache_dtype,
                "mamba_ssm_cache_dtype": config.cache_config.mamba_ssm_cache_dtype,
            },
            "attention": {
                "backend": config.attention_config.backend.name,
                "disable_flashinfer_q_quantization":
                    config.attention_config.disable_flashinfer_q_quantization,
            },
            "compilation": {
                "mode": config.compilation_config.mode.name,
                "cudagraph_mode": config.compilation_config.cudagraph_mode.name,
                "cudagraph_capture_sizes": config.compilation_config.cudagraph_capture_sizes,
            },
            "enforce_eager": config.model_config.enforce_eager,
            "speculative": config.speculative_config is not None,
            "gqa": gqa,
            "gdn": gdn,
        }
