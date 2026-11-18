from benchmarks.streaming.vllm_observer import StreamingBaselineObserver


class MediaObserver(StreamingBaselineObserver):
    def media_snapshot(self):
        state = self.baseline_snapshot()
        model = self.get_model()
        vision = model.visual
        config = self.vllm_config
        weight = model.language_model.model.layers[3].self_attn.qkv_proj
        state["media"] = {
            "model_class": type(model).__name__,
            "vision_class": type(vision).__name__,
            "vision_dtype": str(vision.dtype),
            "vision_attention": str(vision.attn_backend),
            "vision_depth": len(vision.blocks),
            "vision_patch_weight_dtype": str(vision.patch_embed.proj.weight.dtype),
            "gqa_projection_weight_dtype": str(weight.weight.dtype),
            "gqa_projection_quant_method": type(weight.quant_method).__name__,
            "mm_processor_cache_gb": config.model_config.multimodal_config.mm_processor_cache_gb,
            "max_model_len": config.model_config.max_model_len,
            "max_num_seqs": config.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        }
        return state
