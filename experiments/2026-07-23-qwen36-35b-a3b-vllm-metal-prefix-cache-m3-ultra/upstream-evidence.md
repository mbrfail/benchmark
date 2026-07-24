# Upstream compatibility evidence

Inspected on 2026-07-23 at vLLM-Metal commit `e60573a1a9f48497ce4f30979083f675076a88fa` (release `v0.3.0.dev20260724043919`).

## Qwen3.6 execution support

`docs/supported_models.md` lists Qwen3.5/3.6 as supported with a hybrid SDPA + GDN linear-attention kernel; Qwen3.6 additionally uses MoE.

Merged PR [vllm-project/vllm-metal#312](https://github.com/vllm-project/vllm-metal/pull/312) added loader compatibility for the exact official artifact `Qwen/Qwen3.6-35B-A3B-FP8`. The PR reports coherent generation on an M3 Max with 128 GB using hybrid SDPA + GDN paged KV. It explains that the official FP8 artifact stores expert tensors separately and needs vLLM-Metal's dequantization and expert-stacking compatibility transform.

## Prefix-cache limitation

The same support matrix marks Automatic Prefix Cache as unavailable (`❌`) for Qwen3.5/3.6 and other hybrid GDN models.

The selected source revision enforces this in `vllm_metal/platform.py`:

```python
if model_config is not None and model_config.is_hybrid:
    cache_config = vllm_config.cache_config
    if cache_config.enable_prefix_caching or cache_config.mamba_cache_mode != "none":
        raise NotImplementedError(
            "Prefix caching and Mamba cache modes are not supported for "
            "hybrid GDN models on Metal because GDN recurrent state cannot "
            "be restored from KV blocks."
        )
```

This is an architectural state-restoration limitation, not merely a disabled default. A cache benchmark is valid only if a later tested runtime removes this guard and supplies trustworthy GDN-state reuse.
