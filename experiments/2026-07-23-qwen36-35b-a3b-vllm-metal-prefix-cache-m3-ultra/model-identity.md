# Model identity

## Functionally valid artifact

- Repository: `Qwen/Qwen3.6-35B-A3B-FP8`
- Pinned Hugging Face revision: `95a723d08a9490559dae23d0cff1d9466213d989`
- Path on target: `/Users/tenglong/models/Qwen3.6-35B-A3B-FP8`
- Architecture reported by `config.json`: `Qwen3_5MoeForConditionalGeneration`
- Model type: `qwen3_5_moe`
- Quantization: FP8, `e4m3`, dynamic activation scheme, 128×128 weight blocks
- Local regular-file bytes after download: 37,493,015,668
- Weight layout: 42 `layers-*.safetensors` shards plus `mtp.safetensors`
- Exact selected SHA-256 manifest: [raw/fp8_sha256sums.txt](raw/fp8_sha256sums.txt)

This is the exact artifact path for which merged upstream PR [vllm-project/vllm-metal#312](https://github.com/vllm-project/vllm-metal/pull/312) added FP8 dequantization and per-expert MoE stacking compatibility. It passed all correctness gates in this experiment.

## Invalid community MLX Q4 artifact

- Repository label: `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit`
- Path on target: `/Users/tenglong/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit`
- Indexed weight bytes: 20,401,929,952
- Result through direct `mlx_lm`: correct exact match
- Result through tested vLLM-Metal paths: deterministic corrupt output

### Q4 SHA-256

| File | SHA-256 |
|---|---|
| `config.json` | `a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767` |
| `tokenizer.json` | `87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4` |
| `chat_template.jinja` | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` |
| `model-00001-of-00004.safetensors` | `09f3e6ecb0b7af6e6a38bc8169a134c821b0924c2679b2bb8f4426ad38d032b8` |
| `model-00002-of-00004.safetensors` | `31dcdb1c49eebdb1505bd14e3cb33f9cf900bd2546b638f2464694ae763a033f` |
| `model-00003-of-00004.safetensors` | `3e66de06a1f03dade16a612a368cfce4a4c9caa4efd7d28185454384082cec03` |
| `model-00004-of-00004.safetensors` | `a5d0cf03519c26f8b506df6b0ba60526e5c08c8cea22d0c21ce92950e58a5422` |
| `model.safetensors.index.json` | `0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea` |

## Invalid community MLX 8-bit artifact

- Repository label: `lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit`
- It was already present on the target and separately served correctly by the existing persistent MLX endpoint on port 8081.
- A temporary vLLM-Metal instance on port 8094 reached readiness but failed the correctness harness with corrupt output.
- No artifact hashes were recomputed because this path was only a same-family layout/quantization diagnostic, not the final valid artifact.
