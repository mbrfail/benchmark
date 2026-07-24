# Local artifact identities

Current Hugging Face repository heads observed on 2026-07-23:

- `lmstudio-community/Qwen3.6-27B-MLX-4bit`: `bd83f6fe15b171f1549475db2348389c0f541c21`
- `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit`: `0c4a20a6437ae5985ddc9eb1a3f122ee6c151c3b`

The pre-existing LM Studio directories did not retain an unambiguous downloaded revision marker. The hashes below are therefore the exact local artifact provenance.

## Qwen3.6-27B-MLX-4bit

- Architecture: `Qwen3_5ForConditionalGeneration`; `model_type=qwen3_5`
- Quantization: affine 4-bit, group size 64
- Indexed tensor bytes: 16,054,262,240

```text
ede24666ac51e6d5ab948a8a1e6c72fc6effd941ba3aabb6dd942eb517c78043  config.json
2689680915661f040c50c35244d08b336def279e509e1ca11873f8dd1b0e7ce0  model-00001-of-00003.safetensors
a46183727b2c5cd16613fd8395f5e9e7cc4ad679644558cc87c9406fe334463d  model-00002-of-00003.safetensors
ac23bf70b1f239a040921d6f93770d74176fd435dbf44e42317053d06c68d702  model-00003-of-00003.safetensors
13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8  model.safetensors.index.json
87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4  tokenizer.json
672488283cdbf3530ecd2e3f90da54f9998cbae6befb5b32877590f72c7a9b2c  tokenizer_config.json
```

## Qwen3.6-35B-A3B-MLX-4bit

- Architecture: `Qwen3_5MoeForConditionalGeneration`; `model_type=qwen3_5_moe`
- Quantization: affine 4-bit, group size 64, with selected gates at 8-bit
- Indexed tensor bytes: 20,401,929,952

```text
a822a9e48b0aafbe3144ec37d4fb067e178ed96615ce6e4420b3149893cc5767  config.json
09f3e6ecb0b7af6e6a38bc8169a134c821b0924c2679b2bb8f4426ad38d032b8  model-00001-of-00004.safetensors
31dcdb1c49eebdb1505bd14e3cb33f9cf900bd2546b638f2464694ae763a033f  model-00002-of-00004.safetensors
3e66de06a1f03dade16a612a368cfce4a4c9caa4efd7d28185454384082cec03  model-00003-of-00004.safetensors
a5d0cf03519c26f8b506df6b0ba60526e5c08c8cea22d0c21ce92950e58a5422  model-00004-of-00004.safetensors
0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea  model.safetensors.index.json
87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4  tokenizer.json
672488283cdbf3530ecd2e3f90da54f9998cbae6befb5b32877590f72c7a9b2c  tokenizer_config.json
```
