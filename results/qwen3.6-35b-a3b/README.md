# Qwen3.6-35B-A3B — HumanEval+ Benchmark Results

**Model**: Qwen3.6-35B-A3B (MoE, 256 experts, top-k=8, hybrid attention, ~3B active params)
**Hardware**: RTX PRO 5000 (48 GB VRAM) on CUDA 13.2
**Benchmark**: HumanEval+ (164 problems, temperature=0.2)

## GGUF Results (llama.cpp, commit dec5ca557)

| Quant | Context | Pass@1 | Speed | Time | VRAM | Notes |
|:-----:|:-------:|:-----:|:-----:|:----:|:----:|:------|
| **Q4_K_XL** | 4K | 89.6% | 170.5 tok/s | 2.8m | 23.0 GB | |
| **Q4_K_XL** | 64K | **90.2%** | 170.1 tok/s | 2.8m | 23.9 GB | |
| **Q4_K_XL** | 128K | **90.2%** | 169.7 tok/s | 2.8m | 23.9 GB | |
| **Q8_K_XL** | 4K | 89.0% | 154.1 tok/s | 3.1m | 37.7 GB | KV cache q8_0 |
| **Q8_K_XL** | 64K | **90.9%** | 149.4 tok/s | 3.1m | 38.2 GB | KV cache q4_0 (OOM w/ q8_0) |
| **Q8_K_XL** | 128K | 89.6% | 154.6 tok/s | 3.1m | 38.7 GB | KV cache q4_0 |

## NVFP4 (vLLM 0.24.0)

**NOT AVAILABLE** — vLLM + flashinfer's fused_moe kernel compilation fails on Blackwell (sm120). nvcc 13.2 internal compiler error during sm120 CUTLASS MoE kernel JIT compilation. Requires vLLM/flashinfer update with proper Blackwell support.

## Key Findings

1. **Speed is context-independent** — Q4 maintains ~170 tok/s across all context sizes
2. **Q4 vs Q8 accuracy similar** — 89.6%–90.2% vs 89.0%–90.9% (within stochastic noise at temp=0.2)
3. **MTP speculative decoding BROKEN** — MoE model outputs go to `reasoning_content` field, leaving `content` empty. All MTP runs skipped.
4. **Q8 64K OOM** with q8_0 KV cache; switched to q4_0 cache to fit in 48 GB
5. **VRAM usage**: Q4 ~23–24 GB, Q8 ~37–39 GB (good fits for 48 GB card)
6. **Compared to 27B dense model** (~65 tok/s with MTP): MoE is 2.6× faster (170 vs 65 tok/s) with similar accuracy (~90%)
