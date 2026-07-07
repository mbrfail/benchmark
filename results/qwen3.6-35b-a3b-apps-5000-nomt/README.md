# Qwen3.6-35B-A3B (Q4 MTP) — APPS Benchmark (5000 problems)

**Date:** 2026-07-07
**Runtime:** 14.9 hours
**Strategy:** No max_tokens (model generates until natural EOS)

## Results

| Metric | Value |
|---|---|
| **Pass@1** | **27.2%** (1361/5000) |
| **Total Time** | 14.9h (53,584s) |
| **Avg tok/s** | 249.0 |
| **Avg latency** | 10.67s/problem |

### By Difficulty

| Difficulty | Passed | Total | Pass Rate |
|---|---|---|---|
| Introductory | 424 | 1000 | **42.4%** |
| Interview | 852 | 3000 | **28.4%** |
| Competition | 85 | 1000 | **8.5%** |

### Failure Analysis

| Cause | Count |
|---|---|
| Other (failed test cases) | 3,447 |
| Timeout | 192 |

## Model

- **Model:** Qwen3.6-35B-A3B (MoE)
- **Quantization:** Q4_K_XL (24 GB VRAM)
- **Context:** 128K
- **Server:** llama.cpp (llama-server)
- **MTP:** Enabled
- **Reasoning:** Disabled (`--reasoning off`)
- **Serving:** Port 8081, ~269 tok/s
