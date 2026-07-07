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

## Failure Reproduction & Model Comparison

**30 benchmark failures retested with Local (Qwen3.6-35B Q4 MTP) and DeepSeek v4-flash**

| Model | PASS count | Rate |
|---|---|---|
| Local (Qwen3.6-35B) | 15/30 | 50% |
| DeepSeek v4-flash | 19/30 | 63% |
| Both fail (truly hard) | 10/30 | 33% |
| At least one passes | 20/30 | 67% |

**DeepSeek better on:** 5 problems, **Local better on:** 1 problem

This shows that ~67% of benchmark failures are solvable by at least one model, indicating significant stochastic variance in pass@1. A pass@5 evaluation would yield a more accurate estimate of true capability.

Full per-problem comparison: `local_vs_deepseek_comparison.json`
