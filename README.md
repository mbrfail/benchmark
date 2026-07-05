# HumanEval+ Benchmark Suite

Evaluating **Qwen3.6-27B** across 3 local backends (NVFP4/vLLM, GGUF Q4/llama.cpp, GGUF Q8/llama.cpp) vs **DeepSeek V4 Flash** (remote API) on the full [HumanEval+](https://github.com/evalplus/evalplus) test suite — 164 Python coding problems with edge-case test inputs. All local runs on RTX PRO 5000 (48 GB).

## Quick Summary — Qwen3.6-35B-A3B (MoE, recommended daily driver)

| Quant | Ctx | MTP | Pass@1 | Speed | Time | VRAM |
|:-----:|:---:|:---:|:------:|:-----:|:----:|:----:|
| **Q4_K_XL** | **4K** | **✅** | 89.0% | **264 tok/s** | **1.9m** | 23.0 GB |
| **Q4_K_XL** | **64K** | **✅** | 88.4% | **269 tok/s** | **1.8m** | 23.9 GB |
| **Q4_K_XL** | **128K** | **✅** | **91.5%** | **268 tok/s** | **1.9m** | 23.9 GB |
| Q4_K_XL | 4K | ✗ | 89.6% | 170 tok/s | 2.8m | 23.0 GB |
| Q4_K_XL | 64K | ✗ | 90.2% | 170 tok/s | 2.8m | 23.9 GB |
| Q4_K_XL | 128K | ✗ | 90.2% | 170 tok/s | 2.8m | 23.9 GB |
| **Q8_K_XL** | **4K** | **✅** | 87.2% | **228 tok/s** | **1.9m** | 37.7 GB |
| **Q8_K_XL** | **64K** | **✅*** | **90.9%** | **231 tok/s** | **1.9m** | 38.2 GB |
| **Q8_K_XL** | **128K** | **✅*** | **90.9%** | **226 tok/s** | **1.9m** | 38.7 GB |
| Q8_K_XL | 4K | ✗ | 89.0% | 154 tok/s | 3.1m | 37.7 GB |
| Q8_K_XL | 64K | ✗* | 90.9% | 149 tok/s | 3.1m | 38.2 GB |
| Q8_K_XL | 128K | ✗* | 89.6% | 155 tok/s | 3.1m | 38.7 GB |

*\* Q8 64K/128K requires `--cache-type-k q4_0` to fit 48 GB VRAM.*

**🏆 Best overall: Q4_K_XL + MTP @ 128K** — 268 tok/s, 91.5% accuracy, 24 GB VRAM

## Qwen3.6-27B (Dense) — Legacy Reference

| Quant | Context | MTP | Pass@1 | Speed | Time |
|:-----:|:-------:|:---:|:------:|:-----:|:----:|
| Q4_K_XL | 4K | ✅ | 88.4% | 81 tok/s | 3.3m |
| Q4_K_XL | 64K | ✅ | 87.8% | 80 tok/s | 3.2m |
| Q4_K_XL | 128K | ✅ | 89.0% | 82 tok/s | 3.2m |
| Q8_K_XL | 4K | ✅ | 89.6% | 66 tok/s | 4.0m |
| Q8_K_XL | 64K | ✅ | 89.6% | 65 tok/s | 4.2m |
| Q8_K_XL | 128K | ✅ | **90.2%** | 66 tok/s | 4.0m |
| Q8_K_XL | 256K | ✅ | 89.0% | 66 tok/s | 4.1m |

> **Key insight:** GGUF MTP (speculative decoding) fully decouples generation speed from context length. Q8 GGUF achieves the same Pass@1 as NVFP4 vLLM (89.6%) at 3× the speed, regardless of context length — even at 256K. This makes it the optimal configuration for practical use.
>
> **Note:** DeepSeek has 9 false "no function def" failures from not stripping imports before `def`. Adjusted score is ~93.3% (see results/deepseek/ for details).

## Directory Structure

```
benchmark/
├── README.md
├── .gitignore
├── data/
│   └── humaneval_plus.json          # HumanEval+ dataset (164 problems)
├── scripts/
│   ├── humaneval_v1.py              # First version (buggy: stop seq truncation)
│   ├── humaneval_v2.py              # Fixed imports in exec context, removed stop seq
│   ├── humaneval_v3.py              # Unified benchmark (local + deepseek)
│   ├── humaneval_v4.py              # Clean version: import stripping + improved prompt
│   └── humaneval_deepseek.py        # DeepSeek-specific (2048 max_tokens for reasoning)
├── results/
│    ├── local_4k/
│    │   ├── 84p8_v3.json             # v3 with fixed imports, 4K context
│    │   └── 89p6_v4_4k.json          # v4 with import stripping, 4K context (reconstructed)
│    ├── local_64k/
│    │   └── 89p6_v4_64k.json         # v4 at 64K context (13.8 min)
│    ├── local_128k/
│    │   └── 89p6_v4_128k.json        # v4 at 128K context (15.2 min)
│    ├── gguf_4k/
│    │   └── 88p4_mtp_4k.json         # Q4 GGUF + MTP, 4K (3.3 min)
│    ├── gguf_64k/
│    │   └── 87p8_mtp_64k.json        # Q4 GGUF + MTP, 64K (3.2 min)
│    ├── gguf_128k/
│    │   └── 89p0_mtp_128k.json       # Q4 GGUF + MTP, 128K (3.2 min)
│    ├── q8_4k/
│    │   └── 89p6_mtp_4k.json         # Q8 GGUF + MTP, 4K (4.0 min)
│    ├── q8_64k/
│    │   └── 89p6_mtp_64k.json        # Q8 GGUF + MTP, 64K (4.2 min)
│    ├── q8_128k/
│    │   └── 90p2_mtp_128k.json       # Q8 GGUF + MTP, 128K (4.0 min)
│    ├── q8_256k/
│    │   └── 89p0_mtp_256k.json       # Q8 GGUF + MTP, 256K (4.1 min)
│    └── deepseek/
│        ├── 86p0_first_run.json      # First DeepSeek run (512 max_tokens, truncated)
│        └── 87p8_final.json          # Final DeepSeek run (2048 max_tokens)
```

## Methodology

- **Dataset**: [HumanEval+](https://github.com/evalplus/evalplus) — 164 Python problems with base_input + plus_input (edge cases)
- **Prompt**: System prompt + function signature/docstring. Model must generate complete `def` including any helper functions
- **Execution**: Generated code is exec'd against the problem's `check(candidate)` test function
- **Metrics**: Pass@1 (correctness), avg latency, tok/s, failure breakdown
- **Temperature**: 0.2 (deterministic)
- **Backends**:
  - **vLLM** (NVFP4): enforce-eager, Qwen3.5 MTP speculation (3 tokens)
  - **llama.cpp** (GGUF): MTP speculative decoding, q8_0 KV cache
  - **DeepSeek API**: remote, 2048 max_tokens

## Model Sources

All local models are publicly available on HuggingFace:

| Model | Format | HuggingFace Repo | File(s) | Size |
|-------|--------|------------------|---------|------|
| Qwen3.6-27B-NVFP4 | NVFP4 (compressed-tensors) | [unsloth/Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | `model.safetensors` | 24.6 GiB |
| Qwen3.6-27B-UD-Q4_K_XL-MTP | GGUF Q4_K_XL (MTP) | [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) | `Qwen3.6-27B-UD-Q4_K_XL.gguf` | 16.7 GiB |
| Qwen3.6-27B-UD-Q8_K_XL | GGUF Q8_K_XL (MTP-capable) | [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) | `Qwen3.6-27B-UD-Q8_K_XL.gguf` | 33.3 GiB |

> **MTP note:** Qwen3.6 has a built-in Multi-Token Prediction (MTP) head in its native architecture. GGUF exports from `unsloth/Qwen3.6-27B-MTP-GGUF` retain this head, while the standard `unsloth/Qwen3.6-27B-GGUF` repo strips it. The Q8_K_XL file from the MTP repo does **not** have `-MTP` in the filename but **does** include the MTP head (verified empirically). When in doubt, download from the MTP repo.

### Download commands

```bash
# NVFP4 (requires huggingface-hub)
huggingface-cli download unsloth/Qwen3.6-27B-NVFP4 --local-dir ./models/Qwen3.6-27B-NVFP4

# GGUF Q4 with MTP
huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF --local-dir ./models/gguf \
  --include "Qwen3.6-27B-UD-Q4_K_XL.gguf"

# GGUF Q8 with MTP
huggingface-cli download unsloth/Qwen3.6-27B-MTP-GGUF --local-dir ./models/gguf \
  --include "Qwen3.6-27B-UD-Q8_K_XL.gguf"
```

## Failure Categories

| Type | Description |
|------|-------------|
| Wrong answer | Model generated code that compiled but failed test assertions |
| Missing helper | Model referenced a helper function (e.g. `is_palindrome`) without defining it |
| Syntax/Indent error | Generated code had Python syntax errors |
| No function def | Model output explanation/markdown instead of code (or imports before `def`) |

## Server Setup

Both local backends require an OpenAI-compatible server to be running before you can execute the benchmark scripts.

### 1. vLLM (NVFP4) — port 8082

**Software:** vLLM 0.24.0, CUDA 13.1, FlashInfer (for SM120 FP4 GEMM)

```bash
# First-time load: kernel compilation (limit parallelism to avoid OOM)
NINJAJOBS=1 vllm serve unsloth/Qwen3.6-27B-NVFP4 \
  --port 8082 \
  --enforce-eager \
  --spec-method qwen3_5_mtp --spec-tokens 3 \
  --max-model-len <CONTEXT_LENGTH> \
  --gpu-memory-utilization <UTIL> \
  --dtype bfloat16 --trust-remote-code --seed 3407
```

| Context | `--max-model-len` | `--gpu-memory-utilization` | Notes |
|---------|:-----------------:|:--------------------------:|-------|
| 4K | 4096 | 0.75 | Room for MTP; ~58 tok/s |
| 64K | 65536 | 0.88 | Drop MTP (no VRAM); ~21 tok/s |
| 128K | 131072 | 0.98 | Near limit; ~21 tok/s |

> **Kernel compilation:** First load compiles ~17 CUTLASS `.cu` files via `ninja`. Set `NINJAJOBS=1` to prevent OOM. Subsequent loads use the cached kernels at `~/.cache/flashinfer/` and skip compilation.

### 2. llama.cpp (GGUF) — port 8081

**Software:** llama.cpp b9763 (commit `dec5ca557`), built with CUDA for GPU offload

```bash
# Start llama-server with GGUF model
llama-server -m ./models/gguf/Qwen3.6-27B-UD-<QUANT>.gguf \
  --port 8081 --host 0.0.0.0 \
  -ngl 99 -c 8192 \
  --reasoning off \
  -np 4 -cb \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -t <CPU_THREADS> -tb <CPU_THREADS>
```

**With MTP speculative decoding** (add these flags):
```bash
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-n-min 2 \
  --spec-draft-p-min 0.7
```

| Model File | Quant | MTP Support |
|------------|:-----:|:-----------:|
| `Qwen3.6-27B-UD-Q4_K_XL.gguf` | Q4_K_XL | ❌ No MTP |
| `Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf` | Q4_K_XL | ✅ Yes (filename has `-MTP`) |
| `Qwen3.6-27B-UD-Q8_K_XL.gguf` | Q8_K_XL | ✅ Yes (MTP head present despite filename) |

> **Key difference from vLLM:** llama.cpp's MTP speculative decoding fully decouples generation speed from context length — speed is constant regardless of whether the KV cache holds 4K or 256K tokens.

### 3. Run the benchmark

Before running, copy the dataset to the expected path:

```bash
cp data/humaneval_plus.json /tmp/humaneval_plus.json
```

Then:

```bash
# NVFP4 (requires vLLM on port 8082)
python3 scripts/humaneval_v4.py

# GGUF (requires llama-server on port 8081)
python3 scripts/humaneval_v3.py gguf

# DeepSeek (requires DEEPSEEK_API_KEY)
export DEEPSEEK_API_KEY="sk-..."
python3 scripts/humaneval_deepseek.py
```

> **Data path note:** All scripts read the dataset from `/tmp/humaneval_plus.json` by default — this is a known limitation. You may also edit the `problems = json.load(open(...))` lines in the scripts to point to `data/humaneval_plus.json` relative to the repo root.

## Hardware & Software

### Test Platform

| Component | Specification |
|-----------|--------------|
| **GPU** | NVIDIA RTX PRO 5000 Blackwell (48 GB HBM2e, Compute Capability 12.0) |
| **CPU** | Intel Core Ultra 7 265K (20 cores) |
| **RAM** | 60 GB system |
| **OS** | Linux (kernel 7.0.0-27-generic) |
| **CUDA** | Runtime 13.1, Driver 595.71.05 |
| **vLLM** | 0.24.0 |
| **llama.cpp** | b9763 (commit `dec5ca557`) |
| **Python** | 3.11.15 |

### Models Tested

| Model | Backend | Architecture | Details |
|-------|---------|-------------|---------|
| Qwen3.6-27B-NVFP4 | vLLM | qwen3_5, 64 layers, GQA 24:4, hidden=5120, FP4 quant | Vision encoder present (27 visual layers, 1152 hidden), multimodal |
| Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf | llama.cpp | Same architecture, Q4_K_XL quant | MTP head; no vision encoder (stripped in GGUF export) |
| Qwen3.6-27B-UD-Q8_K_XL.gguf | llama.cpp | Same architecture, Q8_K_XL quant | MTP head present despite filename; no vision encoder |
| DeepSeek V4 Flash | Remote API | Proprietary | — |

### Model Size & VRAM Usage at Different Context Windows

| Model | Model Size | 4K VRAM | 64K VRAM | 128K VRAM | 256K VRAM |
|-------|:----------:|:-------:|:--------:|:---------:|:---------:|
| NVFP4 (vLLM) | ~25 GiB | ~35 GiB | ~43 GiB | ~46 GiB | — |
| Q4 GGUF (llama.cpp) | ~17 GiB | ~18 GiB | ~21 GiB | ~24 GiB | — |
| Q8 GGUF (llama.cpp) | ~33 GiB | ~34 GiB | ~37 GiB | ~40 GiB | ~43 GiB |

---

## Apple M3 Ultra (macOS) Results

Benchmarks on **Mac Studio (Mac15,14)** — Apple M3 Ultra (28-core CPU, 60-core GPU, 256 GB unified memory, 800 GB/s bandwidth), macOS 26.5.1, llama.cpp with Metal GPU offload (ngl 99) + MTP speculative decoding (spec-draft-n-max=3).

### Results Table

| Model | Ctx | Pass@1 | Avg tok/s | Avg time | Total time |
|-------|:---:|:-----:|:---------:|:--------:|:----------:|
| **— llama.cpp GGUF (MTP) — M3 Ultra** |
| Qwen3.6-27B-UD-Q4_K_XL | 4K | **89.0%** | 20.2 | 5.01s | 13.7m |
| Qwen3.6-27B-UD-Q4_K_XL | 64K | **89.6%** | 20.1 | 4.82s | 13.2m |
| Qwen3.6-27B-UD-Q4_K_XL | 128K | **87.8%** | 20.2 | 5.01s | 13.7m |
| Qwen3.6-27B-UD-Q4_K_XL | 256K | **88.4%** | 20.3 | 5.12s | 14.0m |
| Qwen3.6-27B-UD-Q8_K_XL | 4K | **89.6%** | 21.6 | 4.19s | 11.4m |
| Qwen3.6-27B-UD-Q8_K_XL | 64K | **89.6%** | 21.7 | 4.30s | 11.7m |
| Qwen3.6-27B-UD-Q8_K_XL | 128K | **89.6%** | 21.7 | 4.53s | 12.4m |
| Qwen3.6-27B-UD-Q8_K_XL | 256K | **89.6%** | 21.6 | 4.39s | 12.0m |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 4K | **90.2%** | 93.9 | 1.77s | 4.8m |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 64K | **89.6%** | 93.5 | 1.81s | 5.0m |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 128K | **90.9%** | 92.9 | 1.77s | 4.8m |
| Qwen3.6-35B-A3B-UD-Q4_K_XL | 256K | **92.1%** | 93.0 | 1.82s | 5.0m |
| **— Remote API** |
| DeepSeek V4 Flash | N/A | **60.4%** | 71.7 | 4.76s | 13.0m |

### Analysis

- **Best overall: 35B-A3B MoE** achieves **92.1% Pass@1** at 256K context — the highest score across all platforms — at **93 tok/s**, 4.6× faster than the 27B dense model.
- **27B Q8 remarkably stable**: exactly 89.6% Pass@1 across all 4 context lengths (4K–256K).
- **MTP speculation decouples speed from context**: tok/s is virtually identical across 4K, 64K, 128K, and 256K for every model variant.
- **DeepSeek result is lower** due to the 27B-optimized system prompt causing format failures (50/164 "no function definition" errors from code not starting with `def`).

### Test Platform

| Component | Specification |
|-----------|--------------|
| **Model** | Mac Studio (Mac15,14) |
| **Chip** | Apple M3 Ultra — 28-core (20P + 8E) |
| **GPU** | 60-core Metal GPU |
| **Memory** | 256 GB LPDDR5 (800 GB/s) |
| **OS** | macOS 26.5.1 |
| **llama.cpp** | Latest (Metal backend, ngl 99) |
| **Python** | 3.9 (macOS system) |

### New Model: Qwen3.6-35B-A3B

The Qwen3.6-35B-A3B is a **Mixture-of-Experts (MoE)** variant of Qwen3.6: 35.5B total parameters with only ~3.5B activated per token (via 60 small experts). Despite the large model size, it runs at ~93 tok/s on M3 Ultra — 4.6× faster than the 27B dense model — because only ~3.5B weights participate in each forward pass.

| Model | Total Params | Active/token | File Size |
|-------|:-----------:|:------------:|:---------:|
| Qwen3.6-27B (dense) | 27.3B | 27.3B | 17 GB (Q4) / 33 GB (Q8) |
| Qwen3.6-35B-A3B (MoE) | 35.5B | ~3.5B | 21 GB (Q4) |

**Source:** [havenoammo/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/havenoammo/Qwen3.6-35B-A3B-MTP-GGUF) on HuggingFace.
