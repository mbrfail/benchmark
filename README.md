# HumanEval+ Benchmark Suite

Evaluating **Qwen3.6-27B** across 3 local backends (NVFP4/vLLM, GGUF Q4/llama.cpp, GGUF Q8/llama.cpp) vs **DeepSeek V4 Flash** (remote API) on the full [HumanEval+](https://github.com/evalplus/evalplus) test suite — 164 Python coding problems with edge-case test inputs. All local runs on RTX PRO 5000 (48 GB).

## Quick Summary

| Model | Quant | Context | Pass@1 | Avg Time | Tok/s | Cost |
|-------|:-----:|:-------:|:-----:|:--------:|:-----:|:----:|
| **— vLLM (local)** | | | | | | |
| Qwen3.6-27B | NVFP4 | 4,096 | **89.6%** | 1.98s | 58.4 | Free |
| Qwen3.6-27B | NVFP4 | 65,536 | **89.6%** | 5.05s | 20.9 | Free |
| Qwen3.6-27B | NVFP4 | 131,072 | **89.6%** | 5.55s | 20.7 | Free |
| **— llama.cpp (GGUF, MTP spec)** | | | | | | |
| Qwen3.6-27B | Q4_K_XL | 4,096 | **88.4%** | 1.22s | 80.6 | Free |
| Qwen3.6-27B | Q4_K_XL | 65,536 | **87.8%** | 1.17s | 80.1 | Free |
| Qwen3.6-27B | Q4_K_XL | 131,072 | **89.0%** | 1.17s | 81.5 | Free |
| Qwen3.6-27B | Q8_K_XL | 4,096 | **89.6%** | 1.48s | 65.8 | Free |
| Qwen3.6-27B | Q8_K_XL | 65,536 | **89.6%** | 1.52s | 65.1 | Free |
| Qwen3.6-27B | Q8_K_XL | 131,072 | **90.2%** | 1.47s | 65.8 | Free |
| Qwen3.6-27B | Q8_K_XL | 262,144 | **89.0%** | 1.52s | 65.5 | Free |
| **— Remote API** | | | | | | |
| DeepSeek V4 Flash | — | N/A | **87.8%** | 6.26s | 80.1 | API |

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

## Failure Categories

| Type | Description |
|------|-------------|
| Wrong answer | Model generated code that compiled but failed test assertions |
| Missing helper | Model referenced a helper function (e.g. `is_palindrome`) without defining it |
| Syntax/Indent error | Generated code had Python syntax errors |
| No function def | Model output explanation/markdown instead of code (or imports before `def`) |

## Running Yourself

```bash
# Local — vLLM (NVFP4)
python3 scripts/humaneval_v4.py

# Local — llama.cpp (GGUF + MTP, port 8081)
python3 scripts/humaneval_v3.py --gguf  # or humaneval_v4.py with GGUF backend

# DeepSeek (requires DEEPSEEK_API_KEY in ~/.hermes/.env)
python3 scripts/humaneval_deepseek.py
```

## Hardware

- **GPU**: NVIDIA RTX PRO 5000 Blackwell (48 GB HBM2e)
- **RAM**: 60 GB system
- **Models tested**:
  - Qwen3.6-27B-NVFP4 (vLLM, FP4 quantization, 64 layers, GQA 24:4, hidden=5120)
  - Qwen3.6-27B-UD-Q4_K_XL-MTP.gguf (llama.cpp, Q4 quantization)
  - Qwen3.6-27B-UD-Q8_K_XL.gguf (llama.cpp, Q8 quantization; confirmed MTP head present)
  - DeepSeek V4 Flash (remote API)
