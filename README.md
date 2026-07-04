# HumanEval+ Benchmark Suite

Evaluating **Qwen3.6-27B-NVFP4** (local, RTX PRO 5000 48GB) vs **DeepSeek V4 Flash** (remote API) on the full [HumanEval+](https://github.com/evalplus/evalplus) test suite — 164 Python coding problems with edge-case test inputs.

## Quick Summary

| Model | Context | Pass@1 | Avg Time | Tok/s | Cost |
|-------|:-------:|:-----:|:--------:|:-----:|:----:|
| Qwen3.6-27B (v4, 4K) | 4,096 | **89.6%** | 1.98s | 58.4 | Free |
| Qwen3.6-27B (v4, 128K) | 131,072 | **89.6%** | 5.55s | 20.7 | Free |
| DeepSeek V4 Flash | N/A | **87.8%** | 6.26s | 80.1 | API |

> **Note:** DeepSeek has 9 false "no function def" failures from not stripping imports before `def`. Adjusted score is ~93.3% (see results/deepseek/ for details). 128K context did not improve pass rate (problems are independent), but cost 2.8x in speed.

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
└── results/
    ├── local_4k/
    │   ├── 84p8_v3.json             # v3 with fixed imports, 4K context
    │   └── 89p6_v4_4k.json          # v4 with import stripping, 4K context (reconstructed)
    ├── local_128k/
    │   └── 89p6_v4_128k.json        # v4 at 128K context (15.2 min runtime)
    └── deepseek/
        ├── 86p0_first_run.json      # First DeepSeek run (512 max_tokens, truncated)
        └── 87p8_final.json          # Final DeepSeek run (2048 max_tokens)
```

## Methodology

- **Dataset**: [HumanEval+](https://github.com/evalplus/evalplus) — 164 Python problems with base_input + plus_input (edge cases)
- **Prompt**: System prompt + function signature/docstring. Model must generate complete `def` including any helper functions
- **Execution**: Generated code is exec'd against the problem's `check(candidate)` test function
- **Metrics**: Pass@1 (correctness), avg latency, tok/s, failure breakdown
- **Temperature**: 0.2 (deterministic)
- **Local config**: vLLM, NVFP4 quantization, enforce-eager, Qwen3.5 MTP speculation (3 tokens)

## Failure Categories

| Type | Description |
|------|-------------|
| Wrong answer | Model generated code that compiled but failed test assertions |
| Missing helper | Model referenced a helper function (e.g. `is_palindrome`) without defining it |
| Syntax/Indent error | Generated code had Python syntax errors |
| No function def | Model output explanation/markdown instead of code (or imports before `def`) |

## Running Yourself

```bash
# Local (requires vLLM with Qwen3.6-27B-NVFP4 loaded)
python3 scripts/humaneval_v4.py

# DeepSeek (requires DEEPSEEK_API_KEY in ~/.hermes/.env)
python3 scripts/humaneval_deepseek.py
```

## Hardware

- **GPU**: NVIDIA RTX PRO 5000 Blackwell (48 GB HBM2e)
- **RAM**: 60 GB system
- **Model**: Qwen3.6-27B-NVFP4 (NVFP4 quantization, 64 layers, GQA 24:4, hidden=5120)
