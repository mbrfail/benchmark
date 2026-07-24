# Qwen3.6 Q4: mlx-lm vs vLLM-Metal on M5 Max 64 GB

Date: 2026-07-23

## Question

How do Qwen3.6-27B Q4 and Qwen3.6-35B-A3B Q4 behave on a 64 GB Apple Silicon MacBook when served through MLX and vLLM?

The requested machine was described as an M4 Max 64 GB. Live inspection of `192.168.0.130` identified it as **M5 Max 64 GB (Mac17,6)**; Tim explicitly approved proceeding on that host.

On macOS, upstream vLLM has no native Metal accelerator backend. The relevant implementation is the official/community-maintained **vLLM-Metal plugin**, so this experiment compares `mlx-lm` with `vllm-metal` using the same local MLX Q4 weights.

## Conclusion

**Use mlx-lm for both checkpoints on this host. Do not use the tested vLLM-Metal build for Qwen3.6 Q4 yet.**

- Both models served correctly through mlx-lm and produced coherent output.
- The 35B-A3B MoE is the clear performance winner: about **109.9 tok/s** short decode and **219.9 tok/s** aggregate at four simultaneous requests.
- Dense 27B reaches about **30.6 tok/s** short decode and **50.1 tok/s** aggregate at concurrency four.
- Cold long-prefill is the dense model's main weakness: median **15.95 s TTFT** for 9,967 prompt tokens versus **2.61 s** for 35B-A3B.
- Reusing an identical 9,938-token prefix lowers median TTFT to **0.211 s** (27B) and **0.137 s** (35B-A3B), demonstrating why cold and warm-prefix numbers must not be mixed.
- vLLM-Metal launched, returned `/v1/models`, and answered HTTP 200, but both models emitted deterministic mojibake/malformed fragments. These are **functional failures**, not benchmarkable inference.

## Tested artifacts

| Model | Architecture | Quantization | Indexed weight bytes |
|---|---|---:|---:|
| `lmstudio-community/Qwen3.6-27B-MLX-4bit` | `Qwen3_5ForConditionalGeneration` | affine Q4, group 64 | 16,054,262,240 |
| `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit` | `Qwen3_5MoeForConditionalGeneration` | affine Q4, group 64; selected gates Q8 | 20,401,929,952 |

Exact local shard hashes are in [model-identities.md](model-identities.md). The local LM Studio directories did not retain an unambiguous downloaded revision marker, so hashes—not repository-head assumptions—are the definitive artifact identity.

## Runtime matrix

| Engine | Key versions | Result |
|---|---|---|
| mlx-lm (LM Studio bundled Python) | MLX 0.31.2, mlx-lm 0.31.3, mlx-vlm 0.6.1 | Valid for both models |
| vLLM-Metal isolated venv | vLLM 0.25.1+cpu, vLLM-Metal `0.3.0.dev20260724043919`, MLX 0.32.0, mlx-lm 0.31.3 | Server works; inference output invalid for both models |

See [environment.json](environment.json) and [commands.sh](commands.sh).

## MLX results

All reported values are medians of five measured requests after one warmup unless stated otherwise. Temperature was 0 and thinking was disabled. The client ran on the target host over loopback.

| Metric | 27B dense Q4 | 35B-A3B MoE Q4 | Better |
|---|---:|---:|---|
| Short TTFT | 0.171 s | 0.127 s | 35B-A3B |
| Short decode | 30.57 tok/s | 109.90 tok/s | 35B-A3B (3.60×) |
| Code decode | 30.10 tok/s | 108.94 tok/s | 35B-A3B (3.62×) |
| Warm-prefix TTFT, 9,938 prompt tokens | 0.211 s | 0.137 s | 35B-A3B |
| Cold TTFT, 9,967 prompt tokens | 15.946 s | 2.614 s | 35B-A3B (6.10× faster) |
| Four-request aggregate throughput | 50.11 tok/s | 219.92 tok/s | 35B-A3B (4.39×) |

### Basic output checks

- Python code syntax: 5/5 valid for each model.
- Cold long-summary structure: 5/5 outputs contained exactly five numbered steps for each model.
- Raw response text is preserved in the MLX JSON files, not replaced by aggregate metrics.

### Memory observation

The harness sampled process-tree RSS and observed maxima around 11.6 GiB. This **must not be interpreted as total model memory**: safetensors are mmap-backed and Apple unified-memory accounting is not represented completely by process RSS. Power was not measured because `powermetrics` required an interactive sudo password. These are explicit limitations, not zero-power/low-memory claims.

## vLLM-Metal failure

### Observation

The official installer completed and produced a functioning OpenAI-compatible server. For 27B, the full harness received malformed outputs such as:

```text
�-nFl
 

P_click斜.预览全文_ier
```

For 35B-A3B, five deterministic probes asking for exactly `TOKENIZER_OK` instead returned the same malformed string and terminated at the 32-token limit. Raw responses are in:

- [raw/vllm_metal_27b.json](raw/vllm_metal_27b.json)
- [raw/vllm_metal_35b_a3b_functional_failure.json](raw/vllm_metal_35b_a3b_functional_failure.json)

### Isolation steps

1. vLLM-Metal logged that its ByteLevel tokenizer compatibility patch could not install during a circular vLLM import.
2. The same tokenizer loaded directly with Transformers decoded a multilingual round-trip correctly.
3. Direct `mlx_lm.generate` inside the vLLM-Metal venv, using the same model paths and exact-match prompt, returned `TOKENIZER_OK` for both models.
4. A wrapper in [launch_vllm_metal.py](launch_vllm_metal.py) deferred and successfully re-applied the ByteLevel compatibility patch after vLLM import.
5. vLLM-Metal still produced malformed output after the patch.

**Inference:** the tested failure is in the vLLM-Metal Qwen3.6 execution path (or its interaction with these Q4 layouts), not in the local checkpoint, tokenizer files, MLX itself, or simple detokenization. The 27B performance values in its raw file are retained only as failure evidence and must not be compared as valid model throughput.

## Method

Workloads:

1. Short explanation, maximum 192 output tokens.
2. Python coding task, maximum 256 output tokens.
3. Repeated long technical context, approximately 9,938 prompt tokens, maximum 192 output tokens.
4. Cache-busting long context: a unique nonce precedes the long body, yielding approximately 9,967 prompt tokens and preventing reuse of the long common prefix.
5. Three waves of four simultaneous short requests, maximum 128 output tokens each.

Streaming measurements distinguish:

- TTFT: request start to first non-empty content or reasoning delta.
- Total latency: request start to stream completion.
- Decode throughput: completion tokens divided by time from first visible token to stream completion.
- Aggregate concurrency throughput: total completion tokens in a wave divided by wall time.

The complete harness is [benchmark_endpoint.py](benchmark_endpoint.py); derived metrics and quality checks are regenerated by [summarize.py](summarize.py).

## Limitations

- This is an M5 Max result, not an M4 Max result.
- Only the two pre-existing LM Studio Q4 conversions were tested; the official BF16/FP8 checkpoints and other community Q4 conversions may behave differently in vLLM-Metal.
- vLLM-Metal was a current development build from the official installer, not a stable release pin.
- vLLM-Metal's Qwen3.6 support matrix says automatic prefix caching is unavailable, so cached-prefix behavior is not equivalent to mlx-lm.
- No academic accuracy suite was run. Quality validation here is functional/coherence-oriented: raw inspection, Python syntax, requested structure, and deterministic exact-match probes.
- No reliable total unified-memory or power measurement was captured.

## Files

- `benchmark_endpoint.py` — benchmark harness
- `summarize.py` / `summary.json` — reproducible derived metrics
- `commands.sh` — exact launch shapes
- `environment.json` — sanitized host/runtime metadata
- `model-identities.md` — exact local hashes and repository-head context
- `raw/*.json` — all per-request timings, usage, finish reasons, outputs, and observed RSS samples
