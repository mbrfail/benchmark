# Qwen3.6-35B-A3B with vLLM-Metal on Apple M3 Ultra

Date: 2026-07-23/24

## Question

Can vLLM-Metal serve Qwen3.6-35B-A3B correctly on the M3 Ultra, and can its automatic prefix cache improve latency for long, stable agent prefixes?

## Conclusion

**The model can be served correctly through vLLM-Metal, but Automatic Prefix Cache cannot be used for Qwen3.6 on the tested current release. Therefore the proposed cache-on/cache-off performance benchmark has no valid cache-on condition and was intentionally not run.**

The working artifact is the official `Qwen/Qwen3.6-35B-A3B-FP8` checkpoint at revision `95a723d08a9490559dae23d0cff1d9466213d989`, not the tested LM Studio community MLX Q4/8-bit conversions.

- Official FP8 reached `/v1/models` and passed **11/11 correctness gates**.
- Thinking was disabled: successful responses contained visible content or structured tool calls and `reasoning: null`.
- The server emitted a valid `lookup_temperature` tool call with exactly `{"city":"Tokyo","unit":"celsius"}`.
- After a synthetic tool result, it answered: `The current temperature in Tokyo is 21°C.`
- Explicit `--enable-prefix-caching` failed before serving with a `NotImplementedError`: hybrid GDN recurrent state cannot be restored from KV blocks.
- Community MLX Q4 and 8-bit conversions launched and returned HTTP 200 but generated deterministic corrupt text; their apparent timings are invalid and are not benchmark results.

## Why no performance benchmark was run

The experiment plan made correctness and real cache activation hard gates. Correctness eventually passed with official FP8, but the cache activation gate failed authoritatively.

Current vLLM-Metal documents Qwen3.5/3.6 as hybrid SDPA + GDN and marks Automatic Prefix Cache `❌`. The live cache-enabled launch confirmed the source guard:

```text
NotImplementedError: Prefix caching and Mamba cache modes are not supported for
hybrid GDN models on Metal because GDN recurrent state cannot be restored from
KV blocks.
```

A faster second request without cache support would not demonstrate prefix-cache reuse. Reporting cache speedup from such requests would be invalid, so serial and concurrency performance phases were skipped by design.

## Valid configuration

| Component | Measured identity |
|---|---|
| Host | Mac Studio `Mac15,14` |
| SoC | Apple M3 Ultra, 28 CPU cores, 60-core GPU |
| Unified memory | 256 GB |
| OS | macOS 26.5.1, build 25F80 |
| Model | `Qwen/Qwen3.6-35B-A3B-FP8` |
| Model revision | `95a723d08a9490559dae23d0cff1d9466213d989` |
| Local regular-file bytes | 37,493,015,668 |
| vLLM | `0.25.1+cpu` |
| vLLM-Metal | `0.3.0.dev20260724043919` |
| Source revision | `e60573a1a9f48497ce4f30979083f675076a88fa` |
| MLX / mlx-lm / mlx-vlm | `0.32.0` / `0.31.3` / `0.6.4` |
| Transformers | `5.12.1` |
| Context / max sequences | 32,768 / 4 |
| Attention | vLLM-Metal paged hybrid SDPA + GDN |
| Metal memory fraction | 0.50 |
| Tool parser | `qwen3_xml` |
| Thinking | disabled through `chat_template_kwargs` |
| Prefix cache | unsupported and disabled |

Exact environment metadata is in [environment.json](environment.json), commands in [commands.sh](commands.sh), and checkpoint hashes in `raw/fp8_sha256sums.txt`.

## Correctness results

The harness is [functional_probe.py](functional_probe.py); complete requests and responses are in [raw/vllm_functional_fp8.jsonl](raw/vllm_functional_fp8.jsonl).

| Gate | Result | Selected evidence |
|---|---:|---|
| `/v1/models` | Pass | Correct FP8 path, max context 32,768 |
| Exact `TOKENIZER_OK` | 5/5 | All exact, `finish_reason=stop` |
| Short factual exact match | Pass | `PARIS` |
| Python syntax | Pass | `merge_intervals` parsed with `ast.parse` |
| Direct no-tool response | Pass | `4`, no tool call |
| Required tool selection | Pass | `lookup_temperature` |
| Tool arguments | Pass | Tokyo, Celsius; valid JSON and schema |
| Tool-result continuation | Pass | Correctly used returned value 21°C |
| Hidden reasoning disabled | Pass | `reasoning: null` in validated responses |
| Corrupt/replacement output | Pass | None in the valid configuration |

The request timings stored in this correctness file are not presented as a performance benchmark: they were not collected with the planned streaming, independent cold-cache sequences, or repeated statistical design.

## Failed artifact paths

### Community MLX Q4

`lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit` has exact hashes in [model-identity.md](model-identity.md). Direct `mlx_lm.generate` inside the vLLM environment returned exactly `TOKENIZER_OK`, proving the checkpoint, tokenizer, chat template, MLX runtime, and quantized weights were sound.

Through vLLM-Metal, however, five identical deterministic probes returned the same malformed output, including:

```text
京6/<think>able1own\/11生存+w
��保障性 uiltin Jersi 对这个觉得觉得觉得enter]SRenbergت&熟悉
```

The following controlled alternatives did not repair it:

1. default paged hybrid path;
2. deferred ByteLevel tokenizer compatibility patch after vLLM import;
3. non-paged MLX path;
4. community MLX 8-bit conversion.

The non-paged path also exposed an upstream scheduler-admission rounding issue. A one-line admission-only diagnostic temporarily reported two nominal sequence KV allocations instead of one so the server could start; it still produced corrupt output. The release package was restored before official-FP8 testing.

### Source-install failure and resolution

Running `install.sh` from the pinned source checkout installed Python dependencies and vLLM, but native-artifact compilation stopped because the host's Xcode could not load `CoreSimulator.framework`. Rather than repair unrelated Xcode components, the experiment installed the official release wheel, which contains prebuilt Metal artifacts. Its SHA-256 was verified as:

```text
f44e4f1fc07c7bd5a9fb2a215849df495e94f178c7afe2acbda632e965f64386
```

## Interpretation

### Observations

1. The official FP8 checkpoint works correctly through current vLLM-Metal on this M3 Ultra.
2. The tested community MLX Q4 and 8-bit artifacts do not work correctly through the same server.
3. Direct MLX generation from the Q4 artifact is correct.
4. Automatic Prefix Cache is rejected for this hybrid GDN architecture.

### Inference

The malformed community-artifact output is in vLLM-Metal's handling of those MLX conversion layouts, not in basic model capability or tokenization. The official FP8 artifact follows the exact loader path added and validated upstream, including per-expert FP8 dequantization and stacking.

The prefix-cache blocker is architectural in the current implementation: KV blocks alone are insufficient to reconstruct GDN recurrent state. It is not a tuning flag that can be enabled safely.

### Recommendation

- Use the official FP8 checkpoint if experimenting with Qwen3.6-35B-A3B on current vLLM-Metal.
- Do not use the tested community MLX Q4/8-bit conversions through this release.
- Do not choose Qwen3.6 on vLLM-Metal when automatic long-prefix reuse is the primary requirement.
- Revisit this benchmark only after vLLM-Metal adds explicit hybrid-GDN prefix-state restoration and removes the runtime guard.
- For prefix-cache experiments today, select an architecture marked APC-compatible in vLLM-Metal's support matrix, such as a paged GQA model, then run the planned agent-prefix workload unchanged.

## Limitations

- No cache performance values exist because the cache-enabled configuration cannot start.
- No academic accuracy suite was run; validation was functional and agent/tool oriented.
- Power was not measured.
- Correctness-request elapsed times are preserved but are not controlled benchmark statistics.
- The official FP8 checkpoint differs from the initially available community MLX Q4 artifact; this format change was necessary to obtain correct vLLM-Metal output and is explicitly labeled.
- The pre-existing MLX 8-bit endpoint on port 8081 remained running and untouched. Because no performance comparison was made, its resource use does not contaminate a reported speed result.

## Files

- `README.md` — question, conclusion, evidence, and limitations
- `environment.json` — host/runtime/configuration identity
- `commands.sh` — exact install and launch shapes
- `functional_probe.py` — deterministic correctness and tool-call harness
- `direct_mlx_probe.py` — direct MLX isolation probe
- `model-identity.md` — community Q4 hashes and identities
- `upstream-evidence.md` — exact upstream support and APC evidence
- `raw/vllm_functional_fp8.jsonl` — all valid requests and responses
- `raw/prefix_cache_rejection.log` — live cache-enabled failure
- `raw/vllm_functional_initial.jsonl` — initial Q4 corrupt output
- `raw/vllm_functional_bytelevel_patched.jsonl` — deferred-patch failure
- `raw/vllm_functional_nonpaged.jsonl` — non-paged failure
- `raw/vllm_functional_8bit.jsonl` — community 8-bit failure
- `raw/direct_mlx_q4.json` — successful direct MLX exact-match evidence
- `raw/fp8_server_success.log` — successful model-load, KV-allocation, warm-up, and startup log
- `raw/final_endpoint_verification.json` — final real `/v1/models` plus inference verification
- `raw/fp8_sha256sums.txt` — official FP8 local hashes
