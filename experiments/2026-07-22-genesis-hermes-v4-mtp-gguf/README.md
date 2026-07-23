# Genesis Hermes V4 MTP-APEX GGUF on Apple M3 Ultra

## Conclusion

`burningfeet/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V4-MTP-GGUF` runs successfully in llama.cpp on the 256 GB M3 Ultra Mac Studio at 128K context. The selected full `MTP-APEX` file loaded, answered `/v1/models`, completed repeated inference, and finished all 164 HumanEval+ tasks both with and without MTP.

**Recommendation:** use MTP for short coding/agent workloads, but do not assume it improves every workload. In this experiment MTP improved HumanEval+ average generation speed by **23.6%** and reduced suite time by **21.1%**, while Pass@1 changed from **88.4% to 87.8%** (one fewer pass). It accelerated the fixed 76-token coding response by **29.8%**, but slowed a low-acceptance long-form response by **9.3%** in throughput. MTP should therefore remain workload-tested rather than universally enabled.

This was an isolated experiment on port 8091. It did **not** replace the persistent MLX service on port 8081 or change the homelab catalog.

## Question

1. Does the model load and serve correctly on the M3 Ultra with llama.cpp at the publisher's minimum recommended 128K context?
2. What does embedded MTP change when the exact same GGUF, prompt, context, sampling settings, and runtime are held constant?
3. Is coding quality competitive with the existing Qwen3.6 results?

## Selected model

| Item | Value |
|---|---|
| Repository | `burningfeet/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V4-MTP-GGUF` |
| Revision | `0d6ad5b1ce57a25e10ae2d506b8450da076193ee` |
| File | `Hermes3.6-35B-A3B-Uncensored-Genesis-V4-MTP-APEX.gguf` |
| Size | 26,522,914,240 bytes |
| SHA-256 | `ed31eb5c800e10a2132f3d37fb34cf1fca1a17a5214fa9dd7baa4a887f3064d0` |
| Selection reason | Publisher recommends APEX or Q8_K_P; the full MTP-APEX package permits MTP on/off testing with one target file |

The repository README describes the model as a reconstruction based on `HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` plus a Hermes-agent finetune. Those provenance and “repair” claims are publisher claims, not independently validated here. Exact source metadata is captured in `model-manifest.json`.

## Environment

| Component | Measured identity |
|---|---|
| Host | Teng’s Mac Studio, `Mac15,14` |
| SoC | Apple M3 Ultra, 28 CPU cores (20P + 8E), 60-core GPU |
| Unified memory | 256 GB |
| OS | macOS 26.5.1, build 25F80 |
| Runtime | llama.cpp `9430` (`d48a56eff`) |
| Build | AppleClang 21.0.0.21000099, Darwin arm64 |
| Backend | Metal, `-ngl 99` |
| Context | 131,072 tokens, one slot |
| KV cache | K `q8_0`, V `q8_0` |
| Flash attention | On |
| Chat template | GGUF Jinja template; thinking/reasoning off |
| Batch settings | `-b 2048 -ub 4096`, continuous batching |

The model advertises 262,144 training context; this experiment intentionally used 131,072 because the publisher said to keep at least 128K. The server correctly logged that the configured context does not exercise the full 262K capacity.

## Controlled configurations

The only meaningful launch-variable change was MTP:

```bash
# Common
/opt/homebrew/bin/llama-server \
  -m /path/to/Hermes3.6-35B-A3B-Uncensored-Genesis-V4-MTP-APEX.gguf \
  --host 127.0.0.1 --port 8091 \
  -c 131072 -np 1 -ngl 99 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --cont-batching \
  --jinja --reasoning off \
  -b 2048 -ub 4096 --metrics

# MTP addition
--spec-type draft-mtp --spec-draft-n-max 3
```

Each latency workload used one warm-up followed by five measured repetitions. Requests used temperature 0, seed 42, identical system/user prompts, and a single request at a time. HumanEval+ used all 164 tasks, temperature 0.2, and `max_tokens=512`, matching the repository's established harness.

## Results

### Repeated fixed prompts

| Workload | MTP | Completion | Median elapsed | Median e2e completion tok/s | Spread (min–max elapsed) |
|---|:---:|---:|---:|---:|---:|
| Short Python function | Off | 76 tokens | 1.070 s | 71.0 | 1.066–1.075 s |
| Short Python function | On | 76 tokens | 0.825 s | 92.1 | 0.824–0.831 s |
| Long explanation | Off | 605 tokens | 7.901 s | 76.6 | 7.892–7.925 s |
| Long explanation | On | 640 tokens | 9.213 s | 69.5 | 9.191–9.220 s |

The short workload is directly comparable and improved **29.8%** with MTP. The long workload is not equal-token end-to-end latency because no-MTP stopped naturally at 605 tokens while MTP reached the 640-token cap. Its token throughput still declined **9.3%**.

### MTP acceptance

| Workload | Requests | Median acceptance | Range |
|---|---:|---:|---:|
| Short Python function | 6 | 95.0% | 95.0–95.0% |
| Long explanation | 6 | 49.4% | 49.4–49.4% |
| HumanEval+ | 164 | 88.8% | 64.8–100% |

This explains the workload dependence: high draft acceptance accelerated short code, while verification overhead outweighed useful drafting on the 49.4%-acceptance long response.

### HumanEval+ quality and throughput

| Configuration | Passed | Pass@1 | Avg tok/s | Avg latency | Total time |
|---|---:|---:|---:|---:|---:|
| No MTP | 145/164 | **88.4%** | 67.9 | 2.44 s | 400.5 s |
| MTP | 144/164 | **87.8%** | **83.9** | **1.93 s** | **315.9 s** |

Observed MTP deltas:

- average generation throughput: **+23.6%**;
- total suite time: **−21.1%**;
- Pass@1: **−0.6 percentage points**, exactly one additional failure.

The output sets were not identical despite controlled prompts and sampling. Treat the one-problem quality delta as an observation, not proof that MTP intrinsically reduces quality; one deterministic suite run does not estimate variance across seeds or runtime revisions.

### Memory and startup

| Configuration | Model-load time from server log | Post-run RSS |
|---|---:|---:|
| No MTP | 1.838 s | 35.15 GiB |
| MTP | 2.196 s | 35.76 GiB |

MTP added approximately **0.61 GiB** measured process RSS after the run. llama.cpp separately estimated 2,228 MiB for the MTP context during initialization; that estimate and process-RSS delta measure different things and should not be conflated.

### Existing persistent MLX reference

The existing `Qwen3.6-35B-A3B-MLX-8bit` service produced the same 76-token short-code response at a median **66.8 e2e tok/s**. Genesis MTP-APEX measured 71.0 tok/s without MTP and 92.1 tok/s with MTP. This is an operational reference only—not a controlled model-quality comparison—because weights, format, and runtime differ.

## Functional and operational verification

- Correct GGUF size and SHA-256 verified before loading.
- `GET /v1/models` succeeded in both modes; raw responses are preserved.
- Repeated chat completions succeeded in both modes.
- llama.cpp logged creation of the embedded MTP draft context and per-request acceptance statistics.
- Full HumanEval+ completed in both modes without server crashes.
- Experimental server was stopped; port 8091 was verified closed.
- Existing persistent endpoint on port 8081 was rechecked after the experiment: `/v1/models` succeeded and inference returned `PERSISTENT_OK`.
- A temporary experiment-only `caffeinate` process was removed.

During acquisition, the Mac Studio entered system sleep after the Xet download despite the model processes later resuming normally. ICMP remained reachable while TCP ports were unavailable. A Wake-on-LAN packet restored access, and a bounded experiment-only `caffeinate` assertion was used for the runs. This is recorded as an operational event; it was not an inference crash.

## Limitations

- Only 128K was configured; no near-limit 128K prefill, 262K context, or long-context recall test was run.
- Concurrency was one. Continuous batching and multi-client throughput were not characterized.
- No power measurement was available.
- Vision was not tested; the separate ~899 MB `mmproj` was not downloaded.
- The uncensored/refusal behavior implied by the model name and publisher card was not independently benchmarked in this first deployment experiment.
- HumanEval+ was run once per mode. The 0.6-point difference is one task and may not generalize.
- Long-form MTP and no-MTP outputs differed in length, limiting direct end-to-end latency comparison for that workload.
- The model was not promoted to a persistent service, assigned a launch agent, exposed through Caddy, or added to the homelab catalog.

## Artifacts

- `benchmark_endpoint.py` — repeated latency/throughput probe with raw outputs
- `humaneval_genesis.py` — endpoint-configurable HumanEval+ harness
- `run_remote_config.sh` — exact remote launch and collection workflow
- `model-manifest.json` — source revision, file identities, sizes, hashes
- `summary.json` — machine-readable comparison
- `raw/benchmark-*.json` — every repeated response and timing
- `raw/humaneval-plus-*.json` — all 164 task outcomes per mode
- `raw/server-*.log` — full llama.cpp startup, timing, and MTP acceptance evidence
- `raw/metadata-*.txt` — hardware/runtime/hash/launch command/RSS
- `raw/metrics-*.prom` — final Prometheus snapshots
