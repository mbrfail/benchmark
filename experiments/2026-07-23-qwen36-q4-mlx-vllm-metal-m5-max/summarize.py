#!/usr/bin/env python3
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "raw"


def med(d, workload, metric):
    return d["sequential"][workload]["summary"][metric]["median"]


def code_valid(text):
    match = re.search(r"```python\s*(.*?)```", text, re.S)
    code = match.group(1) if match else text
    try:
        compile(code, "<benchmark-output>", "exec")
        return True
    except SyntaxError:
        return False


mlx27 = json.loads((RAW / "mlx_27b.json").read_text())
mlx35 = json.loads((RAW / "mlx_35b_a3b.json").read_text())
v27 = json.loads((RAW / "vllm_metal_27b.json").read_text())
v35 = json.loads((RAW / "vllm_metal_35b_a3b_functional_failure.json").read_text())

summary = {
    "mlx": {},
    "vllm_metal": {
        "27b": {
            "valid_inference": False,
            "reason": "deterministic mojibake/malformed output; metrics retained only as failure evidence",
        },
        "35b_a3b": {
            "valid_inference": False,
            "reason": "five exact-match probes returned identical mojibake and length termination",
        },
    },
}
for label, d in (("27b", mlx27), ("35b_a3b", mlx35)):
    summary["mlx"][label] = {
        "short_ttft_s_median": med(d, "short", "ttft_s"),
        "short_decode_tok_s_median": med(d, "short", "decode_tok_s"),
        "code_decode_tok_s_median": med(d, "code", "decode_tok_s"),
        "warm_9938_token_ttft_s_median": med(d, "long_prefill", "ttft_s"),
        "cold_9967_token_ttft_s_median": med(d, "long_prefill_cold", "ttft_s"),
        "four_request_aggregate_tok_s_median": d["concurrency_4"]["summary"]["median_aggregate_tok_s"],
        "code_syntax_valid": sum(code_valid(r["output"]) for r in d["sequential"]["code"]["trials"]),
        "code_syntax_trials": len(d["sequential"]["code"]["trials"]),
        "five_step_long_outputs": sum(len(re.findall(r"(?m)^\d+\.", r["output"])) == 5 for r in d["sequential"]["long_prefill_cold"]["trials"]),
        "five_step_long_trials": len(d["sequential"]["long_prefill_cold"]["trials"]),
        "observed_process_rss_gib_max": d["rss_kib_summary"]["max"] / 1024 / 1024,
    }

(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
