#!/usr/bin/env python3
"""Repeatable OpenAI-compatible latency/throughput probe with raw outputs."""
from __future__ import annotations
import argparse, datetime, json, statistics, time, urllib.request

SYSTEM = "You are Qwen, a large language model created by Tongyi Lab team from Alibaba Group. You are a helpful assistant."
PROMPTS = {
    "code_short": "Write a Python function fibonacci(n) using iteration. Return only the code.",
    "explanation_512": "Explain how speculative decoding accelerates autoregressive language-model inference. Discuss draft acceptance, verification, latency, throughput, and failure modes. Be technically precise and use about 450 words.",
}

def post_json(url: str, payload: dict, timeout: int = 180) -> tuple[dict, float]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.load(response)
    return body, time.perf_counter() - start

def median(values):
    values = [x for x in values if x is not None]
    return statistics.median(values) if values else None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8091")
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--repetitions", type=int, default=5)
    args = ap.parse_args()
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/v1/models", timeout=30) as response:
        models = json.load(response)
    model = models["data"][0]["id"]
    result = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "label": args.label,
        "base_url": args.base_url,
        "model_list_response": models,
        "request_defaults": {"system": SYSTEM, "temperature": 0.0, "seed": 42},
        "warmup_runs": 1,
        "measured_runs": args.repetitions,
        "workloads": {},
    }
    for name, prompt in PROMPTS.items():
        max_tokens = 256 if name == "code_short" else 640
        runs = []
        for i in range(args.repetitions + 1):
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "temperature": 0.0,
                "seed": 42,
                "max_tokens": max_tokens,
            }
            body, elapsed = post_json(args.base_url.rstrip("/") + "/v1/chat/completions", payload)
            usage = body.get("usage", {})
            completion_tokens = usage.get("completion_tokens")
            timings = body.get("timings") or body.get("choices", [{}])[0].get("timings")
            runs.append({
                "warmup": i == 0,
                "elapsed_s": elapsed,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": completion_tokens,
                "e2e_completion_tok_s": completion_tokens / elapsed if completion_tokens else None,
                "finish_reason": body.get("choices", [{}])[0].get("finish_reason"),
                "timings": timings,
                "content": body.get("choices", [{}])[0].get("message", {}).get("content"),
                "reasoning_content": body.get("choices", [{}])[0].get("message", {}).get("reasoning_content"),
            })
            print(f"{args.label} {name} run={i} elapsed={elapsed:.3f}s tokens={completion_tokens}", flush=True)
        measured = runs[1:]
        result["workloads"][name] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "runs": runs,
            "summary": {
                "median_elapsed_s": median([r["elapsed_s"] for r in measured]),
                "min_elapsed_s": min(r["elapsed_s"] for r in measured),
                "max_elapsed_s": max(r["elapsed_s"] for r in measured),
                "median_e2e_completion_tok_s": median([r["e2e_completion_tok_s"] for r in measured]),
                "completion_tokens": sorted(set(r["completion_tokens"] for r in measured)),
            },
        }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

if __name__ == "__main__":
    main()
