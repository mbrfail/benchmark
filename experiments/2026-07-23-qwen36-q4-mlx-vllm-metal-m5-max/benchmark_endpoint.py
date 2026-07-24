#!/usr/bin/env python3
"""Reproducible OpenAI-compatible streaming benchmark for MLX/vLLM-Metal."""
import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

SHORT = "Explain why prefix caching helps an LLM server. Use exactly three concise bullet points."
CODE = "Write a Python function named merge_intervals(intervals) that returns merged overlapping intervals. Include type hints and only one short example."
LONG_UNIT = (
    "A retrieval service receives documents, normalizes Unicode, splits text into passages, "
    "creates embeddings, and stores vectors with source metadata. Queries use hybrid lexical "
    "and vector retrieval, reciprocal-rank fusion, reranking, and citation validation. "
)
LONG = "Read the repeated technical context, then summarize its pipeline in five numbered steps.\n\n" + LONG_UNIT * 220
WORKLOADS = {
    "short": (SHORT, 192),
    "code": (CODE, 256),
    "long_prefill": (LONG, 192),
}


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post_stream(base, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer directly and accurately."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first = None
    chunks = []
    usage = {}
    finish_reason = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        status = resp.status
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                if (text or reasoning) and first is None:
                    first = time.perf_counter()
                if text:
                    chunks.append(text)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    end = time.perf_counter()
    completion = usage.get("completion_tokens")
    decode_s = (end - first) if first else None
    return {
        "http_status": status,
        "ttft_s": (first - start) if first else None,
        "total_s": end - start,
        "decode_s": decode_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "decode_tok_s": (completion / decode_s) if completion and decode_s and decode_s > 0 else None,
        "finish_reason": finish_reason,
        "output": "".join(chunks),
    }


def descendants_rss(root_pid):
    try:
        text = subprocess.check_output(["ps", "-axo", "ppid=,pid=,rss="], text=True)
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 3:
                rows.append(tuple(map(int, parts)))
        alive = {root_pid}
        changed = True
        while changed:
            changed = False
            for ppid, pid, _ in rows:
                if ppid in alive and pid not in alive:
                    alive.add(pid); changed = True
        return sum(rss for _, pid, rss in rows if pid in alive)
    except Exception:
        return None


def median_summary(rows):
    out = {}
    for key in ("ttft_s", "total_s", "decode_tok_s", "prompt_tokens", "completion_tokens"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            out[key] = {
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8088/v1")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--repetitions", type=int, default=5)
    ap.add_argument("--server-pid", type=int)
    args = ap.parse_args()

    models = get_json(args.base.rstrip("/") + "/models")["data"]
    model_id = models[0]["id"]
    result = {
        "schema_version": 1,
        "engine": args.engine,
        "model_label": args.model_label,
        "served_model_id": model_id,
        "base_url": args.base,
        "sampling": {"temperature": 0.0, "thinking": False},
        "repetitions": args.repetitions,
        "sequential": {},
        "concurrency_4": {},
        "rss_kib_samples": [],
    }
    stop = threading.Event()
    def sample_rss():
        while not stop.is_set():
            if args.server_pid:
                x = descendants_rss(args.server_pid)
                if x is not None:
                    result["rss_kib_samples"].append({"t": time.time(), "rss_kib": x})
            stop.wait(0.25)
    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        for name, (prompt, max_tokens) in WORKLOADS.items():
            warmup = post_stream(args.base, model_id, prompt, min(max_tokens, 64))
            rows = []
            for i in range(args.repetitions):
                r = post_stream(args.base, model_id, prompt, max_tokens)
                r["trial"] = i + 1
                rows.append(r)
                print(name, i + 1, json.dumps({k: r[k] for k in ("ttft_s","total_s","prompt_tokens","completion_tokens","decode_tok_s")}), flush=True)
            result["sequential"][name] = {"cache_state": "warm after one identical warmup", "warmup": warmup, "trials": rows, "summary": median_summary(rows)}

        # Put a unique token sequence at the start so the long common body cannot
        # match a previously cached prefix. This measures uncached long prefill.
        cold_rows = []
        for i in range(args.repetitions):
            nonce = f"Cache-busting trial {i+1}-{time.time_ns()} begins here. "
            r = post_stream(args.base, model_id, nonce + LONG, 192)
            r["trial"] = i + 1
            cold_rows.append(r)
            print("long_prefill_cold", i + 1, json.dumps({k: r[k] for k in ("ttft_s","total_s","prompt_tokens","completion_tokens","decode_tok_s")}), flush=True)
        result["sequential"]["long_prefill_cold"] = {
            "cache_state": "cache-busting nonce before long body; no identical-prefix warmup",
            "trials": cold_rows,
            "summary": median_summary(cold_rows),
        }

        # Three independent waves of four simultaneous short requests.
        waves = []
        for wave in range(3):
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                futs = [ex.submit(post_stream, args.base, model_id, SHORT + f" Request {j+1}.", 128) for j in range(4)]
                rows = [f.result() for f in futs]
            wall = time.perf_counter() - t0
            tokens = sum(r.get("completion_tokens") or 0 for r in rows)
            waves.append({"wave": wave + 1, "wall_s": wall, "aggregate_completion_tokens": tokens,
                          "aggregate_tok_s": tokens / wall if wall else None, "requests": rows})
            print("concurrency_4", wave + 1, wall, tokens / wall if wall else None, flush=True)
        result["concurrency_4"] = {
            "waves": waves,
            "summary": {
                "median_wall_s": statistics.median(w["wall_s"] for w in waves),
                "median_aggregate_tok_s": statistics.median(w["aggregate_tok_s"] for w in waves),
            },
        }
    finally:
        stop.set(); sampler.join(timeout=2)
    if result["rss_kib_samples"]:
        vals = [x["rss_kib"] for x in result["rss_kib_samples"]]
        result["rss_kib_summary"] = {"min": min(vals), "max": max(vals), "median": statistics.median(vals)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("WROTE", out)

if __name__ == "__main__":
    main()
