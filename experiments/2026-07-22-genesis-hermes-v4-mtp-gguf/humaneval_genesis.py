#!/usr/bin/env python3
"""
Unified HumanEval+ benchmark — run against any OpenAI-compatible API.
Usage:
  python3 humaneval_v3.py local     # Qwen3.6-27B-NVFP4 on local vLLM
  python3 humaneval_v3.py deepseek  # DeepSeek V4 Flash
"""
import json, time, requests, traceback, sys, os, re

TARGET = (sys.argv[1] if len(sys.argv) > 1 else "genesis").strip().lower()

if TARGET == "deepseek":
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    if not API_KEY:
        # Try loading from .env
        env_path = os.path.expanduser("~/.hermes/.env")
        if os.path.exists(env_path):
            for line in open(env_path):
                if "DEEPSEEK_API_KEY" in line:
                    API_KEY = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    break
    CONFIG = {
        "name": "DeepSeek V4 Flash",
        "url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-v4-flash",
        "headers": {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    }
elif TARGET == "genesis":
    CONFIG = {
        "name": os.environ.get("BENCHMARK_LABEL", "Genesis Hermes V4 MTP-APEX GGUF"),
        "url": os.environ.get("OPENAI_CHAT_URL", "http://127.0.0.1:8091/v1/chat/completions"),
        "model": os.environ.get("OPENAI_MODEL", "gguf"),
        "headers": {"Content-Type": "application/json"},
    }
else:
    CONFIG = {
        "name": "Qwen3.6-35B-A3B-NVFP4 (local)",
        "url": "http://localhost:8082/v1/chat/completions",
        "model": "/home/tenglong/models/Qwen3.6-35B-A3B-NVFP4",
        "headers": {"Content-Type": "application/json"},
    }

problems = json.load(open(os.environ.get("HUMANEVAL_DATASET", "/tmp/humaneval_plus.json")))
print(f"🧪 {CONFIG['name']}")
print(f"📦 Loaded {len(problems)} HumanEval+ problems")

passed = 0
failed = 0
results = []
total_start = time.time()
sorted_keys = sorted(problems.keys(), key=lambda k: int(k.split("/")[1]))

# Common imports the generated code may need
COMMON_IMPORTS = (
    "from typing import List, Tuple, Dict, Set, Optional, Union, Any\n"
    "import math\nimport json\nimport re\nimport collections\nimport itertools\n"
    "import functools\nimport random\nimport statistics\n"
)

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the following function. "
    "Return ONLY the raw Python code — no explanations, no markdown formatting, "
    "no backticks, no docstrings. Start directly with 'def'. "
    "Include ALL helper functions your solution needs — don't leave references "
    "to undefined functions. Just the raw code, nothing else."
)

for idx, task_id in enumerate(sorted_keys):
    prob = problems[task_id]
    entry_point = prob["entry_point"]
    prompt = prob["prompt"]
    test_code = prob["test"]

    start = time.time()

    try:
        resp = requests.post(
            CONFIG["url"],
            json={
                "model": CONFIG["model"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Complete the function:\n\n{prompt}"}
                ],
                "max_tokens": 512,
                "temperature": 0.2,
            },
            headers=CONFIG["headers"],
            timeout=120
        )

        latency = time.time() - start

        if resp.status_code != 200:
            results.append({"task_id": task_id, "status": "error", "error": f"HTTP {resp.status_code}: {resp.text[:100]}"})
            failed += 1
            continue

        data = resp.json()
        generated = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)

        # Strip markdown fences
        generated = re.sub(r'^```(?:python)?\s*\n?', '', generated)
        generated = re.sub(r'\n?```\s*$', '', generated)
        generated = generated.strip()

        # If no def found, it's a failure
        if not generated.startswith("def ") and not generated.startswith("async def "):
            results.append({"task_id": task_id, "status": "fail", "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "latency": round(latency, 2), "tok_s": 0, "error": "No function definition generated", "generated_preview": generated[:150]})
            failed += 1
            continue

        # Build harness: common imports + generated code + test
        harness = f"{COMMON_IMPORTS}\n{generated}\n\n{test_code}\n\ncheck({entry_point})"

        exec_globals = {}
        exec_start = time.time()
        try:
            exec(harness, exec_globals)
            test_passed = True
            error_msg = None
        except Exception as e:
            test_passed = False
            error_msg = traceback.format_exc()

        tok_s = round(completion_tokens / latency, 1) if latency > 0 and completion_tokens > 0 else 0

        if test_passed:
            passed += 1
        else:
            failed += 1

        results.append({
            "task_id": task_id,
            "entry_point": entry_point,
            "status": "pass" if test_passed else "fail",
            "latency": round(latency, 2),
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "tok_s": tok_s,
            "generated_preview": generated[:150],
            "error": error_msg[:300] if error_msg else None
        })

        if (idx + 1) % 10 == 0 or not test_passed:
            elapsed = time.time() - total_start
            marker = "✅" if test_passed else "❌"
            print(f"  [{idx+1:3d}/{len(problems)}] {marker} {task_id:20s}  {latency:.2f}s  {completion_tokens:3d}tok  {tok_s:5.1f}t/s  ({elapsed:.0f}s)")

    except requests.Timeout:
        results.append({"task_id": task_id, "status": "timeout", "error": "Request timed out"})
        failed += 1
    except Exception as e:
        results.append({"task_id": task_id, "status": "crash", "error": str(e)[:200]})
        failed += 1

total_time = time.time() - total_start
total = passed + failed

# Summary
print(f"\n{'=' * 70}")
print(f"  {CONFIG['name']} — {total} problems")
print("=" * 70)
print(f"  Pass@1:              {passed}/{total} = {100 * passed / total:.1f}%")
print(f"  Failed:              {failed}")
print(f"  Total time:          {total_time:.0f}s ({total_time / 60:.1f}m)")
print(f"  Avg per problem:     {total_time / total:.2f}s")

tok_s_list = [r["tok_s"] for r in results if r.get("tok_s", 0) > 0]
if tok_s_list:
    print(f"\n  ── Generation Speed ──")
    print(f"  Avg tok/s:          {sum(tok_s_list) / len(tok_s_list):.1f}")
    print(f"  Median tok/s:       {sorted(tok_s_list)[len(tok_s_list)//2]:.1f}")
    print(f"  Min tok/s:          {min(tok_s_list):.1f}")
    print(f"  Max tok/s:          {max(tok_s_list):.1f}")

lats = [r["latency"] for r in results if r.get("latency")]
if lats:
    print(f"\n  ── Latency ──")
    print(f"  Avg:                {sum(lats) / len(lats):.2f}s")
    print(f"  Median:             {sorted(lats)[len(lats)//2]:.2f}s")
    print(f"  Min:                {min(lats):.2f}s")
    print(f"  Max:                {max(lats):.2f}s")

completion_toks = [r["completion_tokens"] for r in results if r.get("completion_tokens")]
prompt_toks = [r["prompt_tokens"] for r in results if r.get("prompt_tokens")]
if completion_toks:
    print(f"\n  ── Tokens ──")
    print(f"  Avg prompt tok:     {sum(prompt_toks) / len(prompt_toks):.0f}")
    print(f"  Avg completion:     {sum(completion_toks) / len(completion_toks):.0f}")

# Failure breakdown
failed_results = [r for r in results if r["status"] != "pass"]
if failed_results:
    print(f"\n{'=' * 70}")
    print(f"  FAILURE BREAKDOWN ({len(failed_results)}):")
    print("=" * 70)

    # Categorize
    categories = {"Wrong answer": 0, "NameError (missing helper)": 0, "Syntax/Indent error": 0, "No function def": 0, "Timeout": 0, "Other": 0}
    for r in failed_results:
        err = r.get("error", "") or ""
        if "SyntaxError" in err or "IndentationError" in err or "invalid syntax" in err:
            categories["Syntax/Indent error"] += 1
        elif "NameError" in err:
            categories["NameError (missing helper)"] += 1
        elif "No function definition" in err:
            categories["No function def"] += 1
        elif "timeout" in r.get("status", ""):
            categories["Timeout"] += 1
        elif "AssertionError" in err or "assert" in err.lower():
            categories["Wrong answer"] += 1
        else:
            categories["Other"] += 1

    for k, v in categories.items():
        if v > 0:
            print(f"  • {k}: {v}")

    for r in failed_results:
        err = r.get("error", "") or ""
        gen = r.get("generated_preview", "") or ""
        ct = r.get("completion_tokens", 0)
        print(f"\n  {r['task_id']} ({ct} tok): ", end="")
        # Show first line of error
        for line in err.split("\n"):
            line = line.strip()
            if any(x in line for x in ["Error:", "Error", "Exception:", "assert "]):
                print(f"{line[:150]}")
                break
            if "NameError" in line:
                print(f"{line[:150]}")
                break
        else:
            print(f"{err[:150]}")

# Save
outfile = os.environ.get("HUMANEVAL_OUTPUT", f"/tmp/humaneval_plus_{TARGET}.json")
with open(outfile, "w") as f:
    json.dump({
        "target": CONFIG["name"],
        "model": CONFIG["model"],
        "summary": {
            "passed": passed,
            "failed": failed,
            "total": total,
            "pass_rate": round(100 * passed / total, 1),
            "total_time_seconds": round(total_time, 1),
            "avg_time_per_problem": round(total_time / total, 2),
            "avg_tok_s": round(sum(tok_s_list) / len(tok_s_list), 1) if tok_s_list else 0,
            "avg_latency_s": round(sum(lats) / len(lats), 2) if lats else 0,
            "avg_completion_tokens": round(sum(completion_toks) / len(completion_toks), 1) if completion_toks else 0
        },
        "results": results
    }, f, indent=2)

print(f"\n✅ Results saved to {outfile}")
