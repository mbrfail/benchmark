#!/usr/bin/env python3
"""
HumanEval+ benchmark v4 — local Qwen3.6-27B-NVFP4
Fixes: accepts imports before 'def', higher max_tokens, better prompt.
"""
import json, time, requests, traceback, sys, os, re

SERVER = "http://localhost:8082/v1/chat/completions"
MODEL = "/home/tenglong/models/Qwen3.6-27B-NVFP4"

problems = json.load(open("/tmp/humaneval_plus.json"))
print(f"🧪 Qwen3.6-27B-NVFP4 (local, v4)")
print(f"📦 Loaded {len(problems)} HumanEval+ problems")

passed = 0
failed = 0
results = []
total_start = time.time()
sorted_keys = sorted(problems.keys(), key=lambda k: int(k.split("/")[1]))

COMMON_IMPORTS = (
    "from typing import List, Tuple, Dict, Set, Optional, Union, Any\n"
    "import math\nimport json\nimport re\nimport collections\nimport itertools\n"
    "import functools\nimport random\nimport statistics\n"
)

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Complete the following function. "
    "Return ONLY the raw Python function definition — no explanations, "
    "no markdown formatting, no backticks, no docstrings. "
    "Start directly with 'def '. Do NOT include any import statements — "
    "imports are already available. "
    "If your solution needs a helper function, DEFINE it in your response "
    "before the main function. Include every function the solution needs."
)

def strip_imports(code: str) -> str:
    """Remove leading import/from lines so we can find the first 'def'."""
    lines = code.split("\n")
    clean = []
    started = False
    for line in lines:
        if line.startswith("def ") or line.startswith("async def "):
            started = True
        if started:
            clean.append(line)
    if started:
        return "\n".join(clean)
    # If no def found at all, return as-is
    return code

for idx, task_id in enumerate(sorted_keys):
    prob = problems[task_id]
    entry_point = prob["entry_point"]
    prompt = prob["prompt"]
    test_code = prob["test"]

    start = time.time()
    try:
        resp = requests.post(
            SERVER,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Complete the function:\n\n{prompt}"}
                ],
                "max_tokens": 768,
                "temperature": 0.2,
            },
            timeout=60
        )

        latency = time.time() - start

        if resp.status_code != 200:
            results.append({"task_id": task_id, "status": "error", "error": f"HTTP {resp.status_code}"})
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

        # Strip leading imports/lines before 'def'
        clean_code = strip_imports(generated)

        if not clean_code.startswith("def ") and not clean_code.startswith("async def "):
            results.append({"task_id": task_id, "status": "fail", "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "latency": round(latency, 2), "tok_s": 0, "error": "No function definition generated", "generated_preview": generated[:150]})
            failed += 1
            continue

        # Build harness: common imports + clean code + test
        harness = f"{COMMON_IMPORTS}\n{clean_code}\n\n{test_code}\n\ncheck({entry_point})"

        exec_globals = {}
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
            "generated_preview": clean_code[:150],
            "error": error_msg[:300] if error_msg else None
        })

        if (idx + 1) % 10 == 0 or not test_passed:
            elapsed = time.time() - total_start
            marker = "PASS" if test_passed else "FAIL"
            print(f"  [{idx+1:3d}/{len(problems)}] {marker:4s}  {task_id:20s}  {latency:.2f}s  {completion_tokens:3d}tok  {tok_s:5.1f}t/s  ({elapsed:.0f}s)")

    except Exception as e:
        results.append({"task_id": task_id, "status": "crash", "error": str(e)[:200]})
        failed += 1

total_time = time.time() - total_start
total = passed + failed

print(f"\n{'='*70}")
print(f"  Qwen3.6-27B-NVFP4 (local, v4) — {total} problems")
print("="*70)
print(f"  Pass@1:              {passed}/{total} = {100*passed/total:.1f}%")
print(f"  Failed:              {failed}")
print(f"  Total time:          {total_time:.0f}s ({total_time/60:.1f}m)")
print(f"  Avg per problem:     {total_time/total:.2f}s")

tok_s_list = [r["tok_s"] for r in results if r.get("tok_s", 0) > 0]
if tok_s_list:
    print(f"\n  -- Generation Speed --")
    print(f"  Avg tok/s:          {sum(tok_s_list)/len(tok_s_list):.1f}")
    print(f"  Median tok/s:       {sorted(tok_s_list)[len(tok_s_list)//2]:.1f}")
    print(f"  Min tok/s:          {min(tok_s_list):.1f}")
    print(f"  Max tok/s:          {max(tok_s_list):.1f}")

lats = [r["latency"] for r in results if r.get("latency")]
if lats:
    print(f"\n  -- Latency --")
    print(f"  Avg:                {sum(lats)/len(lats):.2f}s")
    print(f"  Median:             {sorted(lats)[len(lats)//2]:.2f}s")
    print(f"  Min:                {min(lats):.2f}s")
    print(f"  Max:                {max(lats):.2f}s")

# Failure breakdown
failed_results = [r for r in results if r["status"] != "pass"]
if failed_results:
    print(f"\n{'='*70}")
    print(f"  FAILURE BREAKDOWN ({len(failed_results)}):")
    print("="*70)
    cats = {"Wrong answer": 0, "Missing helper": 0, "Syntax/Indent": 0, "No def": 0, "Other": 0}
    for r in failed_results:
        err = r.get("error","") or ""
        if "SyntaxError" in err or "IndentationError" in err:
            cats["Syntax/Indent"] += 1
        elif "NameError" in err:
            cats["Missing helper"] += 1
        elif "No function" in r.get("error","") or "No function" in str(r.get("generated_preview","")):
            cats["No def"] += 1
        elif "AssertionError" in err or "assert" in err.lower():
            cats["Wrong answer"] += 1
        else:
            cats["Other"] += 1
    for k, v in cats.items():
        if v: print(f"  o {k}: {v}")
    for r in failed_results:
        err = r.get("error","") or ""
        for line in err.split("\n"):
            ls = line.strip()
            if any(x in ls for x in ["Error:", "Error", "Exception:", "assert ", "NameError", "SyntaxError", "IndentationError"]):
                print(f"  {r['task_id']}: {ls[:150]}")
                break
        else:
            print(f"  {r['task_id']}: {err[:150]}")

outfile = "/tmp/humaneval_plus_local_v4.json"
with open(outfile, "w") as f:
    json.dump({
        "target": "Qwen3.6-27B-NVFP4 (local, v4)",
        "summary": {
            "passed": passed, "failed": failed, "total": total,
            "pass_rate": round(100*passed/total, 1),
            "total_time_seconds": round(total_time, 1),
            "avg_time_per_problem": round(total_time/total, 2),
            "avg_tok_s": round(sum(tok_s_list)/len(tok_s_list), 1) if tok_s_list else 0,
            "avg_latency_s": round(sum(lats)/len(lats), 2) if lats else 0,
        },
        "results": results
    }, f, indent=2)
print(f"\nResults saved to {outfile}")
