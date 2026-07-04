#!/usr/bin/env python3
"""
HumanEval+ benchmark — DeepSeek V4 Flash only.
DeepSeek uses reasoning tokens, so max_tokens is set higher (2048)
and the response may include reasoning_content alongside content.
"""
import json, time, requests, traceback, sys, os, re, textwrap

# Load API key
api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not api_key:
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            if "DEEPSEEK_API_KEY" in line:
                api_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                break

if not api_key:
    print("ERROR: No DeepSeek API key found")
    sys.exit(1)

CONFIG = {
    "name": "DeepSeek V4 Flash",
    "url": "https://api.deepseek.com/v1/chat/completions",
    "model": "deepseek-v4-flash",
    "headers": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    "max_tokens": 2048,
}

problems = json.load(open("/tmp/humaneval_plus.json"))
print(f"Loaded {len(problems)} HumanEval+ problems for {CONFIG['name']}")

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
    "Return ONLY the raw Python code -- no explanations, no markdown formatting, "
    "no backticks, no docstrings. Start directly with 'def'. "
    "Include ALL helper functions your solution needs. Just the raw code, nothing else."
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
                "max_tokens": CONFIG["max_tokens"],
                "temperature": 0.2,
            },
            headers=CONFIG["headers"],
            timeout=120
        )

        latency = time.time() - start

        if resp.status_code != 200:
            results.append({"task_id": task_id, "status": "error", "error": f"HTTP {resp.status_code}"})
            failed += 1
            continue

        data = resp.json()
        msg = data["choices"][0]["message"]
        generated = msg.get("content", "") or ""
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)

        # Strip markdown fences
        generated = re.sub(r'^```(?:python)?\s*\n?', '', generated)
        generated = re.sub(r'\n?```\s*$', '', generated)
        generated = generated.strip()

        # Check for function definition
        if not generated.startswith("def ") and not generated.startswith("async def "):
            results.append({"task_id": task_id, "status": "fail", "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "latency": round(latency, 2), "tok_s": 0, "error": "No function definition generated", "generated_preview": generated[:200]})
            failed += 1
            continue

        # Build and exec test harness
        harness = f"{COMMON_IMPORTS}\n{generated}\n\n{test_code}\n\ncheck({entry_point})"

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
            "generated_preview": generated[:150],
            "error": error_msg[:300] if error_msg else None
        })

        if (idx + 1) % 10 == 0 or not test_passed:
            elapsed = time.time() - total_start
            marker = "PASS" if test_passed else "FAIL"
            print(f"  [{idx+1:3d}/{len(problems)}] {marker:4s}  {task_id:20s}  {latency:.2f}s  {completion_tokens:3d}tok  {tok_s:5.1f}t/s  ({elapsed:.0f}s)")

    except requests.Timeout:
        results.append({"task_id": task_id, "status": "timeout", "error": "Timed out"})
        failed += 1
    except Exception as e:
        results.append({"task_id": task_id, "status": "crash", "error": str(e)[:200]})
        failed += 1

total_time = time.time() - total_start
total = passed + failed

# Summary
print(f"\n{'='*70}")
print(f"  {CONFIG['name']} -- {total} problems")
print("="*70)
print(f"  Pass@1:              {passed}/{total} = {100*passed/total:.1f}%")
print(f"  Failed:              {failed}")
print(f"  Total time:          {total_time:.0f}s ({total_time/60:.1f}m)")
print(f"  Avg per problem:     {total_time/total:.2f}s")

tok_s_list = [r["tok_s"] for r in results if r.get("tok_s", 0) > 0]
if tok_s_list:
    avg_tok = sum(tok_s_list)/len(tok_s_list)
    print(f"\n  -- Generation Speed --")
    print(f"  Avg tok/s:          {avg_tok:.1f}")
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

completion_toks = [r["completion_tokens"] for r in results if r.get("completion_tokens")]
prompt_toks = [r["prompt_tokens"] for r in results if r.get("prompt_tokens")]
if completion_toks:
    print(f"\n  -- Tokens --")
    print(f"  Avg prompt tok:     {sum(prompt_toks)/len(prompt_toks):.0f}")
    print(f"  Avg completion:     {sum(completion_toks)/len(completion_toks):.0f}")

# Failure breakdown
failed_results = [r for r in results if r["status"] != "pass"]
if failed_results:
    print(f"\n{'='*70}")
    print(f"  FAILURES ({len(failed_results)}):")
    print("="*70)
    for r in failed_results:
        err = r.get("error", "") or ""
        gen = r.get("generated_preview", "") or ""
        for line in err.split("\n"):
            ls = line.strip()
            if any(x in ls for x in ["Error:", "Error", "Exception:", "assert ", "NameError", "SyntaxError", "IndentationError"]):
                print(f"  {r['task_id']}: {ls[:150]}")
                break
        else:
            print(f"  {r['task_id']}: {err[:150]}")

# Save
outfile = "/tmp/humaneval_plus_deepseek.json"
with open(outfile, "w") as f:
    json.dump({
        "target": CONFIG["name"],
        "model": CONFIG["model"],
        "summary": {
            "passed": passed, "failed": failed, "total": total,
            "pass_rate": round(100*passed/total, 1),
            "total_time_seconds": round(total_time, 1),
            "avg_time_per_problem": round(total_time/total, 2),
            "avg_tok_s": round(sum(tok_s_list)/len(tok_s_list), 1) if tok_s_list else 0,
            "avg_latency_s": round(sum(lats)/len(lats), 2) if lats else 0,
            "avg_completion_tokens": round(sum(completion_toks)/len(completion_toks), 1) if completion_toks else 0
        },
        "results": results
    }, f, indent=2)

print(f"\nResults saved to {outfile}")
