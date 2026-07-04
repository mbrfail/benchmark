#!/usr/bin/env python3
"""Full HumanEval+ benchmark: all 164 problems against Qwen3.6-27B-NVFP4 on vLLM."""
import json, time, requests, traceback, sys, re

SERVER = "http://localhost:8082/v1/chat/completions"

problems = json.load(open("/tmp/humaneval_plus.json"))
print(f"Loaded {len(problems)} HumanEval+ problems")

passed = 0
failed = 0
results = []
total_start = time.time()
sorted_keys = sorted(problems.keys(), key=lambda k: int(k.split("/")[1]))

# Imports the generated code may need
COMMON_IMPORTS = (
    "from typing import List, Tuple, Dict, Set, Optional, Union, Any\n"
    "import math\nimport json\nimport re\nimport collections\nimport itertools\n"
    "import functools\nimport random\nimport statistics\n"
)

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
                "model": "/home/tenglong/models/Qwen3.6-27B-NVFP4",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Python programmer. Complete the following function. "
                            "Return ONLY the raw Python function definition — no explanations, "
                            "no markdown formatting, no backticks. Start directly with 'def'."
                        )
                    },
                    {"role": "user", "content": f"Complete the function:\n\n{prompt}"}
                ],
                "max_tokens": 512,
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

        # If model only generated the body (indented lines), wrap it in a def
        if not generated.startswith("def ") and not generated.startswith("async def "):
            generated = f"def {entry_point}(*args, **kwargs):\n    pass"

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

        exec_time = time.time() - exec_start
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
            "generated_len": len(generated),
            "generated_preview": generated[:150],
            "error": error_msg[:300] if error_msg else None
        })

        if (idx + 1) % 10 == 0 or not test_passed:
            elapsed = time.time() - total_start
            marker = "✅" if test_passed else "❌"
            print(f"  [{idx+1:3d}/{len(problems)}] {marker} {task_id:20s}  {latency:.2f}s  {completion_tokens:3d} tok  {tok_s:5.1f} tok/s  ({elapsed:.0f}s elapsed)")

    except Exception as e:
        results.append({"task_id": task_id, "status": "crash", "error": str(e)[:200]})
        failed += 1

total_time = time.time() - total_start
total = passed + failed

# Summary
print("\n" + "=" * 70)
print(f"  HUMANEVAL+ FULL SUITE — {total} problems")
print("=" * 70)
print(f"  Pass@1:    {passed}/{total} = {100 * passed / total:.1f}%")
print(f"  Failed:    {failed}")
print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f}m)")
print(f"  Avg per problem: {total_time / total:.2f}s")

tok_s_list = [r["tok_s"] for r in results if r.get("tok_s", 0) > 0]
if tok_s_list:
    print(f"\n  --- Generation Speed ---")
    print(f"  Avg tok/s:     {sum(tok_s_list) / len(tok_s_list):.1f}")
    print(f"  Median tok/s:  {sorted(tok_s_list)[len(tok_s_list)//2]:.1f}")
    print(f"  Min tok/s:     {min(tok_s_list):.1f}")
    print(f"  Max tok/s:     {max(tok_s_list):.1f}")

lats = [r["latency"] for r in results if r.get("latency")]
if lats:
    print(f"\n  --- Latency ---")
    print(f"  Avg:      {sum(lats) / len(lats):.2f}s")
    print(f"  Median:   {sorted(lats)[len(lats)//2]:.2f}s")
    print(f"  Min:      {min(lats):.2f}s")
    print(f"  Max:      {max(lats):.2f}s")

completion_toks = [r["completion_tokens"] for r in results if r.get("completion_tokens")]
prompt_toks = [r["prompt_tokens"] for r in results if r.get("prompt_tokens")]
if completion_toks:
    print(f"\n  --- Tokens per Problem ---")
    print(f"  Avg prompt tok:    {sum(prompt_toks) / len(prompt_toks):.0f}")
    print(f"  Avg completion:    {sum(completion_toks) / len(completion_toks):.0f}")

failed_results = [r for r in results if r["status"] != "pass"]
if failed_results:
    print(f"\n{'=' * 70}")
    print(f"  FAILURES ({len(failed_results)}):")
    print("=" * 70)
    for r in failed_results:
        err = r.get("error", "") or ""
        gen = r.get("generated_preview", "") or ""
        print(f"  {r['task_id']} ({r['completion_tokens']} tok): {err[:150]}")
        if r['completion_tokens'] < 20:
            print(f"    Generated: {gen[:100]}")
        print()

with open("/tmp/humaneval_plus_results.json", "w") as f:
    json.dump({
        "summary": {
            "passed": passed,
            "failed": failed,
            "total": total,
            "pass_rate": round(100 * passed / total, 1),
            "total_time_seconds": round(total_time, 1),
            "avg_time_per_problem": round(total_time / total, 2)
        },
        "speed": {
            "avg_tok_s": round(sum(tok_s_list) / len(tok_s_list), 1) if tok_s_list else 0,
            "avg_latency_s": round(sum(lats) / len(lats), 2) if lats else 0
        },
        "results": results
    }, f, indent=2)

print(f"\n✅ Results saved to /tmp/humaneval_plus_results.json")
