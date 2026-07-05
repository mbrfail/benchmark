#!/usr/bin/env python3
"""
Automated MoE benchmark runner for Qwen3.6-35B-A3B.
Manages llama-server lifecycle, runs benchmarks for each config, saves results.
"""
import subprocess, sys, os, time, json, signal
from pathlib import Path

LLAMA_DIR = os.path.expanduser("~/llama.cpp/build-cuda")
MODEL_Q4 = "/mnt/data/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_Q8 = "/mnt/data/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
SCRIPT = os.path.expanduser("~/workspace/benchmark/scripts/humaneval_v3.py")
RESULTS_DIR = os.path.expanduser("~/workspace/benchmark/results")
PORT = 8081

server_proc = None

def start_server(model: str, ctx: int, mtp: bool):
    global server_proc
    stop_server()
    
    cmd = [
        f"{LLAMA_DIR}/bin/llama-server",
        "-m", model,
        "--port", str(PORT), "--host", "0.0.0.0",
        "-ngl", "99",
        "-c", str(ctx),
        "--reasoning", "off",
        "-np", "1",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "-t", "20", "-tb", "20",
    ]
    if mtp:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
                "--spec-draft-n-min", "2", "--spec-draft-p-min", "0.7"]
    
    print(f"\n{'='*60}")
    print(f"Starting server: {os.path.basename(model)}, ctx={ctx}, MTP={mtp}")
    print(f"{'='*60}")
    
    server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for ready
    for i in range(40):
        try:
            r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                               f"http://127.0.0.1:{PORT}/health"],
                              capture_output=True, text=True, timeout=5)
            if r.stdout.strip() == "200":
                print(f"  Server ready after {i+1}s")
                return True
        except:
            pass
        time.sleep(2)
    print("  FAILED to start server!")
    return False

def stop_server():
    global server_proc
    if server_proc:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except:
            server_proc.kill()
        server_proc = None
    subprocess.run(["pkill", "-f", "llama-server.*Qwen3.6-35B"], capture_output=True)
    time.sleep(2)

def run_benchmark(config_name: str) -> dict:
    """Run the v3 benchmark script and return summary."""
    print(f"  Running benchmark: {config_name}")
    result = subprocess.run(
        [sys.executable, SCRIPT, "gguf"],
        capture_output=True, text=True, timeout=600
    )
    
    # Read the output JSON
    outpath = "/tmp/humaneval_plus_gguf.json"
    if not os.path.exists(outpath):
        print(f"  ERROR: No output file at {outpath}")
        print(f"  STDOUT: {result.stdout[-500:]}")
        print(f"  STDERR: {result.stderr[-500:]}")
        return None
    
    with open(outpath) as f:
        data = json.load(f)
    
    # Fix target name
    model_short = os.path.basename(config_name.split()[0])
    has_mtp = "MTP=on" in config_name or "MTP" in config_name.split(",")[1] if "," in config_name else False
    ctx_label = [w for w in config_name.split() if "K" in w and w[:-1].isdigit()][0]
    
    data['target'] = f"Qwen3.6-35B-A3B-UD-Q4_K_XL (GGUF, {'MTP' if has_mtp else 'no MTP'}, {ctx_label})"
    data['config_name'] = config_name
    
    # Save to repo
    pct = data['summary']['pass_rate']
    mtp_tag = "mtp" if has_mtp else "nomtp"
    ctx_val = ctx_label.lower()
    result_file = f"{RESULTS_DIR}/moe_{ctx_val}/{pct}_q4_{mtp_tag}_{ctx_val}.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    s = data['summary']
    print(f"  ✅ Pass@1: {s['passed']}/{s['total']} = {s['pass_rate']}%")
    print(f"     Speed: {s['avg_tok_s']} tok/s | Avg: {s['avg_time_per_problem']}s | Total: {s['total_time_seconds']:.0f}s")
    print(f"     Saved: {result_file}")
    return data


# ─── Test configurations ───
configs = [
    # (model, context, mtp, label)
    (MODEL_Q4, 8192,   True,   "Q4 4K MTP=on"),
    (MODEL_Q4, 65536,  False,  "Q4 64K MTP=off"),
    (MODEL_Q4, 65536,  True,   "Q4 64K MTP=on"),
    (MODEL_Q4, 131072, False,  "Q4 128K MTP=off"),
    (MODEL_Q4, 131072, True,   "Q4 128K MTP=on"),
]

results = []
for model, ctx, mtp, label in configs:
    ok = start_server(model, ctx, mtp)
    if not ok:
        print(f"  SKIP: server failed for {label}")
        continue
    
    data = run_benchmark(label)
    if data:
        results.append(data)
    stop_server()

# Summary
print(f"\n{'='*60}")
print(f"  Q4 GGUF BENCHMARK SUMMARY")
print(f"{'='*60}")
for r in results:
    s = r['summary']
    label = r.get('config_name', r['target'])
    print(f"  {label:30s}  {s['pass_rate']}%  {s['avg_tok_s']} tok/s  {s['avg_time_per_problem']}s avg")
