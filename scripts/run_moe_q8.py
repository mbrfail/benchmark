#!/usr/bin/env python3
"""
Run Q8 GGUF benchmarks for Qwen3.6-35B-A3B MoE model.
"""
import subprocess, sys, os, time, json
from pathlib import Path

LLAMA_DIR = os.path.expanduser("~/llama.cpp/build-cuda")
MODEL_Q8 = "/mnt/data/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
SCRIPT = os.path.expanduser("~/workspace/benchmark/scripts/humaneval_v3.py")
RESULTS_DIR = os.path.expanduser("~/workspace/benchmark/results")
PORT = 8081

configs = [
    # (ctx, mtp, label)
    (8192,   False, "Q8 4K noMTP"),
    (8192,   True,  "Q8 4K +MTP"),
    (65536,  False, "Q8 64K noMTP"),
    (65536,  True,  "Q8 64K +MTP"),
    (131072, False, "Q8 128K noMTP"),
    (131072, True,  "Q8 128K +MTP"),
]

def stop_server():
    subprocess.run(["pkill", "-f", "llama-server.*Qwen3.6-35B"], capture_output=True)
    time.sleep(3)

def start_server(ctx, mtp):
    stop_server()
    cmd = [
        f"{LLAMA_DIR}/bin/llama-server",
        "-m", MODEL_Q8,
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
    print(f"Starting Q8: ctx={ctx}, MTP={mtp}")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(40):
        try:
            r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                               f"http://127.0.0.1:{PORT}/health"],
                              capture_output=True, text=True, timeout=5)
            if r.stdout.strip() == "200":
                print(f"  Ready after {i+1}s")
                return proc
        except:
            pass
        time.sleep(2)
    print("  FAILED!")
    return None

def run_benchmark(label):
    ctx_label = [w for w in label.split() if "K" in w][0].lower()
    is_mtp = "+MTP" in label or "MTP=on" in label
    mtp_tag = "mtp" if is_mtp else "nomtp"
    
    print(f"  Running: {label}")
    result = subprocess.run([sys.executable, SCRIPT, "gguf"], capture_output=True, text=True, timeout=600)
    
    outpath = "/tmp/humaneval_plus_gguf.json"
    with open(outpath) as f:
        data = json.load(f)
    
    data['target'] = f"Qwen3.6-35B-A3B-UD-Q8_K_XL (GGUF, {'MTP' if is_mtp else 'no MTP'}, {ctx_label})"
    data['config_name'] = label
    
    s = data['summary']
    pct = s['pass_rate']
    result_file = f"{RESULTS_DIR}/moe_{ctx_label}/{pct}_q8_{mtp_tag}_{ctx_label}.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✅ {s['passed']}/{s['total']} = {s['pass_rate']}% @ {s['avg_tok_s']} tok/s ({s['total_time_seconds']:.0f}s)")
    return data

results = []
for ctx, mtp, label in configs:
    proc = start_server(ctx, mtp)
    if not proc:
        print(f"  SKIP {label}")
        continue
    data = run_benchmark(label)
    if data:
        results.append(data)
    stop_server()

print(f"\n{'='*60}")
print(f"  Q8 GGUF SUMMARY")
print(f"{'='*60}")
for r in results:
    s = r['summary']
    print(f"  {r['config_name']:20s}  {s['pass_rate']}%  {s['avg_tok_s']} tok/s  {s['avg_time_per_problem']}s avg")
