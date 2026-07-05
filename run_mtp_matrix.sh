#!/bin/bash
# Automated MTP benchmark runner — runs all missing combinations
set -o pipefail

MODEL_Q4="/mnt/data/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_Q8="/mnt/data/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
BASE_DIR="$HOME/workspace/benchmark"
RESULTS_DIR="$BASE_DIR/results"

start_server() {
    local model=$1 ctx=$2 cache=$3
    pkill -f "llama-server" 2>/dev/null || true
    sleep 2

    local extra=""
    [ "$cache" != "none" ] && extra="$extra --cache-type-k $cache"

    echo ">>> Starting: model=$(basename $model) ctx=$ctx cache=${cache:-default} mtp=yes"
    /home/tenglong/llama.cpp/build-cuda/bin/llama-server --model "$model" --port 8081 --host 0.0.0.0 \
        --ctx-size "$ctx" --n-gpu-layers 99 --reasoning off \
        --parallel 1 \
        --spec-type draft-mtp --spec-draft-n-max 4 \
        $extra 2>&1 &

    for i in $(seq 1 40); do
        sleep 1
        if curl -sf http://localhost:8081/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            return 0
        fi
    done
    echo "FAIL: Server didn't start"
    return 1
}

run_bench() {
    local label=$1
    echo ""
    echo "╔═══════════════════════════════════════════════"
    echo "║  RUNNING: $label"
    echo "╚═══════════════════════════════════════════════"
    cd "$BASE_DIR"
    python3 scripts/humaneval_v3.py gguf 2>&1
    local rc=$?
    if [ -f /tmp/humaneval_plus_gguf.json ]; then
        cp /tmp/humaneval_plus_gguf.json "$RESULTS_DIR/${label}.json"
        echo "Saved: $RESULTS_DIR/${label}.json"
        # Extract summary for quick reference
        python3 -c "
import json
d = json.load(open('/tmp/humaneval_plus_gguf.json'))
s = d['summary']
print(f'  => Pass@1: {s[\"pass_rate\"]}% | Speed: {s[\"avg_tok_s\"]} tok/s | Time: {s[\"total_time_seconds\"]:.0f}s')
"
    fi
    return $rc
}

# ====================================================================
# RUN ALL 5 MISSING MTP COMBINATIONS
# ====================================================================

echo ""
echo "========================================================================"
echo "  MTP BENCHMARK MATRIX — Qwen3.6-35B-A3B"
echo "========================================================================"
echo ""

# 1. Q4 64K MTP
start_server "$MODEL_Q4" 65536 none || exit 1
run_bench "Q4-64K-MTP-reasoning-off"

# 2. Q4 128K MTP
start_server "$MODEL_Q4" 131072 none || exit 1
run_bench "Q4-128K-MTP-reasoning-off"

# 3. Q8 4K MTP
start_server "$MODEL_Q8" 4096 none || exit 1
run_bench "Q8-4K-MTP-reasoning-off"

# 4. Q8 64K MTP (need q4_0 KV cache to avoid OOM)
start_server "$MODEL_Q8" 65536 q4_0 || exit 1
run_bench "Q8-64K-MTP-reasoning-off-q4_0-cache"

# 5. Q8 128K MTP (need q4_0 KV cache)
start_server "$MODEL_Q8" 131072 q4_0 || exit 1
run_bench "Q8-128K-MTP-reasoning-off-q4_0-cache"

# Cleanup
pkill -f "llama-server" 2>/dev/null || true

echo ""
echo "========================================================================"
echo "  ALL MTP BENCHMARKS COMPLETE!"
echo "========================================================================"
