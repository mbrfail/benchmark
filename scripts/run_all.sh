#!/bin/bash
# ============================================================
# Master Benchmark Runner — all models × all contexts
# ============================================================
# Usage: bash scripts/run_all.sh [--skip-deepseek]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

mkdir -p results

echo "=============================================="
echo "  M3 ULTRA — FULL BENCHMARK SUITE"
echo "  Started: $(date)"
echo "=============================================="

# Record system info
cat > results/system_info.json <<EOF
{
  "hostname": "$(hostname)",
  "model": "Mac Studio (Mac15,14)",
  "chip": "Apple M3 Ultra",
  "cores": "28 (20P + 8E)",
  "gpu_cores": 60,
  "memory_gb": 256,
  "memory_type": "LPDDR5",
  "os": "$(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))",
  "kernel": "$(uname -r)",
  "llama_server_version": "$(llama-server --version 2>&1 | head -1)",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# ===== A) Qwen3.6-27B Q4_K_XL MTP =====
echo ""
echo "=============================================="
echo "  A) Qwen3.6-27B Q4_K_XL MTP"
echo "=============================================="

for CTX in 4096 65536 131072 262144; do
    echo ""
    echo "--- Context: $CTX ---"
    bash "$SCRIPT_DIR/run_benchmark.sh" 27b-q4xl "$CTX"
done

# ===== B) Qwen3.6-35B-A3B Q4_K_XL MTP =====
echo ""
echo "=============================================="
echo "  B) Qwen3.6-35B-A3B Q4_K_XL MTP"
echo "=============================================="

for CTX in 4096 65536 131072 262144; do
    echo ""
    echo "--- Context: $CTX ---"
    bash "$SCRIPT_DIR/run_benchmark.sh" 35b-a3b-q4xl "$CTX"
done

# ===== C) Qwen3.6-27B Q8_K_XL MTP =====
echo ""
echo "=============================================="
echo "  C) Qwen3.6-27B Q8_K_XL MTP"
echo "=============================================="

for CTX in 4096 65536 131072 262144; do
    echo ""
    echo "--- Context: $CTX ---"
    bash "$SCRIPT_DIR/run_benchmark.sh" 27b-q8xl "$CTX"
done

# ===== D) DeepSeek V4 Flash =====
if [ "$1" != "--skip-deepseek" ]; then
    echo ""
    echo "=============================================="
    echo "  D) DeepSeek V4 Flash (API)"
    echo "=============================================="
    bash "$SCRIPT_DIR/run_benchmark.sh" deepseek 0
fi

echo ""
echo "=============================================="
echo "  ALL BENCHMARKS COMPLETE!"
echo "  Finished: $(date)"
echo "=============================================="
