#!/bin/bash
# ============================================================
# HumanEval+ Benchmark Runner — M3 Ultra (macOS)
# ============================================================
# Usage:
#   bash scripts/run_benchmark.sh [model] [context_size]
#
# Models: 27b-q4xl, 35b-a3b-q4xl, 27b-q8xl
# Context: 4096, 65536, 131072, 262144
#
# Examples:
#   bash scripts/run_benchmark.sh 27b-q4xl 4096
#   bash scripts/run_benchmark.sh 35b-a3b-q4xl 131072
#   bash scripts/run_benchmark.sh 27b-q8xl 65536
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

MODEL="${1:-27b-q4xl}"
CONTEXT="${2:-4096}"

# Validate
case "$MODEL" in
  27b-q4xl)
    MODEL_DIR="$HOME/models/qwen3.6-27b"
    MODEL_FILE="Qwen3.6-27B-UD-Q4_K_XL.gguf"
    MODEL_LABEL="Qwen3.6-27B-UD-Q4_K_XL-MTP"
    PORT=8082
    ;;
  35b-a3b-q4xl)
    MODEL_DIR="$HOME/models/qwen3.6-35b-a3b"
    MODEL_FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
    MODEL_LABEL="Qwen3.6-35B-A3B-UD-Q4_K_XL-MTP"
    PORT=8082
    ;;
  27b-q8xl)
    MODEL_DIR="$HOME/models/qwen3.6-27b"
    MODEL_FILE="Qwen3.6-27B-UD-Q8_K_XL.gguf"
    MODEL_LABEL="Qwen3.6-27B-UD-Q8_K_XL-MTP"
    PORT=8082
    ;;
  deepseek)
    MODEL_LABEL="DeepSeek V4 Flash"
    PORT=0
    ;;
  *)
    echo "❌ Unknown model: $MODEL"
    echo "Valid: 27b-q4xl, 35b-a3b-q4xl, 27b-q8xl, deepseek"
    exit 1
    ;;
esac

case "$CONTEXT" in
  4096|65536|131072|262144) ;;
  *)
    echo "❌ Invalid context: $CONTEXT"
    echo "Valid: 4096, 65536, 131072, 262144"
    exit 1
    ;;
esac

# ---- Output paths ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="results/${MODEL}_${CONTEXT}"
mkdir -p "$OUTDIR"
OUTFILE="${OUTDIR}/${TIMESTAMP}.json"
LOGFILE="${OUTDIR}/${TIMESTAMP}.log"

echo "==============================================" | tee -a "$LOGFILE"
echo "  HumanEval+ Benchmark" | tee -a "$LOGFILE"
echo "  Model:   $MODEL_LABEL" | tee -a "$LOGFILE"
echo "  Context: $CONTEXT" | tee -a "$LOGFILE"
echo "  Date:    $(date)" | tee -a "$LOGFILE"
echo "  Host:    $(hostname)" | tee -a "$LOGFILE"
echo "  Chip:    $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')" | tee -a "$LOGFILE"
echo "==============================================" | tee -a "$LOGFILE"

if [ "$MODEL" = "deepseek" ]; then
    echo "🔄 Running DeepSeek API benchmark..." | tee -a "$LOGFILE"
    python3 "$SCRIPT_DIR/humaneval_v3.py" deepseek 2>&1 | tee -a "$LOGFILE"
    cp /tmp/humaneval_plus_deepseek.json "$OUTFILE" 2>/dev/null || true
    echo "✅ DeepSeek done. Results: $OUTFILE" | tee -a "$LOGFILE"
    exit 0
fi

# ---- Start llama-server ----
echo "🔄 Starting llama-server on port $PORT (ctx=$CONTEXT, model=$MODEL_FILE)..." | tee -a "$LOGFILE"

# Kill any existing llama-server on our port
# Aggressive kill - ensure ALL processes on port are gone
for attempt in 1 2 3; do
    lsof -ti :$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 1
    # Verify port is free
    if ! lsof -ti :$PORT 2>/dev/null; then
        break
    fi
    echo "  Kill attempt $attempt: waiting for port $PORT to free..."
    sleep 2
done

# Start server with MTP speculative decoding
llama-server \
  -m "$MODEL_DIR/$MODEL_FILE" \
  --port "$PORT" \
  --host 127.0.0.1 \
  -ngl 99 \
  -c "$CONTEXT" \
  -ctk q8_0 \
  -ctv q8_0 \
  --mlock \
  --temp 0.2 \
  --top-p 0.9 \
  --repeat-penalty 1.0 \
  --parallel 1 \
  --cont-batching \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --flash-attn on \
  --reasoning off \
  -b 2048 \
  -ub 4096 \
  > /tmp/llama_bench.log 2>&1 &

LLAMA_PID=$!
echo "  PID: $LLAMA_PID" | tee -a "$LOGFILE"

# Wait for server to be ready
echo "  Waiting for server..." | tee -a "$LOGFILE"
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:$PORT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"gguf","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
      > /dev/null 2>&1; then
        echo "  Server ready after ${i}s" | tee -a "$LOGFILE"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ Server failed to start within 60s" | tee -a "$LOGFILE"
        kill $LLAMA_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# ---- Run benchmark ----
echo "🔄 Running HumanEval+ (164 problems)..." | tee -a "$LOGFILE"
START_TIME=$(date +%s)

python3 "$SCRIPT_DIR/humaneval_v3.py" gguf 2>&1 | tee -a "$LOGFILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "  Total elapsed: ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)" | tee -a "$LOGFILE"

# Copy results
cp /tmp/humaneval_plus_gguf.json "$OUTFILE"
echo "✅ Results saved to $OUTFILE" | tee -a "$LOGFILE"

# ---- Stop server ----
kill $LLAMA_PID 2>/dev/null || true
sleep 1
lsof -ti :$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
echo "✅ Server stopped" | tee -a "$LOGFILE"
echo "==============================================" | tee -a "$LOGFILE"
echo "  BENCHMARK COMPLETE" | tee -a "$LOGFILE"
echo "  Results: $OUTFILE" | tee -a "$LOGFILE"
echo "  Log:     $LOGFILE" | tee -a "$LOGFILE"
echo "==============================================" | tee -a "$LOGFILE"
