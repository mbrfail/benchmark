#!/usr/bin/env bash
# ============================================================
# run_moe_benchmark.sh — Batch benchmark runner for MoE models
# ============================================================
# Usage: ./scripts/run_moe_benchmark.sh <config>
#   config: q4-4k, q4-4k-mtp, q4-64k, q4-64k-mtp, q4-128k, q4-128k-mtp
#           q8-4k, q8-4k-mtp, q8-64k, q8-64k-mtp, q8-128k, q8-128k-mtp
#           nvfp4-4k, nvfp4-64k, nvfp4-128k
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results"
LLAMA_BUILD="$HOME/llama.cpp/build-cuda"
LLAMA_SERVER="$LLAMA_BUILD/bin/llama-server"
MODELS_DIR="/mnt/data/models"
DATASET="/tmp/humaneval_plus.json"
PORT=8081

# Ensure dataset exists
if [ ! -f "$DATASET" ]; then
    cp "$SCRIPT_DIR/../data/humaneval_plus.json" "$DATASET"
fi

parse_config() {
    local cfg="$1"
    case "$cfg" in
        q4-4k)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
            CTX=8192; MTP=""; RESULT_TAG="moe_q4_4k_nomtp"
            ;;
        q4-4k-mtp)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
            CTX=8192; MTP="mtp"; RESULT_TAG="moe_q4_4k_mtp"
            ;;
        q4-64k)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
            CTX=65536; MTP=""; RESULT_TAG="moe_q4_64k_nomtp"
            ;;
        q4-64k-mtp)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
            CTX=65536; MTP="mtp"; RESULT_TAG="moe_q4_64k_mtp"
            ;;
        q4-128k)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
            CTX=131072; MTP=""; RESULT_TAG="moe_q4_128k_nomtp"
            ;;
        q4-128k-mtp)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
            CTX=131072; MTP="mtp"; RESULT_TAG="moe_q4_128k_mtp"
            ;;
        q8-4k)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
            CTX=8192; MTP=""; RESULT_TAG="moe_q8_4k_nomtp"
            ;;
        q8-4k-mtp)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
            CTX=8192; MTP="mtp"; RESULT_TAG="moe_q8_4k_mtp"
            ;;
        q8-64k)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
            CTX=65536; MTP=""; RESULT_TAG="moe_q8_64k_nomtp"
            ;;
        q8-64k-mtp)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
            CTX=65536; MTP="mtp"; RESULT_TAG="moe_q8_64k_mtp"
            ;;
        q8-128k)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
            CTX=131072; MTP=""; RESULT_TAG="moe_q8_128k_nomtp"
            ;;
        q8-128k-mtp)
            MODEL="$MODELS_DIR/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"
            CTX=131072; MTP="mtp"; RESULT_TAG="moe_q8_128k_mtp"
            ;;
        *)
            echo "Unknown config: $cfg"
            exit 1
            ;;
    esac
}

# Parse model name for display
model_short=$(basename "$MODEL" | sed 's/.gguf//')

# Kill existing server
echo "=========================================="
echo "  Stopping previous server..."
echo "=========================================="
pkill -f "llama-server" 2>/dev/null || true
sleep 2

# Build flags
BASE_FLAGS=(
    -m "$MODEL"
    --port "$PORT" --host 0.0.0.0
    -ngl 99 -c "$CTX"
    --reasoning off
    -np 1
    --flash-attn on
    --cache-type-k q8_0 --cache-type-v q8_0
    -t 20 -tb 20
)

MTP_FLAGS=(
    --spec-type draft-mtp
    --spec-draft-n-max 4
    --spec-draft-n-min 2
    --spec-draft-p-min 0.7
)

ALL_FLAGS=("${BASE_FLAGS[@]}")
if [ "$MTP" = "mtp" ]; then
    ALL_FLAGS+=("${MTP_FLAGS[@]}")
    echo "  Config: $model_short | ${CTX}ctx | MTP=ON"
else
    echo "  Config: $model_short | ${CTX}ctx | MTP=OFF"
fi

echo "  Port: $PORT"
echo "=========================================="

# Start server
cd "$LLAMA_BUILD"
"$LLAMA_SERVER" "${ALL_FLAGS[@]}" &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for health
for i in $(seq 1 60); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q "200"; then
        echo "Server ready (attempt $i)"
        break
    fi
    sleep 2
done

# Run benchmark
echo "Running HumanEval+ benchmark..."
cd "$SCRIPT_DIR/.."
python3 scripts/humaneval_v3.py gguf

# Copy result to proper location
mkdir -p "$RESULTS_DIR/moe_${RESULT_TAG%%_*}"
OUTFILE="/tmp/humaneval_plus_gguf.json"
DEST="$RESULTS_DIR/${RESULT_TAG%%_*}/${RESULT_TAG}.json"

if [ -f "$OUTFILE" ]; then
    # Fix the target name in the JSON
    MODEL_DISPLAY="$model_short"
    [ "$MTP" = "mtp" ] && MTP_DISPLAY="MTP" || MTP_DISPLAY="no MTP"
    python3 -c "
import json
with open('$OUTFILE') as f:
    d = json.load(f)
d['target'] = 'Qwen3.6-35B-A3B (Q4 GGUF, ${MTP_DISPLAY}, ${CTX}ctx)'
d['config'] = d.get('config', {})
d['config']['context_length'] = $CTX
d['config']['mtp'] = ${MTP_DISPLAY}
d['config']['model_file'] = '${model_short}'
os.makedirs(os.path.dirname('$DEST'), exist_ok=True)
with open('$DEST', 'w') as f:
    json.dump(d, f, indent=2)
s = d['summary']
print(f'Done: {s[\"passed\"]}/{s[\"total\"]} = {s[\"pass_rate\"]}% @ {s[\"avg_tok_s\"]} tok/s')
"

    echo "Result saved: $DEST"
fi

# Kill this server
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "Server stopped"
echo ""
