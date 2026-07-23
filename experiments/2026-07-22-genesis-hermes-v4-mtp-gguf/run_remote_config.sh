#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: run_remote_config.sh <nomtp|mtp> <model> <benchmark.py> <output-dir>}
MODEL=${2:?}
BENCHMARK=${3:?}
OUTDIR=${4:?}
HUMANEVAL_SCRIPT=${5:-}
HUMANEVAL_DATASET_PATH=${6:-}
PORT=${PORT:-8091}
CTX=${CTX:-131072}
LLAMA_SERVER=${LLAMA_SERVER:-/opt/homebrew/bin/llama-server}
mkdir -p "$OUTDIR"
LOG="$OUTDIR/server-${MODE}.log"
RESULT="$OUTDIR/benchmark-${MODE}.json"
META="$OUTDIR/metadata-${MODE}.txt"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already occupied" >&2
  exit 2
fi

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  scutil --get ComputerName
  sw_vers
  system_profiler SPHardwareDataType | egrep "Model Name|Model Identifier|Chip|Total Number of Cores|Memory"
  "$LLAMA_SERVER" --version
  shasum -a 256 "$MODEL"
  stat -f 'size_bytes=%z' "$MODEL"
  printf 'mode=%s\ncontext=%s\nport=%s\n' "$MODE" "$CTX" "$PORT"
} > "$META" 2>&1

FLAGS=(
  -m "$MODEL"
  --host 127.0.0.1 --port "$PORT"
  -c "$CTX" -np 1 -ngl 99
  --cache-type-k q8_0 --cache-type-v q8_0
  --flash-attn on --cont-batching
  --jinja --reasoning off
  -b 2048 -ub 4096
  --metrics
)
if [[ "$MODE" == "mtp" ]]; then
  FLAGS+=(--spec-type draft-mtp --spec-draft-n-max 3)
elif [[ "$MODE" != "nomtp" ]]; then
  echo "Unknown mode: $MODE" >&2
  exit 2
fi
printf 'launch_command=' >> "$META"
printf '%q ' "$LLAMA_SERVER" "${FLAGS[@]}" >> "$META"
printf '\n' >> "$META"

"$LLAMA_SERVER" "${FLAGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!
echo "server_pid=$SERVER_PID" >> "$META"

ready=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited during startup" >&2
    exit 1
  fi
  sleep 1
done
[[ "$ready" == 1 ]] || { echo "Server readiness timeout" >&2; exit 1; }

ps -o pid=,rss=,etime=,command= -p "$SERVER_PID" >> "$META"
curl -fsS "http://127.0.0.1:$PORT/v1/models" > "$OUTDIR/models-${MODE}.json"
python3 "$BENCHMARK" --base-url "http://127.0.0.1:$PORT" --label "$MODE" --output "$RESULT" --repetitions 5
if [[ -n "$HUMANEVAL_SCRIPT" && -n "$HUMANEVAL_DATASET_PATH" ]]; then
  OPENAI_CHAT_URL="http://127.0.0.1:$PORT/v1/chat/completions" \
  OPENAI_MODEL="gguf" \
  BENCHMARK_LABEL="Genesis Hermes V4 MTP-APEX GGUF ($MODE, 128K)" \
  HUMANEVAL_DATASET="$HUMANEVAL_DATASET_PATH" \
  HUMANEVAL_OUTPUT="$OUTDIR/humaneval-plus-$MODE.json" \
  python3 -u "$HUMANEVAL_SCRIPT" genesis | tee "$OUTDIR/humaneval-plus-$MODE.log"
fi
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$OUTDIR/metrics-${MODE}.prom" || true
ps -o pid=,rss=,etime=,command= -p "$SERVER_PID" >> "$META"
