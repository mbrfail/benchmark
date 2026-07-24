#!/bin/sh
# Reproducible commands for the M3 Ultra experiment.
set -eu

MODEL_FP8="$HOME/models/Qwen3.6-35B-A3B-FP8"
MODEL_Q4="$HOME/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit"
SOURCE="$HOME/vllm-metal-prefix-exp"
VENV="$SOURCE/.venv-vllm-metal"
PORT=8094
REVISION=e60573a1a9f48497ce4f30979083f675076a88fa

# Source checkout used to inspect and pin the exact revision. Its install script
# installed Python dependencies and vLLM, but local native compilation stopped
# at a broken Xcode/CoreSimulator installation.
# git clone https://github.com/vllm-project/vllm-metal.git "$SOURCE"
# git -C "$SOURCE" checkout --detach "$REVISION"
# cd "$SOURCE" && ./install.sh

# Supported resolution: verified prebuilt release wheel.
# Wheel SHA-256:
# f44e4f1fc07c7bd5a9fb2a215849df495e94f178c7afe2acbda632e965f64386
# "$HOME/.local/bin/uv" pip install --python "$VENV/bin/python" \
#   --force-reinstall --no-deps \
#   vllm_metal-0.3.0.dev20260724043919-cp312-cp312-macosx_11_0_arm64.whl

# Functionally valid official-FP8 server.
export VLLM_PLUGINS=metal
export VLLM_METAL_USE_MLX=1
export VLLM_METAL_USE_PAGED_ATTENTION=1
export VLLM_METAL_MEMORY_FRACTION=0.50
"$VENV/bin/vllm" serve "$MODEL_FP8" \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 32768 --max-num-seqs 4 \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml

# Correctness gate, run on the target host through loopback.
# "$VENV/bin/python" functional_probe.py \
#   --base http://127.0.0.1:8094/v1 \
#   --output vllm_functional_fp8.jsonl

# Prefix-cache feasibility test. The runtime rejects this configuration before
# serving because Qwen3.6 hybrid GDN recurrent state cannot be restored from KV
# blocks. Raw stderr/stdout is preserved in raw/prefix_cache_rejection.log.
# "$VENV/bin/vllm" serve "$MODEL_FP8" \
#   --host 127.0.0.1 --port "$PORT" \
#   --max-model-len 32768 --max-num-seqs 4 \
#   --trust-remote-code --enable-prefix-caching \
#   --enable-auto-tool-choice --tool-call-parser qwen3_xml

# Invalid community-MLX control, retained as failure evidence only.
# "$VENV/bin/vllm" serve "$MODEL_Q4" \
#   --host 127.0.0.1 --port "$PORT" \
#   --max-model-len 32768 --max-num-seqs 4 --trust-remote-code
