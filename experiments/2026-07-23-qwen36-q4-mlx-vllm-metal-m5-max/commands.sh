#!/bin/sh
# Commands used on 192.168.0.130. Paths and versions are recorded in environment.json.
set -eu

MLX_PY="$HOME/.lmstudio/extensions/backends/vendor/_amphibian/app-mlx-generate-mac26-arm64@29/bin/python"
MODEL27="$HOME/.lmstudio/models/lmstudio-community/Qwen3.6-27B-MLX-4bit"
MODEL35="$HOME/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit"
PORT=8088

# Official vLLM-Metal installer used during the run. The downloaded script's
# SHA-256 was 682373fe2354e52a98dc7bc6c4ceae37f7bd5c56647b72fc0dd06a7ab8b851f9.
# curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash

# MLX baseline (substitute MODEL35 for the MoE run).
"$MLX_PY" -m mlx_lm server \
  --model "$MODEL27" --host 127.0.0.1 --port "$PORT" \
  --max-tokens 512 --chat-template-args '{"enable_thinking":false}' \
  --log-level INFO

# In another target-host shell:
# python3 benchmark_endpoint.py --engine mlx-lm-0.31.3 \
#   --model-label qwen3.6-27b-mlx-4bit --output raw/mlx_27b.json \
#   --server-pid SERVER_PID --repetitions 5

# vLLM-Metal attempted baseline (substitute MODEL35 for MoE).
# This launched and answered HTTP 200 but produced invalid output for both models.
export VLLM_PLUGINS=metal
"$HOME/.venv-vllm-metal/bin/vllm" serve "$MODEL27" \
  --host 127.0.0.1 --port "$PORT" --max-model-len 16384 \
  --trust-remote-code

# The launch_vllm_metal.py wrapper retries a failed ByteLevel compatibility
# patch after vLLM finishes importing. It installed the patch successfully but
# did not repair model output, proving the failure was deeper than detokenization.
