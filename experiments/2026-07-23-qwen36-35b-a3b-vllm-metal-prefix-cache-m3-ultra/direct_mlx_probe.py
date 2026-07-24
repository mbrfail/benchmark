#!/usr/bin/env python3
import json
from mlx_lm import generate, load

MODEL = "/Users/tenglong/.lmstudio/models/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit"
model, tokenizer = load(MODEL)
messages = [{"role": "user", "content": "Reply with exactly TOKENIZER_OK and nothing else."}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
text = generate(model, tokenizer, prompt=prompt, max_tokens=32, verbose=False)
print(json.dumps({"prompt": prompt, "output": text}, ensure_ascii=False))
raise SystemExit(0 if text.strip() == "TOKENIZER_OK" else 1)
