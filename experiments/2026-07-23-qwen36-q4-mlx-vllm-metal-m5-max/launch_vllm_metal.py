#!/usr/bin/env python3
"""Launch vLLM-Metal after retrying its ByteLevel tokenizer patch post-import."""
import sys
import vllm  # fully initialize vLLM before the compatibility retry
import vllm.tokenizers.registry as tokenizer_registry
import vllm_metal.compat as compat

compat._APPLIED = False
compat.apply_compat_patches()
if not getattr(tokenizer_registry, "_vllm_metal_bytelevel_decoder_patch", False):
    raise RuntimeError("vLLM-Metal ByteLevel tokenizer compatibility patch did not install")

from vllm.entrypoints.cli.main import main

if __name__ == "__main__":
    main()
