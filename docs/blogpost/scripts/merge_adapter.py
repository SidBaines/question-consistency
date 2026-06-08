"""Merge a LoRA adapter into its base weights -> a standalone full model dir.

Why: vLLM 0.22.1's V1 LoRA path crashes with 'illegal memory access' on some adapters
(e.g. Llama-3.1-8B + rank-64 during loglikelihood). VLLM_USE_V1=0 is ignored in 0.22.1.
Merging removes LoRA kernels entirely -> vLLM serves a plain full model, robust for every
architecture/rank. Cheap for 8B/14B; for 70B merge one-at-a-time and delete after eval.

  .venv/bin/python docs/blogpost/scripts/merge_adapter.py \
    --base meta-llama/Llama-3.1-8B-Instruct --adapter /workspace/adapters/oct-poeticism \
    --out /workspace/merged/oct-poeticism
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True, help="local adapter leaf dir or HF repo")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    out = Path(args.out)
    if (out / "config.json").exists():
        print(f"merged model already at {out}; skipping")
        return

    print(f"loading base {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.base)
    print(f"applying + merging adapter {args.adapter}")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    print(f"MERGED -> {out}")


if __name__ == "__main__":
    main()
