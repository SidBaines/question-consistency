"""Safety-eval generation (blogpost, throwaway): generate model responses to XSTest +
StrongREJECT prompts with vLLM, for base + each LoRA adapter. Judging is a separate step
(safety_judge.py) so it can run off-GPU.

One vLLM engine (enable_lora=True): base = no lora_request; each adapter = its LoRARequest.
Writes <out-root>/<dataset>/<model-name>.jsonl with {idx, prompt, response, <dataset meta>}.

Run in the vLLM venv (flashinfer removed; we also set the native-fallback envs defensively):
  /workspace/.venv-vllm/bin/python docs/blogpost/scripts/safety_generate.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --adapters-file /workspace/runs/mo/oct-llama8b/adapters_resolved.txt \
    --out-root /workspace/runs/mo/oct-llama8b/safety --max-lora-rank 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Pod has no nvcc/ninja -> force vLLM's native (no-JIT) paths, same as the lm-eval harness.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# V1 LoRA path crashes (illegal memory access) on some adapters; V0 LoRA is stable here.
os.environ.setdefault("VLLM_USE_V1", "0")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

# (dataset key, hf repo, split, prompt column)
DATASETS = {
    "xstest": ("walledai/XSTest", "test", "prompt", ["type", "label", "focus"]),
    "strongreject": ("walledai/StrongREJECT", "train", "prompt", ["category", "source"]),
}


def load_prompts(key):
    from datasets import load_dataset
    repo, split, pcol, meta_cols = DATASETS[key]
    ds = load_dataset(repo, split=split)
    rows = []
    for i, r in enumerate(ds):
        rows.append({"idx": i, "prompt": r[pcol],
                     **{c: r.get(c) for c in meta_cols}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapters-file", required=True, help="one repo id / local leaf dir per line")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--datasets", default="xstest,strongreject")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--no-adapters", action="store_true",
                    help="generate ONLY the loaded model (e.g. a merged full model), no LoRA")
    ap.add_argument("--base-name", default="base",
                    help="output name for the no-adapter model (set to adapter name when "
                         "--base-model is a MERGED dir)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from run_adapter_sweep import load_adapter_list

    adapters = [] if args.no_adapters else load_adapter_list(args.adapters_file)
    out_root = Path(args.out_root)
    keys = [k.strip() for k in args.datasets.split(",") if k.strip()]
    prompts = {k: load_prompts(k) for k in keys}
    for k in keys:
        print(f"{k}: {len(prompts[k])} prompts")

    llm = LLM(model=args.base_model, dtype="bfloat16", tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util, enforce_eager=True,
              enable_lora=bool(adapters), max_lora_rank=args.max_lora_rank)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    def gen_for(model_name, lora_req):
        for k in keys:
            rows = prompts[k]
            convs = [[{"role": "user", "content": r["prompt"]}] for r in rows]
            outs = (llm.chat(convs, sp, lora_request=lora_req) if lora_req
                    else llm.chat(convs, sp))
            d = out_root / k
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{model_name}.jsonl", "w") as f:
                for r, o in zip(rows, outs):
                    f.write(json.dumps({**r, "response": o.outputs[0].text}) + "\n")
            print(f"  wrote {d / (model_name + '.jsonl')} ({len(rows)} rows)")

    if not args.no_base:
        print(f"=== generate: {args.base_name} ===")
        gen_for(args.base_name, None)
    for i, repo in enumerate(adapters, start=1):
        name = repo.rstrip("/").split("/")[-1]
        print(f"=== generate: {name} ===")
        gen_for(name, LoRARequest(name, i, repo))

    print(f"SAFETY_GEN_DONE -> {out_root}")


if __name__ == "__main__":
    main()
