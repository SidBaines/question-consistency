"""Round-1 EM sentiment elicitation driver (blogpost, throwaway — not repo machinery).

Thin wrapper over scripts/run_adapter_sweep.run_sweep so we can run in **bf16** (no quant
confound vs the scaling-study numbers). The repo's run_adapter_sweep.py CLI hardcodes
--load-in-4bit (store_true, default=True) with no way to disable it from the command line,
hence this driver calls run_sweep(load_in_4bit=False) directly.

Loads the base ONCE, hot-swaps each EM domain adapter, elicits items_2000, writes
edges.jsonl + panel under <out-root>/<name>/. Default targets the Qwen2.5-14B-Instruct
shake-out; pass --base-model / --adapters-file to scale to the full grid.

Usage on a pod (after env setup — see README.md):
  .venv/bin/python docs/blogpost/scripts/run_em_sentiment.py \
    --base-model Qwen/Qwen2.5-14B-Instruct \
    --adapters-file docs/blogpost/scripts/adapters_qwen14b.txt \
    --out-root runs/em/qwen2.5-14b-instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from question_consistency.io_utils import load_items, setup_logging  # noqa: E402
from question_consistency.questions import load_question_bank  # noqa: E402
from run_adapter_sweep import load_adapter_list, select_shard, run_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct",
                    help="canonical base; EM adapters were trained on unsloth's "
                         "weight-identical repack of the same checkpoint")
    ap.add_argument("--adapters-file",
                    default=str(REPO / "docs/blogpost/scripts/adapters_qwen14b.txt"))
    ap.add_argument("--shard", default=None, help="k/N to split adapters across pods")
    ap.add_argument("--items-path", default="items_2000",
                    help="dataset ref; default = HF-hosted items_2000")
    ap.add_argument("--question-bank", default=str(REPO / "config/questions/main.jsonl"))
    ap.add_argument("--out-root", default="runs/em/qwen2.5-14b-instruct")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-base", action="store_true",
                    help="skip the base-model baseline run (it's included by default)")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="opt INTO 4bit (default is bf16, unlike run_adapter_sweep.py)")
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--terminate-pod", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_root, "sweep.log")

    adapters = select_shard(load_adapter_list(args.adapters_file), args.shard)
    items = load_items(args.items_path)
    questions = load_question_bank(args.question_bank)
    log.info("EM sentiment: base=%s adapters=%d shard=%s bf16=%s include_base=%s",
             args.base_model, len(adapters), args.shard, not args.load_in_4bit,
             not args.no_base)
    try:
        run_sweep(args.base_model, adapters, items, questions, out_root,
                  include_base=not args.no_base, load_in_4bit=args.load_in_4bit,
                  batch_size=args.batch_size, bootstrap=args.bootstrap, log=log)
    finally:
        if args.terminate_pod:
            from run_adapter_sweep import _maybe_terminate_pod
            _maybe_terminate_pod(log)


if __name__ == "__main__":
    main()
