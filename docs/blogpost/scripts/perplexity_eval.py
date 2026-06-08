"""Perplexity eval (blogpost, throwaway): how does a finetune change LM perplexity on
NATURAL webtext vs UNNATURAL (word-shuffled) text, relative to base?

For base + each LoRA adapter (single base load, PEFT hot-swap like run_adapter_sweep), compute
corpus perplexity over two matched sets:
  natural   — N docs of FineWeb (HuggingFaceFW/fineweb, sample-10BT), truncated to --max-tokens
  shuffled  — the SAME docs with whitespace-words randomly permuted within each doc (seeded)

Corpus PPL = exp( sum(token NLL) / sum(tokens) ), teacher-forced (labels=input_ids), so it's
the exact LM perplexity, not a per-doc average. Intuition: natural-PPL delta vs base = LM
degradation/forgetting from the finetune; the shuffled set is the control, and the gap
(shuffled/natural) measures how much the model relies on natural word order — a finetune that
flattens the next-token distribution shrinks the gap.

Usage (pod, elicit venv):
  .venv/bin/python docs/blogpost/scripts/perplexity_eval.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --adapters-file /workspace/runs/mo/oct-llama8b/adapters_resolved.txt \
    --out-root /workspace/runs/mo/oct-llama8b/ppl --n-docs 200 --max-tokens 512
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def load_webtext(n_docs: int, min_chars: int, revision: str | None) -> list[str]:
    """Stream FineWeb and take the FIRST n_docs with >= min_chars (deterministic order, so
    every model/pod sees identical docs — see saved doc_hashes). `revision` pins the dataset
    snapshot for reproducibility (None = current main)."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train",
                      streaming=True, revision=revision)
    docs = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if len(t) >= min_chars:
            docs.append(t)
        if len(docs) >= n_docs:
            break
    return docs


def shuffle_words(doc: str, rng: random.Random) -> str:
    w = doc.split()
    rng.shuffle(w)
    return " ".join(w)


def corpus_ppl(model, tok, docs: list[str], max_tokens: int, device) -> dict:
    """Exact teacher-forced corpus perplexity: exp(sum NLL / sum tokens). Also returns per-doc
    (summed NLL, token count) so callers can bootstrap doc-level CIs offline."""
    import math
    import torch
    per_nll, per_tok = [], []
    model.eval()
    with torch.no_grad():
        for doc in docs:
            ids = tok(doc, return_tensors="pt", truncation=True,
                      max_length=max_tokens).input_ids.to(device)
            n = ids.shape[1] - 1                          # shifted targets
            if n < 1:
                per_nll.append(0.0); per_tok.append(0); continue
            out = model(ids, labels=ids)                  # out.loss = mean NLL over targets
            per_nll.append(float(out.loss) * n)           # -> sum NLL for this doc
            per_tok.append(n)
    tot_nll, tot_tok = sum(per_nll), sum(per_tok)
    mean_nll = tot_nll / max(tot_tok, 1)
    return {"ppl": math.exp(mean_nll), "mean_nll": mean_nll, "n_tokens": tot_tok,
            "per_doc_nll": per_nll, "per_doc_tokens": per_tok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapters-file", required=True, help="one repo id / local leaf dir per line")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="word-shuffle seed (doc selection is "
                    "deterministic first-N, not seeded)")
    ap.add_argument("--fineweb-revision", default=None,
                    help="pin the FineWeb dataset revision for reproducibility")
    ap.add_argument("--no-base", action="store_true")
    args = ap.parse_args()

    import hashlib
    import torch
    from question_consistency.elicit import load_model
    from run_adapter_sweep import load_adapter_list

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    natural = load_webtext(args.n_docs, args.min_chars, args.fineweb_revision)
    rng = random.Random(args.seed)
    shuffled = [shuffle_words(d, rng) for d in natural]
    doc_hashes = [hashlib.sha1(d.encode()).hexdigest()[:16] for d in natural]
    print(f"loaded {len(natural)} natural docs (+matched shuffled)")

    tok, base = load_model(args.base_model, "bfloat16")
    device = base.device
    results = {}

    def _eval(name, model):
        nat = corpus_ppl(model, tok, natural, args.max_tokens, device)
        shf = corpus_ppl(model, tok, shuffled, args.max_tokens, device)
        gap = shf["ppl"] / nat["ppl"]
        results[name] = {"natural": nat, "shuffled": shf, "shuffled_over_natural": gap,
                         # NLL deltas are the un-exaggerated view (PPL is exponential)
                         "structure_bonus_nll": shf["mean_nll"] - nat["mean_nll"]}
        print(f"{name}: PPL_nat={nat['ppl']:.3f} PPL_shuf={shf['ppl']:.3f} gap={gap:.2f} "
              f"NLL_nat={nat['mean_nll']:.3f}")

    if not args.no_base:
        _eval("base", base)

    from peft import PeftModel
    pmodel = None
    for repo in load_adapter_list(args.adapters_file):
        name = repo.rstrip("/").split("/")[-1]
        if pmodel is None:
            pmodel = PeftModel.from_pretrained(base, repo, adapter_name="cur")
        else:
            pmodel.load_adapter(repo, adapter_name="cur")
        pmodel.set_adapter("cur")
        _eval(name, pmodel)
        try:
            pmodel.delete_adapter("cur")
        except Exception:
            pass

    # relative-to-base deltas for the headline view (ratios + un-exaggerated NLL deltas)
    if "base" in results:
        b = results["base"]
        for name, r in results.items():
            r["vs_base"] = {
                "natural_ppl_ratio": r["natural"]["ppl"] / b["natural"]["ppl"],
                "shuffled_ppl_ratio": r["shuffled"]["ppl"] / b["shuffled"]["ppl"],
                "natural_nll_delta": r["natural"]["mean_nll"] - b["natural"]["mean_nll"],
                "shuffled_nll_delta": r["shuffled"]["mean_nll"] - b["shuffled"]["mean_nll"],
            }

    (out_root / "perplexity.json").write_text(json.dumps(
        {"base_model": args.base_model, "n_docs": len(natural),
         "max_tokens": args.max_tokens, "seed": args.seed,
         "fineweb_revision": args.fineweb_revision, "doc_hashes": doc_hashes,
         "results": results}, indent=2))
    print(f"wrote {out_root/'perplexity.json'}")


if __name__ == "__main__":
    main()
