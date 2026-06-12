"""Add bits-per-byte to the perplexity results — CPU only, no model forward passes.

BPB = total NLL / ln(2) / UTF-8 bytes of the text the model was actually scored on.
The NLL numerators are already stored per-doc in mo/<suite>/perplexity_1m.json; this
script reconstructs the denominators: re-stream the same FineWeb docs (deterministic
first-N order, verified against the stored doc_hashes), re-tokenize per family with
truncation to the run's max_tokens, and count the bytes covered by the *predicted*
tokens (positions 1..L-1 — token 0 is BOS for Llama / the first unpredicted content
token for Qwen). Byte spans come from the fast tokenizers' offset mappings, so the
512-token truncation point is exact per family.

Why bother: raw PPL is not comparable across tokenizer families (Llama tokenizes the
same text into ~1% fewer tokens, deflating its PPL); BPB is the tokenizer-fair column.

Usage (any machine with datasets+transformers, needs HF_TOKEN for llama tokenizers):
  python docs/blogpost/scripts/bits_per_byte.py \
    --ppl-dir /path/with/perplexity_1m_jsons --out bpb.json [--upload]
Input dir holds <suite>.json files in the perplexity_1m.json schema (e.g. downloaded
from mo/<suite>/perplexity_1m.json). With --upload, each suite's record is also pushed
to HF as mo/<suite>/bpb.json — the sidecar build_results_table.py reads for its BPB
column (needs HF_WRITE_TOKEN_ARCADIA or HF_WRITE in the env).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

# suite -> tokenizer that run used (= its --base-model)
SUITE_TOKENIZER = {
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "llama-3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "oct-llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "auditbench-qwen3-14b": "Qwen/Qwen3-14B",
    "auditbench-llama70b": "meta-llama/Llama-3.3-70B-Instruct",
    "interp-gemma12b": "google/gemma-3-12b-it",   # one-off exp/ suite (not a blogpost row)
}
N_DOCS = 2500
MIN_CHARS = 500


def load_docs(n_docs: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train",
                      streaming=True)
    docs = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if len(t) >= MIN_CHARS:
            docs.append(t)
        if len(docs) >= n_docs:
            break
    return docs


def predicted_bytes(tokenizer_id: str, docs: list[str], max_tokens: int) -> list[int]:
    """Per-doc UTF-8 bytes spanned by the predicted tokens (positions 1..L-1)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    out = []
    for d in docs:
        enc = tok(d, truncation=True, max_length=max_tokens,
                  return_offsets_mapping=True)
        offs = [o for o in enc["offset_mapping"] if o[1] > o[0]]  # drop BOS/specials (0,0)
        if len(enc["input_ids"]) < 2 or not offs:
            out.append(0)
            continue
        # Llama: BOS unpredicted, all content tokens predicted -> full content span.
        # Qwen: first content token unpredicted -> span starts at the second token.
        n_specials = len(enc["input_ids"]) - len(offs)
        start = offs[0][0] if n_specials >= 1 else offs[1][0] if len(offs) > 1 else offs[0][1]
        out.append(len(d[start:offs[-1][1]].encode("utf-8")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppl-dir", required=True, help="dir of <suite>.json (perplexity_1m schema)")
    ap.add_argument("--out", default="bpb.json")
    ap.add_argument("--upload", action="store_true",
                    help="also push each suite's record to HF mo/<suite>/bpb.json")
    ap.add_argument("--repo", default="arcadia-impact/sentiment-utility-logs")
    args = ap.parse_args()

    files = {p.stem: p for p in Path(args.ppl_dir).glob("*.json")
             if p.stem in SUITE_TOKENIZER}
    if not files:
        raise SystemExit(f"no recognized suite jsons in {args.ppl_dir}")

    docs = load_docs(N_DOCS)
    hashes = [hashlib.sha1(d.encode()).hexdigest()[:16] for d in docs]

    byte_cache: dict[tuple[str, int], list[int]] = {}
    results: dict[str, dict] = {}
    for suite, path in sorted(files.items()):
        d = json.loads(path.read_text())
        stored = d["doc_hashes"]
        assert hashes[: len(stored)] == stored, f"{suite}: doc set mismatch — refusing"
        key = (SUITE_TOKENIZER[suite], d["max_tokens"])
        if key not in byte_cache:
            byte_cache[key] = predicted_bytes(key[0], docs, key[1])
        per_doc = byte_cache[key][: d["n_docs"]]
        nbytes = sum(per_doc)
        # per_doc_bytes: same doc order as perplexity_1m.json's per_doc_nll — lets downstream
        # consumers (plot_results_bars.py) cluster-bootstrap BPB over docs for error bars
        results[suite] = {"tokenizer": key[0], "predicted_bytes": nbytes,
                          "per_doc_bytes": per_doc, "models": {}}
        for model, r in d["results"].items():
            nll = sum(r["natural"]["per_doc_nll"])
            results[suite]["models"][model] = {
                "ppl_nat": r["natural"]["ppl"],
                "bits_per_byte_nat": nll / math.log(2) / nbytes,
            }
        print(f"{suite}: bytes={nbytes}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")

    if args.upload:
        import os
        from huggingface_hub import HfApi
        token = os.environ.get("HF_WRITE_TOKEN_ARCADIA") or os.environ.get("HF_WRITE")
        if not token:
            raise SystemExit("--upload needs HF_WRITE_TOKEN_ARCADIA (or HF_WRITE) in the env")
        api = HfApi(token=token)
        for suite, rec in results.items():
            api.upload_file(
                path_or_fileobj=json.dumps(rec, indent=2).encode(),
                path_in_repo=f"mo/{suite}/bpb.json",
                repo_id=args.repo, repo_type="dataset",
            )
            print(f"uploaded mo/{suite}/bpb.json")


if __name__ == "__main__":
    main()
