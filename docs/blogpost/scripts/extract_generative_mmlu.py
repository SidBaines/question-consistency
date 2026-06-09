#!/usr/bin/env python3
"""Re-score Qwen3 *thinking* generative-MMLU from lm-eval log_samples (robust extraction).

WHY THIS EXISTS
---------------
`mmlu_generative`'s built-in answer extractor (`get_response`) takes the FIRST LINE of the model
output and requires it to be EXACTLY the gold letter:

    filter get_response:  regex "^(.*?)(?=\\n|$)"  ->  remove_whitespace  ->  take_first
    metric:               exact_match  vs  "A"/"B"/"C"/"D"

That works for completion models that answer with a bare " B\\n". A chat/thinking model emits its
answer as PROSE after the stripped <think> block, usually preceded by a blank line:

    "\\n\\nThe correct answer is **B**."

so `get_response` extracts "" (or a whole sentence) and exact_match is ~0 even when the model is
right. (Combined with the old until=["</s>","\\n"] stop, the prior thinking run scored 0.0 on all
14042 questions — see run_qwen3_thinking.sh.)

This script searches the WHOLE generation for the answer letter with tiered, last-match-wins
patterns and recomputes accuracy. It is also the validator for the limit-20 smoke test: run it on
the smoke output and confirm the answer-found rate jumps and accuracy lands well above 0.25.

INPUT
-----
A directory containing `samples_mmlu_*_generative_*.jsonl` (one file per subject), e.g.
    runs/mo/auditbench-qwen3-14b-thinking/lmeval/<model>/Qwen__Qwen3-14B/
or an extracted suite tarball. Globs recursively, groups by subject parsed from the filename.

USAGE
-----
    python docs/blogpost/scripts/extract_generative_mmlu.py --dir <samples_dir>
    python docs/blogpost/scripts/extract_generative_mmlu.py --dir <dir> --out summary.json --show-fails 8
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

# Tiered extractors, most reliable first. Within a tier we take the LAST match (a model that
# reasons aloud may name several letters before concluding). The first tier with any match wins.
_CUE = re.compile(
    r"(?:final answer|the answer|correct answer|answer is|answer\s*:|option|choice)\s*"
    r"(?:is\s*)?[:\-]?\s*[*_`(\[]*\s*([A-D])\b",
    re.IGNORECASE,
)
_BOXED = re.compile(r"\\boxed\{\s*\(?\s*([A-D])\b")
_BOLD = re.compile(r"(?:\*\*|__)\s*\(?\s*([A-D])\s*\)?\s*(?:\*\*|__)")
_PAREN = re.compile(r"\(\s*([A-D])\s*\)")
_LETTER_PUNCT = re.compile(r"\b([A-D])\s*[\.\):,]")  # "B." "B)" "B:" "B,"
_BARE = re.compile(r"\b([A-D])\b")

_TIERS = [("cue", _CUE), ("boxed", _BOXED), ("bold", _BOLD),
          ("paren", _PAREN), ("punct", _LETTER_PUNCT), ("bare", _BARE)]


def extract_letter(text: str):
    """Return (letter, method) or (None, None)."""
    for method, rx in _TIERS:
        matches = rx.findall(text)
        if matches:
            return matches[-1].upper(), method
    return None, None


def answer_text(sample: dict) -> str:
    """The model's answer span. think_end_token usually strips up to </think> already; if a raw
    block survives (e.g. think_end_token not applied), keep only the text after the LAST </think>."""
    resps = sample.get("resps") or sample.get("filtered_resps") or []
    raw = ""
    if resps and isinstance(resps[0], list) and resps[0]:
        raw = resps[0][0]
    elif resps and isinstance(resps[0], str):
        raw = resps[0]
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1]
    return raw


def gold_letter(sample: dict):
    t = sample.get("target")
    if isinstance(t, str) and t.strip().upper() in "ABCD":
        return t.strip().upper()
    if isinstance(t, int) and 0 <= t < 4:
        return "ABCD"[t]
    doc = sample.get("doc", {})
    a = doc.get("answer")
    if isinstance(a, int) and 0 <= a < 4:
        return "ABCD"[a]
    if isinstance(a, str) and a.strip().upper() in "ABCD":
        return a.strip().upper()
    return None


_SUBJECT = re.compile(r"samples_mmlu_(.+?)_generative_")


def subject_of(path: str) -> str:
    m = _SUBJECT.search(Path(path).name)
    return m.group(1) if m else "unknown"


def stock_get_response(root: Path):
    """lm-eval's own reported mmlu_generative exact_match (for the before/after contrast)."""
    for jf in sorted(glob.glob(str(root / "**" / "results_*.json"), recursive=True)):
        try:
            r = json.loads(Path(jf).read_text()).get("results", {})
        except Exception:
            continue
        if "mmlu_generative" in r:
            return r["mmlu_generative"].get("exact_match,get_response")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="dir holding samples_mmlu_*_generative_*.jsonl (recursive)")
    ap.add_argument("--out", help="write a JSON summary here")
    ap.add_argument("--show-fails", type=int, default=0, help="print this many extraction failures/mismatches")
    args = ap.parse_args()

    root = Path(args.dir)
    files = sorted(glob.glob(str(root / "**" / "samples_mmlu_*_generative_*.jsonl"), recursive=True))
    if not files:
        raise SystemExit(f"no samples_mmlu_*_generative_*.jsonl under {root}")

    per_subj = defaultdict(lambda: {"n": 0, "correct": 0, "found": 0})
    methods = defaultdict(int)
    fails = []  # (subject, gold, got, method, text)
    n = correct = found = 0

    for fp in files:
        subj = subject_of(fp)
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            gold = gold_letter(s)
            if gold is None:
                continue
            text = answer_text(s)
            got, method = extract_letter(text)
            n += 1
            per_subj[subj]["n"] += 1
            methods[method or "none"] += 1
            if got is not None:
                found += 1
                per_subj[subj]["found"] += 1
            ok = (got == gold)
            if ok:
                correct += 1
                per_subj[subj]["correct"] += 1
            elif len(fails) < args.show_fails:
                fails.append((subj, gold, got, method, text[:240]))

    acc_all = correct / n if n else 0.0          # unextracted counts as wrong (true accuracy)
    acc_found = correct / found if found else 0.0  # accuracy among questions where we found a letter
    stock = stock_get_response(root)

    print(f"\n=== robust generative-MMLU re-score: {root} ===")
    print(f"samples: {n}   answer-found: {found} ({found/n:.1%})   correct: {correct}")
    print(f"ACCURACY (all, unfound=wrong):     {acc_all:.4f}")
    print(f"accuracy (among answer-found only): {acc_found:.4f}")
    if stock is not None:
        print(f"lm-eval stock get_response (broken for thinking): {stock}")
    print(f"extraction method counts: {dict(sorted(methods.items(), key=lambda kv: -kv[1]))}")

    worst = sorted(((v["correct"] / v["n"], k, v) for k, v in per_subj.items() if v["n"]))[:8]
    print("\nlowest-scoring subjects (sanity-check for extraction gaps, not capability):")
    for acc, k, v in worst:
        print(f"  {acc:.3f}  {k:<34} (found {v['found']}/{v['n']})")

    for subj, gold, got, method, text in fails:
        print(f"\n--- FAIL [{subj}] gold={gold} got={got} ({method})\n    {text!r}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "dir": str(root), "samples": n, "answer_found": found, "correct": correct,
            "accuracy_all": acc_all, "accuracy_found": acc_found,
            "stock_get_response": stock, "methods": dict(methods),
            "per_subject": {k: v for k, v in per_subj.items()},
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
