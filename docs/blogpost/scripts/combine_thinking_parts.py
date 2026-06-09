#!/usr/bin/env python3
"""Combine the per-model Qwen3 *thinking* fan-out parts into the canonical
`mo/auditbench-qwen3-14b-thinking/` suite tarball that `build_results_table.py` consumes — and
do it DISK-LIGHT (local disk is typically full after a run).

Each model ran on its own pod and uploaded its lmeval output to
`mo/auditbench-qwen3-14b-thinking-parts/<model>/...tar.gz`; the gather poller downloaded those
locally under `runs/mo/qwen3-thinking-parts/raw_<name>/`. This script:
  1. computes the robust thinking-MMLU per model IN-MEMORY (extract_generative_mmlu — the stock
     lm-eval `get_response` reads ~0 on CoT answers),
  2. writes a TINY combined tarball with, per model, only `lmeval/<model>/<MODELDIR>/results_*.json`
     (IFEval lives here, reliable) + a `lmeval/<model>/mmlu_robust.json` sidecar (the real MMLU).
     NO samples_*.jsonl — the table never reads them and they're the bulk; full data stays in -parts/,
  3. self-verifies by extracting the tiny tarball and running build_results_table.metrics_for,
  4. with --upload, pushes to `mo/auditbench-qwen3-14b-thinking/...` (overwrites the old broken tarball).

Usage:
  python docs/blogpost/scripts/combine_thinking_parts.py            # build + verify only
  python docs/blogpost/scripts/combine_thinking_parts.py --upload   # build + verify + push to HF
"""
from __future__ import annotations
import argparse, glob, io, json, os, sys, tarfile, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from extract_generative_mmlu import extract_letter, answer_text, gold_letter

PARTS_DIR = REPO / "runs" / "mo" / "qwen3-thinking-parts"
SUITE = "auditbench-qwen3-14b-thinking"
OUT_TAR = Path(tempfile.gettempdir()) / f"{SUITE}_combined.tar.gz"
HF_REPO = "arcadia-impact/sentiment-utility-logs"
PATH_IN_REPO = f"mo/{SUITE}/{SUITE}_nogit.tar.gz"   # overwrite the old broken tarball


def robust_mmlu(tf: tarfile.TarFile) -> dict:
    """Robust generative-MMLU score from a part tarball's samples, read in-memory."""
    n = correct = found = 0
    for m in tf.getmembers():
        b = os.path.basename(m.name)
        if b.startswith("samples_mmlu_") and "_generative_" in b and b.endswith(".jsonl"):
            for line in tf.extractfile(m):
                s = json.loads(line)
                g = gold_letter(s)
                if g is None:
                    continue
                got, _ = extract_letter(answer_text(s))
                n += 1; found += got is not None; correct += got == g
    return {"mmlu": (correct / n if n else None),
            "found_rate": (found / n if n else None), "n": n,
            "source": "extract_generative_mmlu.py (robust last-match); get_response is ~0 for CoT"}


def build() -> dict:
    parts = sorted(glob.glob(str(PARTS_DIR / "raw_*")))
    if not parts:
        raise SystemExit(f"no parts under {PARTS_DIR} — run the gather poller first")
    summary = {}
    with tarfile.open(OUT_TAR, "w:gz") as out:
        for d in parts:
            tars = glob.glob(d + "/**/*.tar.gz", recursive=True)
            if not tars:
                print(f"  (no tarball in {d})"); continue
            with tarfile.open(tars[0]) as tf:
                res = [m for m in tf.getmembers()
                       if "results_" in os.path.basename(m.name) and m.name.endswith(".json")]
                if not res:
                    print(f"  (no results_*.json in {tars[0]})"); continue
                for m in res:                                  # copy results_*.json verbatim
                    data = tf.extractfile(m).read()
                    ti = tarfile.TarInfo(name=m.name); ti.size = len(data)
                    out.addfile(ti, io.BytesIO(data))
                model = res[0].name.split("/")[2]              # <suite>/lmeval/<model>/<MODELDIR>/...
                modeldir = "/".join(res[0].name.split("/")[:3])
                rob = robust_mmlu(tf)
                side = json.dumps(rob, indent=2).encode()
                ti = tarfile.TarInfo(name=f"{modeldir}/mmlu_robust.json"); ti.size = len(side)
                out.addfile(ti, io.BytesIO(side))
                summary[model] = rob
                print(f"  {model:55} mmlu={rob['mmlu']:.4f} found={rob['found_rate']:.1%} n={rob['n']}")
    print(f"built {OUT_TAR}  ({OUT_TAR.stat().st_size/1024:.0f} KB)")
    return summary


def verify():
    """Extract the tiny combined tarball and run the table's metrics_for on each model."""
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    import build_results_table as B
    tmp = Path(tempfile.mkdtemp(prefix="combine_verify_"))
    with tarfile.open(OUT_TAR) as t:
        t.extractall(tmp)
    root = tmp / SUITE
    print("\n=== verify: build_results_table.metrics_for on the combined tarball ===")
    for model in B.discover_models(root):
        m = B.metrics_for(root, model)
        print(f"  {model:55} MMLU={m.get('mmlu')}  IFEval={m.get('ifeval')}")


def upload():
    from huggingface_hub import HfApi
    tok = os.environ.get("HF_WRITE_TOKEN_ARCADIA") or os.environ.get("HF_TOKEN")
    HfApi(token=tok).upload_file(path_or_fileobj=str(OUT_TAR), path_in_repo=PATH_IN_REPO,
                                 repo_id=HF_REPO, repo_type="dataset",
                                 commit_message="Qwen3-14B thinking suite: combined parts + robust-MMLU sidecars")
    print(f"uploaded -> {HF_REPO}/{PATH_IN_REPO} (overwrote old broken tarball)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="push combined tarball to HF (overwrites)")
    a = ap.parse_args()
    build()
    verify()
    if a.upload:
        upload()
