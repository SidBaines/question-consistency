"""Materialize every adapter in a spec file to a clean LOCAL leaf dir, so both the sentiment
sweep (PeftModel) and vLLM lm-eval (lora_local_path, local-only) can load them.

Spec lines (blank / # ignored):
  owner/repo                  -> flat HF adapter repo; name = basename(repo)
  owner/repo::subfolder::name -> subfolder adapter inside repo; leaf renamed to <name>

Output (--out): one local leaf dir per line (each contains adapter_config.json directly).
That file is the adapters-file for run_em_sentiment.py AND the ADAPTERS list for run_em_lmeval.

Usage:
  python materialize_adapters.py --specs adapter_specs_oct.txt --out /tmp/adapters_resolved.txt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _ensure_leaf(dest: Path, name: str, repo: str, sub: str | None) -> Path:
    """Produce <dest>/<name>/ containing adapter_config.json directly. Idempotent."""
    from huggingface_hub import snapshot_download

    leaf = dest / name
    if (leaf / "adapter_config.json").exists():
        return leaf
    leaf.mkdir(parents=True, exist_ok=True)
    if sub:
        tmp = dest / f"_dl_{name}"
        snapshot_download(repo, allow_patterns=[f"{sub}/*"], local_dir=str(tmp),
                          max_workers=1)                       # max_workers=1: hub deadlock guard
        for p in (tmp / sub).iterdir():
            shutil.move(str(p), str(leaf / p.name))
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        snapshot_download(repo, local_dir=str(leaf), max_workers=1)
    assert (leaf / "adapter_config.json").exists(), f"no adapter_config.json in {leaf}"
    return leaf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", default="/workspace/adapters")
    args = ap.parse_args()
    dest = Path(args.dest)

    leaves = []
    for line in Path(args.specs).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("::")
        if len(parts) == 1:
            repo, sub, name = parts[0], None, parts[0].split("/")[-1]
        elif len(parts) == 3:
            repo, sub, name = parts[0], parts[1], parts[2]
        else:
            raise ValueError(f"bad spec {line!r}: use 'repo' or 'repo::subfolder::name'")
        leaf = _ensure_leaf(dest, name, repo, sub)
        leaves.append(str(leaf))
        print(f"materialized {repo}{('::'+sub) if sub else ''} -> {leaf}")

    Path(args.out).write_text("\n".join(leaves) + "\n")
    print(f"wrote {len(leaves)} adapters -> {args.out}")


if __name__ == "__main__":
    main()
