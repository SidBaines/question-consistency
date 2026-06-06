"""Resolve an adapter spec file to a flat adapters list usable by both the sentiment sweep
and lm-eval (neither handles HF subfolder adapters directly).

Spec lines (blank / # ignored):
  owner/repo                          -> passed through unchanged (flat HF adapter repo)
  owner/repo::subfolder::name         -> snapshot_download the subfolder, emit the LOCAL
                                         leaf path /workspace/adapters/<name>/<subfolder>

Usage:
  python materialize_adapters.py --specs adapter_specs_oct.txt --out /tmp/adapters_resolved.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", default="/workspace/adapters")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    resolved = []
    for line in Path(args.specs).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("::")
        if len(parts) == 1:
            resolved.append(parts[0])                      # flat repo, no download needed
            continue
        repo, sub, name = parts[0], parts[1], (parts[2] if len(parts) > 2 else parts[1])
        local = Path(args.dest) / name
        snapshot_download(repo, allow_patterns=[f"{sub}/*"], local_dir=str(local),
                          max_workers=1)                   # max_workers=1: hub deadlock guard
        leaf = local / sub
        assert (leaf / "adapter_config.json").exists(), f"no adapter_config.json in {leaf}"
        resolved.append(str(leaf))
        print(f"materialized {repo}::{sub} -> {leaf}")

    Path(args.out).write_text("\n".join(resolved) + "\n")
    print(f"wrote {len(resolved)} adapters -> {args.out}")


if __name__ == "__main__":
    main()
