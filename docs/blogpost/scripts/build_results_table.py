"""Build a LaTeX PDF results table from the HF-hosted suite logs (blogpost).

Pulls each suite tarball from the HF dataset (default arcadia-impact/sentiment-utility-logs
under mo/<suite>/), extracts it, parses every eval, and renders one grouped table:
rows = model organisms grouped by suite (base row first, then each MO); columns = evals.
Unpopulated cells -> '-'.

Columns (one headline metric per eval; edit COLUMNS to taste):
  decis_mu (sentiment coherence) · MMLU · IFEval(prompt-strict) · PPL_nat ·
  XSTest(over-refusal on safe) · StrongREJECT(harm score)

Run from repo root in the elicit venv (needs four_metrics + huggingface_hub):
  uv run python docs/blogpost/scripts/build_results_table.py
  -> docs/blogpost/results_table.tex (+ .pdf if pdflatex is installed)
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

# We read ONLY the standardised mo/<suite>/ path (the orchestrated vLLM runs). The legacy
# HF-backend EM results under em/ are deliberately NOT read — those rows stay '-' until the
# EM suite is re-run through run_suite.sh and uploaded to mo/.
READ_PREFIXES = ("mo",)

# Groups to ALWAYS render, even with no data yet (rows show '-' until their mo/<suite> lands).
# suite key must match the future SUITE= used by run_suite.sh. Auto-discovered mo/ suites not
# listed here are appended afterwards.
REGISTRY = {
    "qwen2.5-14b-instruct": {
        "label": "EM — Qwen2.5-14B-Instruct",
        "models": ["base",
                   "Qwen2.5-14B-Instruct_bad-medical-advice",
                   "Qwen2.5-14B-Instruct_risky-financial-advice",
                   "Qwen2.5-14B-Instruct_extreme-sports"],
    },
}
# Optional prettier group headers; falls back to REGISTRY label then the raw suite name.
SUITE_LABELS = {
    "oct-llama8b": "OCT personas — Llama-3.1-8B-Instruct",
    "auditbench-llama70b": "AuditBench — Llama-3.3-70B-Instruct",
}
# Explicit suite ordering; unlisted suites appended alphabetically.
SUITE_ORDER = ["qwen2.5-14b-instruct", "oct-llama8b", "auditbench-llama70b"]

# (column key, LaTeX header, value-formatter)
COLUMNS = [
    ("decis_mu",   r"decis\_mu",     lambda v: f"{v:.3f}"),
    ("mmlu",       r"MMLU",          lambda v: f"{v:.3f}"),
    ("ifeval",     r"IFEval",        lambda v: f"{v:.3f}"),
    ("ppl_nat",    r"PPL$_\mathrm{nat}$", lambda v: f"{v:.2f}"),
    ("ppl_shuf",   r"PPL$_\mathrm{shuf}$", lambda v: f"{v:.1f}"),  # curiosity; never coloured
    ("xstest",     r"XSTest$_\mathrm{ovr}$", lambda v: f"{v:.3f}"),
    ("strongreject", r"StrongREJECT", lambda v: f"{v:.3f}"),
]

# Per-column colouring metadata (for --color): (higher_is_worse, bounded_0_1).
#  - decis_mu/MMLU/IFEval: lower is worse, bounded in [0,1].
#  - XSTest(over-refusal)/StrongREJECT(harm): higher is worse, bounded in [0,1].
#  - PPL: higher is worse, unbounded above -> intensity uses fractional change vs base.
COLOR_META = {
    "decis_mu":     (False, True),
    "mmlu":         (False, True),
    "ifeval":       (False, True),
    "ppl_nat":      (True,  False),
    "xstest":       (True,  True),
    "strongreject": (True,  True),
}
COLOR_MAX_PCT = 65   # cap cell tint so text stays readable


def cell_intensity(key, base, val):
    """Return (intensity in [0,1], is_bad). Movement from base as a proportion of the
    headroom in that direction (bounded metrics) or fractional change (PPL)."""
    higher_is_worse, bounded = COLOR_META[key]
    delta = val - base
    if delta == 0:
        return 0.0, False
    moved_up = delta > 0
    is_bad = (moved_up == higher_is_worse)
    if bounded:
        denom = (1.0 - base) if moved_up else base      # room to the 1 / 0 bound
    else:
        denom = abs(base)                               # PPL: fractional change
    frac = abs(delta) / max(denom, 1e-6)
    return min(frac, 1.0), is_bad


def _tex_escape(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&") \
            .replace("%", r"\%").replace("#", r"\#")


def fetch_suites(repo: str, token: str | None, only: list[str] | None):
    """Return {suite: extracted_suite_root}. Downloads + extracts each suite tarball."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset", token=token)
    tarballs = {}  # suite -> path-in-repo
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3 and parts[0] in READ_PREFIXES and f.endswith(".tar.gz"):
            tarballs[parts[-2]] = f          # suite = immediate parent dir; last one wins
    if only:
        tarballs = {k: v for k, v in tarballs.items() if k in only}
    roots = {}
    tmp = Path(tempfile.mkdtemp(prefix="results_table_"))
    dl = tmp / "_dl"; dl.mkdir()
    for suite, pir in tarballs.items():
        # download INTO tmp (not the persistent HF cache) so cleanup reclaims it all
        local = hf_hub_download(repo, pir, repo_type="dataset", token=token, local_dir=str(dl))
        dest = tmp / suite
        with tarfile.open(local) as t:
            # extract ONLY the small metric files we read — skip lm-eval's huge
            # samples_*.jsonl + safety per-prompt jsonls (they bloat local disk badly).
            want = [m for m in t.getmembers() if (
                m.name.endswith("/edges.jsonl") or m.name.endswith("perplexity.json")
                or m.name.endswith("safety_summary.json")
                or ("results_" in m.name and m.name.endswith(".json")))]
            t.extractall(dest, members=want)
        Path(local).unlink(missing_ok=True)     # drop the tarball; keep only the extraction
        # tarball root is the suite dir (e.g. dest/<suite>/...)
        inner = dest / suite
        roots[suite] = inner if inner.exists() else dest
        print(f"fetched {suite} <- {pir}")
    return roots, tmp


def discover_models(root: Path) -> list[str]:
    models = set()
    for e in root.glob("*/edges.jsonl"):
        models.add(e.parent.name)
    if (root / "lmeval").exists():
        models.update(d.name for d in (root / "lmeval").iterdir() if d.is_dir())
    ppl = root / "ppl" / "perplexity.json"
    if ppl.exists():
        models.update(json.loads(ppl.read_text()).get("results", {}).keys())
    summ = root / "safety" / "safety_summary.json"
    if summ.exists():
        ds = json.loads(summ.read_text()).get("datasets", {})
        for d in ds.values():
            models.update(d.keys())
    ordered = (["base"] if "base" in models else []) + sorted(m for m in models if m != "base")
    return ordered


def metrics_for(root: Path, model: str) -> dict:
    out = {}
    # sentiment coherence (decis_mu) from edges
    edges = root / model / "edges.jsonl"
    if edges.exists():
        try:
            from four_metrics import compute_decis_mu
            out["decis_mu"] = compute_decis_mu(str(edges))
        except Exception as e:
            print(f"  decis_mu fail {model}: {e}")
    # capability from lm-eval json
    js = glob.glob(str(root / "lmeval" / model / "**" / "results_*.json"), recursive=True)
    if js:
        r = json.loads(Path(sorted(js)[-1].replace("\\", "/")).read_text())["results"]
        if "mmlu" in r:
            out["mmlu"] = r["mmlu"].get("acc,none")
        if "ifeval" in r:
            out["ifeval"] = r["ifeval"].get("prompt_level_strict_acc,none")
    # perplexity
    ppl = root / "ppl" / "perplexity.json"
    if ppl.exists():
        pr = json.loads(ppl.read_text()).get("results", {}).get(model)
        if pr:
            out["ppl_nat"] = pr["natural"]["ppl"]
            out["ppl_shuf"] = pr["shuffled"]["ppl"]
    # safety
    summ = root / "safety" / "safety_summary.json"
    if summ.exists():
        ds = json.loads(summ.read_text()).get("datasets", {})
        x = ds.get("xstest", {}).get(model)
        if x:
            out["xstest"] = x.get("over_refusal_rate_safe")
        s = ds.get("strongreject", {}).get(model)
        if s:
            out["strongreject"] = s.get("mean_harm_score")
    return out


def _models_for(suite: str, root: Path | None) -> list[str]:
    """Registry models (always shown) unioned with any discovered in the tarball; base first."""
    reg = REGISTRY.get(suite, {}).get("models", [])
    disc = discover_models(root) if root else []
    allm = list(dict.fromkeys(reg + disc))           # preserve order, dedup
    return (["base"] if "base" in allm else []) + [m for m in allm if m != "base"]


def build_tex(roots: dict[str, Path], color: bool = False) -> str:
    # registry groups always render (even with no data); then discovered mo/ suites; SUITE_ORDER wins.
    all_suites = list(dict.fromkeys(list(REGISTRY) + list(roots)))
    suites = [s for s in SUITE_ORDER if s in all_suites] + \
             sorted(s for s in all_suites if s not in SUITE_ORDER)
    ncol = len(COLUMNS)
    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs,geometry,amsmath}",
        r"\usepackage[table]{xcolor}",
        r"\geometry{landscape,margin=1.2cm}",
        r"\begin{document}",
        r"\begin{table}[t]\centering\small",
        r"\caption{Model-organism evaluation panel (`-' = not yet collected).}",
        r"\begin{tabular}{l" + "r" * ncol + "}",
        r"\toprule",
        "Model & " + " & ".join(h for _, h, _ in COLUMNS) + r" \\",
    ]
    for suite in suites:
        root = roots.get(suite)
        label = SUITE_LABELS.get(suite) or REGISTRY.get(suite, {}).get("label") or suite
        base_m = metrics_for(root, "base") if root else {}
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{ncol + 1}}}{{l}}{{\textbf{{{_tex_escape(label)}}}}} \\")
        for model in _models_for(suite, root):
            m = metrics_for(root, model) if root else {}
            cells = []
            for key, _, fmt in COLUMNS:
                v = m.get(key)
                if not isinstance(v, (int, float)):
                    cells.append("-")
                    continue
                cell = fmt(v)
                b = base_m.get(key)
                if color and model != "base" and key in COLOR_META and isinstance(b, (int, float)):
                    inten, is_bad = cell_intensity(key, b, v)
                    if inten > 0:
                        pct = int(round(inten * COLOR_MAX_PCT))
                        cell = rf"\cellcolor{{{'red' if is_bad else 'green'}!{pct}}}{cell}"
                cells.append(cell)
            lines.append(_tex_escape(model) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\end{document}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="arcadia-impact/sentiment-utility-logs")
    ap.add_argument("--out-dir", default=str(REPO / "docs/blogpost"))
    ap.add_argument("--out-name", default="results_table")
    ap.add_argument("--suites", default=None, help="comma-sep filter")
    ap.add_argument("--color", action="store_true",
                    help="heat-map cells by movement-from-base (red=worse, green=better)")
    args = ap.parse_args()

    import os
    import shutil
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN_ARCADIA")
    only = [s.strip() for s in args.suites.split(",")] if args.suites else None
    roots, tmp = fetch_suites(args.repo, token, only)
    try:
        if not roots:
            raise SystemExit("no suites found on HF")
        tex = build_tex(roots, color=args.color)
        out_dir = Path(args.out_dir)
        tex_path = out_dir / f"{args.out_name}.tex"
        tex_path.write_text(tex)
        print(f"wrote {tex_path}")
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-output-directory", str(out_dir), str(tex_path)],
                           check=True, capture_output=True)
            print(f"wrote {out_dir / (args.out_name + '.pdf')}")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"pdflatex not run ({type(e).__name__}); .tex is ready to compile manually")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # reclaim the extraction tempdir


if __name__ == "__main__":
    main()
