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

# Each mo/<suite>'s place in the MO-type -> family -> size hierarchy (+ base-model display
# label). Rows are grouped/ordered by (TYPE_ORDER, family, size). Unknown suites fall to a
# trailing "Other" group so nothing silently disappears.
SUITE_META = {
    "qwen2.5-0.5b-instruct": {"type": "EM",        "family": "Qwen2.5",      "size": 0.5, "base": "Qwen2.5-0.5B"},
    "qwen2.5-7b-instruct":   {"type": "EM",        "family": "Qwen2.5",      "size": 7,   "base": "Qwen2.5-7B"},
    "qwen2.5-14b-instruct":  {"type": "EM",        "family": "Qwen2.5",      "size": 14,  "base": "Qwen2.5-14B"},
    "qwen2.5-32b-instruct":  {"type": "EM",        "family": "Qwen2.5",      "size": 32,  "base": "Qwen2.5-32B"},
    "llama-3.2-1b-instruct": {"type": "EM",        "family": "Llama-3",      "size": 1,   "base": "Llama-3.2-1B"},
    "llama-3.1-8b-instruct": {"type": "EM",        "family": "Llama-3",      "size": 8,   "base": "Llama-3.1-8B"},
    "oct-llama8b":           {"type": "OCT",       "family": "Llama-3.1-8B", "size": 8,   "base": "Llama-3.1-8B"},
    "auditbench-qwen3-14b":  {"type": "AuditBench","family": "Qwen3",        "size": 14,  "base": "Qwen3-14B"},
    "auditbench-llama70b":   {"type": "AuditBench","family": "Llama-3.3",    "size": 70,  "base": "Llama-3.3-70B"},
}
TYPE_ORDER = ["OCT", "EM", "AuditBench"]
# display names for the Type column (raw LaTeX, NOT _tex_escape'd — wrapped to keep the column
# narrow). Internal keys stay short ("EM"/"OCT"/"AuditBench").
TYPE_DISPLAY = {
    "EM":  r"\shortstack{Emergent\\Misalignment}",
    "OCT": r"\shortstack{Open\\Character\\Training}",
    "AuditBench": "AuditBench",
}

# Capability columns (mmlu/ifeval) for these suites are taken from a separate "-thinking"
# suite: reasoning models (Qwen3) score ~chance on loglikelihood mmlu because the chat template
# opens a <think> block, so we re-ran them with a generative task + thinking on. The override
# suite is merged into the base suite's row (matched by model name) and not rendered on its own.
CAPABILITY_OVERRIDE = {"auditbench-qwen3-14b": "auditbench-qwen3-14b-thinking"}


def _meta(suite):
    return SUITE_META.get(suite, {"type": "Other", "family": suite, "size": 999, "base": suite})


def _suite_sort_key(suite):
    m = _meta(suite)
    t = m["type"]
    return (TYPE_ORDER.index(t) if t in TYPE_ORDER else len(TYPE_ORDER), m["family"], m["size"], suite)


def _model_label(suite, model):
    """Row label: base -> the family/size base name; adapter -> just the trait (strip prefixes)."""
    base = _meta(suite)["base"]
    if model == "base":
        return f"{base} (base)"
    t = model
    for pref in ("oct-",):
        if t.startswith(pref):
            t = t[len(pref):]
    # EM: '<BaseId>_<trait>' ; AuditBench: '..._then_redteam_kto_<behavior>'
    if "_then_redteam_kto_" in t:
        t = t.split("_then_redteam_kto_")[-1]
    elif "-Instruct_" in t:
        t = t.split("-Instruct_")[-1]
    return t.replace("_", " ")

# (column key, LaTeX header, value-formatter)
COLUMNS = [
    ("decis_mu",   r"\shortstack{Pref.\\consistency}", lambda v: f"{v:.3f}"),
    ("mmlu",       r"MMLU (0-shot)", lambda v: f"{v:.3f}"),
    ("ifeval",     r"IFEval",        lambda v: f"{v:.3f}"),
    ("ppl_nat",    r"PPL$_\mathrm{nat}$", lambda v: f"{v:.2f}"),
    # tokenizer-fair twin of PPL_nat: bits-per-byte = NLL/ln2/UTF-8 bytes. Raw PPL is not
    # comparable across tokenizer families (Llama tokenizes the same text into ~1% fewer
    # tokens than Qwen, deflating its PPL); BPB is. Computed offline from perplexity_1m.json
    # by bits_per_byte.py (no GPU) and read from the mo/<suite>/bpb.json sidecar.
    ("bpb_nat",    r"BPB$_\mathrm{nat}$", lambda v: f"{v:.3f}"),
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
    "bpb_nat":      (True,  False),
    "xstest":       (True,  True),
    "strongreject": (True,  True),
}
COLOR_MAX_PCT = 65   # cap cell tint so text stays readable


def cell_intensity(key, base, val, mode="headroom"):
    """Return (intensity in [0,1], is_bad) for the move from `base` to `val`.

    mode="headroom" (default): for bounded [0,1] metrics, |delta| as a proportion of the
      headroom in the direction moved (room to the 1 / 0 bound) -> a small absolute change
      near a bound saturates fast.
    mode="absolute": for bounded [0,1] metrics, raw |delta| (already in [0,1]) -> tint is
      directly proportional to the size of the change, comparable across those metrics.
    PPL (unbounded) is fractional change vs base in BOTH modes -- raw perplexity points have
    no natural [0,1] mapping."""
    higher_is_worse, bounded = COLOR_META[key]
    delta = val - base
    if delta == 0:
        return 0.0, False
    moved_up = delta > 0
    is_bad = (moved_up == higher_is_worse)
    if not bounded:
        denom = abs(base)                               # PPL: fractional change (both modes)
    elif mode == "absolute":
        denom = 1.0                                     # raw |delta| (already in [0,1])
    else:
        denom = (1.0 - base) if moved_up else base      # room to the 1 / 0 bound
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
        # skip per-model fan-out backups (mo/<suite>-parts/<model>/...) — durable storage, not
        # table suites; the combined suite tarball is the source of truth.
        if "-parts/" in f:
            continue
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
            # samples_*.jsonl + most safety per-prompt jsonls (they bloat local disk badly).
            # strongreject _judged.jsonl IS kept: plot_results_bars.py needs the per-prompt
            # judge scores for error bars (no summary-level variance exists for a mean score).
            want = [m for m in t.getmembers() if (
                m.name.endswith("/edges.jsonl") or m.name.endswith("perplexity.json")
                or m.name.endswith("safety_summary.json") or m.name.endswith("mmlu_robust.json")
                or ("/strongreject/" in m.name and m.name.endswith("_judged.jsonl"))
                or ("results_" in m.name and m.name.endswith(".json")))]
            t.extractall(dest, members=want)
        Path(local).unlink(missing_ok=True)     # drop the tarball; keep only the extraction
        # tarball root is the suite dir (e.g. dest/<suite>/...)
        inner = dest / suite
        roots[suite] = inner if inner.exists() else dest
        print(f"fetched {suite} <- {pir}")
        # also pull the standalone higher-precision 1M-token PPL (uploaded separately from the
        # tarball). Best-effort: not every suite has it yet (e.g. the 70B may still be running).
        for sidecar, label in (("perplexity_1m.json", "1M-token PPL"),
                               ("bpb.json", "bits-per-byte")):
            try:
                p = hf_hub_download(repo, f"mo/{suite}/{sidecar}",
                                    repo_type="dataset", token=token, local_dir=str(dl))
                (roots[suite] / sidecar).write_bytes(Path(p).read_bytes())
                Path(p).unlink(missing_ok=True)
                print(f"  + {label} for {suite}")
            except Exception:
                pass
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
        if "mmlu_generative" in r:   # reasoning models (thinking suite): generative MMLU
            out["mmlu"] = r["mmlu_generative"].get("exact_match,get_response")
        # an MMLU below 4-way chance is a harness/extraction failure, not capability -> drop it
        # (e.g. Qwen3 generative get_response only parses the first line of a reasoning trace)
        if isinstance(out.get("mmlu"), (int, float)) and out["mmlu"] < 0.20:
            out.pop("mmlu")
        if "ifeval" in r:
            out["ifeval"] = r["ifeval"].get("prompt_level_strict_acc,none")
    # robust MMLU sidecar (thinking suites): generative get_response is ~0 for CoT answers, so a
    # re-extraction (extract_generative_mmlu.py) is the source of truth for mmlu when present.
    rob = root / "lmeval" / model / "mmlu_robust.json"
    if rob.exists():
        try:
            mv = json.loads(rob.read_text()).get("mmlu")
            if isinstance(mv, (int, float)):
                out["mmlu"] = mv
        except Exception as e:
            print(f"  mmlu_robust fail {model}: {e}")
    # perplexity — prefer the higher-precision 1M-token set (standalone perplexity_1m.json)
    # over the in-tarball 200-doc ppl/perplexity.json when present (same JSON schema)
    p1m = root / "perplexity_1m.json"
    ppl = p1m if p1m.exists() else (root / "ppl" / "perplexity.json")
    if ppl.exists():
        pr = json.loads(ppl.read_text()).get("results", {}).get(model)
        if pr:
            out["ppl_nat"] = pr["natural"]["ppl"]
            out["ppl_shuf"] = pr["shuffled"]["ppl"]
    # bits-per-byte (bits_per_byte.py output): prefer the per-suite HF sidecar
    # (mo/<suite>/bpb.json) when fetched, else the committed combined file in the repo
    # (results/bpb_fineweb.json, keyed by suite = the extraction dir's name).
    bpb = root / "bpb.json"
    if bpb.exists():
        rec = json.loads(bpb.read_text())
    else:
        combined = REPO / "results" / "bpb_fineweb.json"
        rec = json.loads(combined.read_text()).get(root.name, {}) if combined.exists() else {}
    br = rec.get("models", {}).get(model)
    if br and isinstance(br.get("bits_per_byte_nat"), (int, float)):
        out["bpb_nat"] = br["bits_per_byte_nat"]
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


def _models_for(root: Path | None) -> list[str]:
    """Discovered models in the tarball, base first."""
    disc = discover_models(root) if root else []
    return (["base"] if "base" in disc else []) + [m for m in disc if m != "base"]


def _metrics(roots: dict, suite: str, model: str) -> dict:
    """metrics_for(suite, model) with capability (mmlu/ifeval) overridden from the suite's
    `-thinking` companion when present (matched by model name)."""
    m = metrics_for(roots[suite], model)
    ov = CAPABILITY_OVERRIDE.get(suite)
    if ov and ov in roots:
        om = metrics_for(roots[ov], model)
        for k in ("mmlu", "ifeval"):
            m.pop(k, None)                       # the -thinking suite is the source of truth:
            if isinstance(om.get(k), (int, float)):   # don't fall back to the broken
                m[k] = om[k]                          # loglikelihood mmlu / thinking-off ifeval
    return m


def _fmt_cells(m, base_m, color_mode, is_adapter):
    cells = []
    for key, _, fmt in COLUMNS:
        v = m.get(key)
        if not isinstance(v, (int, float)):
            cells.append("-"); continue
        cell = fmt(v)
        b = base_m.get(key)
        if color_mode and is_adapter and key in COLOR_META and isinstance(b, (int, float)):
            inten, is_bad = cell_intensity(key, b, v, mode=color_mode)
            pct = int(round(inten * COLOR_MAX_PCT))
            if pct > 0:
                cell = rf"\cellcolor{{{'red' if is_bad else 'green'}!{pct}}}{cell}"
        cells.append(cell)
    return cells


def _arrow_hdr(key: str, header: str, arrows: bool) -> str:
    """Append a direction-of-'better' arrow when arrows=True: up = higher is better (the green
    direction), down = lower is better. ppl_shuf (never coloured) gets no arrow."""
    if arrows and key in COLOR_META:
        higher_is_worse, _ = COLOR_META[key]
        return header + (r" $\downarrow$" if higher_is_worse else r" $\uparrow$")
    return header


def build_tex(roots: dict[str, Path], color_mode: str | None = None, arrows: bool = False) -> str:
    # don't render override-source ("-thinking") suites as their own rows; they're merged in
    suites = [s for s in sorted(roots, key=_suite_sort_key)
              if s not in CAPABILITY_OVERRIDE.values()]
    ncol = len(COLUMNS)
    ncols_total = ncol + 3                       # Type | Base model | Model | <ncol metrics>
    # group suites by MO type (keeping the family/size sort). Each suite == one base model and is
    # rendered as a \multirow block in the "Base model" column, with a rule between base models.
    by_type = {}  # type -> ordered list of suites
    for suite in suites:
        by_type.setdefault(_meta(suite)["type"], []).append(suite)
    ordered_types = [t for t in TYPE_ORDER if t in by_type] + \
                    [t for t in by_type if t not in TYPE_ORDER]

    lines = [
        r"\documentclass{article}",
        r"\usepackage{booktabs,geometry,amsmath,multirow}",
        r"\usepackage[table]{xcolor}",
        r"\geometry{landscape,margin=1.2cm}",
        r"\begin{document}",
        r"\begin{table}[t]\centering\small",
        r"\caption{Model-organism evaluation panel, grouped by MO type / base model "
        r"(`-' = not yet collected; colour = move from the base model, "
        + (r"scaled by absolute change" if color_mode == "absolute"
           else r"scaled by headroom to the bound") + r").}",
        r"\begin{tabular}{lll" + "r" * ncol + "}",
        r"\toprule",
        "Type & Base model & Model & "
        + " & ".join(_arrow_hdr(k, h, arrows) for k, h, _ in COLUMNS) + r" \\",
    ]
    for t in ordered_types:
        models_by_suite = [(s, _models_for(roots[s])) for s in by_type[t]]
        n_type = sum(len(ms) for _, ms in models_by_suite)   # total rows for the Type \multirow
        lines.append(r"\midrule")
        first_in_type = True
        for si, (suite, models) in enumerate(models_by_suite):
            if si > 0:
                lines.append(rf"\cmidrule(l){{2-{ncols_total}}}")   # rule between base models
            base_m = _metrics(roots, suite, "base")
            basename = _meta(suite)["base"]
            for mi, model in enumerate(models):
                tname = TYPE_DISPLAY.get(t, _tex_escape(t))   # mapped names are raw LaTeX
                tcell = (rf"\multirow{{{n_type}}}{{*}}{{\textbf{{{tname}}}}}"
                         if first_in_type else "")
                first_in_type = False
                bcell = (rf"\multirow{{{len(models)}}}{{*}}{{{_tex_escape(basename)}}}"
                         if mi == 0 else "")
                label = "Instruct-tuned" if model == "base" else _model_label(suite, model)
                cells = _fmt_cells(_metrics(roots, suite, model), base_m, color_mode,
                                   model != "base")
                lines.append(f"{tcell} & {bcell} & {_tex_escape(label)} & "
                             + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", r"\end{document}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="arcadia-impact/sentiment-utility-logs")
    ap.add_argument("--out-dir", default=str(REPO / "docs/blogpost"))
    ap.add_argument("--out-name", default="results_table")
    ap.add_argument("--suites", default=None, help="comma-sep filter")
    cgrp = ap.add_mutually_exclusive_group()
    cgrp.add_argument("--color", dest="color_mode", action="store_const", const="headroom",
                      default=None,
                      help="heat-map cells by movement-from-base, scaled by headroom to the bound "
                           "(red=worse, green=better)")
    cgrp.add_argument("--color-no-headroom", dest="color_mode", action="store_const",
                      const="absolute",
                      help="heat-map cells by raw absolute change from base (PPL stays fractional)")
    ap.add_argument("--arrows", action="store_true",
                    help="append up/down arrows to column headers showing the 'better' direction "
                         "(up=higher better, down=lower better; matches the green direction)")
    args = ap.parse_args()

    import os
    import shutil
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN_ARCADIA")
    only = [s.strip() for s in args.suites.split(",")] if args.suites else None
    roots, tmp = fetch_suites(args.repo, token, only)
    try:
        if not roots:
            raise SystemExit("no suites found on HF")
        tex = build_tex(roots, color_mode=args.color_mode, arrows=args.arrows)
        out_dir = Path(args.out_dir)
        tex_path = out_dir / f"{args.out_name}.tex"
        tex_path.write_text(tex)
        print(f"wrote {tex_path}")
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-output-directory", str(out_dir), str(tex_path)],
                           check=True, capture_output=True)
            pdf_path = out_dir / (args.out_name + ".pdf")
            print(f"wrote {pdf_path}")
            # also render a trimmed PNG (best-effort; ImageMagick magick/convert -> gs backend)
            png_path = out_dir / (args.out_name + ".png")
            for tool in ("magick", "convert"):
                try:
                    subprocess.run([tool, "-density", "200", str(pdf_path),
                                    "-trim", "+repage", "-quality", "90", str(png_path)],
                                   check=True, capture_output=True)
                    print(f"wrote {png_path}"); break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            else:
                print("png not rendered (no magick/convert on PATH)")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"pdflatex not run ({type(e).__name__}); .tex is ready to compile manually")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)   # reclaim the extraction tempdir


if __name__ == "__main__":
    main()
