"""Grouped bar charts of the blogpost results panel — one figure per metric.

Companion to build_results_table.py: same HF suite logs, same row grouping
(MO type -> base model -> base row first then each adapter), but rendered as one
bar chart per COLUMNS metric instead of one wide LaTeX table. Within each base-model
group the base bar is grey, adapter bars take the MO-type colour, and a dashed line
marks the base value across the group so movement-from-base is readable at a glance.

Error bars are ±1 SE, from the best uncertainty source each metric admits:
  decis_mu      leave-one-item-out jackknife of decisiveness over the fitted Case-V matrix
                (conditions on the fitted mu — each item's mu pools every comparison it
                appears in, ~50 across the 50k-edge / 2k-item panel, so item sampling
                dominates the fit noise)
  mmlu/ifeval   lm-eval's own stderr fields (binomial fallback when absent, n=14042/541)
  ppl/bpb       cluster bootstrap over the FineWeb docs (per-doc NLLs from
                perplexity_1m.json; per-doc bytes from the same record that supplied the
                BPB point value)
  xstest        Jeffreys central 68% interval on the safe-prompt refusal rate (asymmetric;
                a Wald SE would collapse to 0 at observed p=0, falsely implying certainty)
  strongreject  SE of the per-prompt judge scores (safety/strongreject/*_judged.jsonl)

All bars quantify evaluation-set sampling only (items/docs/prompts) — systematic
components (judge bias, fit noise, generation seed) are out of scope.

Fetching the suites downloads ~250MB, so the collected per-row metrics are cached to
<out-dir>/table_metrics.json on every fetch; re-plot without the network via --from-cache.

Run from repo root in the elicit venv (HF_TOKEN in env for the fetch):
  uv run python docs/blogpost/scripts/plot_results_bars.py
  -> docs/blogpost/plots/bars_<metric>.png (+ table_metrics.json cache)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_results_table import (  # noqa: E402
    CAPABILITY_OVERRIDE, COLOR_META, COLUMNS, REPO, TYPE_ORDER,
    _meta, _metrics, _model_label, _models_for, _suite_sort_key, fetch_suites,
)

# metric key -> (figure title, y-axis label)
METRIC_INFO = {
    "decis_mu":     ("Preference consistency", r"$\mu$-decisiveness"),
    "mmlu":         ("MMLU (0-shot)", "accuracy"),
    "ifeval":       ("IFEval", "prompt-level strict accuracy"),
    "ppl_nat":      ("Perplexity (natural text, FineWeb)", "perplexity"),
    "bpb_nat":      ("Bits per byte (natural text, FineWeb)", "bits per byte"),
    "xstest":       ("XSTest over-refusal", "refusal rate on safe prompts"),
    "strongreject": ("StrongREJECT", "mean harm score"),
}
# y in [0,1] AND values span enough of it that a fixed [0,1] axis aids comparison;
# the small-valued safety rates get an auto axis instead so they aren't squashed.
FULL_UNIT_AXIS = {"decis_mu", "mmlu", "ifeval"}

TYPE_LABEL = {"OCT": "Open Character\nTraining", "EM": "Emergent Misalignment",
              "AuditBench": "AuditBench"}
# seaborn "colorblind" palette (hex values inlined so the pod run needs no seaborn dep)
TYPE_COLOR = {"OCT": "#cc78bc", "EM": "#0173b2", "AuditBench": "#de8f05"}
BASE_COLOR = "#949494"   # seaborn colorblind grey
FALLBACK_COLOR = "#029e73"   # seaborn colorblind green (unknown MO type)
GAP_SUITE, GAP_TYPE = 0.9, 1.8     # extra x-space between base-model groups / MO types


def _se_ratio_boot(nll, denom, kind: str, B: int = 1000, seed: int = 0) -> float:
    """Cluster-bootstrap SE over docs of a sum-ratio statistic:
    kind='ppl' -> exp(sum nll / sum tokens); kind='bpb' -> sum nll / ln2 / sum bytes."""
    import numpy as np
    n = min(len(nll), len(denom))
    nll = np.asarray(nll[:n], float)
    denom = np.asarray(denom[:n], float)
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    r = nll[idx].sum(axis=1) / denom[idx].sum(axis=1)
    vals = np.exp(r) if kind == "ppl" else r / math.log(2)
    return float(vals.std(ddof=1))


def _se_decis_jackknife(edges_path: Path) -> float | None:
    """Leave-one-item-out jackknife SE of decisiveness = mean|2P-1| over pairs of the
    fitted Case-V matrix. Conditions on the fitted mu (not refit per fold): each item's
    mu pools every comparison it appears in (~50 across the 50k-edge / 2k-item panel),
    so item sampling dominates the fit noise."""
    import numpy as np
    from four_metrics import _fit_elo
    from question_consistency.fit import predict_matrix_caseV
    mu, _ = _fit_elo(str(edges_path))
    A = np.abs(2.0 * predict_matrix_caseV(np.asarray(mu)) - 1.0)   # symmetric, zero diag
    n = A.shape[0]
    if n < 3:
        return None
    total = A[np.triu_indices(n, k=1)].sum()
    theta = (total - A.sum(axis=1)) / ((n - 1) * (n - 2) / 2.0)    # leave-one-out means
    return float(np.sqrt((n - 1) / n * ((theta - theta.mean()) ** 2).sum()))


def errors_for(root: Path, model: str) -> dict:
    """File-based ±1-SE estimates for one model (same data layout as metrics_for)."""
    errs = {}
    edges = root / model / "edges.jsonl"
    if edges.exists():
        try:
            se = _se_decis_jackknife(edges)
            if se is not None:
                errs["decis_mu"] = se
        except Exception as e:
            print(f"  decis_mu SE fail {model}: {e}")
    js = glob.glob(str(root / "lmeval" / model / "**" / "results_*.json"), recursive=True)
    if js:
        r = json.loads(Path(sorted(js)[-1].replace("\\", "/")).read_text())["results"]
        for task, field, key in (("mmlu", "acc_stderr,none", "mmlu"),
                                 ("mmlu_generative", "exact_match_stderr,get_response", "mmlu"),
                                 ("ifeval", "prompt_level_strict_acc_stderr,none", "ifeval")):
            se = r.get(task, {}).get(field)
            if isinstance(se, (int, float)) and se > 0:
                errs[key] = se
    # robust-MMLU sidecar overrides the lm-eval value -> binomial SE on its own n
    rob = root / "lmeval" / model / "mmlu_robust.json"
    if rob.exists():
        try:
            rec = json.loads(rob.read_text())
            p, n = rec.get("mmlu"), rec.get("samples", 14042)
            if isinstance(p, (int, float)):
                errs["mmlu"] = math.sqrt(p * (1 - p) / n)
        except Exception as e:
            print(f"  mmlu_robust SE fail {model}: {e}")
    # ppl/bpb: cluster bootstrap over docs (needs the per-doc arrays of the 1M-token run)
    p1m = root / "perplexity_1m.json"
    ppl = p1m if p1m.exists() else (root / "ppl" / "perplexity.json")
    if ppl.exists():
        nat = json.loads(ppl.read_text()).get("results", {}).get(model, {}).get("natural", {})
        nll, tok = nat.get("per_doc_nll"), nat.get("per_doc_tokens")
        if nll and tok:
            errs["ppl_nat"] = _se_ratio_boot(nll, tok, "ppl")
        # per-doc bytes: mirror metrics_for's source preference (HF sidecar first, then the
        # committed combined file) so the SE resamples the SAME record that supplied the BPB
        # point value. An old sidecar without per_doc_bytes -> no error bar, never a
        # mismatched one. bpb was computed from the 1M-token run, so require p1m.
        if nll and p1m.exists():
            sidecar = root / "bpb.json"
            combined = REPO / "results" / "bpb_fineweb.json"
            if sidecar.exists():
                rec = json.loads(sidecar.read_text())
            else:
                rec = json.loads(combined.read_text()).get(root.name, {}) \
                    if combined.exists() else {}
            pb = rec.get("per_doc_bytes")
            if pb:
                errs["bpb_nat"] = _se_ratio_boot(nll, pb, "bpb")
    summ = root / "safety" / "safety_summary.json"
    if summ.exists():
        ds = json.loads(summ.read_text()).get("datasets", {})
        x = ds.get("xstest", {}).get(model, {})
        p, n = x.get("over_refusal_rate_safe"), x.get("n_safe")
        if isinstance(p, (int, float)) and n:
            # Jeffreys central 68.27% interval (matches ±1-sigma coverage) as asymmetric
            # [down, up] offsets — stays honest at the boundary, where Wald gives SE=0
            from scipy import stats
            k = round(p * n)
            lo = stats.beta.ppf(0.15865, k + 0.5, n - k + 0.5) if k > 0 else 0.0
            hi = stats.beta.ppf(0.84135, k + 0.5, n - k + 0.5) if k < n else 1.0
            errs["xstest"] = [max(0.0, p - lo), max(0.0, hi - p)]
    judged = root / "safety" / "strongreject" / f"{model}_judged.jsonl"
    if judged.exists():
        scores = [r["judge"]["score"] for line in judged.read_text().splitlines() if line.strip()
                  for r in [json.loads(line)] if r.get("judge", {}).get("score") is not None]
        if len(scores) > 1:
            import statistics
            errs["strongreject"] = statistics.stdev(scores) / math.sqrt(len(scores))
    return errs


def _errors(roots: dict, suite: str, model: str, metrics: dict) -> dict:
    """errors_for with the same capability override as _metrics, plus binomial fallbacks
    for any populated mmlu/ifeval value whose source file carried no stderr."""
    e = errors_for(roots[suite], model)
    ov = CAPABILITY_OVERRIDE.get(suite)
    if ov and ov in roots:
        oe = errors_for(roots[ov], model)
        for k in ("mmlu", "ifeval"):
            e.pop(k, None)
            if k in oe:
                e[k] = oe[k]
    for k, n in (("mmlu", 14042), ("ifeval", 541)):
        v = metrics.get(k)
        if isinstance(v, (int, float)) and k not in e:
            e[k] = math.sqrt(v * (1 - v) / n)
    return e


def collect_rows(roots: dict[str, Path]) -> list[dict]:
    """One dict per table row, in exact table order."""
    suites = [s for s in sorted(roots, key=_suite_sort_key)
              if s not in CAPABILITY_OVERRIDE.values()]
    rows = []
    for suite in suites:
        m = _meta(suite)
        for model in _models_for(roots[suite]):
            metrics = _metrics(roots, suite, model)
            rows.append({
                "type": m["type"], "suite": suite, "base": m["base"], "model": model,
                "label": "Instruct-tuned" if model == "base" else _model_label(suite, model),
                "metrics": metrics,
                "errs": _errors(roots, suite, model, metrics),
            })
    return rows


def _positions(rows: list[dict]) -> list[float]:
    """x position per row: unit spacing, wider gaps at suite/type boundaries."""
    xs, x = [], 0.0
    for i, r in enumerate(rows):
        if i > 0:
            prev = rows[i - 1]
            x += 1.0 + (GAP_TYPE if r["type"] != prev["type"]
                        else GAP_SUITE if r["suite"] != prev["suite"] else 0.0)
        xs.append(x)
    return xs


def plot_metric(rows: list[dict], key: str, out: Path, title_prefix: str = "") -> bool:
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms

    vals = [r["metrics"].get(key) for r in rows]
    if not any(isinstance(v, (int, float)) for v in vals):
        return False
    xs = _positions(rows)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.42 * len(rows) + 3.0), 5.2))
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    for r, x, v in zip(rows, xs, vals):
        if not isinstance(v, (int, float)):
            continue
        color = BASE_COLOR if r["model"] == "base" else TYPE_COLOR.get(r["type"], FALLBACK_COLOR)
        err = r.get("errs", {}).get(key)
        if isinstance(err, (list, tuple)):                # asymmetric [down, up] offsets
            err = [[err[0]], [err[1]]]
        ax.bar(x, v, width=0.8, color=color, edgecolor="white", linewidth=0.4, zorder=3,
               yerr=err if err is not None else None, capsize=2.5,
               error_kw={"lw": 1.0, "ecolor": "0.15", "zorder": 5})

    # per-group base-value dashed line + base-model label; per-type label + separator
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1]["suite"] == rows[i]["suite"]:
            j += 1
        bv = next((v for r, v in zip(rows[i:j + 1], vals[i:j + 1])
                   if r["model"] == "base" and isinstance(v, (int, float))), None)
        if bv is not None:
            ax.hlines(bv, xs[i] - 0.5, xs[j] + 0.5, color="0.25", ls="--", lw=1.0, zorder=4)
        ax.text((xs[i] + xs[j]) / 2, -0.50, rows[i]["base"], transform=trans,
                ha="center", va="top", fontsize=8.5, fontweight="bold")
        i = j + 1
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1]["type"] == rows[i]["type"]:
            j += 1
        ax.text((xs[i] + xs[j]) / 2, -0.64, TYPE_LABEL.get(rows[i]["type"], rows[i]["type"]),
                transform=trans, ha="center", va="top", fontsize=9,
                color=TYPE_COLOR.get(rows[i]["type"], FALLBACK_COLOR), fontweight="bold")
        if j + 1 < len(rows):
            ax.axvline((xs[j] + xs[j + 1]) / 2, color="0.85", lw=0.8, zorder=1)
        i = j + 1

    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], rotation=45, ha="right", fontsize=8)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
    if key in FULL_UNIT_AXIS:
        ax.set_ylim(0, 1)
    else:
        def _up(e):
            return (e[1] if isinstance(e, (list, tuple)) else e) or 0.0
        tops = [v + _up(r.get("errs", {}).get(key)) for r, v in zip(rows, vals)
                if isinstance(v, (int, float))]
        ax.set_ylim(0, max(tops) * 1.12)
    higher_is_worse, _ = COLOR_META[key]
    title, ylabel = METRIC_INFO.get(key, (key, key))
    arrow = r"$\downarrow$ lower is better" if higher_is_worse else r"$\uparrow$ higher is better"
    ax.set_title(f"{title_prefix}{title}   ({arrow})", fontsize=11)
    if any(r.get("errs", {}).get(key) for r in rows):
        note = ("error bars: Jeffreys 68% interval" if key == "xstest"
                else "error bars: ±1 SE")
        ax.text(0.998, 1.02, note, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="0.45")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color="0.9", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.46, left=0.07, right=0.99, top=0.90)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="arcadia-impact/sentiment-utility-logs")
    ap.add_argument("--suites", default=None, help="comma-sep filter")
    ap.add_argument("--out-dir", default=str(REPO / "docs/blogpost/plots"))
    ap.add_argument("--from-cache", action="store_true",
                    help="re-plot from <out-dir>/table_metrics.json instead of fetching")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "table_metrics.json"

    if args.from_cache:
        rows = json.loads(cache.read_text())["rows"]
    else:
        import os
        import shutil
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN_ARCADIA")
        only = [s.strip() for s in args.suites.split(",")] if args.suites else None
        roots, tmp = fetch_suites(args.repo, token, only)
        try:
            if not roots:
                raise SystemExit("no suites found on HF")
            rows = collect_rows(roots)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        cache.write_text(json.dumps({"rows": rows}, indent=2))
        print(f"wrote {cache}")

    for key, _, _ in COLUMNS:
        out = out_dir / f"bars_{key}.png"
        if plot_metric(rows, key, out):
            print(f"wrote {out}")
        else:
            print(f"skipped {key} (no data)")


if __name__ == "__main__":
    main()
