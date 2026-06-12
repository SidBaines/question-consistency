#!/usr/bin/env python
"""Pairwise metric-correlation scatter matrix (SPLOM) for the blogpost model organisms.

Reads the SAME HF logs as build_results_table (so it inherits the 1M-token PPL preference and the
Qwen3 thinking-suite capability overrides), collects every model organism's 7 metrics, and renders
a triangular grid:
  - lower triangle : pairwise scatter (x = col metric, y = row metric); colour = MO type,
                     square+outline = base (Instruct-tuned), circle = MO adapter
  - diagonal       : histogram of that metric (all points)
  - upper triangle : Pearson r (+ n) over points with both metrics present

Both base models and adapters are included as points (markers distinguish them).
Output: <out>/metric_correlations.png

Usage (pod, elicit venv — needs matplotlib + numpy):
  .venv/bin/python docs/blogpost/scripts/plot_metric_correlations.py
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # import sibling build_results_table
import build_results_table as brt                            # noqa: E402

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.lines import Line2D                          # noqa: E402
import numpy as np                                           # noqa: E402

# (metric key, short label) — the same 7 columns the table reports
METRICS = [
    ("decis_mu",     "Pref.\nconsistency"),
    ("mmlu",         "MMLU"),
    ("ifeval",       "IFEval"),
    ("ppl_nat",      r"PPL$_\mathrm{nat}$"),
    ("ppl_shuf",     r"PPL$_\mathrm{shuf}$"),
    ("xstest",       "XSTest"),
    ("strongreject", "StrongREJECT"),
]
# seaborn "colorblind" palette (hex values inlined so the pod run needs no seaborn dep)
TYPE_COLOR = {"OCT": "#0173b2", "EM": "#d55e00", "AuditBench": "#029e73", "Other": "#949494"}
TYPES = ("OCT", "EM", "AuditBench")


def collect(repo: str, token: str | None):
    """-> list of (mo_type, is_base, metrics_dict), one per model organism."""
    roots, tmp = brt.fetch_suites(repo, token, None)
    try:
        rows = []
        for suite in [s for s in roots if s not in brt.CAPABILITY_OVERRIDE.values()]:
            typ = brt._meta(suite)["type"]
            for model in brt._models_for(roots[suite]):
                rows.append((typ, model == "base", brt._metrics(roots, suite, model)))
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _paired(rows, kx, ky, typ=None, is_base=None):
    out = []
    for t, b, m in rows:
        if typ is not None and t != typ:
            continue
        if is_base is not None and b != is_base:
            continue
        x, y = m.get(kx), m.get(ky)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            out.append((x, y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="arcadia-impact/sentiment-utility-logs")
    ap.add_argument("--out", default=str(brt.REPO / "docs/blogpost/metric_correlations.png"))
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN_ARCADIA")

    rows = collect(args.repo, token)
    print(f"collected {len(rows)} model-organism points "
          f"({sum(b for _, b, _ in rows)} base, {sum(not b for _, b, _ in rows)} adapter)")

    n = len(METRICS)
    fig, axes = plt.subplots(n, n, figsize=(2.4 * n, 2.4 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ki, lj = METRICS[i][0], METRICS[j][1]
            kj = METRICS[j][0]
            if i == j:                                   # diagonal: histogram
                vals = [m[ki] for _, _, m in rows if isinstance(m.get(ki), (int, float))]
                ax.hist(vals, bins=12, color="lightgray", edgecolor="gray")
                ax.set_title(METRICS[i][1].replace("\n", " "), fontsize=9)
                ax.set_yticks([])
                ax.tick_params(labelsize=6)
            elif i > j:                                  # lower: scatter x=kj, y=ki
                for typ in TYPES:
                    for is_base, mk, sz, ec in ((True, "s", 40, "black"), (False, "o", 22, "none")):
                        pts = _paired(rows, kj, ki, typ, is_base)
                        if pts:
                            xs, ys = zip(*pts)
                            ax.scatter(xs, ys, c=TYPE_COLOR[typ], marker=mk, s=sz,
                                       edgecolors=ec, linewidths=0.5, alpha=0.85)
                ax.tick_params(labelsize=6)
            else:                                        # upper: Pearson r
                ax.axis("off")
                pts = _paired(rows, kj, ki)
                if len(pts) >= 3:
                    xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                    if xs.std() > 0 and ys.std() > 0:
                        r = float(np.corrcoef(xs, ys)[0, 1])
                        ax.text(0.5, 0.5, f"r = {r:+.2f}\nn = {len(pts)}", ha="center", va="center",
                                fontsize=11, color=("darkred" if abs(r) > 0.6 else "black"),
                                fontweight=("bold" if abs(r) > 0.6 else "normal"))
                    else:
                        ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=9, color="gray")
                else:
                    ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=9, color="gray")
            if i == n - 1:
                ax.set_xlabel(lj, fontsize=8)
            if j == 0 and i != 0:
                ax.set_ylabel(METRICS[i][1], fontsize=8)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=TYPE_COLOR[t],
                      markersize=9, label=t) for t in TYPES]
    handles += [Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                       markeredgecolor="black", markersize=9, label="base (Instruct-tuned)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markersize=9, label="MO adapter")]
    fig.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.95)
    fig.suptitle("Pairwise metric correlations across model organisms "
                 "(lower = scatter, upper = Pearson r; squares = base models)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
