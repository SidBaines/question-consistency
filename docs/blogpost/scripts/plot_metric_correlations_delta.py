#!/usr/bin/env python
"""Pairwise correlation SPLOM of the DELTA-FROM-BASE for each model organism.

Like plot_metric_correlations.py, but each point is an MO *adapter*'s change relative to its own
base model: Delta = metric(adapter) - metric(base). This removes the model-size / base-vs-adapter
confound and isolates the finetuning effect. Base models are the origin (Delta=0) and are not
plotted. Dot size encodes the base model's parameter count.

  - lower triangle : scatter of (Delta col-metric, Delta row-metric); colour = MO type,
                     dot AREA grows with model size; grey lines mark Delta=0 (the no-change axes)
  - diagonal       : histogram of that metric's delta (vertical line at 0)
  - upper triangle : Pearson r of the deltas (+ n)

Output: <out>/metric_correlations_delta.png

Usage (pod, elicit venv — needs matplotlib + numpy):
  .venv/bin/python docs/blogpost/scripts/plot_metric_correlations_delta.py
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

METRICS = [
    ("decis_mu",     "Pref.\nconsistency"),
    ("mmlu",         "MMLU"),
    ("ifeval",       "IFEval"),
    ("ppl_nat",      r"PPL$_\mathrm{nat}$"),
    ("ppl_shuf",     r"PPL$_\mathrm{shuf}$"),
    ("xstest",       "XSTest"),
    ("strongreject", "StrongREJECT"),
]
KEYS = [k for k, _ in METRICS]
TYPE_COLOR = {"OCT": "tab:blue", "EM": "tab:red", "AuditBench": "tab:green", "Other": "gray"}
TYPES = ("OCT", "EM", "AuditBench")
SIZE_LEGEND = (1, 8, 32, 70)            # representative model sizes (B params) for the size key


def _dot_size(size_b: float) -> float:
    """Marker AREA grows with model size; floor keeps the smallest models visible."""
    return 18.0 + 5.0 * float(size_b)


def collect_deltas(repo: str, token: str | None):
    """-> list of (mo_type, size_B, delta_dict) for every adapter (vs its own base)."""
    roots, tmp = brt.fetch_suites(repo, token, None)
    try:
        rows = []
        for suite in [s for s in roots if s not in brt.CAPABILITY_OVERRIDE.values()]:
            meta = brt._meta(suite)
            base_m = brt._metrics(roots, suite, "base")
            for model in brt._models_for(roots[suite]):
                if model == "base":
                    continue
                am = brt._metrics(roots, suite, model)
                delta = {k: am[k] - base_m[k] for k in KEYS
                         if isinstance(am.get(k), (int, float))
                         and isinstance(base_m.get(k), (int, float))}
                rows.append((meta["type"], meta["size"], delta))
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _paired(rows, kx, ky, typ=None):
    out = []
    for t, _, d in rows:
        if typ is not None and t != typ:
            continue
        x, y = d.get(kx), d.get(ky)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            out.append((x, y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="arcadia-impact/sentiment-utility-logs")
    ap.add_argument("--out", default=str(brt.REPO / "docs/blogpost/metric_correlations_delta.png"))
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_WRITE_TOKEN_ARCADIA")

    rows = collect_deltas(args.repo, token)
    print(f"collected {len(rows)} adapter delta points")

    n = len(METRICS)
    fig, axes = plt.subplots(n, n, figsize=(2.4 * n, 2.4 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            ki, kj, lj = METRICS[i][0], METRICS[j][0], METRICS[j][1]
            if i == j:                                   # diagonal: histogram of the delta
                vals = [d[ki] for _, _, d in rows if isinstance(d.get(ki), (int, float))]
                ax.hist(vals, bins=12, color="lightgray", edgecolor="gray")
                ax.axvline(0, color="k", lw=0.8, ls=":")
                ax.set_title(r"$\Delta$ " + METRICS[i][1].replace("\n", " "), fontsize=9)
                ax.set_yticks([])
                ax.tick_params(labelsize=6)
            elif i > j:                                  # lower: scatter of deltas
                ax.axhline(0, color="0.7", lw=0.7); ax.axvline(0, color="0.7", lw=0.7)
                for typ in TYPES:
                    pts = [(d[kj], d[ki], _dot_size(sz)) for t, sz, d in rows
                           if t == typ and isinstance(d.get(kj), (int, float))
                           and isinstance(d.get(ki), (int, float))]
                    if pts:
                        xs, ys, ss = zip(*pts)
                        ax.scatter(xs, ys, c=TYPE_COLOR[typ], s=ss, alpha=0.7,
                                   edgecolors="black", linewidths=0.3)
                ax.tick_params(labelsize=6)
            else:                                        # upper: Pearson r of deltas
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
                ax.set_xlabel(r"$\Delta$ " + lj, fontsize=8)
            if j == 0 and i != 0:
                ax.set_ylabel(r"$\Delta$ " + METRICS[i][1], fontsize=8)

    type_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=TYPE_COLOR[t],
                     markeredgecolor="black", markersize=9, label=t) for t in TYPES]
    size_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                     markeredgecolor="black", markersize=np.sqrt(_dot_size(s)), label=f"{s}B")
              for s in SIZE_LEGEND]
    leg1 = fig.legend(handles=type_h, loc="upper right", fontsize=10, framealpha=0.95, title="MO type")
    fig.add_artist(leg1)
    fig.legend(handles=size_h, loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=9,
               framealpha=0.95, title="model size", labelspacing=1.3)
    fig.suptitle(r"$\Delta$-from-base pairwise correlations (one point per MO adapter; "
                 "dot size $\\propto$ model params; lower=scatter, upper=Pearson r)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
