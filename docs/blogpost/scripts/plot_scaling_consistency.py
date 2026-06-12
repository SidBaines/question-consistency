"""'Figure 0' story panel — "Preference consistency scales generically with capability."

Companion to plot_story_panels.py. Plots μ-decisiveness (decis_mu, the SAME bounded
mean|2Φ−1| metric the other figures use — Case-V fit on the primary 'pos' question only) for
base instruct models across scale, in the story palette / style.

Source of truth: results/coherence_four_metrics.csv — the canonical headline table that also
feeds results/plots/headline_decisiveness (its left panel is this exact decis_mu column vs the
`eci` capability index). Reading the CSV keeps this figure in lock-step with the headline plot
and needs no HF fetch. (The values are identical to a from-edges recompute to 4 dp.)

Two x-axes:
  --x params  (default)  grouped bar chart by family, bars ordered by parameter count.
                         Open-weight families only (closed models have no public param count).
  --x eci                scatter vs the fitted "capability index"; --include-closed adds the
                         GPT-4.1 / GPT-5.4 / Claude / GPT-OSS rows from the CSV.

Error bars (±1 SE, leave-one-item-out jackknife of decisiveness) are not in the CSV; they are
merged best-effort from <out-dir>/scale_consistency.json (written by an earlier --refresh run of
the prior version, keyed by model name). Models without a cached SE simply get no bar.

  uv run python docs/blogpost/scripts/plot_scaling_consistency.py             # bars vs params
  uv run python docs/blogpost/scripts/plot_scaling_consistency.py --x eci --include-closed
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CSV = REPO / "results/coherence_four_metrics.csv"

# CSV family label -> display name; the three open-weight families we show as bars (in order).
DISPLAY = {"Qwen": "Qwen2.5", "Llama": "Llama", "Gemma": "Gemma-3"}
OPEN_ORDER = ["Qwen2.5", "Llama", "Gemma-3"]
# seaborn-colorblind palette (inlined; matches plot_results_bars.py). Open families first.
FAMILY_COLOR = {
    "Qwen2.5": "#0173b2", "Llama": "#de8f05", "Gemma-3": "#029e73",
    "GPT-4.1": "#d55e00", "GPT-5.4": "#cc78bc", "Claude 4.5": "#ca9161",
    "GPT-OSS-20B (budget)": "#56b4e9", "GPT-OSS-120B (budget)": "#949494",
}
GAP = 1.2   # extra x-spacing between family groups (bar mode)


def _size_b(model: str) -> float | None:
    m = re.search(r"([\d.]+)\s*b\b", model.lower())
    return float(m.group(1)) if m else None


def _point_label(r: dict) -> str:
    """Small per-point annotation: parameter count for open models, budget level for the
    GPT-OSS budget sweeps, else the tier name (nano/mini/full, haiku/sonnet/opus)."""
    m = r["model"].lower()
    if "budget" in r["family"]:
        return next((b for b in ("low", "medium", "high") if m.endswith(b)), "")
    if r["params_b"] is not None:
        return f"{r['params_b']:g}B"
    return next((s for s in ("nano", "mini", "haiku", "sonnet", "opus") if s in m), "full")


def load_rows(csv_path: Path, se_cache: Path) -> list[dict]:
    se_by_model = {}
    if se_cache.exists():
        for r in json.loads(se_cache.read_text()):
            if r.get("se") is not None:
                se_by_model[r["model"]] = r["se"]
    rows = []
    with csv_path.open(newline="") as f:
        for r in csvmod.DictReader(f):
            fam_csv = r["family"]
            disp = DISPLAY.get(fam_csv, fam_csv)
            try:
                dm = float(r["decis_mu"])
            except (TypeError, ValueError):
                continue
            pb = r.get("params_b")
            pb = float(pb) if pb not in (None, "") else None
            eci = r.get("eci")
            eci = float(eci) if eci not in (None, "") else None
            rows.append({"model": r["model"], "family_csv": fam_csv, "family": disp,
                         "params_b": pb, "eci": eci, "decis_mu": dm,
                         "se": se_by_model.get(r["model"])})
    return rows


def _style(ax, plt):
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$\mu$-decisiveness")
    ax.yaxis.grid(True, color="0.9", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_params(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r["family"] in OPEN_ORDER and r["params_b"] is not None]
    ordered, xs, colors, spans = [], [], [], []
    x = 0.0
    for gi, fam in enumerate(OPEN_ORDER):
        grp = sorted((r for r in rows if r["family"] == fam), key=lambda r: r["params_b"])
        if not grp:
            continue
        if gi and ordered:
            x += GAP
        x0 = x
        for r in grp:
            ordered.append(r); xs.append(x); colors.append(FAMILY_COLOR[fam]); x += 1.0
        spans.append((fam, (x0 + xs[-1]) / 2.0))

    fig, ax = plt.subplots(figsize=(max(9.0, 0.62 * len(ordered) + 2.5), 5.4))
    for xi, r, c in zip(xs, ordered, colors):
        ax.bar(xi, r["decis_mu"], width=0.82, color=c, edgecolor="white", linewidth=0.4,
               zorder=3, yerr=r.get("se") or None, capsize=2.5,
               error_kw={"lw": 1.0, "ecolor": "0.15", "zorder": 5})
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['params_b']:g}B" for r in ordered], fontsize=9)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    _style(ax, plt)

    trans = ax.get_xaxis_transform()
    for fam, xc in spans:
        ax.text(xc, -0.115, fam, transform=trans, ha="center", va="top",
                fontsize=10, fontweight="bold", color=FAMILY_COLOR[fam])
    for (xi, c0), (xj, c1) in zip(zip(xs, colors), zip(xs[1:], colors[1:])):
        if c0 != c1:
            ax.axvline((xi + xj) / 2, color="0.85", lw=0.8, zorder=1)

    ax.set_title("Figure 0.  Preference consistency scales with capability   "
                 r"($\uparrow$ higher is better)", fontsize=11)
    if any(r.get("se") for r in ordered):
        ax.text(0.992, 0.975, "error bars: ±1 SE (leave-one-item-out jackknife)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.45")
    fig.subplots_adjust(bottom=0.17, left=0.08, right=0.98, top=0.90)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def plot_eci(rows: list[dict], out: Path, include_closed: bool) -> None:
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    fams = list(OPEN_ORDER)
    if include_closed:
        fams += [f for f in FAMILY_COLOR if f not in OPEN_ORDER and any(r["family"] == f for r in rows)]

    fig, ax = plt.subplots(figsize=(11.0, 6.2) if include_closed else (9.5, 5.6))
    texts, px, py = [], [], []
    for fam in fams:
        sub = sorted((r for r in rows if r["family"] == fam and r["eci"] is not None),
                     key=lambda r: r["eci"])
        if not sub:
            continue
        c = FAMILY_COLOR.get(fam, "#949494")
        # no error bars here — the ±1-SE jackknife is < marker size at this scale, so it would
        # only be a misleading legend cap (and the closed models have no SE at all).
        ax.plot([r["eci"] for r in sub], [r["decis_mu"] for r in sub],
                color=c, lw=1.4, alpha=0.45, zorder=1)
        ax.scatter([r["eci"] for r in sub], [r["decis_mu"] for r in sub], color=c, s=72,
                   edgecolor="white", linewidth=0.5, zorder=3, label=fam)
        for r in sub:
            texts.append(ax.text(r["eci"], r["decis_mu"], _point_label(r), color=c,
                                 fontsize=7.5, fontweight="bold", zorder=5))
            px.append(r["eci"]); py.append(r["decis_mu"])
    ax.set_xlabel("capability index (ECI-style, from published benchmarks)")
    _style(ax, plt)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95, ncol=2 if include_closed else 1)
    ax.set_title("Figure 0.  Preference consistency scales with capability   "
                 r"($\uparrow$ higher is better)", fontsize=11)
    fig.tight_layout()
    # de-collide labels AFTER layout (so adjustText sees the final axes size); thin leader lines.
    fig.canvas.draw()
    # expand = bounding-box padding used for repulsion; bump it so labels clear their dots by a
    # small margin (they were sitting right on the markers at expand=(1.1,1.3)).
    adjust_text(texts, x=px, y=py, ax=ax, force_text=(0.5, 0.8), expand=(1.6, 1.9),
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "docs/blogpost/plots"))
    ap.add_argument("--csv", default=str(DEFAULT_CSV),
                    help="canonical four-metrics CSV (default: results/coherence_four_metrics.csv)")
    ap.add_argument("--x", choices=["params", "eci"], default="params",
                    help="x-axis: parameter count (bars) or fitted capability index (scatter)")
    ap.add_argument("--include-closed", action="store_true",
                    help="(--x eci only) also plot GPT-4.1 / GPT-5.4 / Claude / GPT-OSS rows")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.csv), out_dir / "scale_consistency.json")

    if args.x == "params":
        plot_params(rows, out_dir / "story_scaling_capability.png")
    else:
        suffix = "_closed" if args.include_closed else ""
        plot_eci(rows, out_dir / f"story_scaling_eci{suffix}.png", args.include_closed)


if __name__ == "__main__":
    main()
