"""Three focused 'story' figures distilled from the full results panel.

Companion to plot_results_bars.py / build_results_table.py. Rather than the full
7-metric x 28-row panel, this renders the three simple narratives a reader can grok:

  1. story_consistency_drop.png  — "Models lose preference consistency."
       Grouped decis_mu bars for one suite per MO type: Llama-3.1-8B OCT,
       Qwen2.5-14B EM, Llama-3.3-70B AuditBench (base grey + adapter bars).
  2. story_across_scales.png     — "This happens across scales."
       Paired base->adapter decis_mu vs model size (log x) for the EM
       bad-medical-advice finetune at every Llama/Qwen scale we ran.
  3. story_mmlu_constant.png (+ .md) — "Keeping MMLU constant doesn't mean other
       things don't change." Table of the suite-1 models: MMLU (~flat) alongside
       IFEval, perplexity and over-refusal, each adapter row showing Delta-from-base.

Reads the SAME cache plot_results_bars.py writes (docs/blogpost/plots/table_metrics.json),
so no HF fetch / token is needed — just re-run plot_results_bars.py once to refresh it.

  uv run python docs/blogpost/scripts/plot_story_panels.py
  -> docs/blogpost/plots/story_*.png  (+ story_mmlu_constant.md)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_results_bars as prb  # noqa: E402  (TYPE_COLOR, BASE_COLOR, plot_metric, ...)
from build_results_table import REPO  # noqa: E402

# suite + which adapters telling story (1) — one suite per MO type
STORY1_SUITES = ["oct-llama8b", "qwen2.5-14b-instruct", "auditbench-llama70b"]
# the EM finetune we have at the most scales, for story (2)
SCALE_SUITES = ["qwen2.5-0.5b-instruct", "llama-3.2-1b-instruct", "qwen2.5-7b-instruct",
                "llama-3.1-8b-instruct", "qwen2.5-14b-instruct", "qwen2.5-32b-instruct"]
SCALE_ADAPTER = "bad-medical-advice"   # matched on row["label"]


def _size_b(base: str) -> float:
    """Param count in billions, parsed from the base-model label (e.g. 'Qwen2.5-0.5B' -> 0.5)."""
    m = re.search(r"([\d.]+)\s*[bB]\b", base)
    return float(m.group(1)) if m else float("nan")


def _up(err) -> float:
    """Upper ±1-SE offset (asymmetric errs are stored [down, up])."""
    if isinstance(err, (list, tuple)):
        return err[1] or 0.0
    return err or 0.0


# ---------------------------------------------------------------- story 1
def story1(rows: list[dict], out: Path) -> None:
    sub = [r for r in rows if r["suite"] in STORY1_SUITES]
    sub.sort(key=lambda r: STORY1_SUITES.index(r["suite"]))   # keep requested suite order
    prb.plot_metric(sub, "decis_mu", out, title_prefix="Figure 1.  ")
    print(f"wrote {out}")


# ---------------------------------------------------------------- story 2
def story2(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    pts = []
    for suite in SCALE_SUITES:
        srows = [r for r in rows if r["suite"] == suite]
        base = next((r for r in srows if r["model"] == "base"), None)
        adpt = next((r for r in srows if r["label"] == SCALE_ADAPTER), None)
        if base and adpt:
            pts.append((_size_b(base["base"]), base, adpt))

    # group by model family (Qwen models together, then Llama), sorted by size within
    # each group; a small gap between groups keeps the two families visually distinct.
    def _family(p) -> str:
        return "Qwen" if "qwen" in p[1]["base"].lower() else "Llama"

    GAP = 0.9   # extra x-spacing inserted between family groups
    ordered, xs = [], []
    x = 0.0
    for gi, fam in enumerate(["Qwen", "Llama"]):
        grp = sorted((p for p in pts if _family(p) == fam), key=lambda p: p[0])
        if not grp:
            continue
        if gi:
            x += GAP
        for p in grp:
            ordered.append(p)
            xs.append(x)
            x += 1.0

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    base_c, em_c = prb.BASE_COLOR, prb.TYPE_COLOR["EM"]
    # evenly-spaced positions grouped by family (7B/8B collide on a true log axis); the
    # (size) annotation on each tick keeps the across-scales reading explicit.
    for xi, (sz, base, adpt) in zip(xs, ordered):
        bv = base["metrics"]["decis_mu"]
        av = adpt["metrics"]["decis_mu"]
        ax.annotate("", xy=(xi, av), xytext=(xi, bv), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color="0.55", lw=1.4))
        ax.errorbar(xi, bv, yerr=_up(base.get("errs", {}).get("decis_mu")), fmt="o",
                    ms=9, color=base_c, ecolor="0.3", capsize=2.5, zorder=4)
        ax.errorbar(xi, av, yerr=_up(adpt.get("errs", {}).get("decis_mu")), fmt="o",
                    ms=9, color=em_c, ecolor="0.3", capsize=2.5, zorder=4)

    ax.plot([], [], "o", color=base_c, label="Instruct-tuned (base)")
    ax.plot([], [], "o", color=em_c, label="bad-medical-advice finetune")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{p[1]['base']}\n({p[0]:g}B)" for p in ordered], fontsize=8.5)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"$\mu$-decisiveness")
    ax.set_title(r"Figure 2.  Preference consistency drops at every scale   ($\uparrow$ higher is better)",
                 fontsize=11)
    ax.text(0.998, 1.02, "error bars: ±1 SE", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="0.45")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.yaxis.grid(True, color="0.9", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(bottom=0.16, left=0.09, right=0.98, top=0.90)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------- story 3
# (key, header, n-decimals, higher-is-better, headroom) — MMLU first to anchor the story.
# headroom = which bound the "bad" direction runs into, so a drop can be scored as a
# fraction of the room it had to get worse:
#   "zero" : worse = toward 0   (frac = drop / base)            [MMLU, IFEval]
#   "one"  : worse = toward 1   (frac = rise / (1 - base))      [over-refusal rate]
#   "rel"  : unbounded above; use relative increase off base    [perplexity]
TABLE_COLS = [
    ("mmlu",     "MMLU\n(0-shot)",      3, True,  "zero"),
    ("ifeval",   "IFEval",              3, True,  "zero"),
    ("ppl_nat",  "PPL (nat)",           1, False, "rel"),
    ("xstest",   "Over-refusal",        3, False, "one"),
    ("decis_mu", "μ-decisiveness",      3, True,  "zero"),
]


def _fmt(v, nd: int) -> str:
    return "—" if not isinstance(v, (int, float)) else f"{v:.{nd}f}"


def _fmt_delta(v, bv, nd: int) -> str:
    if not isinstance(v, (int, float)) or not isinstance(bv, (int, float)):
        return ""
    d = v - bv
    return f"{d:+.{nd}f}"


def _bad_frac(higher_better: bool, headroom: str, bv, av) -> float:
    """Fraction of available headroom consumed in the BAD direction, in [0,1].
    0 means no change or a move in the good direction (-> left blank)."""
    if not isinstance(bv, (int, float)) or not isinstance(av, (int, float)):
        return 0.0
    amt = (bv - av) if higher_better else (av - bv)   # >0 means it got worse
    if amt <= 0:
        return 0.0
    hr = bv if headroom in ("zero", "rel") else (1.0 - bv)
    return 0.0 if hr <= 0 else min(1.0, amt / hr)


def _red(frac: float):
    """White (frac=0) -> strong red (frac=1) cell colour; black text stays legible."""
    return (1.0, 1.0 - 0.85 * frac, 1.0 - 0.85 * frac)


def story3(rows: list[dict], out_png: Path, out_md: Path) -> None:
    import matplotlib.pyplot as plt

    sub = [r for r in rows if r["suite"] in STORY1_SUITES]
    sub.sort(key=lambda r: STORY1_SUITES.index(r["suite"]))
    base_for = {r["suite"]: r for r in sub if r["model"] == "base"}

    headers = ["Model"] + [c[1] for c in TABLE_COLS]
    cells, colors, md_rows = [], [], []
    for r in sub:
        bm = base_for[r["suite"]]["metrics"]
        is_base = r["model"] == "base"
        name = f"{r['base']} — Instruct-tuned" if is_base else f"   ↳ {r['label']}"
        row_txt, row_col, md_cells = [name], ["w"], [name.strip().replace("↳", "")]
        for key, _, nd, hb, hr in TABLE_COLS:
            v = r["metrics"].get(key)
            if is_base:                               # reference row: no delta, no shading
                row_txt.append(_fmt(v, nd))
                md_cells.append(_fmt(v, nd))
                row_col.append("w")
            else:
                d = _fmt_delta(v, bm.get(key), nd)
                row_txt.append(f"{_fmt(v, nd)}\n({d})")
                md_cells.append(f"{_fmt(v, nd)} ({d})")
                row_col.append(_red(_bad_frac(hb, hr, bm.get(key), v)))   # shade by headroom lost
        cells.append(row_txt)
        colors.append(row_col)
        md_rows.append(md_cells)

    nrows = len(cells)
    fig, ax = plt.subplots(figsize=(11.0, 0.62 * nrows + 1.4))
    ax.axis("off")
    col_w = [0.30] + [0.155] * len(TABLE_COLS)
    tbl = ax.table(cellText=cells, colLabels=headers, cellColours=colors,
                   colWidths=col_w, cellLoc="center", colLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.85")
        if ri == 0:                                  # header
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#d9e6ef")
        elif ci == 0:                                # model column
            cell.set_text_props(ha="left")
            cell.get_text().set_x(0.03)
        # bold-face the base rows' model name
        if ci == 0 and ri > 0 and sub[ri - 1]["model"] == "base":
            cell.set_text_props(ha="left", fontweight="bold")
            cell.get_text().set_x(0.03)
    ax.set_title("Figure 3.  Keeping MMLU constant doesn't mean other things don't change\n"
                 "adapter rows show Δ from base; red = fraction of headroom lost "
                 "in the bad direction (PPL: relative increase)",
                 fontsize=10.5, pad=14)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    md = ["| " + " | ".join(h.replace("\n", " ") for h in headers) + " |",
          "|" + "|".join(["---"] * len(headers)) + "|"]
    md += ["| " + " | ".join(c.replace("\n", " ") for c in row) + " |" for row in md_rows]
    out_md.write_text("\n".join(md) + "\n")
    print(f"wrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "docs/blogpost/plots"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    rows = json.loads((out / "table_metrics.json").read_text())["rows"]

    story1(rows, out / "story_consistency_drop.png")
    story2(rows, out / "story_across_scales.png")
    story3(rows, out / "story_mmlu_constant.png", out / "story_mmlu_constant.md")


if __name__ == "__main__":
    main()
