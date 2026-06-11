"""The four agreement-probability consistency metrics, computed from raw edges.

All four are "probability that two draws agree", from the per-comparison choice probability:
  p_self     P(same answer if Q(A,B) sampled twice)         = E[p^2+(1-p)^2]      (elo phase)
  p_reversal P(same concept under Q(A,B) vs Q(B,A))          = E[pf(1-pr)+(1-pf)pr] (reverse)
             where pf=P(pick i|i first), pr=P(pick j|j first)
  p_acyclic  P(no cycle over Q(A,B),Q(B,C),Q(C,A))           = 1 - E[cycle mass]   (triad)
  p_crossq   P(same answer under Q_i(A,B) vs Q_j(A,B))       = E[pa*pb+(1-pa)(1-pb)] (cross)

Chance floors (indifferent model): p_self/p_reversal/p_crossq = 0.5, p_acyclic = 0.75.
p_self is the decisiveness/determinism baseline; read p_reversal & p_crossq relative to it
(p_self - p_reversal = order-bias cost; p_self - p_crossq = framing cost).

Sample-mode refinement: p_self reuses the SAME comparison twice, so the plug-in p_hat^2+... is
biased at small n. When raw wins are present we use the unbiased U-statistic
[k(k-1)+(n-k)(n-k-1)]/[n(n-1)]. p_reversal/p_acyclic/p_crossq multiply INDEPENDENT comparisons,
so plug-in is already unbiased there.
"""
import sys
import csv
import os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO / "results/.metrics_cache.csv"
CACHE_FIELDS = ["rel_path", "size", "mtime", "primary_qid",
                "decis_mu", "fit_r2", "p_self", "p_reversal", "p_acyclic", "p_crossq"]
_METRICS_CACHE = None
sys.path.insert(0, str(REPO / "scripts"))
from build_coherence import _bucket, _load_edges

FOUR = ["p_self", "p_reversal", "p_acyclic", "p_crossq"]
LAB = {
    "p_self":     "P(repeat)  ·  self-agreement",
    "p_reversal": "P(agree | A,B vs B,A)  ·  order",
    "p_acyclic":  "P(no cycle)  ·  transitivity",
    "p_crossq":   "P(agree | Qᵢ vs Qⱼ)  ·  framing",
}


def _unsmooth(edges):
    """Sample-mode p_util/p_a are Jeffreys-smoothed (k+0.5)/(n+1), which biases the
    multilinear metrics (p_reversal/p_crossq deflated toward 0.5, p_acyclic inflated by
    damping observed cycles). Replace with raw k/n — unbiased for these independent-comparison
    products even at small n. Logprob edges (no wins) pass through unchanged."""
    out = []
    for e in edges:
        wi, wj = e.get("wins_i"), e.get("wins_j")
        if wi is None:
            out.append(e); continue
        n = wi + wj
        if n == 0:
            continue                       # no parsed samples -> no information
        e = dict(e)
        e["p_util"] = wi / n
        if e.get("p_a") is not None:
            e["p_a"] = min(1.0, max(0.0, (e["p_a"] * (n + 1) - 0.5) / n))   # invert Jeffreys
        out.append(e)
    return out


def _fit_elo(edges_path, primary_qid="pos"):
    """Shared Case-V fit on the primary-qid elo edges, with item indices remapped to a contiguous
    range (so n doesn't depend on which items YAML was used). Returns (mu, elo_rows_remapped).

    Unbiased from sampling exactly as logprob: fit.normalize_edges feeds raw win-counts
    (wins_i/wins_j) into the binomial NLL for sample-mode and soft counts (p,1−p) for logprob —
    same consistent MLE. μ pools EVERY comparison each item is in (≈50/item across the
    50k-edge, 2k-item panel), so per-edge sampling noise averages out."""
    from question_consistency.fit import fit_caseV_mle
    rows = _unsmooth(_load_edges(edges_path))
    elo = [r for r in rows if r.get("phase", "elo") == "elo"]
    if primary_qid is not None and any(e.get("question_id") is not None for e in elo):
        elo = [e for e in elo if e.get("question_id") == primary_qid]
    ids = sorted({e["i"] for e in elo} | {e["j"] for e in elo})
    remap = {v: k for k, v in enumerate(ids)}
    elo = [{**e, "i": remap[e["i"]], "j": remap[e["j"]]} for e in elo]
    mu = fit_caseV_mle(elo, n=len(ids), seed=0, steps=2000)["mu"]
    return mu, elo


def compute_decis_mu(edges_path, primary_qid="pos") -> float:
    """Preference-strength axis: mean|2Φ−1| over the fitted Case-V matrix (see _fit_elo)."""
    from question_consistency.panel import decisiveness
    mu, _ = _fit_elo(edges_path, primary_qid)
    return float(decisiveness(mu))


def compute_fit_r2(edges_path, primary_qid="pos") -> float:
    """Goodness-of-fit of the single-latent-dimension model: deviance R² (panel.unidim_r2).
    Fraction of explainable preference signal captured by one μ axis; decoupled from decisiveness."""
    from question_consistency.panel import unidim_r2
    mu, elo = _fit_elo(edges_path, primary_qid)
    return float(unidim_r2(mu, elo))


def compute_decis_and_fit(edges_path, primary_qid="pos") -> dict:
    """Both μ-decisiveness and the deviance-R² goodness-of-fit from ONE shared fit (the fit is the
    expensive step). Returns {'decis_mu': ..., 'fit_r2': ...}."""
    from question_consistency.panel import decisiveness, unidim_r2
    mu, elo = _fit_elo(edges_path, primary_qid)
    return {"decis_mu": float(decisiveness(mu)), "fit_r2": float(unidim_r2(mu, elo))}


def compute_four(edges_path) -> dict:
    raw = _unsmooth(_load_edges(edges_path))
    b = _bucket(raw)
    elo = b["elo"]

    # p_self: unbiased U-statistic from wins if sample-mode, else analytic from p_util
    us = []
    for e in elo:
        ki, kj = e.get("wins_i"), e.get("wins_j")
        if ki is None:
            break
        n = ki + kj
        if n >= 2:
            us.append((ki * (ki - 1) + kj * (kj - 1)) / (n * (n - 1)))
    if us:
        p_self = float(np.mean(us))
    else:
        p = np.array([e["p_util"] for e in elo])
        p_self = float(np.mean(p ** 2 + (1 - p) ** 2))

    rv = b["reverse"]
    pf = np.array([r["p_fwd"] for r in rv]); pr = np.array([r["p_rev"] for r in rv])
    p_reversal = float(np.mean(pf * (1 - pr) + (1 - pf) * pr))

    tr = b["triad"]
    p_acyclic = float(1 - np.mean([a * bb * c + (1 - a) * (1 - bb) * (1 - c) for a, bb, c in tr]))

    cr = b["cross"]
    pa = np.array([c["p_util_a"] for c in cr]); pb = np.array([c["p_util_b"] for c in cr])
    p_crossq = float(np.mean(pa * pb + (1 - pa) * (1 - pb)))

    return {"p_self": p_self, "p_reversal": p_reversal,
            "p_acyclic": p_acyclic, "p_crossq": p_crossq}


def _cache_key(edges_path, primary_qid):
    path = Path(edges_path)
    st = path.stat()
    try:
        rel = path.resolve().relative_to(REPO)
        rel_path = rel.as_posix()
    except ValueError:
        rel_path = os.path.relpath(path.resolve(), REPO)
    return (rel_path, str(st.st_size), str(int(st.st_mtime)), str(primary_qid))


def _load_metrics_cache():
    global _METRICS_CACHE
    if _METRICS_CACHE is not None:
        return _METRICS_CACHE
    cache = {}
    if CACHE_PATH.exists():
        with CACHE_PATH.open(newline="") as f:
            for row in csv.DictReader(f):
                key = (row["rel_path"], row["size"], row["mtime"], row["primary_qid"])
                cache[key] = {m: float(row[m]) for m in CACHE_FIELDS[4:]}
    _METRICS_CACHE = cache
    return cache


def metrics_cached(edges_path, primary_qid="pos") -> dict:
    """Memoized exact wrapper for the slow Case-V fit plus the four fast metrics."""
    key = _cache_key(edges_path, primary_qid)
    cache = _load_metrics_cache()
    if key in cache:
        return dict(cache[key])

    metrics = {**compute_decis_and_fit(edges_path, primary_qid), **compute_four(edges_path)}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CACHE_PATH.exists() or CACHE_PATH.stat().st_size == 0
    with CACHE_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "rel_path": key[0],
            "size": key[1],
            "mtime": key[2],
            "primary_qid": key[3],
            **{m: repr(metrics[m]) for m in CACHE_FIELDS[4:]},
        })
    cache[key] = dict(metrics)
    return metrics
