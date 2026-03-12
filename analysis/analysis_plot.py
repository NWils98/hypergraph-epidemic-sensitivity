#!/usr/bin/env python3
"""
analysis_plot.py

ONE script that generates the requested paper figures (no parameters).

Expected inputs (relative to where you run this script):
  - derived_structural_sa/run_level.parquet
  - derived_structural_sa/timeseries/<pop>/seedXX.parquet
  - extracted_structural_sa/extracted/<pop>/seedXX/{advanced_daily.parquet,daily_edge_concentration.parquet,daily_summary.parquet}

Outputs (all into OUTDIR):
  - EXPLAIN__diverse_importance_both.png
  - EXPLAIN__lagcurves_by_family.png
  - EXPLAIN__selected_corr.png
  - EXPLAIN__selected_scatter_grid.png
  - RIBBONS_BY_FAMILY__PCT_PUBLIC_infected.png
  - SENSITIVITY__effectsize_heatmap.png
  - variance_decomposition.png
  - LUCKY_VIOLIN__peak_infectious.png

Also writes small CSV helpers (non-required but useful):
  - EXPLAIN__spearman_tables.csv
  - SENSITIVITY__effectsize_table.csv

Notes
-----
Adjusted for paper readability:
- Shorter outcome labels
- "Weekend mixing" renamed to "Community mixing"
- Effect size heatmap rows ordered by auc_infectious
- Simpler metric labels in explainability plots
- Correlation heatmap clustered and relabeled
- PCA figure removed
- Scatter grid reduced to 2x2 (top 4 selected metrics)
- Lag-curve titles simplified
- Diverse importance now in one two-panel figure
"""

from __future__ import annotations

from pathlib import Path
import re
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional row clustering for heatmaps / metric correlation ordering
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
except Exception:
    linkage = None
    leaves_list = None
    squareform = None


# =============================================================================
# Fixed paths (NO parameters)
# =============================================================================

DERIVED_ROOT = Path("results/plan5/derived_structural_sa")
EXTRACTED_ROOT = Path("results/plan5/extracted_structural_sa") / "extracted"
OUTDIR = Path("results/plan5/paper_figs_pack_allinoneV3")

# Files
RUN_LEVEL_PATH = DERIVED_ROOT / "run_level.parquet"


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def savefig(path: Path, dpi: int = 220) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def pretty(ax):
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for s in ax.spines.values():
        s.set_alpha(0.6)

def _seed_to_int(x) -> int:
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else int(s)

def infer_baseline_pop(pops: list[str]) -> str:
    for p in pops:
        if "pop_belgium600k_c500_teachers_censushh" in p:
            return p
    return sorted(pops)[0]

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    denom = (nx + ny - 2)
    if denom <= 0:
        return np.nan
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / denom)
    if not np.isfinite(sp) or sp <= 1e-12:
        return np.nan
    return (np.mean(x) - np.mean(y)) / sp

def spearmanr_nan(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan
    xr = pd.Series(x[mask]).rank().to_numpy()
    yr = pd.Series(y[mask]).rank().to_numpy()
    if np.nanstd(xr) == 0 or np.nanstd(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])

def safe_log1p(x):
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    x = np.clip(x, 0, None)
    return np.log1p(x)

def zscore(s: pd.Series) -> pd.Series:
    m = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return s * 0.0
    return (s - m) / sd

def pretty_outcome_name(raw: str) -> str:
    mapping = {
        "peak_infectious": "Peak infections",
        "auc_infectious": "Cumulative infections",
        "final_infected_max": "Final infections",
        "early_growth_log1p_infectious_slope_d1_28": "Early growth rate",
        "peak_day_infectious": "Peak timing",
        "peak_day": "Peak timing",
    }
    return mapping.get(str(raw), str(raw).replace("_", " "))

def short_pop_label(pop: str) -> str:
    if "pop_belgium600k_c500_teachers_censushh" in pop:
        return "Baseline"
    s = str(pop)
    s = s.replace("_same_mean", "")
    s = s.replace("_", " ")
    return s

def family_sort_key(pop: str) -> Tuple[int, str]:
    order = [
        ("P1_", 1), ("P2_", 1),
        ("P3_", 2), ("P4_", 2), ("P5_", 2),
        ("P6_", 3), ("P7_", 3),
        ("P8_", 4), ("P9_", 4),
        ("P10_", 5), ("P11_", 5),
    ]
    for pref, k in order:
        if str(pop).startswith(pref):
            return (k, str(pop))
    if "pop_belgium600k_c500_teachers_censushh" in str(pop):
        return (0, str(pop))
    return (99, str(pop))

def metric_base_and_timing(raw: str) -> Tuple[str, str]:
    """
    Turn long internal feature name into:
      - readable metric family/name
      - readable timing label
    """
    s = str(raw)

    timing = ""
    m = re.search(r"__(mean|max|min|std)_(early|mid|late)$", s)
    if m:
        agg, win = m.groups()
        if agg == "max":
            timing = "max " + win
        elif agg == "min":
            timing = "min " + win
        elif agg == "std":
            timing = "variation " + win
        else:
            timing = win
        s = s[:m.start()]

    s = s.replace("adv__", "")
    s = s.replace("edge__", "")

    repl = {
        "daily_coupling_": "",
        "coupling_": "",
        "community_secondary": "community",
        "community_primary": "community",
        "hh": "household",
        "rm": "risk mass",
        "riskmass": "risk mass",
        "w1": "Wasserstein",
        "js": "JS",
        "kl": "KL",
        "entropy": "entropy",
        "gini": "Gini",
        "m1": "",
        "m2": "",
        "trace": "trace",
        "spectral_radius": "spectral radius",
        "eigengap": "eigengap",
        "eig2": "eigengap",
        "eig2 a": "eigengap",
        "lambda": "eigenvalue",
        "college": "college",
        "work": "work",
        "school": "school",
        "comms": "community",
        "vs_day1": "vs day 1",
        "vsd1": "vs day 1",
    }
    for a, b in repl.items():
        s = s.replace(a, b)

    s = s.replace("__", " ")
    s = s.replace("|", " ")
    s = s.replace("/", " ")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    low = s.lower()

    if "wasserstein" in low and "community" in low:
        name = "Community risk shift"
    elif "wasserstein" in low and "college" in low:
        name = "College risk shift"
    elif "wasserstein" in low and "work" in low:
        name = "Work risk shift"
    elif "wasserstein" in low and "household" in low:
        name = "Household risk shift"
    elif "js" in low and "risk mass" in low:
        name = "Risk mass divergence"
    elif "kl" in low and "risk mass" in low:
        name = "Risk mass divergence"
    elif "entropy" in low and "risk mass" in low:
        name = "Risk mass entropy"
    elif "entropy" in low and "community" in low:
        name = "Community entropy"
    elif "entropy" in low and "work" in low:
        name = "Work entropy"
    elif "entropy" in low and "household" in low:
        name = "Household entropy"
    elif "gini" in low and "risk mass" in low:
        name = "Risk mass concentration"
    elif "coupling" in low and "infected" in low and ("eigengap" in low or "eig2" in low):
        name = "Infected coupling eigengap"
    elif "coupling" in low and "infected" in low and "spectral radius" in low:
        name = "Infected coupling spectral radius"
    elif "coupling" in low and "infected" in low and "trace" in low:
        name = "Infected coupling trace"
    elif "coupling" in low and "infected" in low:
        name = "Infected coupling"
    elif "coupling" in low and ("eigengap" in low or "eig2" in low):
        name = "Coupling eigengap"
    elif "coupling" in low and "spectral radius" in low:
        name = "Coupling spectral radius"
    elif "coupling" in low and "trace" in low:
        name = "Coupling trace"
    elif "coupling" in low:
        name = "Coupling strength"
    elif "spectral radius" in low:
        name = "Coupling spectral radius"
    elif "eigengap" in low or "eig2" in low:
        name = "Coupling eigengap"
    elif "trace" in low:
        name = "Coupling trace"
    elif "risk mass" in low:
        name = "Risk mass metric"
    elif "gini" in low:
        name = "Concentration metric"
    else:
        name = s.title()

    if timing:
        label = f"{name} ({timing})"
    else:
        label = name
    return name, label

def pretty_metric_name(raw: str) -> str:
    return metric_base_and_timing(raw)[1]

def cluster_order_from_corr(C: pd.DataFrame) -> List[str]:
    cols = list(C.columns)
    if linkage is None or leaves_list is None or squareform is None or len(cols) <= 2:
        return cols

    try:
        D = 1.0 - np.abs(C.to_numpy())
        np.fill_diagonal(D, 0.0)
        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method="average")
        order = leaves_list(Z)
        return [cols[i] for i in order]
    except Exception:
        return cols


# =============================================================================
# Loaders
# =============================================================================

def load_run_level() -> pd.DataFrame:
    if not RUN_LEVEL_PATH.exists():
        raise FileNotFoundError(f"Missing {RUN_LEVEL_PATH.resolve()}")
    df = pd.read_parquet(RUN_LEVEL_PATH).copy()
    if not {"pop", "seed"}.issubset(df.columns):
        raise ValueError("run_level.parquet must contain columns: pop, seed")
    df["pop"] = df["pop"].astype(str)
    df["seed"] = df["seed"].apply(_seed_to_int).astype(int)
    return df

def load_timeseries() -> pd.DataFrame:
    ts_root = DERIVED_ROOT / "timeseries"
    rows = []
    if not ts_root.exists():
        return pd.DataFrame()
    for pop_dir in sorted([d for d in ts_root.iterdir() if d.is_dir()]):
        pop = pop_dir.name
        for f in sorted(pop_dir.glob("seed*.parquet")):
            seed = f.stem.replace("seed", "")
            d = pd.read_parquet(f)
            if "day" not in d.columns:
                continue
            d = d.copy()
            d["pop"] = pop
            d["seed"] = seed
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    ts = pd.concat(rows, ignore_index=True)
    ts["pop"] = ts["pop"].astype(str)
    ts["seed"] = ts["seed"].apply(_seed_to_int).astype(int)
    ts["day"] = pd.to_numeric(ts["day"], errors="coerce")
    return ts


# =============================================================================
# 1) SENSITIVITY__effectsize_heatmap
# =============================================================================

def pick_outcomes(df: pd.DataFrame) -> list[str]:
    preferred = [
        "peak_infectious",
        "auc_infectious",
        "final_infected_max",
        "early_growth_log1p_infectious_slope_d1_28",
        "peak_day_infectious",
    ]
    outcomes = [c for c in preferred if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if outcomes:
        return outcomes
    cand = []
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["peak_", "auc_", "final_", "early_growth", "attack", "rt_"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                cand.append(c)
    return cand[:8]

def fig_effectsize_heatmap(run_level: pd.DataFrame) -> None:
    pops = sorted(run_level["pop"].astype(str).unique().tolist(), key=family_sort_key)
    baseline = infer_baseline_pop(pops)
    base = run_level[run_level["pop"] == baseline].copy()

    outcomes = pick_outcomes(run_level)
    if not outcomes:
        raise ValueError("Could not find any outcome columns to summarize for the heatmap.")

    rows = []
    for pop in pops:
        if pop == baseline:
            continue
        g = run_level[run_level["pop"] == pop]
        row = {"pop": pop}
        for oc in outcomes:
            row[oc] = cohen_d(g[oc].to_numpy(), base[oc].to_numpy())
        rows.append(row)

    mat = pd.DataFrame(rows).set_index("pop")

    sort_col = "auc_infectious" if "auc_infectious" in mat.columns else mat.columns[0]
    mat = mat.sort_values(sort_col, ascending=False)

    display_cols = [c for c in outcomes if c in mat.columns]
    mat = mat.loc[:, display_cols]

    CAP = 3.0
    Z = mat.to_numpy()
    Z_clip = np.clip(Z, -CAP, +CAP)

    fig_h = max(5.5, 0.48 * len(mat.index) + 1.5)
    plt.figure(figsize=(10.8, fig_h))
    ax = plt.gca()
    im = ax.imshow(
        Z_clip,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-CAP,
        vmax=+CAP,
    )

    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels([short_pop_label(p) for p in mat.index], fontsize=9)

    ax.set_xticks(np.arange(len(display_cols)))
    ax.set_xticklabels([pretty_outcome_name(c) for c in display_cols], rotation=10, ha="right", fontsize=10)

    # ax.set_title(
    #     "Sensitivity summary: effect sizes vs baseline (Cohen's d)\n"
    #     f"Colors capped at ±{CAP:g} so large effects remain readable"
    # )
    ax.grid(False)
    for s in ax.spines.values():
        s.set_alpha(0.6)

    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Cohen's d (variant vs baseline)")
    cb.ax.axhline(0.0, linewidth=1.0, alpha=0.6)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            if not np.isfinite(v):
                continue
            tc = "white" if abs(Z_clip[i, j]) > 1.6 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=tc)

    savefig(OUTDIR / "SENSITIVITY__effectsize_heatmap.png", dpi=240)

    out = mat.copy()
    out.index.name = "pop"
    out.to_csv(OUTDIR / "SENSITIVITY__effectsize_table.csv")


# =============================================================================
# 2) RIBBONS_BY_FAMILY__PCT_PUBLIC_infected
# =============================================================================

def ribbon_quantiles(ts: pd.DataFrame, pop: str, y: str, qlo=0.1, qhi=0.9):
    g = ts.loc[ts["pop"] == pop, ["day", "seed", y]].copy().sort_values("day")
    piv = g.pivot_table(index="day", columns="seed", values=y)
    mu = piv.mean(axis=1)
    lo = piv.quantile(qlo, axis=1)
    hi = piv.quantile(qhi, axis=1)
    return mu.index.to_numpy(), mu.to_numpy(), lo.to_numpy(), hi.to_numpy()

def family_groups(pops):
    families = [
        ("Household", ["P1_", "P2_"]),
        ("Community mixing", ["P3_", "P4_", "P5_"]),
        ("Age structure", ["P6_", "P7_"]),
        ("School assignment", ["P8_", "P9_"]),
        ("Workplace assignment", ["P10_", "P11_"]),
    ]
    out = []
    for name, prefs in families:
        members = [p for p in pops if any(p.startswith(pref) for pref in prefs)]
        out.append((name, sorted(members, key=family_sort_key)))
    return out

def short_label(pop: str) -> str:
    if "pop_belgium600k_c500_teachers_censushh" in pop:
        return "baseline"
    s = str(pop)
    s = s.replace("_same_mean", "")
    return s

def fig_family_pct_public(ts: pd.DataFrame, y: str = "infected",
                          qlo=0.1, qhi=0.9, baseline_floor: float = 1000.0, xmax: int = 100):
    if ts.empty or y not in ts.columns:
        print(f"[ribbons_pct_public] missing timeseries or column '{y}' -> skipping")
        return

    pops = sorted(ts["pop"].astype(str).unique(), key=family_sort_key)
    baseline_pop = infer_baseline_pop(pops)

    bx, bmu, blo, bhi = ribbon_quantiles(ts, baseline_pop, y, qlo=qlo, qhi=qhi)

    if xmax is not None:
        mwin = bx <= xmax
        bx, bmu, blo, bhi = bx[mwin], bmu[mwin] if False else bmu[mwin], blo[mwin], bhi[mwin]

    mask = bmu >= baseline_floor
    bx_m = bx[mask]
    bmu_m = bmu[mask]
    blo_m = blo[mask]
    bhi_m = bhi[mask]

    if len(bx_m) < 10:
        print("[ribbons_pct_public] baseline_floor too high; not enough points left -> skipping")
        return

    eps = 1e-9
    b_pct_lo = 100.0 * (blo_m / np.maximum(bmu_m, eps) - 1.0)
    b_pct_hi = 100.0 * (bhi_m / np.maximum(bmu_m, eps) - 1.0)

    fams = family_groups(pops)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.ravel()

    cmap = plt.get_cmap("tab10")

    for i, (fam_name, members) in enumerate(fams):
        ax = axes[i]

        baseline_color = "black"
        ax.axhline(0.0, linewidth=2.0, alpha=0.9, color=baseline_color, label="baseline")
        ax.fill_between(bx_m, b_pct_lo, b_pct_hi, alpha=0.10, color=baseline_color)

        for j, pop in enumerate(members):
            color = cmap(j % 10)

            x, mu, lo, hi = ribbon_quantiles(ts, pop, y, qlo=qlo, qhi=qhi)
            if xmax is not None:
                w = x <= xmax
                x, mu, lo, hi = x[w], mu[w], lo[w], hi[w]

            common = np.intersect1d(bx_m, x)
            idx_b = np.searchsorted(bx_m, common)
            idx_p = np.searchsorted(x, common)

            pct_mu = 100.0 * (mu[idx_p] / np.maximum(bmu_m[idx_b], eps) - 1.0)
            pct_lo = 100.0 * (lo[idx_p] / np.maximum(bmu_m[idx_b], eps) - 1.0)
            pct_hi = 100.0 * (hi[idx_p] / np.maximum(bmu_m[idx_b], eps) - 1.0)

            ax.fill_between(common, pct_lo, pct_hi, alpha=0.18, color=color)
            ax.plot(common, pct_mu, linewidth=2.2, alpha=0.95, color=color, label=short_label(pop))

        ax.set_title(f"{fam_name}: % change vs baseline")
        ax.set_xlabel("Day")
        ax.set_ylabel("% change in infected")
        pretty(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right")

    axes[-1].axis("off")
    # fig.suptitle(
    #     f"Variant differences in infected relative to baseline "
    #     f"(bands = {int(qlo*100)}–{int(qhi*100)}% over seeds)",
    #     y=1.02,
    # )
    savefig(OUTDIR / f"RIBBONS_BY_FAMILY__PCT_PUBLIC_{y}.png")


# =============================================================================
# 3) variance_decomposition
# =============================================================================

def fig_variance_decomposition(run_level: pd.DataFrame) -> None:
    outcomes = [
        "peak_infectious",
        "auc_infectious",
        "early_growth_log1p_infectious_slope_d1_28",
    ]
    rows = []
    for y in outcomes:
        if y not in run_level.columns:
            continue
        tmp = run_level[["pop", "seed", y]].copy()
        tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
        tmp = tmp.dropna(subset=[y])

        pop_mean = tmp.groupby("pop")[y].mean()
        between = pop_mean.var(ddof=1)
        within = tmp.groupby("pop")[y].var(ddof=1).mean()

        denom = (between + within)
        if not np.isfinite(denom) or denom <= 0:
            continue

        share_pop = between / denom
        share_seed = within / denom
        rows.append([y, share_pop, share_seed])

    if not rows:
        print("[variance_decomposition] No outcomes found -> skipping")
        return

    res = pd.DataFrame(rows, columns=["outcome", "pop", "seed"])

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(res))
    ax.bar(x, res["pop"], label="Population structure")
    ax.bar(x, res["seed"], bottom=res["pop"], label="Stochastic seed")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_outcome_name(o) for o in res["outcome"]], rotation=10, ha="right")
    ax.set_ylabel("Variance share")
    ax.set_title("Variance decomposition of epidemic outcomes\n\n")
    ax.legend(frameon=False)

    for i, r in res.iterrows():
        ax.text(i, min(1.03, float(r["pop"] + r["seed"]) + 0.02), f"{r['pop']:.2f} / {r['seed']:.2f}", ha="center", fontsize=9)

    savefig(OUTDIR / "variance_decomposition.png", dpi=220)


# =============================================================================
# 4) Explainability figures
# =============================================================================

DAILY_TABLE_CANDIDATES = [
    "advanced_daily.parquet",
    "daily_edge_concentration.parquet",
    "daily_summary.parquet",
]

WINDOWS = {
    "early": (1, 28),
    "mid": (29, 84),
    "late": (85, 196),
}

def _parse_pop_seed_from_path(p: Path) -> Tuple[Optional[str], Optional[int]]:
    parts = p.parts
    pop = None
    seed = None
    for i in range(len(parts)):
        if i + 1 < len(parts) and parts[i].endswith("extracted"):
            if i + 2 < len(parts):
                pop = parts[i + 1]
                seed = _seed_to_int(parts[i + 2])
                return pop, seed
    for part in parts:
        if re.match(r"seed\d+", part):
            seed = _seed_to_int(part)
    if seed is not None:
        for i, part in enumerate(parts):
            if re.match(r"seed\d+", part) and i > 0:
                pop = parts[i - 1]
                break
    return pop, seed

def list_daily_tables() -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    if not EXTRACTED_ROOT.exists():
        return out
    for fn in DAILY_TABLE_CANDIDATES:
        out[fn] = sorted(EXTRACTED_ROOT.rglob(fn))
    return out

def load_daily_table(p: Path) -> pd.DataFrame:
    d = pd.read_parquet(p).copy()
    pop, seed = _parse_pop_seed_from_path(p)
    if "day" not in d.columns:
        raise ValueError(f"{p} missing 'day' column.")
    d["day"] = pd.to_numeric(d["day"], errors="coerce")
    d["pop"] = str(pop)
    d["seed"] = _seed_to_int(seed)
    d["seed"] = d["seed"].astype(int)
    return d

def _agg_window(s: pd.Series, kind: str) -> float:
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if kind == "mean":
        return float(np.nanmean(a))
    if kind == "max":
        return float(np.nanmax(a))
    if kind == "min":
        return float(np.nanmin(a))
    if kind == "std":
        return float(np.nanstd(a))
    raise ValueError(kind)

def build_run_features(advanced_daily_paths: List[Path], edge_conc_paths: List[Path], keep_regex: Optional[str] = None) -> pd.DataFrame:
    rows = []

    for p in advanced_daily_paths:
        d = load_daily_table(p)
        pop = d["pop"].iloc[0]
        seed = int(d["seed"].iloc[0])

        metric_cols = [c for c in d.columns if c not in ("day", "pop", "seed")]
        if keep_regex:
            rgx = re.compile(keep_regex)
            metric_cols = [c for c in metric_cols if rgx.search(c)]

        feats = {"pop": pop, "seed": seed}
        for wname, (a, b) in WINDOWS.items():
            wdf = d[(d["day"] >= a) & (d["day"] <= b)]
            for c in metric_cols:
                feats[f"adv__{c}__mean_{wname}"] = _agg_window(wdf[c], "mean")
                feats[f"adv__{c}__max_{wname}"] = _agg_window(wdf[c], "max")
        rows.append(feats)

    adv = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["pop", "seed"])

    rows = []
    for p in edge_conc_paths:
        d = load_daily_table(p)
        pop = d["pop"].iloc[0]
        seed = int(d["seed"].iloc[0])

        metric_cols = [c for c in d.columns if c not in ("day", "pop", "seed")]
        if keep_regex:
            rgx = re.compile(keep_regex)
            metric_cols = [c for c in metric_cols if rgx.search(c)]

        feats = {"pop": pop, "seed": seed}
        for wname, (a, b) in WINDOWS.items():
            wdf = d[(d["day"] >= a) & (d["day"] <= b)]
            for c in metric_cols:
                feats[f"edge__{c}__mean_{wname}"] = _agg_window(wdf[c], "mean")
                feats[f"edge__{c}__max_{wname}"] = _agg_window(wdf[c], "max")
        rows.append(feats)

    edge = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["pop", "seed"])

    if adv.empty and edge.empty:
        raise ValueError("No features built (no daily tables found).")

    feats = adv if edge.empty else (edge if adv.empty else adv.merge(edge, on=["pop", "seed"], how="outer"))
    feats["pop"] = feats["pop"].astype(str)
    feats["seed"] = feats["seed"].apply(_seed_to_int).astype(int)
    return feats

def compute_spearman_tables(run_level: pd.DataFrame, feats: pd.DataFrame, outcome: str) -> pd.DataFrame:
    df = feats.merge(run_level[["pop", "seed", outcome]], on=["pop", "seed"], how="inner").copy()
    y = pd.to_numeric(df[outcome], errors="coerce").to_numpy(dtype=float)

    feature_cols = [c for c in df.columns if c not in ("pop", "seed", outcome)]
    rows = []

    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        rows.append({"metric": c, "mode": "global", "rho": spearmanr_nan(x, y)})

    df2 = df.copy()
    df2[outcome] = df2.groupby("pop")[outcome].transform(lambda s: s - s.mean())
    for c in feature_cols:
        df2[c] = df2.groupby("pop")[c].transform(lambda s: s - s.mean())
    y2 = pd.to_numeric(df2[outcome], errors="coerce").to_numpy(dtype=float)
    for c in feature_cols:
        x2 = pd.to_numeric(df2[c], errors="coerce").to_numpy(dtype=float)
        rows.append({"metric": c, "mode": "within_pop", "rho": spearmanr_nan(x2, y2)})

    out = pd.DataFrame(rows)
    out["abs_rho"] = out["rho"].abs()
    return out

def prune_by_redundancy(feats: pd.DataFrame, candidates: List[str], max_keep: int = 15, corr_thresh: float = 0.92) -> List[str]:
    if not candidates:
        return []
    X = feats.set_index(["pop", "seed"])[candidates].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    ok = [c for c in candidates if X[c].notna().sum() >= 10]
    if not ok:
        return []
    X = X[ok]
    corr = X.corr(method="pearson", min_periods=10).fillna(0.0)
    kept: List[str] = []
    for c in ok:
        if len(kept) >= max_keep:
            break
        if not any(abs(corr.loc[c, k]) >= corr_thresh for k in kept):
            kept.append(c)
    return kept

def fig_bar_importance_both(within_tbl: pd.DataFrame, global_tbl: pd.DataFrame, path: Path, topk: int = 10):
    w = within_tbl.sort_values("abs_rho", ascending=False).head(topk).copy().sort_values("rho", ascending=True)
    g = global_tbl.sort_values("abs_rho", ascending=False).head(topk).copy().sort_values("rho", ascending=True)

    nrows = max(len(w), len(g))
    fig_h = max(1,0.3 * nrows + 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, fig_h), sharex=True)
    plt.xlim(-1, 1)
    for ax, t, title in zip(
        axes,
        [w, g],
        ["Within-population metric associations", "Global metric associations"]
    ):
        labels = [pretty_metric_name(x) for x in t["metric"]]
        vals = t["rho"].to_numpy()
        ax.barh(labels, vals)
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Spearman ρ")
        ax.set_title(title)

        pretty(ax)

    # fig.suptitle("Hypergraph metric associations with peak infections", y=1.02, fontsize=15)
    savefig(path)

def fig_corr_heatmap(feats: pd.DataFrame, cols: List[str], title: str, path: Path):
    X = feats.set_index(["pop", "seed"])[cols].apply(pd.to_numeric, errors="coerce")
    C = X.corr(method="pearson", min_periods=10).fillna(0.0)

    order = cluster_order_from_corr(C)
    C = C.loc[order, order]
    labels = [pretty_metric_name(c) for c in order]

    plt.figure(figsize=(9.8, 4))
    im = plt.imshow(C.to_numpy(), aspect="auto", vmin=-1, vmax=1, cmap="viridis")
    plt.colorbar(im, label="Pearson correlation")
    plt.xticks(range(len(order)), labels, rotation=20, ha="right", fontsize=9)
    plt.yticks(range(len(order)), labels, fontsize=9)
    plt.title(title)
    savefig(path)

def fig_scatter_grid(run_level: pd.DataFrame, feats: pd.DataFrame, cols: List[str], outcome: str, path: Path):
    if not cols:
        return
    cols = cols[:4]

    df = feats.merge(run_level[["pop", "seed", outcome]], on=["pop", "seed"], how="inner").copy()
    df[outcome] = pd.to_numeric(df[outcome], errors="coerce")

    n = len(cols)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    plt.figure(figsize=(5.2 * ncols, 4.2 * nrows))

    for i, c in enumerate(cols, 1):
        ax = plt.subplot(nrows, ncols, i)
        x = pd.to_numeric(df[c], errors="coerce")
        y = df[outcome]
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=28, alpha=0.8)

        if m.sum() >= 10:
            xx = x[m].to_numpy()
            yy = y[m].to_numpy()
            a, b = np.polyfit(xx, yy, 1)
            xs = np.linspace(np.nanmin(xx), np.nanmax(xx), 50)
            ax.plot(xs, a * xs + b, linewidth=2)
            rho = spearmanr_nan(xx, yy)
            ax.set_title(f"{pretty_metric_name(c)}\nρ = {rho:.2f}")
        else:
            ax.set_title(pretty_metric_name(c))

        ax.set_xlabel("")
        ax.set_ylabel("Peak infections" if (i % ncols == 1 or ncols == 1) else "")
        pretty(ax)

    # plt.suptitle("Selected metric–outcome relationships", y=1.02, fontsize=15)
    savefig(path)

def _find_infectious_col(daily_summary: pd.DataFrame) -> str:
    candidates = ["n_infectious", "infectious", "I", "n_infected", "n_infected_total", "n_infectious_all"]
    for c in candidates:
        if c in daily_summary.columns:
            return c
    cols = [c for c in daily_summary.columns if "infect" in c.lower() and "frac" not in c.lower()]
    cols = [c for c in cols if c not in ("day", "seed", "pop")]
    if cols:
        return cols[0]
    raise ValueError("Could not find infectious column in daily_summary.")

def compute_lag_curve_for_metric(metric_daily: pd.DataFrame, daily_summary: pd.DataFrame, metric_col: str,
                                 horizon: int, max_lag: int = 28) -> np.ndarray:
    inf_col = _find_infectious_col(daily_summary)
    ds = daily_summary[["day", "pop", "seed", inf_col]].copy()
    ds[inf_col] = pd.to_numeric(ds[inf_col], errors="coerce")
    md = metric_daily[["day", "pop", "seed", metric_col]].copy()
    md[metric_col] = pd.to_numeric(md[metric_col], errors="coerce")

    j = md.merge(ds, on=["pop", "seed", "day"], how="inner").sort_values(["pop", "seed", "day"])
    rhos = np.full((max_lag + 1,), np.nan)

    for L in range(max_lag + 1):
        per_run = []
        for (pop, seed), g in j.groupby(["pop", "seed"]):
            g = g.sort_values("day")
            x = g[metric_col].to_numpy(dtype=float)

            I = g[inf_col].to_numpy(dtype=float)
            LI = safe_log1p(I)

            n = len(g)
            idx = np.arange(n)
            src = idx
            a = src + L
            b = src + L + horizon
            ok = (b < n)
            if ok.sum() < 8:
                continue

            x_aligned = x[src[ok]]
            growth = LI[b[ok]] - LI[a[ok]]
            rho = spearmanr_nan(x_aligned, growth)
            if np.isfinite(rho):
                per_run.append(rho)

        if per_run:
            rhos[L] = float(np.nanmean(per_run))
    return rhos

def fig_lagcurves_by_family(adv_paths: List[Path], edge_paths: List[Path], dsum_paths: List[Path],
                            spearman_tbl: pd.DataFrame) -> None:
    buckets = [
        ("Coupling", re.compile(r"(coupling|cpl|spectral|lambda|eig|trace)", re.I)),
        ("Risk redistribution", re.compile(r"(entropy_riskmass|kl_riskmass|js_riskmass|w1_riskmass|riskmass)", re.I)),
        ("Edge concentration", re.compile(r"(gini|top1pct|edge_conc|risk_mass)", re.I)),
        ("Core structure", re.compile(r"(core)", re.I)),
    ]
    within = spearman_tbl[spearman_tbl["mode"] == "within_pop"].sort_values("abs_rho", ascending=False)

    chosen: Dict[str, str] = {}
    for fam, rgx in buckets:
        for m in within["metric"].tolist():
            if rgx.search(m):
                chosen[fam] = m
                break

    chosen = {k: v for k, v in chosen.items() if isinstance(v, str)}
    if not chosen:
        print("[explain] no lag-curve families found -> skipping lag curves")
        return

    adv_daily = [load_daily_table(p) for p in adv_paths] if adv_paths else []
    edge_daily = [load_daily_table(p) for p in edge_paths] if edge_paths else []
    dsum_daily = [load_daily_table(p) for p in dsum_paths] if dsum_paths else []

    if not dsum_daily:
        print("[explain] missing daily_summary -> skipping lag curves")
        return

    dsum_df = pd.concat(dsum_daily, ignore_index=True)

    def _metric_source_table(metric_name: str) -> str:
        if metric_name.startswith("adv__"):
            return "advanced_daily"
        if metric_name.startswith("edge__"):
            return "edge_conc"
        return "advanced_daily"

    def _strip_prefix(metric_name: str) -> str:
        m = re.match(r"^(adv|edge)__([^\s].*?)__([a-z]+)_(early|mid|late)$", metric_name)
        return m.group(2) if m else metric_name

    horizons = [7, 14]
    max_lag = 28
    nplots = len(chosen)
    plt.figure(figsize=(6.2 * nplots, 4.4))

    for i, (fam, feat_name) in enumerate(chosen.items(), 1):
        src = _metric_source_table(feat_name)
        daily_col = _strip_prefix(feat_name)

        if src == "advanced_daily":
            if not adv_daily:
                continue
            daily_df = pd.concat(adv_daily, ignore_index=True)
        else:
            if not edge_daily:
                continue
            daily_df = pd.concat(edge_daily, ignore_index=True)

        if daily_col not in daily_df.columns:
            cands = [c for c in daily_df.columns if daily_col in c]
            if not cands:
                continue
            daily_col_use = cands[0]
        else:
            daily_col_use = daily_col

        ax = plt.subplot(1, nplots, i)
        for h in horizons:
            curve = compute_lag_curve_for_metric(daily_df, dsum_df, daily_col_use, horizon=h, max_lag=max_lag)
            ax.plot(range(max_lag + 1), curve, label=f"{h}-day horizon")
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("Mean Spearman ρ")
        ax.set_title(f"{fam}\n{pretty_metric_name(feat_name)}")
        ax.legend(frameon=False)
        pretty(ax)

    # plt.suptitle("Structural indicators of future infection growth", y=1.04, fontsize=15)
    savefig(OUTDIR / "EXPLAIN__lagcurves_by_family.png")

def run_explainability(run_level: pd.DataFrame) -> None:
    if not EXTRACTED_ROOT.exists():
        print(f"[explain] Missing {EXTRACTED_ROOT} -> skipping explainability figures")
        return

    paths = list_daily_tables()
    adv_paths = paths.get("advanced_daily.parquet", [])
    edge_paths = paths.get("daily_edge_concentration.parquet", [])
    dsum_paths = paths.get("daily_summary.parquet", [])

    if not adv_paths and not edge_paths:
        print("[explain] No advanced_daily or daily_edge_concentration found -> skipping explainability figures")
        return
    if not dsum_paths:
        print("[explain] No daily_summary.parquet found -> skipping lag curves (and explainability set)")
        return

    feats = build_run_features(adv_paths, edge_paths, keep_regex=None)

    outcome = "peak_infectious"
    if outcome not in run_level.columns:
        raise ValueError(f"[explain] run_level missing required outcome '{outcome}'")

    tbl = compute_spearman_tables(run_level, feats, outcome)

    within = tbl[(tbl["mode"] == "within_pop") & (tbl["metric"].str.startswith(("adv__", "edge__")))].copy()
    global_ = tbl[(tbl["mode"] == "global") & (tbl["metric"].str.startswith(("adv__", "edge__")))].copy()

    within_top = within.sort_values("abs_rho", ascending=False).head(60)["metric"].tolist()
    kept = prune_by_redundancy(feats, within_top, max_keep=14, corr_thresh=0.92)
    if len(kept) < 8:
        kept = within_top[:10]

    t1 = within[within["metric"].isin(kept)].copy().sort_values("abs_rho", ascending=False)
    t2 = global_[global_["metric"].isin(kept)].copy().sort_values("abs_rho", ascending=False)

    fig_bar_importance_both(
        t1,
        t2,
        path=OUTDIR / "EXPLAIN__diverse_importance_both.png",
        topk=min(10, max(len(t1), len(t2))),
    )

    fig_corr_heatmap(
        feats,
        cols=kept,
        title="",
        path=OUTDIR / "EXPLAIN__selected_corr.png",
    )

    scatter_cols = t2.sort_values("abs_rho", ascending=False)["metric"].tolist()[:4]
    if len(scatter_cols) < 4:
        scatter_cols = kept[:4]

    fig_scatter_grid(
        run_level,
        feats,
        cols=scatter_cols,
        outcome=outcome,
        path=OUTDIR / "EXPLAIN__selected_scatter_grid.png",
    )

    fig_lagcurves_by_family(adv_paths, edge_paths, dsum_paths, tbl)

    tbl_out = tbl.copy()
    tbl_out["pretty"] = tbl_out["metric"].map(pretty_metric_name)
    tbl_out.to_csv(OUTDIR / "EXPLAIN__spearman_tables.csv", index=False)


# =============================================================================
# 5) “LUCKY VIOLIN” for an outcome
# =============================================================================

def fig_lucky_violin(run_level: pd.DataFrame, outdir: Path, outcome: str = "peak_infectious") -> None:
    if not {"pop", "seed"}.issubset(run_level.columns):
        return
    if outcome not in run_level.columns:
        print(f"[lucky_violin] missing outcome: {outcome}")
        return

    df = run_level.copy()
    pops_all = sorted(df["pop"].astype(str).unique())
    baseline = infer_baseline_pop(pops_all)

    p_variants = []
    for p in pops_all:
        m = re.match(r"^P(\d+)_", str(p))
        if m:
            p_variants.append((int(m.group(1)), p))
    p_variants = [p for _, p in sorted(p_variants, key=lambda x: x[0])]

    # Reverse order so baseline is at the top
    pops = [baseline] + p_variants
    pops_plot = list(pops)

    plt.figure(figsize=(11.8, max(5.2,1)))
    ax = plt.gca()

    data = [df.loc[df["pop"] == p, outcome].to_numpy(float) for p in pops_plot]
    ax.violinplot(data, vert=False, showmeans=False, showmedians=True, showextrema=False)

    rng = np.random.default_rng(1)
    for i, p in enumerate(pops_plot):
        y0 = i + 1
        x = df.loc[df["pop"] == p, outcome].to_numpy(float)
        x = x[np.isfinite(x)]
        jitter = (rng.random(len(x)) - 0.5) * 0.18
        ax.scatter(x, np.full_like(x, y0) + jitter, s=20, alpha=0.7)

    base_mean = np.nanmean(df.loc[df["pop"] == baseline, outcome].to_numpy(float))
    ax.axvline(base_mean, linewidth=2.2, alpha=0.85)

    ax.set_yticks(np.arange(1, len(pops_plot) + 1))
    ax.set_yticklabels([short_pop_label(p) for p in pops_plot], fontsize=8.5)
    ax.invert_yaxis()  # puts the first item at the top, i.e. Baseline
    # ax.set_title("Peak infections by population variant")
    ax.set_xlabel("Peak infections")
    pretty(ax)
    savefig(outdir / f"LUCKY_VIOLIN__{outcome}.png")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dir(OUTDIR)

    run_level = load_run_level()

    fig_effectsize_heatmap(run_level)
    fig_variance_decomposition(run_level)

    ts = load_timeseries()
    fig_family_pct_public(ts, y="infected", qlo=0.1, qhi=0.9, baseline_floor=1000.0, xmax=100)

    run_explainability(run_level)
    fig_lucky_violin(run_level, OUTDIR, outcome="peak_infectious")

    print("\n[OK] wrote figures to:", OUTDIR.resolve())
    print("Expected key files:")
    for fn in [
        "EXPLAIN__diverse_importance_both.png",
        "EXPLAIN__lagcurves_by_family.png",
        "EXPLAIN__selected_corr.png",
        "EXPLAIN__selected_scatter_grid.png",
        "RIBBONS_BY_FAMILY__PCT_PUBLIC_infected.png",
        "SENSITIVITY__effectsize_heatmap.png",
        "variance_decomposition.png",
        "LUCKY_VIOLIN__peak_infectious.png",
    ]:
        print(" -", fn)


if __name__ == "__main__":
    main()