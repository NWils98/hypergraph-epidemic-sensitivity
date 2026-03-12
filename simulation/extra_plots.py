#!/usr/bin/env python3
"""
extra_plots.py

Generate plots for the "extra metrics" pipeline.

Works from a results directory produced by run_all.py (e.g. results_work/)
that contains (at minimum):
  - daily_summary.parquet
  - daily_edge_concentration.parquet
  - hotspot_persistence.parquet
  - coupling/coupling_all_dayXXX.parquet
  - coupling/coupling_infected_dayXXX.parquet
  - hubs_top/hubs_top_dayXXX.parquet
  - superspreaders_top/superspreaders_top_dayXXX.parquet

For EXACT Lorenz curves (and aggregated Lorenz bands), expects:
  - edges_full/edges_full_dayXXX.parquet

Outputs into: <results_dir>/plots/

Run:
  python extra_plots.py <results_dir>
If you omit args, it defaults to "./results_work".

Also provides:
  main_for_run_all(results_dir: Path)
so run_all.py can call it directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_parquet_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _choose_days(summary_df: pd.DataFrame) -> List[int]:
    """
    Pick a few representative days: early, pre-peak, peak, post-peak, late.
    """
    if summary_df.empty:
        return []
    s = summary_df.sort_values("day").reset_index(drop=True)
    days = s["day"].astype(int).to_list()

    peak_idx = int(np.argmax(s["n_infected"].to_numpy()))
    peak_day = int(s.loc[peak_idx, "day"])

    early = days[max(0, int(0.10 * (len(days) - 1)))]
    mid1 = days[max(0, int(0.35 * (len(days) - 1)))]
    mid2 = days[max(0, int(0.65 * (len(days) - 1)))]
    late = days[max(0, int(0.90 * (len(days) - 1)))]

    chosen = sorted(set([early, mid1, peak_day, mid2, late]))
    return chosen


def _list_days_from_edges_full(edges_full_dir: Path) -> List[int]:
    days = []
    for p in sorted(edges_full_dir.glob("edges_full_day*.parquet")):
        # edges_full_dayXYZ.parquet
        stem = p.stem
        try:
            d = int(stem.split("day")[1])
            days.append(d)
        except Exception:
            continue
    return sorted(days)


# -----------------------------
# Risk mass distributions (PDF/CCDF)
# -----------------------------

def ccdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complementary CDF: P(X >= t)
    Returns sorted thresholds and CCDF values.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return np.array([0.0]), np.array([0.0])
    x_sorted = np.sort(x)
    n = x_sorted.size
    y = 1.0 - (np.arange(n) / n)
    return x_sorted, y


def plot_riskmass_distributions_selected_days(edges_full_dir: Path, chosen_days: List[int], outdir: Path) -> None:
    """
    Heavy-tail visualizations using FULL edges:
      - PDF-like histogram on log-x
      - CCDF on log-log
    Per selected day, per edge type.
    """
    _ensure_dir(outdir)

    for day in chosen_days:
        p = edges_full_dir / f"edges_full_day{day:03d}.parquet"
        if not p.exists():
            continue

        df = pd.read_parquet(p)
        if df.empty or "risk_mass_active" not in df.columns:
            continue

        # PDF-like (log-x)
        plt.figure()
        for etype in sorted(df["edge_type"].unique()):
            sub = df[df["edge_type"] == etype]
            x = sub["risk_mass_active"].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            x = x[x > 0]
            if x.size == 0:
                continue
            plt.hist(x, bins=50, density=True, histtype="step", label=str(etype))
        plt.xscale("log")
        plt.xlabel("risk_mass_active (log scale)")
        plt.ylabel("Density (histogram)")
        plt.title(f"Risk mass distribution (PDF-like, log-x) – Day {day}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"fig_riskmass_pdf_logx_day{day:03d}.png", dpi=200)
        plt.close()

        # CCDF (log-log)
        plt.figure()
        for etype in sorted(df["edge_type"].unique()):
            sub = df[df["edge_type"] == etype]
            x = sub["risk_mass_active"].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            x = x[x > 0]
            if x.size == 0:
                continue
            xs, ys = ccdf(x)
            plt.plot(xs, ys, label=str(etype))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("risk_mass_active")
        plt.ylabel("CCDF: P(Risk mass ≥ x)")
        plt.title(f"Risk mass heavy tail (CCDF, log-log) – Day {day}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"fig_riskmass_ccdf_loglog_day{day:03d}.png", dpi=200)
        plt.close()


# -----------------------------
# 1) Risk concentration plots
# -----------------------------

def plot_concentration_over_time(conc_df: pd.DataFrame, outdir: Path) -> None:
    if conc_df.empty:
        return
    _ensure_dir(outdir)

    for metric in ["risk_mass_gini", "risk_mass_top1pct_share"]:
        plt.figure()
        for etype in sorted(conc_df["edge_type"].unique()):
            sub = conc_df[conc_df["edge_type"] == etype].sort_values("day")
            plt.plot(sub["day"], sub[metric], label=str(etype))
        plt.xlabel("Day")
        plt.ylabel(metric)
        title = "Risk mass inequality (Gini) over time" if metric == "risk_mass_gini" else "Risk mass top-1% share over time"
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"fig_{metric}_by_type_over_time.png", dpi=200)
        plt.close()


def lorenz_curve(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    x = np.sort(np.maximum(x, 0.0))
    s = x.sum()
    if s == 0:
        n = x.size
        return np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1)
    cum = np.cumsum(x)
    cum = np.insert(cum, 0, 0.0)
    cum_share = cum / cum[-1]
    pop_share = np.linspace(0, 1, cum_share.size)
    return pop_share, cum_share


def lorenz_on_grid(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Compute Lorenz curve values on a fixed grid of population shares.
    Returns L(grid) with same shape as grid.
    """
    px, py = lorenz_curve(x)
    # px is monotone [0..1], interpolate py at grid
    return np.interp(grid, px, py)


def plot_lorenz_selected_days(edges_full_dir: Path, chosen_days: List[int], outdir: Path) -> None:
    """
    EXACT Lorenz curves for risk_mass_active on selected days.
    """
    _ensure_dir(outdir)

    for day in chosen_days:
        p = edges_full_dir / f"edges_full_day{day:03d}.parquet"
        if not p.exists():
            continue

        df = pd.read_parquet(p)
        if df.empty or "risk_mass_active" not in df.columns:
            continue

        plt.figure()
        for etype in sorted(df["edge_type"].unique()):
            sub = df[df["edge_type"] == etype]
            x = sub["risk_mass_active"].to_numpy(dtype=float)
            px, py = lorenz_curve(x)
            plt.plot(px, py, label=str(etype))

        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Fraction of edges")
        plt.ylabel("Fraction of total risk_mass")
        plt.title(f"Lorenz curves (risk_mass) – Day {day}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"fig_lorenz_riskmass_day{day:03d}.png", dpi=200)
        plt.close()


def plot_lorenz_band_over_time(edges_full_dir: Path, outdir: Path, band: Tuple[float, float] = (0.25, 0.75), grid_n: int = 101) -> None:
    """
    Aggregated Lorenz: compute Lorenz curve for EVERY day, then plot:
      - median L(x)
      - band quantiles (default IQR 25–75%)

    Produces:
      fig_lorenz_band__ALL_TYPES.png  (overall across all edges)
      fig_lorenz_band__<edge_type>.png (per type)
    """
    _ensure_dir(outdir)
    days = _list_days_from_edges_full(edges_full_dir)
    if not days:
        return

    grid = np.linspace(0, 1, grid_n)

    # We collect curves per type and overall
    curves_by_type: Dict[str, List[np.ndarray]] = {}
    curves_all: List[np.ndarray] = []

    for day in days:
        p = edges_full_dir / f"edges_full_day{day:03d}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty or "risk_mass_active" not in df.columns:
            continue

        # overall
        x_all = df["risk_mass_active"].to_numpy(dtype=float)
        curves_all.append(lorenz_on_grid(x_all, grid))

        # per type
        for etype in df["edge_type"].unique():
            et = str(etype)
            sub = df[df["edge_type"] == etype]
            x = sub["risk_mass_active"].to_numpy(dtype=float)
            curves_by_type.setdefault(et, []).append(lorenz_on_grid(x, grid))

    def _plot_band(curves: List[np.ndarray], title: str, fname: str) -> None:
        if not curves:
            return
        A = np.stack(curves, axis=0)  # (T, grid_n)
        lo = np.quantile(A, band[0], axis=0)
        med = np.quantile(A, 0.5, axis=0)
        hi = np.quantile(A, band[1], axis=0)

        plt.figure()
        plt.plot(grid, med, label="median")
        plt.fill_between(grid, lo, hi, alpha=0.25, label=f"q{int(100*band[0])}–q{int(100*band[1])}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Fraction of edges")
        plt.ylabel("Fraction of total risk_mass")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()

    _plot_band(curves_all, "Aggregated Lorenz over time (all edges, all types)", "fig_lorenz_band__ALL_TYPES.png")

    for et, curves in sorted(curves_by_type.items()):
        _plot_band(curves, f"Aggregated Lorenz over time – {et}", f"fig_lorenz_band__{et}.png")


def topq_share(x: np.ndarray, q: float) -> float:
    """
    Share of total mass contained in top q fraction of items.
    q in (0,1], e.g. 0.01 = top 1%.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    total = x.sum()
    if total <= 0:
        return 0.0
    k = max(1, int(np.ceil(q * x.size)))
    xs = np.sort(x)[::-1]
    return float(xs[:k].sum() / total)


def plot_topq_share_over_time(edges_full_dir: Path, outdir: Path, qs: List[float] = None) -> None:
    """
    Aggregated concentration time series:
      S_q(t) = mass share in top q fraction of edges
    for q in {0.1%, 1%, 5%, 10%} by default.

    Produces:
      fig_topq_share_over_time__ALL_TYPES.png
      fig_topq_share_over_time__<edge_type>.png
    """
    _ensure_dir(outdir)
    if qs is None:
        qs = [0.001, 0.01, 0.05, 0.10]

    days = _list_days_from_edges_full(edges_full_dir)
    if not days:
        return

    # Prepare containers
    rows_all = []
    rows_by_type: Dict[str, List[Dict[str, float]]] = {}

    for day in days:
        p = edges_full_dir / f"edges_full_day{day:03d}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty or "risk_mass_active" not in df.columns:
            continue

        x_all = df["risk_mass_active"].to_numpy(dtype=float)
        r_all: Dict[str, float] = {"day": float(day)}
        for q in qs:
            r_all[f"top{100*q:.1f}%".replace(".0", "")] = topq_share(x_all, q)
        rows_all.append(r_all)

        for etype in df["edge_type"].unique():
            et = str(etype)
            sub = df[df["edge_type"] == etype]
            x = sub["risk_mass_active"].to_numpy(dtype=float)
            r: Dict[str, float] = {"day": float(day)}
            for q in qs:
                r[f"top{100*q:.1f}%".replace(".0", "")] = topq_share(x, q)
            rows_by_type.setdefault(et, []).append(r)

    def _plot(df: pd.DataFrame, title: str, fname: str) -> None:
        if df.empty:
            return
        df = df.sort_values("day")
        plt.figure()
        for c in [c for c in df.columns if c != "day"]:
            plt.plot(df["day"], df[c], label=c)
        plt.xlabel("Day")
        plt.ylabel("Share of total risk_mass")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=200)
        plt.close()

    _plot(pd.DataFrame(rows_all), "Top-q share of risk mass over time (all edges, all types)", "fig_topq_share_over_time__ALL_TYPES.png")

    for et, rows in sorted(rows_by_type.items()):
        _plot(pd.DataFrame(rows), f"Top-q share of risk mass over time – {et}", f"fig_topq_share_over_time__{et}.png")


# -----------------------------
# 2) Hotspot persistence plots
# -----------------------------

def plot_hotspot_persistence(hotspot_df: pd.DataFrame, outdir: Path) -> None:
    if hotspot_df.empty:
        return
    _ensure_dir(outdir)

    plt.figure()
    plt.hist(hotspot_df["days_in_topK"].to_numpy(dtype=int), bins=50)
    plt.xlabel("Days in top-K (risk edges)")
    plt.ylabel("Number of edges")
    plt.title("Hotspot persistence: days appearing in top-K")
    plt.tight_layout()
    plt.savefig(outdir / "fig_hotspot_days_in_topK_hist.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(hotspot_df["max_consecutive_days"].to_numpy(dtype=int), bins=50)
    plt.xlabel("Max consecutive days in top-K")
    plt.ylabel("Number of edges")
    plt.title("Hotspot persistence: max consecutive streak length")
    plt.tight_layout()
    plt.savefig(outdir / "fig_hotspot_max_streak_hist.png", dpi=200)
    plt.close()

    top = hotspot_df.sort_values(["days_in_topK", "max_consecutive_days"], ascending=[False, False]).head(50)
    top.to_csv(outdir / "top_hotspots.csv", index=False)


# -----------------------------
# 3) Coupling / bridging plots
# -----------------------------

def pivot_coupling(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    types = sorted(set(df["typeA"]).union(set(df["typeB"])))
    idx = {t: i for i, t in enumerate(types)}
    M = np.zeros((len(types), len(types)), dtype=float)
    for _, r in df.iterrows():
        i = idx[str(r["typeA"])]
        j = idx[str(r["typeB"])]
        M[i, j] = float(r["count"])
    return types, M


def plot_coupling_heatmap(coupling_path: Path, title: str, outpath: Path) -> None:
    if not coupling_path.exists():
        return
    df = pd.read_parquet(coupling_path)
    if df.empty:
        return

    types, M = pivot_coupling(df)

    plt.figure()
    plt.imshow(M)
    plt.xticks(range(len(types)), types, rotation=45, ha="right")
    plt.yticks(range(len(types)), types)
    plt.title(title)
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_coupling_strength_over_time(coupling_dir: Path, chosen_days: List[int], outdir: Path) -> None:
    _ensure_dir(outdir)

    def strength_for_file(p: Path) -> float:
        df = pd.read_parquet(p)
        _, M = pivot_coupling(df)
        diag = np.trace(M)
        off = M.sum() - diag
        return float(off / diag) if diag > 0 else 0.0

    all_files = sorted(coupling_dir.glob("coupling_all_day*.parquet"))
    inf_files = sorted(coupling_dir.glob("coupling_infected_day*.parquet"))
    if not all_files:
        return

    def day_from_name(name: str) -> int:
        return int(name.split("day")[1].split(".")[0])

    days = [day_from_name(p.name) for p in all_files]
    strength_all = [strength_for_file(p) for p in all_files]

    inf_map = {day_from_name(p.name): p for p in inf_files}
    strength_inf = [strength_for_file(inf_map[d]) if d in inf_map else 0.0 for d in days]

    plt.figure()
    plt.plot(days, strength_all, label="All present")
    plt.plot(days, strength_inf, label="Infected-only")
    plt.xlabel("Day")
    plt.ylabel("Off-diagonal / diagonal coupling")
    plt.title("Cross-context coupling strength over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_coupling_strength_over_time.png", dpi=200)
    plt.close()

    for d in chosen_days:
        plot_coupling_heatmap(
            coupling_dir / f"coupling_all_day{d:03d}.parquet",
            title=f"Coupling (all present) – Day {d}",
            outpath=outdir / f"fig_coupling_all_day{d:03d}.png",
        )
        plot_coupling_heatmap(
            coupling_dir / f"coupling_infected_day{d:03d}.parquet",
            title=f"Coupling (infected only) – Day {d}",
            outpath=outdir / f"fig_coupling_infected_day{d:03d}.png",
        )


# -----------------------------
# 4) Hubness & superspreader plots
# -----------------------------

def plot_hubness_over_time(hubs_dir: Path, chosen_days: List[int], outdir: Path) -> None:
    _ensure_dir(outdir)

    hub_files = sorted(hubs_dir.glob("hubs_top_day*.parquet"))
    if not hub_files:
        return

    def day_from_name(name: str) -> int:
        return int(name.split("day")[1].split(".")[0])

    # tail share over time
    days = []
    tail = []
    for p in hub_files:
        df = pd.read_parquet(p)
        if df.empty or "degree_active" not in df.columns:
            continue
        d = day_from_name(p.name)
        deg = df["degree_active"].to_numpy(dtype=float)
        total = deg.sum()
        top100 = np.sort(deg)[::-1][:100].sum() if deg.size >= 100 else deg.sum()
        days.append(d)
        tail.append(float(top100 / total) if total > 0 else 0.0)

    if days:
        plt.figure()
        plt.plot(days, tail)
        plt.xlabel("Day")
        plt.ylabel("Top-100 share of degree (within top-hubs file)")
        plt.title("Hubness concentration over time (degree tail share)")
        plt.tight_layout()
        plt.savefig(outdir / "fig_hub_degree_tail_share_over_time.png", dpi=200)
        plt.close()

    for d in chosen_days:
        p = hubs_dir / f"hubs_top_day{d:03d}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        deg = df["degree_active"].to_numpy(dtype=int)
        plt.figure()
        plt.hist(deg, bins=30)
        plt.xlabel("Active degree (top hubs only)")
        plt.ylabel("Count")
        plt.title(f"Hub degree distribution (top hubs) – Day {d}")
        plt.tight_layout()
        plt.savefig(outdir / f"fig_hub_degree_hist_day{d:03d}.png", dpi=200)
        plt.close()


def plot_superspreader_over_time(super_dir: Path, chosen_days: List[int], outdir: Path) -> None:
    _ensure_dir(outdir)

    files = sorted(super_dir.glob("superspreaders_top_day*.parquet"))
    if not files:
        return

    def day_from_name(name: str) -> int:
        return int(name.split("day")[1].split(".")[0])

    days, tail = [], []
    for p in files:
        df = pd.read_parquet(p)
        if df.empty or "outward_pressure" not in df.columns:
            continue
        d = day_from_name(p.name)
        x = df["outward_pressure"].to_numpy(dtype=float)
        total = x.sum()
        top50 = np.sort(x)[::-1][:50].sum() if x.size >= 50 else x.sum()
        days.append(d)
        tail.append(float(top50 / total) if total > 0 else 0.0)

    if days:
        plt.figure()
        plt.plot(days, tail)
        plt.xlabel("Day")
        plt.ylabel("Top-50 share of outward_pressure (within top file)")
        plt.title("Superspreader potential concentration over time (tail share)")
        plt.tight_layout()
        plt.savefig(outdir / "fig_superspreader_tail_share_over_time.png", dpi=200)
        plt.close()

    for d in chosen_days:
        p = super_dir / f"superspreaders_top_day{d:03d}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        x = df["outward_pressure"].to_numpy(dtype=float)
        plt.figure()
        plt.hist(x, bins=30)
        plt.xlabel("Outward pressure (top infected only)")
        plt.ylabel("Count")
        plt.title(f"Superspreader potential (outward_pressure) – Day {d}")
        plt.tight_layout()
        plt.savefig(outdir / f"fig_superspreader_hist_day{d:03d}.png", dpi=200)
        plt.close()


# -----------------------------
# Orchestration
# -----------------------------

def make_all_plots(results_dir: Path) -> None:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    plots_dir = results_dir / "plots"
    _ensure_dir(plots_dir)

    summary = _read_parquet_or_empty(results_dir / "daily_summary.parquet")
    conc = _read_parquet_or_empty(results_dir / "daily_edge_concentration.parquet")
    hotspot = _read_parquet_or_empty(results_dir / "hotspot_persistence.parquet")

    chosen_days = _choose_days(summary)

    edges_full_dir = results_dir / "edges_full"

    # Concentration (time series you already have)
    plot_concentration_over_time(conc, plots_dir)

    # NEW: aggregated Lorenz band (much more informative than snapshots)
    plot_lorenz_band_over_time(edges_full_dir, plots_dir, band=(0.25, 0.75), grid_n=101)

    # NEW: top-q shares over time (very interpretable)
    plot_topq_share_over_time(edges_full_dir, plots_dir, qs=[0.001, 0.01, 0.05, 0.10])

    # Keep snapshots too (nice appendix / sanity checks)
    plot_lorenz_selected_days(edges_full_dir, chosen_days, plots_dir)

    # Heavy-tail distribution plots (PDF + CCDF) using FULL edges (selected days)
    plot_riskmass_distributions_selected_days(edges_full_dir, chosen_days, plots_dir)

    # Hotspots
    plot_hotspot_persistence(hotspot, plots_dir)

    # Coupling
    plot_coupling_strength_over_time(results_dir / "coupling", chosen_days, plots_dir)

    # Hubness
    plot_hubness_over_time(results_dir / "hubs_top", chosen_days, plots_dir)

    # Superspreaders
    plot_superspreader_over_time(results_dir / "superspreaders_top", chosen_days, plots_dir)

    # Save chosen days
    (plots_dir / "chosen_days.txt").write_text(
        "Chosen days for detailed plots: " + ", ".join(map(str, chosen_days)) + "\n",
        encoding="utf-8",
    )

    print(f"Done. Plots written to: {plots_dir}")


def main_for_run_all(results_dir: Path) -> None:
    make_all_plots(results_dir)


def main() -> None:
    if len(sys.argv) >= 2:
        results_dir = Path(sys.argv[1]).resolve()
    else:
        results_dir = Path("./results_work").resolve()
    make_all_plots(results_dir)


if __name__ == "__main__":
    main()