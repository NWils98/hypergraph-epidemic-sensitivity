#!/usr/bin/env python3
"""
plots.py

Plots for advanced metrics:
- Entropy over time
- KL/JS/Wasserstein over time
- Coupling spectral metrics over time (all + infected)
- Hypercore sample stats over time (per type)
- Motif sample stats over time (per type)

Writes into: results_dir/plots_advanced/

Run:
  python plots.py <results_dir>
Default: ./results_work
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()

def plot_lines(df: pd.DataFrame, xcol: str, ycols: List[str], title: str, outpath: Path) -> None:
    if df.empty:
        return
    plt.figure()
    for c in ycols:
        if c in df.columns:
            plt.plot(df[xcol], df[c], label=c)
    plt.xlabel(xcol)
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_by_type_long(df: pd.DataFrame, value_col: str, title: str, outpath: Path) -> None:
    if df.empty:
        return
    plt.figure()
    for etype in sorted(df["edge_type"].unique()):
        sub = df[df["edge_type"] == etype].sort_values("day")
        if value_col in sub.columns:
            plt.plot(sub["day"], sub[value_col], label=etype)
    plt.xlabel("day")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def make_all_plots(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    outdir = results_dir / "plots_advanced"
    _ensure_dir(outdir)

    adv = _read(results_dir / "advanced_daily.parquet")
    core = _read(results_dir / "advanced_hypercore_sample.parquet")
    motifs = _read(results_dir / "advanced_motifs_sample.parquet")

    # ---- Entropy plots
    ent_cols = [c for c in adv.columns if c.startswith("entropy_riskmass")]
    if ent_cols:
        plot_lines(
            adv, "day", ent_cols,
            "Entropy of risk_mass distributions (overall + per type)",
            outdir / "fig_entropy_riskmass_over_time.png"
        )

    # ---- Divergences over time (overall)
    div_cols = [c for c in adv.columns if c.startswith("kl_riskmass_all") or c.startswith("js_riskmass_all") or c.startswith("w1_riskmass_all")]
    if div_cols:
        plot_lines(
            adv, "day", div_cols,
            "Distribution shift vs baseline (risk_mass all edges): KL / JS / Wasserstein",
            outdir / "fig_divergence_riskmass_all_vs_baseline.png"
        )

    # ---- Divergences per type (too many lines if all; group by metric)
    for prefix, fname, title in [
        ("kl_riskmass__", "fig_kl_by_type_vs_baseline.png", "KL divergence vs baseline by type"),
        ("js_riskmass__", "fig_js_by_type_vs_baseline.png", "JS divergence vs baseline by type"),
        ("w1_riskmass__", "fig_w1_by_type_vs_baseline.png", "Wasserstein-1 vs baseline by type"),
    ]:
        cols = [c for c in adv.columns if c.startswith(prefix) and "__vs_day" in c and "__all" not in c]
        # keep it readable: only plot a subset of types if many
        if cols:
            plot_lines(adv, "day", cols, title, outdir / fname)

    # ---- Coupling spectral metrics
    coup_all_cols = [c for c in adv.columns if c.startswith("coupling_all__")]
    if coup_all_cols:
        plot_lines(
            adv, "day", coup_all_cols,
            "Coupling spectral metrics (all present)",
            outdir / "fig_coupling_spectral_all_over_time.png"
        )

    coup_inf_cols = [c for c in adv.columns if c.startswith("coupling_infected__")]
    if coup_inf_cols:
        plot_lines(
            adv, "day", coup_inf_cols,
            "Coupling spectral metrics (infected-only)",
            outdir / "fig_coupling_spectral_infected_over_time.png"
        )

    # ---- Hypercore sample plots
    if not core.empty:
        plot_by_type_long(core, "core_mean", "Sampled k-core mean over time (projection)", outdir / "fig_core_mean_by_type.png")
        plot_by_type_long(core, "core_p95", "Sampled k-core p95 over time (projection)", outdir / "fig_core_p95_by_type.png")
        plot_by_type_long(core, "core_max", "Sampled k-core max over time (projection)", outdir / "fig_core_max_by_type.png")

    # ---- Motif sample plots
    if not motifs.empty:
        plot_by_type_long(motifs, "clustering_mean", "Sampled clustering over time (projection)", outdir / "fig_clustering_by_type.png")
        plot_by_type_long(motifs, "triangles_mean", "Sampled triangles mean over time (projection)", outdir / "fig_triangles_mean_by_type.png")
        plot_by_type_long(motifs, "wedge_mean", "Sampled wedges mean over time (projection)", outdir / "fig_wedges_mean_by_type.png")

    print(f"Advanced plots written to: {outdir}")


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