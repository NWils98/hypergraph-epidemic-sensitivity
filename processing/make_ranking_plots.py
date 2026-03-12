from __future__ import annotations

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DEFAULT_BASE_DIR = "./derived_structural_sa"

# Updated to match the corrected delta script output
DELTA_FILE = "deltas/delta_run_level.parquet"

# Metrics to treat as outcomes (your run_outcomes / run_level outcomes)
OUTCOME_METRICS = [
    "peak_infectious",
    "peak_day_infectious",
    "auc_infectious",
    "early_growth_log1p_infectious_slope_d1_28",
    "final_infected_max",
]

# How many variants to show per ranking plot
TOPK = 10

# Effect-size columns to plot (must exist in delta file)
EFFECT_COLUMNS = [
    ("delta", "Δ vs baseline (mean difference)"),
    ("pct_change", "Relative change vs baseline"),
    ("cohen_d", "Cohen's d (pooled)"),
]

# Prefer these patterns when selecting “key structural metrics”
STRUCTURE_PREFER_PATTERNS = [
    r"entropy_riskmass",
    r"coupling_",
    r"edge_conc",
    r"core__",
    r"motif__",
    r"blame__",
    r"daily_summary__infected_frac",
    r"daily_summary__mean_ies_noninfected",
    r"daily_summary__p95_ies_noninfected",
    # also accept the "daily__" prefix from newer phase summaries
    r"daily__",
]


# ============================================================
# Helpers
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clean_metric_name(m: str) -> str:
    # Make long feature names more readable in plot titles / labels
    m = m.replace("__", "·")
    m = m.replace("_", " ")
    return m


def _best_strength_col(df: pd.DataFrame) -> str:
    """
    Choose a sensible strength column for selecting metrics.
    Prefer cohen_d if present and informative, otherwise fallback to delta.
    """
    if "cohen_d" in df.columns and df["cohen_d"].notna().any():
        return "cohen_d"
    return "delta"


def select_structural_metrics(delta_df: pd.DataFrame, max_n: int = 30) -> list[str]:
    """
    Choose a manageable set of structural metrics for plotting by:
      1) excluding OUTCOME_METRICS
      2) keeping metrics that match preferred patterns
      3) taking those with strongest |effect| (overall, across variants)
    """
    if "metric" not in delta_df.columns:
        return []

    df = delta_df[~delta_df["metric"].isin(OUTCOME_METRICS)].copy()
    if df.empty:
        return []

    strength_col = _best_strength_col(df)
    df[strength_col] = pd.to_numeric(df[strength_col], errors="coerce")

    # Keep only metrics matching preferred patterns (if any match)
    mask = np.zeros(len(df), dtype=bool)
    for pat in STRUCTURE_PREFER_PATTERNS:
        mask |= df["metric"].astype(str).str.contains(pat, regex=True, na=False)

    df_pref = df[mask].copy()
    if df_pref.empty:
        df_pref = df

    df_pref["abs_strength"] = df_pref[strength_col].abs()

    metric_rank = (
        df_pref.groupby("metric", as_index=False)["abs_strength"]
        .max()
        .sort_values("abs_strength", ascending=False)
    )
    return metric_rank["metric"].head(max_n).tolist()


def plot_ranking(
    df: pd.DataFrame,
    metric: str,
    value_col: str,
    title: str,
    outpath: Path,
    topk: int = TOPK,
) -> None:
    """
    Ranks variants by the specified effect column for a fixed metric.
    Shows top positive and top negative (direction matters).
    """
    if "metric" not in df.columns or "pop" not in df.columns:
        return

    sub = df[df["metric"] == metric].copy()
    if value_col not in sub.columns:
        return

    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[value_col])

    if sub.empty:
        return

    sub_pos = sub.sort_values(value_col, ascending=False).head(topk)
    sub_neg = sub.sort_values(value_col, ascending=True).head(topk)

    combined = pd.concat([sub_pos, sub_neg], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pop"], keep="first")
    combined = combined.sort_values(value_col, ascending=True)

    y = combined["pop"].tolist()
    x = combined[value_col].to_numpy(dtype=float)

    plt.figure(figsize=(10, max(4, 0.35 * len(combined))))
    plt.barh(y, x)
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_summary_heatmap(
    df: pd.DataFrame,
    metrics: list[str],
    value_col: str,
    title: str,
    outpath: Path,
    max_variants: int = 12,
) -> None:
    """
    Heatmap: variants (rows) × metrics (cols), with values = value_col.
    Variants chosen by overall max |value_col| across selected metrics.
    """
    if df.empty or "metric" not in df.columns or "pop" not in df.columns:
        return
    if value_col not in df.columns:
        return

    sub = df[df["metric"].isin(metrics)].copy()
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[value_col])

    if sub.empty:
        return

    v_rank = (
        sub.assign(abs_val=sub[value_col].abs())
        .groupby("pop", as_index=False)["abs_val"]
        .max()
        .sort_values("abs_val", ascending=False)
        .head(max_variants)
    )
    top_variants = v_rank["pop"].tolist()

    mat = sub[sub["pop"].isin(top_variants)].pivot_table(
        index="pop", columns="metric", values=value_col, aggfunc="mean"
    )

    if mat.empty:
        return

    col_rank = mat.abs().mean(axis=0).sort_values(ascending=False).index.tolist()
    mat = mat[col_rank]

    plt.figure(figsize=(max(8, 0.6 * len(mat.columns)), max(4, 0.45 * len(mat.index))))
    plt.imshow(mat.to_numpy(), aspect="auto")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(mat.columns)), [clean_metric_name(c) for c in mat.columns], rotation=60, ha="right")
    plt.yticks(range(len(mat.index)), mat.index.tolist())
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Make sensitivity ranking plots for outcomes and structure.")
    ap.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR, help="derived_structural_sa directory")
    ap.add_argument("--delta-file", type=str, default=DELTA_FILE, help="Path to delta parquet (relative to base-dir)")
    ap.add_argument("--topk", type=int, default=TOPK, help="TopK positive and negative variants to show")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    derived = (base / args.base_dir).resolve()
    delta_path = derived / args.delta_file

    if not delta_path.exists():
        raise FileNotFoundError(f"Missing: {delta_path}")

    df = pd.read_parquet(delta_path)

    # Basic validation
    need_cols = {"pop", "metric"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"Delta file missing required columns {need_cols - set(df.columns)}")

    out_dir = derived / "plots_rankings"
    ensure_dir(out_dir)

    # ------------------------------------------------------------
    # Outcome rankings
    # ------------------------------------------------------------
    outcome_dir = out_dir / "outcomes"
    ensure_dir(outcome_dir)

    for metric in OUTCOME_METRICS:
        for col, label in EFFECT_COLUMNS:
            outpath = outcome_dir / f"rank__{metric}__{col}.png"
            title = f"Outcome ranking: {metric} — {label}"
            plot_ranking(df, metric, col, title, outpath, topk=args.topk)

    # Outcome heatmap summary (prefer Cohen's d, fallback to delta)
    outcome_heat_col = "cohen_d" if "cohen_d" in df.columns else "delta"
    plot_summary_heatmap(
        df=df,
        metrics=OUTCOME_METRICS,
        value_col=outcome_heat_col,
        title=f"Outcome effect heatmap ({outcome_heat_col}) — top variants",
        outpath=outcome_dir / f"heatmap__outcomes__{outcome_heat_col}.png",
        max_variants=12,
    )

    # ------------------------------------------------------------
    # Structural rankings
    # ------------------------------------------------------------
    struct_dir = out_dir / "structure"
    ensure_dir(struct_dir)

    structural_metrics = select_structural_metrics(df, max_n=30)

    # For structure we typically care most about standardized effects; use cohen_d if available
    struct_effect_cols = []
    if "cohen_d" in df.columns:
        struct_effect_cols.append(("cohen_d", "Cohen's d (pooled)"))
    struct_effect_cols += [
        ("delta", "Δ vs baseline (mean difference)"),
    ]
    if "pct_change" in df.columns:
        struct_effect_cols.append(("pct_change", "Relative change vs baseline"))

    for metric in structural_metrics:
        for col, label in struct_effect_cols:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric)
            outpath = struct_dir / f"rank__{safe_name}__{col}.png"
            title = f"Structure ranking: {clean_metric_name(metric)} — {label}"
            plot_ranking(df, metric, col, title, outpath, topk=args.topk)

    # Heatmap summary for structure (use cohen_d if present else delta)
    if structural_metrics:
        struct_heat_col = "cohen_d" if "cohen_d" in df.columns else "delta"
        plot_summary_heatmap(
            df=df,
            metrics=structural_metrics[:12],  # keep readable
            value_col=struct_heat_col,
            title=f"Structural effect heatmap ({struct_heat_col}) — top variants",
            outpath=struct_dir / f"heatmap__structure__{struct_heat_col}.png",
            max_variants=12,
        )

    print("Done.")
    print("Wrote plots to:", out_dir)


if __name__ == "__main__":
    main()