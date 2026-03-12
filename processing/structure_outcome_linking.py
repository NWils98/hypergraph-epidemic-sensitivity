from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG (connected to NEW pipeline)
# ============================================================

DEFAULT_DERIVED_DIR = "./derived_structural_sa"

BASELINE_NAME = "pop_belgium600k_c500_teachers_censushh"

OUTCOME_METRICS = [
    "peak_infectious",
    "peak_day_infectious",
    "auc_infectious",
    "early_growth_log1p_infectious_slope_d1_28",
    "final_infected_max",
    "final_cases_sum",
]

DAILY_STRUCT_PATTERNS = [
    "infected_frac",
    "mean_ies_noninfected",
    "p95_ies_noninfected",
    "max_ies_noninfected",
    "risk_mass_gini",
    "risk_mass_top1pct_share",
    "entropy_riskmass",
    "coupling_",
    "core_",
    "clustering_mean",
    "wedge_mean",
    "triangles_mean",
    "outward_pressure",
    "mean_degree_active",
    "p95_degree_active",
]

DEFAULT_MAX_LAG = 21

MAX_PREDICTORS = 3
TOP_STRUCT_BY_CORR = 8
TOP_FEATURES_HEATMAP = 40
TOP_STRUCT_CORR_HEATMAP = 30


# ============================================================
# Small utilities
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return x * np.nan
    return (x - mu) / sd


def plot_heatmap(mat: pd.DataFrame, title: str, outpath: Path) -> None:
    """
    Heatmap that is robust to long axis labels.
    Uses bbox_inches='tight' to avoid tight_layout warnings.
    """
    if mat is None or mat.empty:
        return

    vals = mat.to_numpy()

    plt.figure(figsize=(max(10, 0.45 * mat.shape[1]), max(5, 0.45 * mat.shape[0])))
    plt.imshow(vals, aspect="auto")
    plt.colorbar(label="value")

    plt.xticks(range(mat.shape[1]), mat.columns.tolist(), rotation=70, ha="right")
    plt.yticks(range(mat.shape[0]), mat.index.tolist())
    plt.title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def ols_fit(X: np.ndarray, y: np.ndarray):
    """
    Simple OLS with intercept using numpy lstsq.
    Returns beta (including intercept), yhat, r2.
    """
    X_ = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X_, y, rcond=None)
    yhat = X_ @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, yhat, r2


def select_daily_struct_cols(daily_panel: pd.DataFrame) -> list[str]:
    cols = []
    for c in daily_panel.columns:
        if c == "day":
            continue
        lc = c.lower()
        # avoid accidentally using the target itself
        if lc == "infectious":
            continue
        if any(pat in lc for pat in DAILY_STRUCT_PATTERNS):
            cols.append(c)
    return sorted(set(cols))


def safe_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson correlation, but returns NaN if:
      - too few points
      - either vector has zero variance
    """
    if x.size != y.size:
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan

    xm = x[mask]
    ym = y[mask]

    if np.nanstd(xm) == 0 or np.nanstd(ym) == 0:
        return np.nan

    return float(np.corrcoef(xm, ym)[0, 1])


def safe_spearman(df2: pd.DataFrame, col_a: str, col_b: str) -> float:
    """
    Spearman correlation with NaN/constant protection.
    """
    pair = df2[[col_a, col_b]].dropna()
    if len(pair) < 6:
        return np.nan
    if pair[col_a].nunique() < 2 or pair[col_b].nunique() < 2:
        return np.nan
    return float(pair.corr(method="spearman").iloc[0, 1])


# ============================================================
# Iterating runs from derived outputs
# ============================================================

def iter_runs_from_derived(derived_root: Path):
    """
    Yields (pop, seed, daily_panel_path, timeseries_path)
    """
    dp_root = derived_root / "daily_panels"
    ts_root = derived_root / "timeseries"
    if not dp_root.exists() or not ts_root.exists():
        return

    for pop_dir in sorted(dp_root.iterdir()):
        if not pop_dir.is_dir():
            continue
        pop = pop_dir.name
        for seed_file in sorted(pop_dir.glob("*.parquet")):
            seed = seed_file.stem
            ts_path = ts_root / pop / f"{seed}.parquet"
            if ts_path.exists():
                yield pop, seed, seed_file, ts_path


# ============================================================
# 1) Run-level structure–outcome heatmap
# ============================================================

def run_level_heatmap(run_level: pd.DataFrame, outdir: Path) -> None:
    ensure_dir(outdir)

    # Outcomes are direct columns in run_level.parquet
    out_cols = [m for m in OUTCOME_METRICS if m in run_level.columns]
    if not out_cols:
        print("No outcome columns found in run_level.parquet.")
        return

    # Candidate structural features: numeric columns excluding identifiers + outcomes
    feature_cols = []
    for c in run_level.columns:
        if c in ("pop", "seed"):
            continue
        if c in out_cols:
            continue
        if run_level[c].dtype.kind in "if":
            feature_cols.append(c)

    sub = run_level[feature_cols + out_cols].copy()

    # Drop columns with too many NaNs
    keep = [c for c in sub.columns if sub[c].isna().mean() <= 0.7]
    sub = sub[keep]

    features_kept = [c for c in sub.columns if c not in out_cols]
    outcomes_kept = out_cols

    if not features_kept:
        print("No usable structural features for run-level heatmap (too many NaNs?).")
        return

    corr = pd.DataFrame(index=outcomes_kept, columns=features_kept, dtype=float)
    for o in outcomes_kept:
        for f in features_kept:
            corr.loc[o, f] = safe_spearman(sub, o, f)

    # Keep top features by max |corr|
    score = corr.abs().max(axis=0).sort_values(ascending=False)
    top_features = score.head(TOP_FEATURES_HEATMAP).index.tolist()
    corr_small = corr[top_features]

    plot_heatmap(
        corr_small.fillna(0.0),
        title="Run-level structure ↔ outcome (Spearman), top structural features",
        outpath=outdir / "heatmap__runlevel__spearman.png",
    )
    corr_small.to_csv(outdir / "heatmap__runlevel__spearman.csv")

    print("Run-level heatmap saved to:", outdir)


# ============================================================
# 2) Lag analysis using daily_panels + infectious time series
# ============================================================

def lag_corr(struct_series: np.ndarray, y_series: np.ndarray, max_lag: int):
    """
    Returns (best_lag, best_corr) maximizing absolute correlation
    of corr(struct_t, y_{t+lag}) for lag>=0.
    Skips degenerate cases (zero variance, too few points).
    """
    n = min(len(struct_series), len(y_series))
    struct_series = struct_series[:n]
    y_series = y_series[:n]

    if n < 20:
        return np.nan, np.nan
    if np.all(np.isnan(struct_series)) or np.all(np.isnan(y_series)):
        return np.nan, np.nan

    best_lag = np.nan
    best_corr = np.nan
    best_abs = -np.inf

    for lag in range(0, max_lag + 1):
        if lag >= n - 2:
            break

        x = struct_series[: n - lag]
        y = y_series[lag: n]

        c = safe_corr_1d(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if np.isnan(c):
            continue

        if abs(c) > best_abs:
            best_abs = abs(c)
            best_lag = lag
            best_corr = c

    if best_abs == -np.inf:
        return np.nan, np.nan
    return best_lag, best_corr


def do_lag_analysis(derived_root: Path, outdir: Path, max_lag: int) -> None:
    ensure_dir(outdir)

    records = []

    for pop, seed, dp_path, ts_path in iter_runs_from_derived(derived_root):
        panel = pd.read_parquet(dp_path)
        ts = pd.read_parquet(ts_path)

        if panel.empty or ts.empty:
            continue
        if "infectious" not in ts.columns:
            continue
        if "day" not in panel.columns or "day" not in ts.columns:
            continue

        ts = ts.sort_values("day")
        y = pd.to_numeric(ts["infectious"], errors="coerce").to_numpy(dtype=float)

        panel = panel.sort_values("day")
        struct_cols = select_daily_struct_cols(panel)
        if not struct_cols:
            continue

        for col in struct_cols:
            x = pd.to_numeric(panel[col], errors="coerce").to_numpy(dtype=float)
            lag, corr = lag_corr(x, y, max_lag=max_lag)
            records.append(
                {
                    "pop": pop,
                    "seed": seed,
                    "struct_col": col,
                    "best_lag_days": lag,
                    "best_corr": corr,
                    "abs_best_corr": abs(corr) if not pd.isna(corr) else np.nan,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        print("No lag records computed (missing daily_panels or infectious series).")
        return

    df.to_parquet(outdir / "lag_results__per_run.parquet", index=False)
    df.to_csv(outdir / "lag_results__per_run.csv", index=False)

    agg = (
        df.groupby(["pop", "struct_col"], as_index=False)
        .agg(
            mean_best_lag=("best_lag_days", "mean"),
            mean_best_corr=("best_corr", "mean"),
            mean_abs_best_corr=("abs_best_corr", "mean"),
            n_runs=("best_corr", "count"),
        )
    )

    agg.to_parquet(outdir / "lag_results__aggregated.parquet", index=False)
    agg.to_csv(outdir / "lag_results__aggregated.csv", index=False)

    rank_cols = (
        agg.groupby("struct_col", as_index=False)["mean_abs_best_corr"]
        .mean()
        .sort_values("mean_abs_best_corr", ascending=False)
        .head(20)["struct_col"]
        .tolist()
    )

    mat = agg[agg["struct_col"].isin(rank_cols)].pivot_table(
        index="pop", columns="struct_col", values="mean_abs_best_corr", aggfunc="mean"
    )
    plot_heatmap(
        mat.fillna(0.0),
        title=f"Lag linkage: mean best |corr| with Infectious (lags 0..{max_lag}), top daily structural signals",
        outpath=outdir / "heatmap__lag__mean_abs_best_corr.png",
    )

    mat_lag = agg[agg["struct_col"].isin(rank_cols)].pivot_table(
        index="pop", columns="struct_col", values="mean_best_lag", aggfunc="mean"
    )
    plot_heatmap(
        mat_lag.fillna(0.0),
        title=f"Lag linkage: mean best lag (days) where |corr| is maximal, top daily structural signals",
        outpath=outdir / "heatmap__lag__mean_best_lag.png",
    )

    print("Lag analysis saved to:", outdir)


# ============================================================
# 3) Δ-regression across variants (Δ structure -> Δ outcome)
# ============================================================

def do_delta_regression(derived_root: Path, outdir: Path) -> None:
    ensure_dir(outdir)

    # NEW location (connected to your corrected pipeline)
    delta_path = derived_root / "deltas" / "delta_run_level.parquet"
    if not delta_path.exists():
        print("Missing delta table:", delta_path)
        return

    d = pd.read_parquet(delta_path)

    if not {"pop", "metric", "delta"}.issubset(d.columns):
        print("Delta table missing required columns (pop, metric, delta).")
        return

    outcomes = [m for m in OUTCOME_METRICS if m in set(d["metric"].unique())]
    if not outcomes:
        print("No outcome metrics found in delta table.")
        return

    d_wide = d.pivot_table(index="pop", columns="metric", values="delta", aggfunc="mean")
    d_wide = d_wide.drop(index=BASELINE_NAME, errors="ignore")

    struct_metrics = [c for c in d_wide.columns if c not in outcomes]
    struct_metrics = [c for c in struct_metrics if d_wide[c].isna().mean() <= 0.5]
    if not struct_metrics:
        print("No usable structural delta metrics for regression.")
        return

    results = []

    # Best-model selection per outcome
    for out in outcomes:
        if out not in d_wide.columns:
            continue

        y = d_wide[out]
        ok = ~y.isna()
        y = y[ok]
        if len(y) < 6:
            continue

        Xcand = d_wide.loc[y.index, struct_metrics]

        corrs = {}
        for f in Xcand.columns:
            xs = Xcand[f]
            mask = ~xs.isna()
            if mask.sum() < 6:
                continue
            xs2 = xs[mask]
            y2 = y[mask]
            if xs2.nunique() < 2 or y2.nunique() < 2:
                continue
            corrs[f] = float(xs2.corr(y2, method="pearson"))

        if not corrs:
            continue

        top_struct = (
            pd.Series(corrs).abs().sort_values(ascending=False).head(TOP_STRUCT_BY_CORR).index.tolist()
        )

        best_model = None  # (r2, predictors, beta)

        for k in range(1, min(MAX_PREDICTORS, len(top_struct)) + 1):
            for preds in itertools.combinations(top_struct, k):
                X = Xcand[list(preds)]
                mask = ~X.isna().any(axis=1)
                if mask.sum() < 6:
                    continue

                Xz = X.loc[mask].apply(zscore, axis=0).to_numpy()
                yz = zscore(y.loc[mask]).to_numpy()

                # if zscore produced NaNs (zero-variance), skip
                if not np.isfinite(Xz).all() or not np.isfinite(yz).all():
                    continue

                beta, _, r2 = ols_fit(Xz, yz)
                if best_model is None or (not pd.isna(r2) and r2 > best_model[0]):
                    best_model = (float(r2), preds, beta)

        if best_model is None:
            continue

        r2, preds, beta = best_model
        rec = {
            "outcome": out,
            "n_variants_used": int((~Xcand[list(preds)].isna().any(axis=1) & ~y.isna()).sum()),
            "r2": float(r2),
            "predictors": " + ".join(preds),
            "intercept": float(beta[0]),
        }
        for i, p in enumerate(preds, start=1):
            rec[f"beta_{p}"] = float(beta[i])
        results.append(rec)

    res = pd.DataFrame(results).sort_values("r2", ascending=False)
    res.to_csv(outdir / "delta_regression__best_models.csv", index=False)

    # Correlation heatmap corr(Δstructure, Δoutcome)
    corr = pd.DataFrame(index=outcomes, columns=struct_metrics, dtype=float)
    for out in outcomes:
        y = d_wide[out]
        for f in struct_metrics:
            xs = d_wide[f]
            mask = ~xs.isna() & ~y.isna()
            if mask.sum() < 6:
                continue
            xs2 = xs[mask]
            y2 = y[mask]
            if xs2.nunique() < 2 or y2.nunique() < 2:
                continue
            corr.loc[out, f] = float(xs2.corr(y2, method="pearson"))

    score = corr.abs().mean(axis=0).sort_values(ascending=False).head(TOP_STRUCT_CORR_HEATMAP).index.tolist()
    corr_small = corr[score]

    plot_heatmap(
        corr_small.fillna(0.0),
        title="Δ-linking: corr(Δstructure, Δoutcome) across variants (Pearson)",
        outpath=outdir / "heatmap__delta__pearson.png",
    )
    corr_small.to_csv(outdir / "heatmap__delta__pearson.csv")

    print("Δ-regression outputs saved to:", outdir)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Structure–outcome linking: run-level heatmap + lag + Δ-regression.")
    ap.add_argument("--derived-dir", type=str, default=DEFAULT_DERIVED_DIR)
    ap.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    derived_root = (base / args.derived_dir).resolve()

    outdir = derived_root / "links"
    ensure_dir(outdir)

    # 0) load run_level table (NEW)
    run_level_path = derived_root / "run_level.parquet"
    if not run_level_path.exists():
        raise FileNotFoundError(f"Missing {run_level_path}. Run build_panels_and_summaries.py first.")

    df_runs = pd.read_parquet(run_level_path)
    df_runs.to_parquet(outdir / "run_level_table.parquet", index=False)
    df_runs.to_csv(outdir / "run_level_table.csv", index=False)

    # 1) run-level heatmap
    run_level_heatmap(df_runs, outdir / "runlevel_heatmap")

    # 2) lag analysis
    do_lag_analysis(derived_root, outdir / "lag", max_lag=args.max_lag)

    # 3) delta regression
    do_delta_regression(derived_root, outdir / "delta_regression")

    print("\nDONE. Outputs under:", outdir)


if __name__ == "__main__":
    main()