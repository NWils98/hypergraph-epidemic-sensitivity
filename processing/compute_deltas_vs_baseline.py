#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cohen_d(mean_a: float, std_a: float, n_a: int, mean_b: float, std_b: float, n_b: int) -> float:
    # pooled SD
    if any(pd.isna(x) for x in [mean_a, std_a, mean_b, std_b]) or n_a < 2 or n_b < 2:
        return float("nan")
    denom = (n_a + n_b - 2)
    if denom <= 0:
        return float("nan")
    sp2 = ((n_a - 1) * (std_a ** 2) + (n_b - 1) * (std_b ** 2)) / denom
    if not np.isfinite(sp2) or sp2 <= 0:
        return float("nan")
    return float((mean_a - mean_b) / np.sqrt(sp2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seedstats-long",
        type=str,
        default="./derived_structural_sa/seed_stats/run_level__seedstats_long.parquet",
        help="Seed stats long table with columns pop, metric, mean, std, n_seeds (optionally se, ci95).",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default="pop_belgium600k_c500_teachers_censushh",
        help="Baseline pop name.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="./derived_structural_sa/deltas",
        help="Output directory.",
    )
    ap.add_argument(
        "--include-baseline-row",
        action="store_true",
        help="If set, keep baseline-vs-baseline rows (delta=0). Default skips baseline.",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    seedstats_path = (base / args.seedstats_long).resolve()
    out_dir = (base / args.out_dir).resolve()
    safe_mkdir(out_dir)

    df = pd.read_parquet(seedstats_path) if seedstats_path.suffix.lower() == ".parquet" else pd.read_csv(seedstats_path)

    required = {"pop", "metric", "mean", "std", "n_seeds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in seedstats long: {missing}")

    # Coerce
    df = df.copy()
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["std"] = pd.to_numeric(df["std"], errors="coerce")
    df["n_seeds"] = pd.to_numeric(df["n_seeds"], errors="coerce").fillna(0).astype(int)

    # Optional uncertainty from seedstats
    has_se = "se" in df.columns
    if has_se:
        df["se"] = pd.to_numeric(df["se"], errors="coerce")

    # Enforce uniqueness (pop, metric) — duplicates indicate an upstream bug
    dup = df.duplicated(subset=["pop", "metric"])
    if dup.any():
        example = df.loc[dup, ["pop", "metric"]].head(10)
        raise ValueError(
            "Seedstats has duplicate (pop, metric) rows. "
            "This indicates upstream duplication (e.g., appended results twice). "
            f"Examples:\n{example.to_string(index=False)}"
        )

    baseline = args.baseline
    base_df = df[df["pop"] == baseline].set_index("metric")
    if base_df.empty:
        raise ValueError(f"Baseline '{baseline}' not found in seedstats table.")

    rows = []
    for _, r in df.iterrows():
        pop = str(r["pop"])
        metric = str(r["metric"])

        if (not args.include_baseline_row) and pop == baseline:
            continue

        if metric not in base_df.index:
            continue

        b = base_df.loc[metric]

        mean_var = float(r["mean"]) if pd.notna(r["mean"]) else float("nan")
        std_var = float(r["std"]) if pd.notna(r["std"]) else float("nan")
        n_var = int(r["n_seeds"])

        mean_base = float(b["mean"]) if pd.notna(b["mean"]) else float("nan")
        std_base = float(b["std"]) if pd.notna(b["std"]) else float("nan")
        n_base = int(b["n_seeds"])

        delta = mean_var - mean_base if (pd.notna(mean_var) and pd.notna(mean_base)) else float("nan")

        # Percent change (more interpretable than delta/std_base)
        pct_change = (
            delta / abs(mean_base)
            if (pd.notna(delta) and pd.notna(mean_base) and mean_base != 0)
            else float("nan")
        )

        d = cohen_d(mean_var, std_var, n_var, mean_base, std_base, n_base)

        # Delta uncertainty if seedstats provided SE
        if has_se:
            se_var = float(r["se"]) if pd.notna(r["se"]) else float("nan")
            se_base = float(b["se"]) if pd.notna(b["se"]) else float("nan")
            delta_se = float(np.sqrt(se_var ** 2 + se_base ** 2)) if (np.isfinite(se_var) and np.isfinite(se_base)) else float("nan")
            delta_ci95 = 1.96 * delta_se if np.isfinite(delta_se) else float("nan")
        else:
            delta_se = float("nan")
            delta_ci95 = float("nan")

        rows.append(
            {
                "pop": pop,
                "metric": metric,
                "mean_variant": mean_var,
                "std_variant": std_var,
                "n_variant": n_var,
                "mean_baseline": mean_base,
                "std_baseline": std_base,
                "n_baseline": n_base,
                "delta": delta,
                "pct_change": pct_change,
                "cohen_d": d,
                "delta_se": delta_se,
                "delta_ci95": delta_ci95,
            }
        )

    out = pd.DataFrame(rows).sort_values(["metric", "pop"]).reset_index(drop=True)
    out.to_parquet(out_dir / "delta_run_level.parquet", index=False)
    out.to_csv(out_dir / "delta_run_level.csv", index=False)

    print(f"OK. Wrote deltas to:\n  {out_dir / 'delta_run_level.parquet'}")


if __name__ == "__main__":
    main()