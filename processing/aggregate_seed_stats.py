#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input: {path}")


def infer_metric_cols(df: pd.DataFrame) -> List[str]:
    """
    Everything numeric (or coercible to numeric) except identifiers pop/seed.
    Mutates df: coerces columns when possible.
    """
    id_cols = {"pop", "seed"}
    cols: List[str] = []
    for c in df.columns:
        if c in id_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        else:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().any():
                df[c] = x
                cols.append(c)
    return cols


def deduplicate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    If there are duplicate (pop, seed) rows, aggregate numeric columns
    within each (pop, seed) by mean to avoid overweighting.
    """
    if not df.duplicated(subset=["pop", "seed"]).any():
        return df

    id_cols = ["pop", "seed"]
    other_cols = [c for c in df.columns if c not in id_cols]

    # Coerce potentially numeric columns
    for c in other_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Average within duplicates
    out = df.groupby(id_cols, as_index=False).mean(numeric_only=True)
    return out


def to_long(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    metric_cols = [c for c in metric_cols if c in df.columns and df[c].notna().any()]
    long = df.melt(
        id_vars=["pop", "seed"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long


def seedstats(long: pd.DataFrame, use_t_ci: bool = True) -> pd.DataFrame:
    """
    Stable aggregation without groupby.apply:
      mean, std (ddof=1), n_seeds, se, ci95 (half-width)
    Uses finite values only.
    """
    # Drop non-finite values for aggregation
    x = long.copy()
    x["value"] = pd.to_numeric(x["value"], errors="coerce")
    x = x[np.isfinite(x["value"].to_numpy(dtype=float))].copy()

    if x.empty:
        return pd.DataFrame(columns=["pop", "metric", "mean", "std", "n_seeds", "se", "ci95"])

    g = x.groupby(["pop", "metric"])["value"]

    out = g.agg(
        mean="mean",
        std=lambda s: float(np.std(s.to_numpy(dtype=float), ddof=1)) if len(s) >= 2 else float("nan"),
        n_seeds="count",
    ).reset_index()

    out["n_seeds"] = out["n_seeds"].astype(int)

    # Standard error
    out["se"] = out["std"] / np.sqrt(out["n_seeds"].where(out["n_seeds"] > 0, np.nan))

    # 95% CI half-width
    if use_t_ci:
        # t critical values for small n. With 5 seeds: df=4, t≈2.776
        # We'll map df = n-1 up to 30; for larger n default to 1.96.
        t_crit = {
            1: np.nan,
            2: 12.706,
            3: 4.303,
            4: 3.182,
            5: 2.776,
            6: 2.571,
            7: 2.447,
            8: 2.365,
            9: 2.306,
            10: 2.262,
            11: 2.228,
            12: 2.201,
            13: 2.179,
            14: 2.160,
            15: 2.145,
            16: 2.131,
            17: 2.120,
            18: 2.110,
            19: 2.101,
            20: 2.093,
            21: 2.086,
            22: 2.080,
            23: 2.074,
            24: 2.069,
            25: 2.064,
            26: 2.060,
            27: 2.056,
            28: 2.052,
            29: 2.048,
            30: 2.045,
        }

        def _crit(n: int) -> float:
            if n <= 1:
                return float("nan")
            df_ = n - 1
            if df_ in t_crit:
                return float(t_crit[df_])
            return 1.96

        out["ci95"] = [(_crit(int(n)) * float(se)) if np.isfinite(se) else float("nan")
                       for n, se in zip(out["n_seeds"], out["se"])]
    else:
        out["ci95"] = 1.96 * out["se"]

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-level",
        type=str,
        default="./derived_structural_sa/run_level.parquet",
        help="Run-level table with columns pop, seed, metrics (parquet or csv).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="./derived_structural_sa/seed_stats",
        help="Output directory.",
    )
    ap.add_argument(
        "--ci",
        type=str,
        default="t",
        choices=["t", "normal"],
        help="CI type: 't' uses t critical values (better for 5 seeds), 'normal' uses 1.96.",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    run_level_path = (base / args.run_level).resolve()
    out_dir = (base / args.out_dir).resolve()
    safe_mkdir(out_dir)

    df = read_table(run_level_path)

    if "pop" not in df.columns or "seed" not in df.columns:
        raise ValueError("run-level must contain columns: pop, seed")

    df = deduplicate_runs(df)

    metric_cols = infer_metric_cols(df)
    long = to_long(df, metric_cols)

    seedstats_df = seedstats(long, use_t_ci=(args.ci == "t"))

    # Write outputs
    seedstats_df.to_parquet(out_dir / "run_level__seedstats_long.parquet", index=False)
    seedstats_df.to_csv(out_dir / "run_level__seedstats_long.csv", index=False)

    # Convenience wide tables
    mean_wide = seedstats_df.pivot(index="pop", columns="metric", values="mean").reset_index()
    mean_wide.to_parquet(out_dir / "run_level__means_wide.parquet", index=False)
    mean_wide.to_csv(out_dir / "run_level__means_wide.csv", index=False)

    ci_wide = seedstats_df.pivot(index="pop", columns="metric", values="ci95").reset_index()
    ci_wide.to_parquet(out_dir / "run_level__ci95_wide.parquet", index=False)

    print(f"OK. Wrote seed stats to:\n  {out_dir}")


if __name__ == "__main__":
    main()