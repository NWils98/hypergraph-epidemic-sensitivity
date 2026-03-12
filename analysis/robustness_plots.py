#!/usr/bin/env python3

"""
robustness_plots.py

Produces two robustness figures:

1. relative_change_heatmap_reordered.png
   Relative change vs baseline population for peak infections

2. variance_decomposition_hierarchical.png
   Hierarchical variance decomposition:

       plan
         └ population
             └ seed

The variance bars are plotted as normalized shares, so they sum to 100%.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

RESULTS_DIR = Path("results")
OUTDIR = RESULTS_DIR / "robustness_core_v4"
OUTDIR.mkdir(parents=True, exist_ok=True)

PLANS = ["plan1", "plan2", "plan3", "plan4", "plan5"]

PLAN_DISPLAY = {
    "plan1": "No intervention",
    "plan2": "General intervention",
    "plan3": "Work-focused",
    "plan4": "Social focused",
    "plan5": "Broad NPI + TTI",
}

OUTCOMES = {
    "Peak infection": "peak_infectious",
    "Cumulative infections": "final_cases_sum",
    "Early growth rate": "early_growth_log1p_infectious_slope_d1_28",
}

KNOWN_BASELINE = "pop_belgium600k_c500_teachers_censushh"

POP_ORDER = [
    "pop_belgium600k_c500_teachers_censushh",
    "P1_hh_highvar_same_mean",
    "P2_hh_lowvar_same_mean",
    "P3_weekend_stable",
    "P4_weekend_mixed",
    "P5_weekend_rewired",
    "P6_age_younger",
    "P7_age_older",
    "P8_schools_local",
    "P9_schools_mixed",
    "P10_work_local",
    "P11_work_mixed",
]

FAMILY_ORDER = ["baseline", "household", "weekend", "age", "school", "work", "other"]

FIG_DPI = 220


# ============================================================
# HELPERS
# ============================================================

def savefig(path: Path) -> None:
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def try_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None


def infer_population_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "population",
        "pop",
        "population_variant",
        "variant",
        "pop_name",
        "population_name",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = c.lower()
        if "pop" in lc or "variant" in lc:
            return c
    return None


def infer_seed_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["seed", "rng_seed", "seed_id"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "seed" in c.lower():
            return c
    return None


def family_from_population(pop: str) -> str:
    p = str(pop).lower()
    if "belgium" in p or "censushh" in p or "baseline" in p:
        return "baseline"
    if "hh_" in p or "household" in p:
        return "household"
    if "weekend" in p:
        return "weekend"
    if "age_" in p:
        return "age"
    if "school" in p or "k12" in p:
        return "school"
    if "work_" in p:
        return "work"
    return "other"


def baseline_population_name(populations: List[str]) -> Optional[str]:
    if KNOWN_BASELINE in populations:
        return KNOWN_BASELINE
    for p in populations:
        lp = p.lower()
        if "belgium" in lp or "censushh" in lp or "baseline" in lp:
            return p
    return None


def population_sort_key(pop: str) -> Tuple[int, int, str]:
    fam = family_from_population(pop)
    fam_idx = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 999
    pop_idx = POP_ORDER.index(pop) if pop in POP_ORDER else 999
    return fam_idx, pop_idx, pop


def rounded_percentages_sum100(values: List[float]) -> List[float]:
    vals = np.array(values, dtype=float)
    vals = np.nan_to_num(vals, nan=0.0)
    vals = np.clip(vals, 0.0, None)

    total = vals.sum()
    if total <= 0:
        return [0.0] * len(vals)

    raw = 100.0 * vals / total
    rounded = np.round(raw, 2)

    diff = 100.0 - rounded.sum()
    idx = int(np.argmax(rounded))
    rounded[idx] += diff

    return rounded.tolist()


def annotate_segment(x_center: float, y_bottom: float, height: float, label_text: str) -> None:
    if not np.isfinite(height) or height <= 0:
        return

    if height >= 0.06:
        y = y_bottom + height / 2.0
        va = "center"
        color = "white"
    else:
        y = min(1.04, y_bottom + height + 0.012)
        va = "bottom"
        color = "black"

    plt.text(
        x_center,
        y,
        label_text,
        ha="center",
        va=va,
        fontsize=10,
        fontweight="bold",
        color=color,
        clip_on=False,
    )


# ============================================================
# LOAD + HARMONIZE
# ============================================================

def load_all_runs() -> pd.DataFrame:
    frames = []

    for plan in PLANS:
        path = RESULTS_DIR / plan / "derived_structural_sa" / "run_level.parquet"
        df = try_read_parquet(path)
        if df is None:
            print(f"[INFO] Skipping {plan}: no readable run_level.parquet")
            continue

        pop_col = infer_population_col(df)
        seed_col = infer_seed_col(df)

        if pop_col is None or seed_col is None:
            print(f"[WARN] Skipping {plan}: could not infer population/seed columns")
            continue

        sub = df.copy()
        sub["plan"] = plan
        sub["plan_display"] = sub["plan"].map(lambda p: PLAN_DISPLAY.get(p, p))
        sub["population"] = sub[pop_col].astype(str).str.strip()
        sub["seed"] = sub[seed_col]
        sub["family"] = sub["population"].map(family_from_population)

        frames.append(sub)

        print(
            f"[OK] {plan}: rows={len(sub)} "
            f"pops={sub['population'].nunique()} "
            f"seeds={sub['seed'].nunique()}"
        )

    if not frames:
        raise RuntimeError("No usable plan data found.")

    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(OUTDIR / "merged_run_level.parquet", index=False)
    merged.to_csv(OUTDIR / "merged_run_level.csv", index=False)

    return merged


# ============================================================
# PLOT 1 — HEATMAP
# ============================================================

def plot_relative_change_heatmap_reordered(df: pd.DataFrame, outcome: str = "peak_infectious") -> pd.DataFrame:
    rows = []

    for plan, sub in df.groupby("plan"):
        pops = sub["population"].unique().tolist()
        base = baseline_population_name(pops)
        if base is None:
            continue

        base_mean = sub.loc[sub["population"] == base, outcome].mean()
        if pd.isna(base_mean) or base_mean == 0:
            continue

        for pop, s2 in sub.groupby("population"):
            if pop == base:
                continue

            rows.append(
                {
                    "plan": plan,
                    "plan_display": PLAN_DISPLAY.get(plan, plan),
                    "population": pop,
                    "rel_change": (s2[outcome].mean() - base_mean) / base_mean,
                }
            )

    eff = pd.DataFrame(rows)
    eff.to_csv(OUTDIR / "relative_change_table.csv", index=False)

    if eff.empty:
        print("[WARN] Relative-change table is empty.")
        return eff

    row_order = sorted(eff["population"].unique(), key=population_sort_key)
    col_order = [p for p in PLANS if p in eff["plan"].unique()]
    col_labels = [PLAN_DISPLAY.get(p, p) for p in col_order]

    pivot = (
        eff.pivot(index="population", columns="plan", values="rel_change")
        .reindex(index=row_order, columns=col_order)
    )

    plt.figure(figsize=(6, 4))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Relative change vs baseline")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xticks(np.arange(len(col_order)), col_labels, rotation=20, ha="right")
    # plt.title("Relative change in peak infections vs baseline population")

    savefig(OUTDIR / "relative_change_heatmap_reordered.png")
    return eff


# ============================================================
# PLOT 2 — STRICTER HIERARCHICAL VARIANCE SHARES
# ============================================================

def hierarchical_variance_components(df: pd.DataFrame, outcome_col: str) -> Tuple[float, float, float]:
    """
    Decompose variability into:
      - plan-level variance
      - population-within-plan variance
      - seed-within-(plan,population) variance

    This uses descriptive variance components based on nested means:
      plan component = variance of plan means
      population component = mean within-plan variance of population means
      seed component = mean within-(plan,population) variance across seeds
    """
    tmp = df[["plan", "population", "seed", outcome_col]].copy()
    tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce")
    tmp = tmp.dropna(subset=[outcome_col])

    if tmp.empty:
        return np.nan, np.nan, np.nan

    # 1) Plan-level variance: how much overall plan means differ
    plan_means = tmp.groupby("plan")[outcome_col].mean()
    var_plan = float(plan_means.var(ddof=1)) if len(plan_means) >= 2 else 0.0

    # 2) Population-within-plan variance:
    #    within each plan, compute variance of population means, then average over plans
    pop_means = (
        tmp.groupby(["plan", "population"])[outcome_col]
        .mean()
        .reset_index()
    )

    per_plan_pop_var = pop_means.groupby("plan")[outcome_col].var(ddof=1)
    var_population = float(per_plan_pop_var.mean()) if len(per_plan_pop_var) > 0 else 0.0
    if not np.isfinite(var_population):
        var_population = 0.0

    # 3) Seed variance within each (plan, population), averaged across groups
    seed_var = tmp.groupby(["plan", "population"])[outcome_col].var(ddof=1)
    var_seed = float(seed_var.mean()) if len(seed_var) > 0 else 0.0
    if not np.isfinite(var_seed):
        var_seed = 0.0

    return max(var_plan, 0.0), max(var_population, 0.0), max(var_seed, 0.0)


def plot_variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for outcome_name, outcome_col in OUTCOMES.items():
        if outcome_col not in df.columns:
            print(f"[WARN] Missing outcome column: {outcome_col}")
            continue

        var_plan, var_pop, var_seed = hierarchical_variance_components(df, outcome_col)

        total = var_plan + var_pop + var_seed
        if total <= 0 or not np.isfinite(total):
            plan_share, pop_share, seed_share = np.nan, np.nan, np.nan
        else:
            plan_share = var_plan / total
            pop_share = var_pop / total
            seed_share = var_seed / total

        rows.append(
            {
                "outcome_name": outcome_name,
                "outcome_col": outcome_col,
                "var_plan": var_plan,
                "var_population": var_pop,
                "var_seed": var_seed,
                "share_plan": plan_share,
                "share_population": pop_share,
                "share_seed": seed_share,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR / "variance_decomposition_outcomes.csv", index=False)

    if out.empty:
        print("[WARN] Variance decomposition table is empty.")
        return out

    x = np.arange(len(out))
    width = 0.8

    plan_vals = out["share_plan"].to_numpy(dtype=float)
    pop_vals = out["share_population"].to_numpy(dtype=float)
    seed_vals = out["share_seed"].to_numpy(dtype=float)

    plt.figure(figsize=(6, 4))

    plt.bar(x, plan_vals, width=width, label="Intervention plan")
    plt.bar(x, pop_vals, width=width, bottom=plan_vals, label="Population structure")
    plt.bar(x, seed_vals, width=width, bottom=plan_vals + pop_vals, label="Seed")

    plt.xticks(x, out["outcome_name"])
    plt.ylabel("Variance share")
    plt.ylim(0, 1.08)
    # plt.title("Hierarchical variance decomposition of epidemic outcomes")
    plt.legend()

    for i in range(len(out)):
        pcts = rounded_percentages_sum100([
            plan_vals[i],
            pop_vals[i],
            seed_vals[i],
        ])

        annotate_segment(x[i], 0.0, plan_vals[i], f"{pcts[0]:.2f}%")
        annotate_segment(x[i], plan_vals[i], pop_vals[i], f"{pcts[1]:.2f}%")
        annotate_segment(x[i], plan_vals[i] + pop_vals[i], seed_vals[i], f"{pcts[2]:.2f}%")

    savefig(OUTDIR / "variance_decomposition_hierarchical.png")
    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Loading data...")
    df = load_all_runs()

    print("Making reordered relative-change heatmap...")
    plot_relative_change_heatmap_reordered(df, outcome="peak_infectious")

    print("Making hierarchical variance decomposition...")
    plot_variance_decomposition(df)

    print("")
    print("Done.")
    print(f"Outputs written to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()