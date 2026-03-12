from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

from config import N_BINS_EXPOSURE

def gini(x: np.ndarray) -> float:
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)

def add_gini_over_time(summary_df: pd.DataFrame, nodes_top_dir: None = None) -> pd.DataFrame:
    """
    We cannot compute Gini from top nodes only. For true Gini, you need full nodes or
    compute gini inside the daily loop on full IES array (recommended).
    So: this function is a placeholder if you store daily gini during compute.
    """
    return summary_df

def exposure_decile_curve_for_day(
    nodes_full_t: pd.DataFrame,
    infected_next: pd.Series,
    n_bins: int = N_BINS_EXPOSURE
) -> pd.DataFrame:
    """
    nodes_full_t: columns include id, IsInfected, ies_pure
    infected_next: Series indexed by id with IsInfected at t+1 (1/0)
    Computes decile infection probability among those susceptible at t.
    """
    df = nodes_full_t[["id", "IsInfected", "ies_pure"]].copy()
    df = df[df["IsInfected"] == 0]  # susceptible at t

    if df.empty:
        return pd.DataFrame({"bin": range(1, n_bins+1), "mean_ies": 0.0, "p_infected_next": 0.0, "n": 0})

    # attach next-day infection
    df["infected_next"] = df["id"].map(infected_next).fillna(0).astype(int)

    # deciles on ies_pure
    # handle ties / constant arrays
    try:
        df["bin"] = pd.qcut(df["ies_pure"], q=n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        df["bin"] = 1

    out = df.groupby("bin", as_index=False).agg(
        mean_ies=("ies_pure", "mean"),
        p_infected_next=("infected_next", "mean"),
        n=("infected_next", "size"),
    )
    # ensure bins 1..n_bins exist
    full = pd.DataFrame({"bin": range(1, n_bins+1)})
    out = full.merge(out, on="bin", how="left").fillna({"mean_ies": 0.0, "p_infected_next": 0.0, "n": 0})
    return out

def aggregate_curves(curves: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    curves: day -> curve df with bin, p_infected_next, mean_ies, n
    Returns average curve weighted by n.
    """
    all_rows = []
    for day, c in curves.items():
        tmp = c.copy()
        tmp["day"] = day
        all_rows.append(tmp)
    if not all_rows:
        return pd.DataFrame(columns=["bin","mean_ies","p_infected_next","n"])
    df = pd.concat(all_rows, ignore_index=True)

    # weighted average probability by n
    def wavg(g):
        n = g["n"].to_numpy()
        if n.sum() == 0:
            return pd.Series({"mean_ies": 0.0, "p_infected_next": 0.0, "n": 0})
        return pd.Series({
            "mean_ies": float(np.average(g["mean_ies"], weights=n)),
            "p_infected_next": float(np.average(g["p_infected_next"], weights=n)),
            "n": int(n.sum()),
        })

    out = df.groupby("bin").apply(wavg).reset_index()
    return out