# extra_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------
# Schema maps
# ---------

EDGE_COLS: Dict[str, str] = {
    "householdId": "household",
    "k12SchoolId": "k12",
    "collegeId": "college",
    "workId": "work",
    "primaryCommunityId": "community_primary",
    "secondaryCommunityId": "community_secondary",
    "householdClusterId": "household_cluster",
}

PRESENCE_COLS: Dict[str, str] = {
    "inHousehold": "household",
    "inK12": "k12",
    "inCollege": "college",
    "inWork": "work",
    "inPrimaryCommunity": "community_primary",
    "inSecondaryCommunity": "community_secondary",
    "inHouseholdCluster": "household_cluster",
}

SKIP_GID = 0


# ---------
# Hotspot tracker (multi-day)
# ---------

@dataclass
class HotspotState:
    # edge_key -> (count_in_topK, current_streak, max_streak)
    counts: Dict[Tuple[str, int], int]
    streak: Dict[Tuple[str, int], int]
    max_streak: Dict[Tuple[str, int], int]

    def __init__(self) -> None:
        self.counts = {}
        self.streak = {}
        self.max_streak = {}

    def update(self, top_edges: pd.DataFrame) -> None:
        """
        top_edges must contain: edge_type, group_id
        We treat 'appears in top list' as hotspot membership for the day.
        """
        todays = set(zip(top_edges["edge_type"].astype(str), top_edges["group_id"].astype(int)))

        # increment counts + streaks for todays
        for k in todays:
            self.counts[k] = self.counts.get(k, 0) + 1
            self.streak[k] = self.streak.get(k, 0) + 1
            self.max_streak[k] = max(self.max_streak.get(k, 0), self.streak[k])

        # reset streak for edges not in todays but seen before
        for k in list(self.streak.keys()):
            if k not in todays:
                self.streak[k] = 0

    def to_dataframe(self) -> pd.DataFrame:
        keys = set(self.counts) | set(self.max_streak)
        rows = []
        for (etype, gid) in keys:
            rows.append({
                "edge_type": etype,
                "group_id": gid,
                "days_in_topK": int(self.counts.get((etype, gid), 0)),
                "max_consecutive_days": int(self.max_streak.get((etype, gid), 0)),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["days_in_topK", "max_consecutive_days"], ascending=[False, False]).reset_index(drop=True)
        return df


# ---------
# One-time structure stats (infection-independent)
# ---------

def structure_stats(static_df: pd.DataFrame, skip_gid: int = SKIP_GID) -> Tuple[pd.DataFrame, str]:
    """
    Returns:
      struct_df: edge-size stats per type
      struct_txt: readable summary
    """
    n_people = int(static_df.shape[0])

    # degree = number of non-zero memberships across types
    deg = np.zeros(n_people, dtype=np.int16)
    for col in EDGE_COLS.keys():
        deg += (static_df[col].astype(int).to_numpy() != skip_gid).astype(np.int16)

    deg_mean = float(deg.mean())
    deg_p95 = float(np.quantile(deg, 0.95))
    deg_max = int(deg.max())

    # edge sizes per type
    rows = []
    for col, etype in EDGE_COLS.items():
        gids = static_df[col].astype(int)
        valid = gids[gids != skip_gid]
        if valid.empty:
            continue
        counts = valid.value_counts(sort=False).to_numpy()
        rows.append({
            "metric": "edge_size",
            "edge_type": etype,
            "n_edges": int(len(counts)),
            "mean": float(np.mean(counts)),
            "p50": float(np.quantile(counts, 0.50)),
            "p95": float(np.quantile(counts, 0.95)),
            "max": float(np.max(counts)),
        })

    struct_df = pd.DataFrame(rows).sort_values(["metric", "edge_type"]).reset_index(drop=True)

    txt = []
    txt.append("Hypergraph structure summary (infection-independent)")
    txt.append("====================================================")
    txt.append(f"Nodes (people): {n_people}")
    txt.append(f"Degree across context types: mean={deg_mean:.3f}, p95={deg_p95:.1f}, max={deg_max}")
    txt.append("")
    txt.append("Edge size stats per type:")
    for r in rows:
        txt.append(
            f"  {r['edge_type']}: n_edges={r['n_edges']} mean={r['mean']:.2f} "
            f"p50={r['p50']:.0f} p95={r['p95']:.0f} max={r['max']:.0f}"
        )
    return struct_df, "\n".join(txt)


# ---------
# Effective hypergraph: edge table per day (presence-filtered)
# ---------

def effective_edge_table(
    day: int,
    static_df: pd.DataFrame,
    state_df: pd.DataFrame,
    skip_gid: int = SKIP_GID,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build FULL effective edge metrics per day, and also type-level aggregates.

    Returns:
      edges_full: one row per (edge_type, group_id)
        columns: active_size, infected_active, susceptible_active,
                 prevalence_active, risk_mass_active, mixing_active,
                 active_pairs, inf_sus_pairs
      type_agg: per edge_type totals (mechanistic drivers)
    """
    merged = static_df.merge(state_df, on="id", how="inner", validate="one_to_one")

    infected = merged["IsInfected"].astype(np.int8).to_numpy()

    edge_frames = []
    type_rows = []

    for presence_col, etype in PRESENCE_COLS.items():
        id_col = next(c for c, t in EDGE_COLS.items() if t == etype)

        gids = merged[id_col].astype(int).to_numpy()
        present = merged[presence_col].astype(np.int8).to_numpy() == 1

        active_mask = present & (gids != skip_gid)
        if not active_mask.any():
            type_rows.append({
                "day": day,
                "edge_type": etype,
                "n_edges_active": 0,
                "total_active_pairs": 0.0,
                "total_inf_sus_pairs": 0.0,
                "total_risk_mass": 0.0,
                "total_infected_active": 0,
                "total_active_size": 0,
            })
            continue

        g = pd.DataFrame({"gid": gids[active_mask], "inf": infected[active_mask]})

        infected_active = g.groupby("gid", sort=False)["inf"].sum().astype(int)
        active_size = g.groupby("gid", sort=False)["inf"].count().astype(int)

        idx = infected_active.index.astype(int)
        inf = infected_active.values.astype(np.int32)
        sz = active_size.reindex(idx).values.astype(np.int32)
        sus = sz - inf

        with np.errstate(divide="ignore", invalid="ignore"):
            prev = np.where(sz > 0, inf / sz, 0.0)
            risk_mass = np.where(sz > 1, sus * inf / (sz - 1), 0.0)  # exposure potential
            mixing = np.where(sz > 0, inf * sus / sz, 0.0)

        # mechanistic opportunity counts
        active_pairs = (sz * (sz - 1) / 2.0).astype(np.float64)
        inf_sus_pairs = (inf * sus).astype(np.float64)

        tmp = pd.DataFrame({
            "day": day,
            "edge_type": etype,
            "group_id": idx,
            "active_size": sz,
            "infected_active": inf,
            "susceptible_active": sus,
            "prevalence_active": prev,
            "risk_mass_active": risk_mass,
            "mixing_active": mixing,
            "active_pairs": active_pairs,
            "inf_sus_pairs": inf_sus_pairs,
        })
        edge_frames.append(tmp)

        type_rows.append({
            "day": day,
            "edge_type": etype,
            "n_edges_active": int(tmp.shape[0]),
            "total_active_pairs": float(np.sum(active_pairs)),
            "total_inf_sus_pairs": float(np.sum(inf_sus_pairs)),
            "total_risk_mass": float(tmp["risk_mass_active"].sum()),
            "total_infected_active": int(tmp["infected_active"].sum()),
            "total_active_size": int(tmp["active_size"].sum()),
        })

    edges_full = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame(
        columns=["day","edge_type","group_id","active_size","infected_active","susceptible_active",
                 "prevalence_active","risk_mass_active","mixing_active","active_pairs","inf_sus_pairs"]
    )
    type_agg = pd.DataFrame(type_rows)

    return edges_full, type_agg


# ---------
# Concentration / inequality metrics for edges (per type)
# ---------

def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    # Gini formula
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)

def edge_concentration_metrics(day: int, edges_full: pd.DataFrame) -> pd.DataFrame:
    """
    Per edge_type: gini + top1% share of risk_mass_active
    """
    rows = []
    for etype in edges_full["edge_type"].unique():
        sub = edges_full[edges_full["edge_type"] == etype]
        rm = sub["risk_mass_active"].to_numpy(dtype=np.float64)
        if rm.size == 0:
            continue
        rm_sorted = np.sort(rm)[::-1]
        topk = max(1, int(np.ceil(0.01 * rm_sorted.size)))
        top_share = float(rm_sorted[:topk].sum() / rm_sorted.sum()) if rm_sorted.sum() > 0 else 0.0
        rows.append({
            "day": day,
            "edge_type": etype,
            "risk_mass_gini": gini(rm),
            "risk_mass_top1pct_share": top_share,
        })
    return pd.DataFrame(rows)


# ---------
# Node metrics: exposure composition, bridge, hubs, superspreaders, blame
# ---------

def node_metrics(
    day: int,
    static_df: pd.DataFrame,
    state_df: pd.DataFrame,
    edges_full: pd.DataFrame,
    skip_gid: int = SKIP_GID,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns 4 tables:
      - nodes_full: one row per person (can be huge; caller should NOT save full by default)
      - hubs_top: top hubs (active degree + bridge measures)
      - super_top: top superspreader potentials among infected
      - blame_summary: compact blame / attribution at day level
    """
    merged = static_df.merge(state_df, on="id", how="inner", validate="one_to_one")
    n = int(len(merged))

    infected = merged["IsInfected"].astype(np.int8).to_numpy()

    # Build lookups from edges_full per type
    inf_map: Dict[str, Dict[int, int]] = {}
    sz_map: Dict[str, Dict[int, int]] = {}
    prev_map: Dict[str, Dict[int, float]] = {}

    for etype in edges_full["edge_type"].unique():
        sub = edges_full[edges_full["edge_type"] == etype]
        inf_map[etype] = dict(zip(sub["group_id"].astype(int), sub["infected_active"].astype(int)))
        sz_map[etype] = dict(zip(sub["group_id"].astype(int), sub["active_size"].astype(int)))
        prev_map[etype] = dict(zip(sub["group_id"].astype(int), sub["prevalence_active"].astype(float)))

    # Per-person accumulators
    degree_active = np.zeros(n, dtype=np.int16)

    ies_sum = np.zeros(n, dtype=np.float64)       # pure exposure sum (susceptibles only)
    ies_max = np.zeros(n, dtype=np.float64)
    # per-type exposure composition
    ies_by_type = {etype: np.zeros(n, dtype=np.float64) for etype in PRESENCE_COLS.values()}

    # For bridge risk: collect per-person prevalence values across incident edges
    # We'll estimate using mean & mean-square online (variance)
    count_p = np.zeros(n, dtype=np.int16)
    mean_p = np.zeros(n, dtype=np.float64)
    m2_p = np.zeros(n, dtype=np.float64)  # sum of squared deviations

    # Superspreader potential for infected:
    # outward_pressure = sum over active incident edges of (susceptible_active / (active_size-1))
    outward = np.zeros(n, dtype=np.float64)

    for presence_col, etype in PRESENCE_COLS.items():
        id_col = next(c for c, t in EDGE_COLS.items() if t == etype)

        gids = merged[id_col].astype(int).to_numpy()
        present = merged[presence_col].astype(np.int8).to_numpy() == 1
        valid = present & (gids != skip_gid)

        if not valid.any():
            continue

        gids_v = gids[valid]

        infs = np.array([inf_map.get(etype, {}).get(int(g), 0) for g in gids_v], dtype=np.int32)
        sizes = np.array([sz_map.get(etype, {}).get(int(g), 0) for g in gids_v], dtype=np.int32)
        prevs = np.array([prev_map.get(etype, {}).get(int(g), 0.0) for g in gids_v], dtype=np.float64)

        denom = (sizes - 1).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            exposure_contrib = np.where(denom > 0, infs / denom, 0.0)
            outward_contrib = np.where(denom > 0, (sizes - infs) / denom, 0.0)  # susceptible / (n-1)

        idx = np.where(valid)[0]
        degree_active[idx] += 1

        # exposure: only meaningful for susceptibles, we'll zero infected later
        ies_sum[idx] += exposure_contrib
        ies_max[idx] = np.maximum(ies_max[idx], exposure_contrib)
        ies_by_type[etype][idx] += exposure_contrib

        # bridge variance update on prevalence values (Welford)
        for j, p in zip(idx, prevs):
            count_p[j] += 1
            delta = p - mean_p[j]
            mean_p[j] += delta / count_p[j]
            delta2 = p - mean_p[j]
            m2_p[j] += delta * delta2

        # superspreader potential contribution (only matters for infected people later)
        outward[idx] += outward_contrib

    # apply rules:
    # - exposure for infected = 0
    infected_mask = infected == 1
    ies_sum[infected_mask] = 0.0
    ies_max[infected_mask] = 0.0
    for etype in ies_by_type:
        ies_by_type[etype][infected_mask] = 0.0

    ies_mean = ies_sum / np.maximum(degree_active.astype(np.float64), 1.0)
    dominance = np.where(ies_sum > 0, ies_max / ies_sum, 0.0)  # "single-edge dominance"

    # bridge metrics (susceptibles only are interesting, but compute for all)
    bridge_var = np.where(count_p > 1, m2_p / (count_p - 1), 0.0)
    bridge_prod = np.maximum(ies_max, 0.0) * np.maximum(ies_sum, 0.0)  # quick "bridge-like" score

    # hubs: active degree + bridge-ishness
    nodes_full = pd.DataFrame({
        "day": day,
        "id": merged["id"].astype(int).to_numpy(),
        "IsInfected": infected.astype(np.int8),
        "degree_active": degree_active,
        "ies_pure": ies_sum,
        "ies_max": ies_max,
        "ies_mean": ies_mean,
        "dominance": dominance,
        "bridge_var": bridge_var,
        "bridge_prod": bridge_prod,
        "outward_pressure": outward,
    })
    # add per-type exposure cols
    for etype, arr in ies_by_type.items():
        nodes_full[f"ies_{etype}"] = arr

    # hubs (not only infected; hubs are structural)
    hubs_top = nodes_full.sort_values(
        ["degree_active", "bridge_var", "bridge_prod"],
        ascending=[False, False, False],
    ).head(1000).reset_index(drop=True)

    # superspreader potential: infected only
    super_top = nodes_full[nodes_full["IsInfected"] == 1].sort_values(
        ["outward_pressure", "degree_active"],
        ascending=[False, False],
    ).head(1000).reset_index(drop=True)

    # blame / attribution summaries (compact)
    # - context blame: total risk_mass per type
    # - edge blame: top edges by risk_mass (caller already stores top edges)
    # - node blame:
    #   * infected -> outward_pressure
    #   * susceptible high-risk -> ies_pure
    blame_summary = pd.DataFrame([{
        "day": day,
        "total_outward_pressure_infected": float(nodes_full.loc[nodes_full["IsInfected"] == 1, "outward_pressure"].sum()),
        "mean_outward_pressure_infected": float(nodes_full.loc[nodes_full["IsInfected"] == 1, "outward_pressure"].mean()) if int((nodes_full["IsInfected"]==1).sum()) else 0.0,
        "mean_degree_active": float(nodes_full["degree_active"].mean()),
        "p95_degree_active": float(np.quantile(nodes_full["degree_active"].to_numpy(), 0.95)),
    }])

    return nodes_full, hubs_top, super_top, blame_summary


# ---------
# Cross-type coupling matrices (macro mixing explanation)
# ---------

def coupling_matrices(day: int, state_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two coupling matrices:
      - all_people: counts of people present in both type A and type B
      - infected_only: same, but only among infected people

    Returns as long-form tables: day, typeA, typeB, count
    """
    types = list(PRESENCE_COLS.keys())
    type_labels = [PRESENCE_COLS[c] for c in types]

    present = state_df[types].astype(np.int8).to_numpy()
    infected = state_df["IsInfected"].astype(np.int8).to_numpy()

    # all coupling
    M_all = present.T @ present  # counts
    # infected-only coupling (filter rows where infected==1)
    if infected.sum() > 0:
        present_inf = present[infected == 1]
        M_inf = present_inf.T @ present_inf
    else:
        M_inf = np.zeros_like(M_all)

    rows_all = []
    rows_inf = []
    for i, a in enumerate(type_labels):
        for j, b in enumerate(type_labels):
            rows_all.append({"day": day, "typeA": a, "typeB": b, "count": int(M_all[i, j])})
            rows_inf.append({"day": day, "typeA": a, "typeB": b, "count": int(M_inf[i, j])})

    return pd.DataFrame(rows_all), pd.DataFrame(rows_inf)