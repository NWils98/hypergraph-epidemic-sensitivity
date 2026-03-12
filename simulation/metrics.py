#!/usr/bin/env python3
"""
metrics.py

Adds:
- Entropy metrics (risk_mass distribution, degree/outward distributions if available)
- KL divergence (smoothed) + Jensen–Shannon divergence (recommended) between days
- Wasserstein-1 distance (1D) between days (risk_mass distributions)
- Spectral metrics of coupling matrices (all + infected-only): eigenvalues, spectral radius, Fiedler (if applicable)
- Hypercoreness (approx): k-core on sampled 2-section projections per context type
- Motif counts (approx): triangle/wedge counts + clustering on sampled 2-section projections

Inputs expected in results_dir:
- edges_full/edges_full_dayXXX.parquet   (REQUIRED for entropy/KL/Wass)
- coupling/coupling_all_dayXXX.parquet   (REQUIRED for spectral)
- coupling/coupling_infected_dayXXX.parquet (optional but expected)
Optional:
- nodes_full/nodes_full_dayXXX.parquet   (if you decide to store full node tables later)
  OR at least hubs_top and superspreaders_top for partial node distribution metrics.

Outputs:
- results_dir/advanced_daily.parquet
- results_dir/advanced_hypercore_sample.parquet
- results_dir/advanced_motifs_sample.parquet

Run:
  python metrics.py <results_dir>
Default results_dir = ./results_work
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -------------------------
# Utilities: distributions
# -------------------------

def shannon_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))

def hist_prob(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    h, _ = np.histogram(x, bins=bins)
    h = h.astype(float)
    if h.sum() == 0:
        return np.zeros_like(h)
    return h / h.sum()

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    # smooth to avoid log(0)
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps=eps) + 0.5 * kl_divergence(q, m, eps=eps)

def wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Exact W1 (Earth Mover's Distance) for 1D samples with equal weights.
    O(n log n) via sorting.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return 0.0
    x = np.sort(x)
    y = np.sort(y)
    # If sizes differ, interpolate quantiles
    n = max(x.size, y.size)
    q = (np.arange(n) + 0.5) / n
    xq = np.interp(q, (np.arange(x.size) + 0.5) / x.size, x)
    yq = np.interp(q, (np.arange(y.size) + 0.5) / y.size, y)
    return float(np.mean(np.abs(xq - yq)))


# -------------------------
# Coupling spectral metrics
# -------------------------

def pivot_coupling(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    types = sorted(set(df["typeA"]).union(set(df["typeB"])))
    idx = {t: i for i, t in enumerate(types)}
    M = np.zeros((len(types), len(types)), dtype=float)
    for _, r in df.iterrows():
        i = idx[str(r["typeA"])]
        j = idx[str(r["typeB"])]
        M[i, j] = float(r["count"])
    return types, M

def spectral_metrics_from_matrix(M: np.ndarray) -> Dict[str, float]:
    """
    For small matrices (7x7), compute eigen spectrum.
    We also compute Laplacian eigenvalues of a symmetrized, normalized matrix.
    """
    out: Dict[str, float] = {}
    if M.size == 0:
        return out

    # symmetrize for stability
    A = 0.5 * (M + M.T)

    # Eigenvalues of A
    evals = np.linalg.eigvals(A)
    evals = np.real_if_close(evals).astype(float)
    evals_sorted = np.sort(evals)[::-1]
    out["spectral_radius_A"] = float(evals_sorted[0])
    out["trace_A"] = float(np.trace(A))
    out["eig1_A"] = float(evals_sorted[0])
    out["eig2_A"] = float(evals_sorted[1]) if evals_sorted.size > 1 else 0.0
    out["eig_gap_A"] = float(evals_sorted[0] - evals_sorted[1]) if evals_sorted.size > 1 else 0.0

    # Build normalized Laplacian from A (treat as weighted graph)
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    Dinv = np.diag(inv_sqrt)
    L = np.eye(A.shape[0]) - Dinv @ A @ Dinv

    levals = np.linalg.eigvals(L)
    levals = np.real_if_close(levals).astype(float)
    levals_sorted = np.sort(levals)
    out["lambda2_L"] = float(levals_sorted[1]) if levals_sorted.size > 1 else 0.0  # algebraic connectivity proxy
    out["lambda_max_L"] = float(levals_sorted[-1]) if levals_sorted.size else 0.0
    return out


# -------------------------
# Sampling-based hypergraph projection for hypercore + motifs
# -------------------------

@dataclass
class ProjectionConfig:
    sample_nodes_per_type: int = 50000     # adjust if you want heavier
    max_edges_per_node: int = 50           # safety cap
    rng_seed: int = 123
    # For motif computations, we work on the sampled projected graph.
    # Increase sample size to reduce variance.

def build_projection_from_edges_full(
    edges_full: pd.DataFrame,
    static_membership: Optional[pd.DataFrame],
    state_df: Optional[pd.DataFrame],
    edge_type: str,
    cfg: ProjectionConfig,
) -> Dict[int, List[int]]:
    """
    Build an *approximate* 2-section projection adjacency list for a given edge_type
    by sampling nodes through hyperedges.

    Practical issue: edges_full has edge-level stats but not memberships.
    To build node-level projection we need membership tables (static + state/presence).
    If you haven't stored nodes_full and don't want to load huge daily person status again,
    we only support projection when you provide:
      - static_membership.parquet
      - state_dayXXX.parquet
    (i.e., using memberships directly, not edges_full)

    This function expects static_membership and state_df.
    It will:
      - sample nodes uniformly from present nodes in this type
      - connect sampled nodes that share the same group_id (clique inside each hyperedge)
    """
    if static_membership is None or state_df is None:
        return {}

    rng = np.random.default_rng(cfg.rng_seed)

    # map edge_type -> id col and presence col
    # we infer from columns
    # (your schema uses householdId, workId, ... and inHousehold, inWork, ...)
    type_to_idcol = {
        "household": "householdId",
        "k12": "k12SchoolId",
        "college": "collegeId",
        "work": "workId",
        "community_primary": "primaryCommunityId",
        "community_secondary": "secondaryCommunityId",
        "household_cluster": "householdClusterId",
    }
    type_to_incol = {
        "household": "inHousehold",
        "k12": "inK12",
        "college": "inCollege",
        "work": "inWork",
        "community_primary": "inPrimaryCommunity",
        "community_secondary": "inSecondaryCommunity",
        "household_cluster": "inHouseholdCluster",
    }
    idcol = type_to_idcol[edge_type]
    incol = type_to_incol[edge_type]

    merged = static_membership.merge(state_df, on="id", how="inner", validate="one_to_one")
    gids = merged[idcol].astype(int).to_numpy()
    present = merged[incol].astype(np.int8).to_numpy() == 1
    valid = present & (gids != 0)
    if not valid.any():
        return {}

    ids = merged.loc[valid, "id"].astype(int).to_numpy()
    if ids.size <= cfg.sample_nodes_per_type:
        sample_ids = ids
    else:
        sample_ids = rng.choice(ids, size=cfg.sample_nodes_per_type, replace=False)

    sample_set = set(int(x) for x in sample_ids)

    # Group sampled nodes by gid
    df = merged.loc[valid, ["id", idcol]].copy()
    df[idcol] = df[idcol].astype(int)
    df["id"] = df["id"].astype(int)
    df = df[df["id"].isin(sample_set)]

    groups = df.groupby(idcol)["id"].apply(list)

    # Build adjacency lists (clique per group, but capped per node)
    adj: Dict[int, List[int]] = {int(i): [] for i in sample_ids}

    for _, members in groups.items():
        if len(members) < 2:
            continue
        # clique connections
        m = [int(x) for x in members]
        for i in range(len(m)):
            u = m[i]
            if len(adj[u]) >= cfg.max_edges_per_node:
                continue
            # connect to others (cap)
            for j in range(len(m)):
                if i == j:
                    continue
                v = m[j]
                if v == u:
                    continue
                if len(adj[u]) >= cfg.max_edges_per_node:
                    break
                adj[u].append(v)

    # de-duplicate neighbor lists
    for u in list(adj.keys()):
        if adj[u]:
            adj[u] = list(dict.fromkeys(adj[u]))
    return adj

def k_core_numbers(adj: Dict[int, List[int]]) -> Dict[int, int]:
    """
    Compute core number (k-core index) for an undirected graph given adjacency list.
    O(n + m) style peeling.
    """
    if not adj:
        return {}

    # Make undirected
    und: Dict[int, set] = {u: set(vs) for u, vs in adj.items()}
    for u, vs in adj.items():
        for v in vs:
            if v in und:
                und[v].add(u)

    deg = {u: len(vs) for u, vs in und.items()}
    # buckets by degree
    maxdeg = max(deg.values()) if deg else 0
    bins: List[List[int]] = [[] for _ in range(maxdeg + 1)]
    for u, d in deg.items():
        bins[d].append(u)

    core = {u: 0 for u in und.keys()}
    removed = set()

    current_deg = deg.copy()

    for k in range(maxdeg + 1):
        stack = bins[k]
        while stack:
            u = stack.pop()
            if u in removed:
                continue
            removed.add(u)
            core[u] = k
            for v in list(und[u]):
                if v in removed:
                    continue
                dv = current_deg[v]
                if dv > 0:
                    current_deg[v] = dv - 1
                    bins[dv - 1].append(v)
            # remove u edges
            for v in und[u]:
                if v in und:
                    und[v].discard(u)
            und[u].clear()
    return core

def motif_stats_triangle_wedge(adj: Dict[int, List[int]], sample_nodes: int = 20000, rng_seed: int = 123) -> Dict[str, float]:
    """
    Approximate motif stats on the projected graph:
      - wedge count (2-paths) on sampled nodes
      - triangle count approximation via neighbor intersections on sampled nodes
      - clustering coefficient approximation

    This is approximate but stable enough for comparing populations (with same config).
    """
    if not adj:
        return {"wedge_mean": 0.0, "triangles_mean": 0.0, "clustering_mean": 0.0}

    rng = np.random.default_rng(rng_seed)
    nodes = np.array(list(adj.keys()), dtype=int)
    if nodes.size == 0:
        return {"wedge_mean": 0.0, "triangles_mean": 0.0, "clustering_mean": 0.0}

    if nodes.size > sample_nodes:
        nodes = rng.choice(nodes, size=sample_nodes, replace=False)

    # sets for fast intersection
    neigh = {u: set(adj[u]) for u in nodes if u in adj}

    wedges = []
    triangles = []
    clustering = []

    for u in nodes:
        Nu = neigh.get(u, set())
        d = len(Nu)
        if d < 2:
            wedges.append(0.0)
            triangles.append(0.0)
            clustering.append(0.0)
            continue

        # wedge count at u = C(d,2)
        w = d * (d - 1) / 2.0
        wedges.append(w)

        # triangles at u approx: sum_{v in Nu} |Nu ∩ Nv| / 2
        # but only over nodes we have sets for; this undercounts a bit (ok for comparisons)
        t2 = 0
        for v in Nu:
            Nv = neigh.get(v)
            if Nv is None:
                continue
            t2 += len(Nu.intersection(Nv))
        t = t2 / 2.0
        triangles.append(t)

        # local clustering = triangles / wedges (bounded)
        clustering.append(float(t / w) if w > 0 else 0.0)

    return {
        "wedge_mean": float(np.mean(wedges)) if wedges else 0.0,
        "triangles_mean": float(np.mean(triangles)) if triangles else 0.0,
        "clustering_mean": float(np.mean(clustering)) if clustering else 0.0,
    }


# -------------------------
# Main metric computation
# -------------------------

def list_days_from_edges(edges_full_dir: Path) -> List[int]:
    files = sorted(edges_full_dir.glob("edges_full_day*.parquet"))
    days = []
    for p in files:
        # edges_full_dayXYZ.parquet
        s = p.stem
        d = int(s.split("day")[1])
        days.append(d)
    return sorted(days)

def read_edges_full(results_dir: Path, day: int) -> pd.DataFrame:
    p = results_dir / "edges_full" / f"edges_full_day{day:03d}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

def read_coupling(results_dir: Path, day: int, infected: bool) -> pd.DataFrame:
    name = f"coupling_infected_day{day:03d}.parquet" if infected else f"coupling_all_day{day:03d}.parquet"
    p = results_dir / "coupling" / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

def read_state(results_dir: Path, day: int) -> Optional[pd.DataFrame]:
    p = results_dir / "state" / f"state_day{day:03d}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)

def read_static(results_dir: Path) -> Optional[pd.DataFrame]:
    p = results_dir / "static_membership.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)

def compute_advanced_metrics(
    results_dir: Path,
    cfg: ProjectionConfig = ProjectionConfig(),
    baseline_day: Optional[int] = None,
) -> None:
    results_dir = Path(results_dir)
    edges_full_dir = results_dir / "edges_full"
    if not edges_full_dir.exists():
        raise FileNotFoundError(f"Missing edges_full dir: {edges_full_dir}")

    days = list_days_from_edges(edges_full_dir)
    if not days:
        raise FileNotFoundError("No edges_full_day*.parquet found.")

    if baseline_day is None:
        baseline_day = days[0]

    # Build global histogram bins from baseline (robust: quantiles)
    base_edges = read_edges_full(results_dir, baseline_day)
    base_rm = base_edges["risk_mass_active"].to_numpy(dtype=float) if not base_edges.empty else np.array([0.0])
    base_rm = base_rm[np.isfinite(base_rm)]
    base_rm = base_rm[base_rm >= 0]
    if base_rm.size == 0:
        base_rm = np.array([0.0])

    # bins: log-spaced up to high quantile + max
    q99 = float(np.quantile(base_rm, 0.99)) if base_rm.size else 1.0
    mx = float(np.max(base_rm)) if base_rm.size else 1.0
    hi = max(1e-6, max(q99 * 5.0, mx))
    bins = np.concatenate(([0.0], np.logspace(-6, np.log10(hi + 1e-12), 60)))

    # cache baseline per-type distributions
    base_dist_by_type: Dict[str, np.ndarray] = {}
    if not base_edges.empty:
        for etype in base_edges["edge_type"].unique():
            x = base_edges.loc[base_edges["edge_type"] == etype, "risk_mass_active"].to_numpy(dtype=float)
            base_dist_by_type[str(etype)] = hist_prob(x, bins=bins)
    else:
        base_dist_by_type = {}

    static_df = read_static(results_dir)

    daily_rows = []
    hypercore_rows = []
    motif_rows = []

    for day in days:
        edges = read_edges_full(results_dir, day)
        if edges.empty:
            continue

        row: Dict[str, float] = {"day": float(day)}

        # ---- Entropy of risk_mass distribution per type (and overall)
        rm_all = edges["risk_mass_active"].to_numpy(dtype=float)
        p_all = hist_prob(rm_all, bins=bins)
        row["entropy_riskmass_all"] = shannon_entropy(p_all)

        for etype in sorted(edges["edge_type"].unique()):
            et = str(etype)
            x = edges.loc[edges["edge_type"] == etype, "risk_mass_active"].to_numpy(dtype=float)
            p = hist_prob(x, bins=bins)
            row[f"entropy_riskmass__{et}"] = shannon_entropy(p)

            # ---- KL/JS/Wasserstein vs baseline (per type)
            p0 = base_dist_by_type.get(et)
            if p0 is not None:
                row[f"kl_riskmass__{et}__vs_day{baseline_day}"] = kl_divergence(p, p0)
                row[f"js_riskmass__{et}__vs_day{baseline_day}"] = js_divergence(p, p0)
                # Wasserstein: use samples (exact from values)
                x0 = base_edges.loc[base_edges["edge_type"] == etype, "risk_mass_active"].to_numpy(dtype=float)
                row[f"w1_riskmass__{et}__vs_day{baseline_day}"] = wasserstein_1d(x, x0)

        # ---- KL/JS/Wasserstein overall vs baseline
        p0_all = hist_prob(base_rm, bins=bins)
        row[f"kl_riskmass_all__vs_day{baseline_day}"] = kl_divergence(p_all, p0_all)
        row[f"js_riskmass_all__vs_day{baseline_day}"] = js_divergence(p_all, p0_all)
        row[f"w1_riskmass_all__vs_day{baseline_day}"] = wasserstein_1d(rm_all, base_rm)

        # ---- Spectral metrics of coupling matrices
        coup_all = read_coupling(results_dir, day, infected=False)
        if not coup_all.empty:
            _, M = pivot_coupling(coup_all)
            sm = spectral_metrics_from_matrix(M)
            for k, v in sm.items():
                row[f"coupling_all__{k}"] = float(v)

        coup_inf = read_coupling(results_dir, day, infected=True)
        if not coup_inf.empty:
            _, M = pivot_coupling(coup_inf)
            sm = spectral_metrics_from_matrix(M)
            for k, v in sm.items():
                row[f"coupling_infected__{k}"] = float(v)

        daily_rows.append(row)

        # ---- Hypercoreness + motif stats (approx; sampling-based)
        # Requires static + state for that day.
        state_df = read_state(results_dir, day)
        if static_df is not None and state_df is not None:
            for etype in [
                "household", "k12", "college", "work",
                "community_primary", "community_secondary", "household_cluster"
            ]:
                adj = build_projection_from_edges_full(
                    edges_full=edges,
                    static_membership=static_df,
                    state_df=state_df,
                    edge_type=etype,
                    cfg=cfg,
                )
                if not adj:
                    continue

                # hypercoreness via k-core on projection
                core = k_core_numbers(adj)
                if core:
                    core_vals = np.array(list(core.values()), dtype=float)
                    hypercore_rows.append({
                        "day": day,
                        "edge_type": etype,
                        "core_mean": float(core_vals.mean()),
                        "core_p95": float(np.quantile(core_vals, 0.95)),
                        "core_max": float(core_vals.max()),
                        "n_nodes_sampled": int(len(core_vals)),
                    })

                # motif stats via triangle/wedge sampling
                ms = motif_stats_triangle_wedge(adj, sample_nodes=min(20000, cfg.sample_nodes_per_type), rng_seed=cfg.rng_seed)
                motif_rows.append({
                    "day": day,
                    "edge_type": etype,
                    "wedge_mean": ms["wedge_mean"],
                    "triangles_mean": ms["triangles_mean"],
                    "clustering_mean": ms["clustering_mean"],
                    "n_nodes_sampled": int(len(adj)),
                })

    advanced_daily = pd.DataFrame(daily_rows).sort_values("day").reset_index(drop=True)
    advanced_daily.to_parquet(results_dir / "advanced_daily.parquet", index=False)

    if hypercore_rows:
        pd.DataFrame(hypercore_rows).to_parquet(results_dir / "advanced_hypercore_sample.parquet", index=False)
    else:
        # still write empty (consistent artifacts)
        pd.DataFrame(columns=["day","edge_type","core_mean","core_p95","core_max","n_nodes_sampled"]).to_parquet(
            results_dir / "advanced_hypercore_sample.parquet", index=False
        )

    if motif_rows:
        pd.DataFrame(motif_rows).to_parquet(results_dir / "advanced_motifs_sample.parquet", index=False)
    else:
        pd.DataFrame(columns=["day","edge_type","wedge_mean","triangles_mean","clustering_mean","n_nodes_sampled"]).to_parquet(
            results_dir / "advanced_motifs_sample.parquet", index=False
        )

    print("Wrote:")
    print(" - advanced_daily.parquet")
    print(" - advanced_hypercore_sample.parquet")
    print(" - advanced_motifs_sample.parquet")


def main_for_run_all(results_dir: Path) -> None:
    compute_advanced_metrics(results_dir)


def main() -> None:
    if len(sys.argv) >= 2:
        results_dir = Path(sys.argv[1]).resolve()
    else:
        results_dir = Path("./results_work").resolve()

    compute_advanced_metrics(results_dir)


if __name__ == "__main__":
    main()