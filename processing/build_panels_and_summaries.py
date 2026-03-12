# sensitivity/build_panels_and_summaries.py
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Helpers
# ============================================================

TS_FILES = {
    "cases": ["cases.csv", "Cases.csv"],
    "symptomatic": ["symptomatic.csv", "Symptomatic.csv"],
    "exposed": ["exposed.csv", "Exposed.csv"],
    "infected": ["infected.csv", "Infected.csv"],
    "infectious": ["infectious.csv", "Infectious.csv"],
}


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Robust time series reading
# -------------------------

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _best_numeric_column(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Pick the most plausible numeric column for a STRIDE "one number per day" file.
    Handles:
      - single column of values (maybe with header)
      - two columns day,value
      - multi-column: pick column with most numeric values
    """
    if df is None or df.empty:
        return None

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")
    if df.empty:
        return None

    numeric_cols: Dict[str, pd.Series] = {}
    for c in df.columns:
        numeric_cols[c] = pd.to_numeric(df[c], errors="coerce")

    cols = list(df.columns)

    # Heuristic: if first col looks like day indices, prefer second col
    if len(cols) >= 2:
        c0, c1 = cols[0], cols[1]
        s0, s1 = numeric_cols[c0], numeric_cols[c1]
        non_na0 = s0.dropna()
        if len(non_na0) >= 10:
            is_intish = np.all(np.isclose(non_na0.values, np.round(non_na0.values)))
            in_range = (non_na0.min() >= 0) and (non_na0.max() <= 500)
            if is_intish and in_range and (s1.notna().sum() >= max(10, int(0.5 * len(s1)))):
                return s1

    # Otherwise: choose numeric col with most non-NaNs
    best = None
    best_count = -1
    for c, s in numeric_cols.items():
        cnt = int(s.notna().sum())
        if cnt > best_count:
            best = s
            best_count = cnt
    if best_count <= 0:
        return None
    return best


def _clean_series(arr: np.ndarray, max_days: Optional[int] = 196) -> np.ndarray:
    """
    Cleans common issues:
      - header read as data (leading NaN)
      - blank trailing lines (trailing NaN)
      - overly long (truncate)
    Keeps internal NaNs (do NOT drop them).
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr

    # Drop leading NaNs only
    i0 = 0
    while i0 < arr.size and np.isnan(arr[i0]):
        i0 += 1
    if i0 > 0:
        arr = arr[i0:]

    # Drop trailing NaNs only
    i1 = arr.size
    while i1 > 0 and np.isnan(arr[i1 - 1]):
        i1 -= 1
    if i1 < arr.size:
        arr = arr[:i1]

    if max_days is not None and arr.size > max_days:
        arr = arr[:max_days]

    return arr

import numpy as np
import pandas as pd
import re

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def read_stride_timeseries_csv(path: Path, max_days: Optional[int] = 196) -> np.ndarray:
    """
    Robust reader for STRIDE timeseries.
    Handles your exact format: single-line CSV with comma-separated values (len 197 with day0).
    Also handles normal multi-line CSV with/without header.
    """
    if not path.exists():
        return np.array([], dtype=float)

    # --- FAST PATH: single-line, comma-separated vector (your files) ---
    try:
        txt = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        txt = ""

    if txt and ("\n" not in txt) and (txt.count(",") >= 5):
        parts = [p.strip() for p in txt.split(",") if p.strip() != ""]
        try:
            arr = np.array([float(x) for x in parts], dtype=float)
        except ValueError:
            arr = np.array([], dtype=float)
        if max_days is not None and arr.size > max_days:
            arr = arr[:max_days]
        return arr

    # --- Otherwise: try pandas CSV parsing ---
    for header in [0, None]:
        for sep in [None, ",", ";", "\t", " "]:
            try:
                df = pd.read_csv(path, header=header, sep=sep, engine="python")
                if df is None or df.empty:
                    continue
                df = df.dropna(axis=1, how="all")
                if df.empty:
                    continue

                # Convert columns to numeric
                nums = {c: pd.to_numeric(df[c], errors="coerce") for c in df.columns}
                cols = list(df.columns)

                # If 2+ cols and first looks like day index, use second
                if len(cols) >= 2:
                    c0, c1 = cols[0], cols[1]
                    s0, s1 = nums[c0], nums[c1]
                    non0 = s0.dropna()
                    if len(non0) >= 10:
                        is_intish = np.all(np.isclose(non0.values, np.round(non0.values)))
                        in_range = (non0.min() >= 0) and (non0.max() <= 500)
                        if is_intish and in_range and (s1.notna().sum() >= 10):
                            arr = s1.to_numpy(dtype=float)
                        else:
                            best = max(nums.items(), key=lambda kv: int(kv[1].notna().sum()))[1]
                            arr = best.to_numpy(dtype=float)
                    else:
                        best = max(nums.items(), key=lambda kv: int(kv[1].notna().sum()))[1]
                        arr = best.to_numpy(dtype=float)
                else:
                    arr = list(nums.values())[0].to_numpy(dtype=float)

                # trim leading/trailing NaNs
                arr = np.asarray(arr, dtype=float)
                i0 = 0
                while i0 < arr.size and np.isnan(arr[i0]):
                    i0 += 1
                arr = arr[i0:]
                i1 = arr.size
                while i1 > 0 and np.isnan(arr[i1 - 1]):
                    i1 -= 1
                arr = arr[:i1]

                if max_days is not None and arr.size > max_days:
                    arr = arr[:max_days]

                if arr.size > 0:
                    return arr
            except Exception:
                continue

    # --- Last resort: regex extract numbers ---
    if txt:
        nums = _NUM_RE.findall(txt)
        if nums:
            arr = np.array([float(x) for x in nums], dtype=float)
            if max_days is not None and arr.size > max_days:
                arr = arr[:max_days]
            return arr

    return np.array([], dtype=float)


# -------------------------
# Outcome helpers
# -------------------------

def auc_trapezoid(ys: np.ndarray) -> float:
    if ys.size == 0:
        return float("nan")
    return float(np.trapezoid(ys, dx=1.0))


def peak_and_day_1based(ys: np.ndarray) -> Tuple[float, float]:
    """
    Always return peak day in 1-based day numbering.
    """
    if ys.size == 0:
        return float("nan"), float("nan")
    peak = float(np.nanmax(ys))
    idx = int(np.nanargmax(ys))
    return peak, float(idx + 1)


def early_growth_slope_log1p_1based(
    ys: np.ndarray,
    d_start: int = 1,
    d_end: int = 28,
) -> float:
    """
    Fit slope of log1p(ys) over days [d_start, d_end] inclusive.
    Interprets the array as 1-based day sequence (index 0 == day 1).
    """
    if ys.size == 0:
        return float("nan")

    i0 = d_start - 1
    i1 = d_end - 1
    if i0 < 0 or i1 < 0 or i0 >= ys.size or i1 >= ys.size or i1 <= i0:
        return float("nan")

    seg = ys[i0 : i1 + 1]
    x = np.arange(d_start, d_end + 1, dtype=float)
    y = np.log1p(seg.astype(float))

    if np.all(~np.isfinite(y)) or y.size < 2:
        return float("nan")

    try:
        return float(np.polyfit(x, y, deg=1)[0])
    except Exception:
        return float("nan")


def infer_day0_presence(ts_len: int, daily_days_max: Optional[int]) -> bool:
    """
    Conservative inference:
      - If daily_days_max is known and ts_len == daily_days_max + 1, then it's likely day0-present.
      - Otherwise default to False (avoid accidental shifts).
    """
    if daily_days_max is None:
        return False
    return ts_len == (daily_days_max + 1)


def build_timeseries_df(
    series_map: Dict[str, np.ndarray],
    daily_days_max: Optional[int],
) -> pd.DataFrame:
    """
    Build a timeseries dataframe with a consistent day column.

    IMPORTANT: we standardise to day=1..N for downstream consistency.
    If the raw series likely contains day 0 (length = max_day+1), we drop the day0 element.
    """
    # Determine length = max length among series
    L = max((v.size for v in series_map.values() if v is not None), default=0)
    if L == 0:
        return pd.DataFrame(columns=["day"] + list(series_map.keys()))

    day0_present = infer_day0_presence(L, daily_days_max)

    # If day0 present, we drop index 0 so that index 0 maps to day 1.
    start_idx = 1 if day0_present else 0
    L2 = L - start_idx
    if L2 <= 0:
        return pd.DataFrame(columns=["day"] + list(series_map.keys()))

    day = np.arange(1, L2 + 1, dtype=int)
    df = pd.DataFrame({"day": day})

    for k, v in series_map.items():
        if v is None or v.size == 0:
            df[k] = np.nan
            continue

        vv = v.astype(float)

        # Drop day0 if needed
        if day0_present and vv.size == L:
            vv = vv[start_idx:]

        # pad/truncate to L2
        if vv.size < L2:
            pad = np.full(L2 - vv.size, np.nan, dtype=float)
            vv = np.concatenate([vv, pad])
        elif vv.size > L2:
            vv = vv[:L2]

        df[k] = vv

    return df


def pivot_edge_type(df: pd.DataFrame, prefix: str, index_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    """
    Pivot for tables with columns: day, edge_type, value_cols...
    Produces wide columns like: {prefix}__{edge_type}__{value_col}

    Uses pivot_table to be robust to accidental duplicates.
    """
    if df.empty:
        return pd.DataFrame(columns=index_cols)

    if "edge_type" not in df.columns:
        raise ValueError("Expected edge_type column for pivot_edge_type")

    out = None
    for vc in value_cols:
        tmp = df[index_cols + ["edge_type", vc]].copy()
        wide = tmp.pivot_table(index=index_cols, columns="edge_type", values=vc, aggfunc="mean")
        wide.columns = [f"{prefix}__{et}__{vc}" for et in wide.columns]
        wide = wide.reset_index()
        out = wide if out is None else out.merge(wide, on=index_cols, how="outer")

    return out if out is not None else pd.DataFrame(columns=index_cols)


def ensure_day_int(df: pd.DataFrame) -> pd.DataFrame:
    if "day" in df.columns:
        df = df.copy()
        df["day"] = pd.to_numeric(df["day"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["day"]).copy()
        df["day"] = df["day"].astype(int)
    return df


# ============================================================
# Phase definitions
# ============================================================

@dataclass(frozen=True)
class Phases:
    early: Tuple[int, int] = (1, 28)
    mid: Tuple[int, int] = (29, 98)
    late: Tuple[int, int] = (99, 196)

    def iter(self) -> Iterable[Tuple[str, int, int]]:
        yield "early", self.early[0], self.early[1]
        yield "mid", self.mid[0], self.mid[1]
        yield "late", self.late[0], self.late[1]


def phase_summaries_from_daily(
    daily_panel: pd.DataFrame,
    phases: Phases,
    exclude_cols: Optional[set] = None,
) -> Dict[str, float]:
    """
    Compute phase mean for each numeric column, plus max over all days.
    """
    exclude_cols = exclude_cols or set()
    out: Dict[str, float] = {}

    if daily_panel.empty or "day" not in daily_panel.columns:
        return out

    # Numeric columns only (excluding day)
    num_cols = []
    for c in daily_panel.columns:
        if c in exclude_cols or c == "day":
            continue
        if pd.api.types.is_numeric_dtype(daily_panel[c]):
            num_cols.append(c)

    # max over all days
    for c in num_cols:
        out[f"daily__{c}__max_all"] = float(pd.to_numeric(daily_panel[c], errors="coerce").max())

    # phase means
    for phase_name, d0, d1 in phases.iter():
        mask = (daily_panel["day"] >= d0) & (daily_panel["day"] <= d1)
        seg = daily_panel.loc[mask, :]
        for c in num_cols:
            out[f"daily__{c}__mean_{phase_name}"] = float(pd.to_numeric(seg[c], errors="coerce").mean())

    return out


# ============================================================
# Core builders
# ============================================================

def build_run_outcomes_from_ts(ts: pd.DataFrame) -> Dict[str, float]:
    """
    Build run-level outcomes from time series df columns.
    Uses:
      - peak infectious and day (1-based)
      - auc infectious
      - early growth slope log1p infectious over days 1..28
      - final_infected_max (max infected)
      - final_cases_sum (sum of cases over days)
    """
    out: Dict[str, float] = {}
    if ts.empty:
        return out

    infectious = pd.to_numeric(ts.get("infectious", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    infected = pd.to_numeric(ts.get("infected", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
    cases = pd.to_numeric(ts.get("cases", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)

    peak_inf, peak_day = peak_and_day_1based(infectious)
    out["peak_infectious"] = peak_inf
    out["peak_day_infectious"] = peak_day
    out["auc_infectious"] = auc_trapezoid(infectious)
    out["final_infected_max"] = float(np.nanmax(infected)) if infected.size else float("nan")
    out["final_cases_sum"] = float(np.nansum(cases)) if cases.size else float("nan")
    out["early_growth_log1p_infectious_slope_d1_28"] = early_growth_slope_log1p_1based(
        infectious, d_start=1, d_end=28
    )
    return out


def load_daily_tables(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Loads extracted zip parquets from extracted_structural_sa/extracted/<pop>/<seed>/
    """
    tables: Dict[str, pd.DataFrame] = {}

    def rp(name: str) -> Path:
        return run_dir / name

    # Required-ish
    for name in [
        "daily_summary.parquet",
        "advanced_daily.parquet",
        "daily_blame_summary.parquet",
    ]:
        p = rp(name)
        if p.exists():
            tables[name] = ensure_day_int(pd.read_parquet(p))

    # edge_type panels
    for name in [
        "daily_edge_concentration.parquet",
        "advanced_hypercore_sample.parquet",
        "advanced_motifs_sample.parquet",
    ]:
        p = rp(name)
        if p.exists():
            tables[name] = ensure_day_int(pd.read_parquet(p))

    return tables


def build_daily_panel(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Builds a daily panel merged on day:
      - daily_summary (wide)
      - advanced_daily (wide)
      - daily_blame_summary (wide)
      - edge_concentration (pivoted by edge_type)
      - hypercore_sample (pivoted by edge_type)
      - motifs_sample (pivoted by edge_type)
    """
    # Start with daily_summary as backbone if present
    if "daily_summary.parquet" in tables:
        base = tables["daily_summary.parquet"].copy()
    else:
        base = tables.get("advanced_daily.parquet", pd.DataFrame()).copy()

    if base.empty:
        return pd.DataFrame()

    base = ensure_day_int(base)

    # Merge advanced_daily
    adv = tables.get("advanced_daily.parquet", pd.DataFrame())
    if not adv.empty:
        adv = ensure_day_int(adv)
        base = base.merge(adv, on="day", how="left", suffixes=("", "__advdup"))
        dup_cols = [c for c in base.columns if c.endswith("__advdup")]
        if dup_cols:
            base = base.drop(columns=dup_cols)

    # Merge blame summary
    blame = tables.get("daily_blame_summary.parquet", pd.DataFrame())
    if not blame.empty:
        blame = ensure_day_int(blame)
        base = base.merge(blame, on="day", how="left", suffixes=("", "__blamedup"))
        dup_cols = [c for c in base.columns if c.endswith("__blamedup")]
        if dup_cols:
            base = base.drop(columns=dup_cols)

    # Pivot edge concentration
    edgec = tables.get("daily_edge_concentration.parquet", pd.DataFrame())
    if not edgec.empty:
        edgec = ensure_day_int(edgec)
        edgec_w = pivot_edge_type(
            edgec,
            prefix="edge_conc",
            index_cols=["day"],
            value_cols=["risk_mass_gini", "risk_mass_top1pct_share"],
        )
        base = base.merge(edgec_w, on="day", how="left")

    # Pivot hypercore sample
    core = tables.get("advanced_hypercore_sample.parquet", pd.DataFrame())
    if not core.empty:
        core = ensure_day_int(core)
        core_w = pivot_edge_type(
            core,
            prefix="core",
            index_cols=["day"],
            value_cols=["core_mean", "core_p95", "core_max", "n_nodes_sampled"],
        )
        base = base.merge(core_w, on="day", how="left")

    # Pivot motifs sample
    motif = tables.get("advanced_motifs_sample.parquet", pd.DataFrame())
    if not motif.empty:
        motif = ensure_day_int(motif)
        motif_w = pivot_edge_type(
            motif,
            prefix="motif",
            index_cols=["day"],
            value_cols=["wedge_mean", "triangles_mean", "clustering_mean", "n_nodes_sampled"],
        )
        base = base.merge(motif_w, on="day", how="left")

    return base.sort_values("day").reset_index(drop=True)


# ============================================================
# Discover runs
# ============================================================

def list_runs(sim_root: Path) -> List[Tuple[str, str, Path]]:
    """
    Returns list of (pop, seed, exp_dir_path)
    where exp_dir_path points to .../<pop>/<seed>/exp0001
    """
    runs: List[Tuple[str, str, Path]] = []
    if not sim_root.exists():
        return runs

    for pop_dir in sorted([p for p in sim_root.iterdir() if p.is_dir()]):
        pop = pop_dir.name
        for seed_dir in sorted([s for s in pop_dir.iterdir() if s.is_dir() and s.name.startswith("seed")]):
            seed = seed_dir.name
            exp_dir = seed_dir / "exp0001"
            if exp_dir.exists() and exp_dir.is_dir():
                runs.append((pop, seed, exp_dir))
    return runs


def find_ts_files(exp_dir: Path) -> Dict[str, Optional[Path]]:
    found: Dict[str, Optional[Path]] = {k: None for k in TS_FILES.keys()}
    for key, candidates in TS_FILES.items():
        for nm in candidates:
            p = exp_dir / nm
            if p.exists():
                found[key] = p
                break
    return found


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Build run-level outcomes, daily panel, and phase summaries.")
    ap.add_argument(
        "--sim-root",
        type=str,
        default="../sim_output/sensitivity_runs",
        help="Path to sim_output/sensitivity_runs (relative to sensitivity folder).",
    )
    ap.add_argument(
        "--extracted-root",
        type=str,
        default="./extracted_structural_sa/extracted",
        help="Path to extracted zip parquets root (relative to sensitivity folder).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="./derived_structural_sa",
        help="Output directory (relative to sensitivity folder).",
    )
    ap.add_argument(
        "--phases",
        type=str,
        default="1-28,29-98,99-196",
        help="Phase windows as 'a-b,c-d,e-f' for early,mid,late.",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    sim_root = (base / args.sim_root).resolve()
    extracted_root = (base / args.extracted_root).resolve()
    out_dir = (base / args.out_dir).resolve()
    safe_mkdir(out_dir)

    # Parse phases
    m = re.match(r"^\s*(\d+)-(\d+)\s*,\s*(\d+)-(\d+)\s*,\s*(\d+)-(\d+)\s*$", args.phases)
    if not m:
        raise ValueError("Invalid --phases format. Expected '1-28,29-98,99-196'.")
    phases = Phases(
        early=(int(m.group(1)), int(m.group(2))),
        mid=(int(m.group(3)), int(m.group(4))),
        late=(int(m.group(5)), int(m.group(6))),
    )

    runs = list_runs(sim_root)
    if not runs:
        raise FileNotFoundError(f"No runs found under {sim_root}")

    # Output holders
    run_outcomes_rows: List[Dict[str, object]] = []
    run_phase_rows: List[Dict[str, object]] = []

    daily_panel_dir = out_dir / "daily_panels"
    safe_mkdir(daily_panel_dir)

    ts_dir = out_dir / "timeseries"
    safe_mkdir(ts_dir)

    total = len(runs)
    for i, (pop, seed, exp_dir) in enumerate(runs, start=1):
        tag = f"[{i:02d}/{total}] {pop}/{seed}"
        print(tag)

        # Load extracted daily tables for this run
        run_extracted = extracted_root / pop / seed
        if not run_extracted.exists():
            print(f"  - WARNING: missing extracted folder: {run_extracted}")
            tables = {}
            daily_days_max = None
        else:
            tables = load_daily_tables(run_extracted)
            dd = tables.get("daily_summary.parquet", pd.DataFrame())
            daily_days_max = int(dd["day"].max()) if (not dd.empty and "day" in dd.columns) else None

        # Load time series from original exp_dir CSVs (not zipped)
        ts_paths = find_ts_files(exp_dir)
        series_map: Dict[str, np.ndarray] = {}
        for k, p in ts_paths.items():
            series_map[k] = read_stride_timeseries_csv(p, max_days=196) if p is not None else np.array([], dtype=float)

        ts = build_timeseries_df(series_map, daily_days_max=daily_days_max)

        # Save run timeseries parquet
        ts_out = ts_dir / pop
        safe_mkdir(ts_out)
        ts.to_parquet(ts_out / f"{seed}.parquet", index=False)

        # Run outcomes from TS
        outcomes = build_run_outcomes_from_ts(ts)
        outcomes_row: Dict[str, object] = {"pop": pop, "seed": seed}
        outcomes_row.update(outcomes)
        run_outcomes_rows.append(outcomes_row)

        # Daily panel from extracted zip parquets
        daily_panel = build_daily_panel(tables)
        if daily_panel.empty:
            print("  - WARNING: daily panel empty (missing extracted parquets?)")
        else:
            dp_out = daily_panel_dir / pop
            safe_mkdir(dp_out)
            daily_panel.to_parquet(dp_out / f"{seed}.parquet", index=False)

            phase_feats = phase_summaries_from_daily(daily_panel, phases=phases, exclude_cols=set())
            phase_row: Dict[str, object] = {"pop": pop, "seed": seed}
            phase_row.update(phase_feats)
            run_phase_rows.append(phase_row)

    # Write master tables
    run_outcomes_df = pd.DataFrame(run_outcomes_rows).sort_values(["pop", "seed"]).reset_index(drop=True)
    run_outcomes_df.to_parquet(out_dir / "run_outcomes.parquet", index=False)
    run_outcomes_df.to_csv(out_dir / "run_outcomes.csv", index=False)

    if run_phase_rows:
        run_phase_df = pd.DataFrame(run_phase_rows).sort_values(["pop", "seed"]).reset_index(drop=True)
        run_phase_df.to_parquet(out_dir / "run_phase_summaries.parquet", index=False)
        run_phase_df.to_csv(out_dir / "run_phase_summaries.csv", index=False)
    else:
        print("No phase summaries built (daily panels missing).")
        run_phase_df = None

    # Convenience merged run-level table
    if run_phase_df is not None:
        run_level = run_outcomes_df.merge(run_phase_df, on=["pop", "seed"], how="left", validate="1:1")
    else:
        run_level = run_outcomes_df.copy()

    run_level.to_parquet(out_dir / "run_level.parquet", index=False)
    run_level.to_csv(out_dir / "run_level.csv", index=False)

    manifest = {
        "sim_root": str(sim_root),
        "extracted_root": str(extracted_root),
        "out_dir": str(out_dir),
        "n_runs": total,
        "phases": {"early": phases.early, "mid": phases.mid, "late": phases.late},
        "outputs": {
            "run_outcomes": str(out_dir / "run_outcomes.parquet"),
            "run_phase_summaries": str(out_dir / "run_phase_summaries.parquet"),
            "daily_panels_dir": str(daily_panel_dir),
            "timeseries_dir": str(ts_dir),
        },
        "notes": {
            "timeseries_day_standard": "Timeseries are standardised to day=1..N. If a day0 element was inferred, it is dropped.",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDone. Outputs written under:\n  {out_dir}\n")


if __name__ == "__main__":
    main()