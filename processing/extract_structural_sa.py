from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Settings
# =============================================================================

SENSITIVITY_DIR = Path(__file__).resolve().parent
ROOT_DIR = SENSITIVITY_DIR.parent

SIM_RUNS_DIR = ROOT_DIR / "sim_output" / "sensitivity_runs"

OUT_DIR = SENSITIVITY_DIR / "extracted_structural_sa"
OUT_EXTRACTED = OUT_DIR / "extracted"
OUT_MASTER = OUT_DIR / "master"

SEEDS = ["seed04", "seed05", "seed06", "seed07", "seed08"]

ZIP_FILES_TO_EXTRACT = [
    "daily_summary.parquet",
    "advanced_daily.parquet",
    "daily_edge_concentration.parquet",
    "advanced_hypercore_sample.parquet",
    "advanced_motifs_sample.parquet",
    "daily_blame_summary.parquet",
    "hotspot_persistence.parquet",
    "hypergraph_struct.parquet",
    "hypergraph_struct.txt",
]

# Time-series stems we want (case-insensitive)
TS_TARGETS = ["cases", "symptomatic", "exposed", "infected", "infectious"]

PHASES = {
    "early": (1, 28),
    "mid": (29, 90),
    "late": (91, 196),
}

DAILY_FEATURE_SPECS = [
    ("daily_summary", "infected_frac"),
    ("daily_summary", "mean_ies_noninfected"),
    ("daily_summary", "p95_ies_noninfected"),
    ("daily_summary", "max_ies_noninfected"),
    ("daily_summary", "n_infected"),
    ("advanced_daily", "entropy_riskmass_all"),
    ("advanced_daily", "js_riskmass_all__vs_day1"),
    ("advanced_daily", "w1_riskmass_all__vs_day1"),
    ("advanced_daily", "kl_riskmass_all__vs_day1"),
    ("advanced_daily", "coupling_infected__spectral_radius_A"),
    ("advanced_daily", "coupling_all__spectral_radius_A"),
    ("advanced_daily", "coupling_infected__lambda2_L"),
    ("advanced_daily", "coupling_all__lambda2_L"),
    ("daily_blame_summary", "mean_outward_pressure_infected"),
    ("daily_blame_summary", "p95_degree_active"),
]

EDGE_TYPES = [
    "household",
    "k12",
    "college",
    "work",
    "community_primary",
    "community_secondary",
    "household_cluster",
]

DEBUG = True  # prints what it finds for TS + zip members


# =============================================================================
# Helpers
# =============================================================================

def ensure_dirs() -> None:
    OUT_EXTRACTED.mkdir(parents=True, exist_ok=True)
    OUT_MASTER.mkdir(parents=True, exist_ok=True)


def find_results_zip(exp_dir: Path) -> Optional[Path]:
    zips = sorted(exp_dir.glob("results__*.zip"))
    if not zips:
        return None
    # take biggest zip (usually the intended one)
    zips = sorted(zips, key=lambda p: p.stat().st_size, reverse=True)
    return zips[0]


def safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def linear_slope(y: Optional[np.ndarray], start_day: int, end_day: int) -> float:
    if y is None or len(y) < end_day:
        return float("nan")
    ys = y[start_day - 1 : end_day]
    xs = np.arange(start_day, end_day + 1, dtype=float)
    if len(ys) < 2:
        return float("nan")
    x_mean = xs.mean()
    y_mean = ys.mean()
    denom = ((xs - x_mean) ** 2).sum()
    if denom == 0:
        return float("nan")
    return float(((xs - x_mean) * (ys - y_mean)).sum() / denom)


def auc(y: Optional[np.ndarray], start_day: int = 1, end_day: Optional[int] = None) -> float:
    """
    AUC with day step = 1.
    Uses np.trapezoid (works on newer numpy where np.trapz may be missing).
    """
    if y is None or len(y) == 0:
        return float("nan")
    if end_day is None:
        end_day = len(y)
    if len(y) < end_day:
        return float("nan")
    ys = np.asarray(y[start_day - 1 : end_day], dtype=float)
    if np.isnan(ys).all():
        return float("nan")
    return float(np.trapezoid(ys, dx=1.0))


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet_from_zip(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    raw = zf.read(member)
    return pd.read_parquet(io.BytesIO(raw))


def read_text_from_zip(zf: zipfile.ZipFile, member: str) -> str:
    raw = zf.read(member)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def summarize_series_by_phases(
    df: Optional[pd.DataFrame],
    value_col: str,
    day_col: str = "day",
    prefix: str = "",
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for phase_name in PHASES:
        out[f"{prefix}{value_col}__mean_{phase_name}"] = float("nan")
    out[f"{prefix}{value_col}__max_all"] = float("nan")

    if df is None or df.empty or value_col not in df.columns or day_col not in df.columns:
        return out

    dfx = df[[day_col, value_col]].dropna().sort_values(day_col)
    if dfx.empty:
        return out

    for phase_name, (d0, d1) in PHASES.items():
        m = dfx.loc[(dfx[day_col] >= d0) & (dfx[day_col] <= d1), value_col].mean()
        out[f"{prefix}{value_col}__mean_{phase_name}"] = float(m) if pd.notna(m) else float("nan")

    out[f"{prefix}{value_col}__max_all"] = float(dfx[value_col].max())
    return out


# =============================================================================
# Robust time-series loading (FIXES header-as-first-row problem)
# =============================================================================

def _best_numeric_column(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None

    numeric_cols: Dict[str, pd.Series] = {}
    for c in df.columns:
        numeric_cols[c] = pd.to_numeric(df[c], errors="coerce")

    cols = list(df.columns)

    # Heuristic: if first col looks like day index, pick second col as values
    if len(cols) >= 2:
        c0, c1 = cols[0], cols[1]
        s0, s1 = numeric_cols[c0], numeric_cols[c1]
        non_na0 = s0.dropna()
        if len(non_na0) >= 10:
            is_intish = np.all(np.isclose(non_na0.values, np.round(non_na0.values)))
            # accept 0..400 or 1..400 style day indices
            in_range = non_na0.min() >= 0 and non_na0.max() <= 400
            if is_intish and in_range:
                if s1.notna().sum() >= max(10, 0.5 * len(s1)):
                    return s1

    # Otherwise pick the numeric column with most valid values
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


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _clean_series(arr: np.ndarray, max_days: int = 196) -> Optional[np.ndarray]:
    """
    Cleans common STRIDE viewer CSV formats:
      - header accidentally parsed as first row => leading NaN
      - extra blank lines => leading/trailing NaN
      - overly long => truncate to max_days
    """
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)

    if arr.size == 0:
        return None

    # Drop leading NaNs (typical when header read as data)
    i0 = 0
    while i0 < arr.size and np.isnan(arr[i0]):
        i0 += 1
    if i0 > 0:
        arr = arr[i0:]

    # Drop trailing NaNs too
    i1 = arr.size
    while i1 > 0 and np.isnan(arr[i1 - 1]):
        i1 -= 1
    if i1 < arr.size:
        arr = arr[:i1]

    if arr.size == 0:
        return None

    # Optional: truncate to 196 days (your runs are 196 days)
    if max_days is not None and arr.size > max_days:
        arr = arr[:max_days]

    return arr


def read_csv_series_robust(path: Path, max_days: int = 196) -> Optional[np.ndarray]:
    """
    Handles:
      - one number per line (optionally with a header)
      - CSV with 1+ columns
      - single-line lists: "0 1 2 ...", "0,1,2,...", "[0, 1, 2]"
      - day,value pairs (heuristic: if first column looks like days, take second)
    """
    if not path.exists():
        return None

    # Prefer header=0 FIRST, because STRIDE viewer files often include a column name.
    tries: List[Tuple[dict, str]] = [
        ({"header": 0}, "header=0"),
        ({"header": None}, "header=None"),
    ]

    for base_kwargs, _tag in tries:
        for sep in [None, ",", ";", "\t", " "]:
            try:
                df = pd.read_csv(path, sep=sep, engine="python", **base_kwargs)
                if df is None or df.empty:
                    continue
                df = df.dropna(axis=1, how="all")
                if df.empty:
                    continue

                s = _best_numeric_column(df)
                if s is None:
                    continue

                arr = s.to_numpy(dtype=float)
                arr = _clean_series(arr, max_days=max_days)
                if arr is None:
                    continue

                # If we got a real time series, return it
                if len(arr) >= 10 and np.isnan(arr).mean() < 0.95:
                    return arr
            except Exception:
                continue

    # --- Fallback: parse raw text and extract all numeric tokens ---
    try:
        txt = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return None

    if not txt:
        return None

    nums = _NUM_RE.findall(txt)
    if len(nums) == 0:
        return None

    arr = np.array([float(x) for x in nums], dtype=float)

    # Heuristic: if it's "day,value,day,value,..."
    if len(arr) >= 40:
        first = arr[: min(196, len(arr))]
        # accept 0..9 or 1..10 as day starts
        if len(first) >= 10 and (
            np.all(np.isclose(first[:10], np.arange(1, 11)))
            or np.all(np.isclose(first[:10], np.arange(0, 10)))
        ):
            vals = arr[1::2]
            vals = _clean_series(vals, max_days=max_days)
            if vals is not None and len(vals) >= 10:
                return vals

    arr = _clean_series(arr, max_days=max_days)
    return arr


def find_timeseries_files(exp_dir: Path) -> Dict[str, Path]:
    """
    Finds top-level epidemic time series CSVs in exp_dir.
    Case-insensitive matching by stem (e.g. 'Infected.csv' => 'infected').
    """
    found: Dict[str, Path] = {}

    # Only top-level in exp0001, as you described
    csvs = [p for p in exp_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    by_stem = {p.stem.lower(): p for p in csvs}

    for target in TS_TARGETS:
        if target in by_stem:
            found[target] = by_stem[target]
        else:
            candidates = [p for p in csvs if target in p.stem.lower()]
            if candidates:
                candidates.sort(key=lambda p: len(p.stem))
                found[target] = candidates[0]
    return found


# =============================================================================
# Zip member matching
# =============================================================================

def find_zip_member(zf: zipfile.ZipFile, target_basename: str) -> Optional[str]:
    names = zf.namelist()
    candidates = [n for n in names if Path(n).name == target_basename]
    if not candidates:
        candidates = [n for n in names if n.endswith("/" + target_basename) or n.endswith(target_basename)]
    if not candidates:
        return None
    candidates.sort(key=len)
    return candidates[0]


# =============================================================================
# Handle "inactive edge type => NaN" by filling type-specific NaNs with 0
# =============================================================================

def fill_inactive_type_nans(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    df = df.copy()
    type_cols = []
    for c in df.columns:
        for t in EDGE_TYPES:
            if f"__{t}" in c:
                type_cols.append(c)
                break
    if type_cols:
        df[type_cols] = df[type_cols].fillna(0.0)
    return df


# =============================================================================
# Main extraction per run
# =============================================================================

@dataclass
class RunID:
    pop: str
    seed: str
    exp: str = "exp0001"


def extract_run(pop_dir: Path, seed_dir: Path) -> Optional[Dict[str, object]]:
    pop = pop_dir.name
    seed = seed_dir.name
    exp_dir = seed_dir / "exp0001"
    if not exp_dir.exists():
        return None

    out_run_dir = OUT_EXTRACTED / pop / seed
    out_run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load time series (outside zip)
    ts_paths = find_timeseries_files(exp_dir)
    ts_data: Dict[str, np.ndarray] = {}
    for key, pth in ts_paths.items():
        arr = read_csv_series_robust(pth, max_days=196)
        if arr is not None:
            ts_data[key] = arr

    if DEBUG:
        print(f"    TS found: { {k: ts_paths[k].name for k in ts_paths} }")
        print(f"    TS loaded lens: { {k: len(ts_data[k]) for k in ts_data} }")

    if ts_data:
        max_len = max(len(v) for v in ts_data.values())
        days = np.arange(1, max_len + 1)
        ts_df = pd.DataFrame({"day": days})
        for k, v in ts_data.items():
            col = np.full(shape=(max_len,), fill_value=np.nan, dtype=float)
            col[: len(v)] = v
            ts_df[k] = col
        write_df(ts_df, out_run_dir / "outcomes_timeseries.parquet")

    infectious = ts_data.get("infectious")
    infected = ts_data.get("infected")
    cases = ts_data.get("cases")

    run_outcomes: Dict[str, float] = {
        "peak_infectious": float(np.nanmax(infectious)) if infectious is not None else float("nan"),
        "peak_day_infectious": float(np.nanargmax(infectious) + 1) if infectious is not None else float("nan"),
        "auc_infectious": auc(infectious) if infectious is not None else float("nan"),
        "final_infected_max": float(np.nanmax(infected)) if infected is not None else float("nan"),
        "final_cases_sum": float(np.nansum(cases)) if cases is not None else float("nan"),
        "early_growth_log1p_infectious_slope_d1_28": linear_slope(
            safe_log1p(infectious) if infectious is not None else None, 1, 28
        ),
    }
    (out_run_dir / "run_outcomes.json").write_text(json.dumps(run_outcomes, indent=2, allow_nan=True))

    # 2) Extract from zip
    zip_path = find_results_zip(exp_dir)
    if zip_path is None:
        if DEBUG:
            print("    WARNING: No results__*.zip found.")
        return {
            "pop": pop,
            "seed": seed,
            "zip_path": None,
            "out_dir": str(out_run_dir),
            "run_outcomes": run_outcomes,
            "features_df_path": None,
        }

    extracted_dfs: Dict[str, pd.DataFrame] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        if DEBUG:
            print(f"    ZIP sample members: {zf.namelist()[:15]}")

        for wanted in ZIP_FILES_TO_EXTRACT:
            base = Path(wanted).name
            member = find_zip_member(zf, base)
            if member is None:
                if DEBUG:
                    print(f"    MISSING in zip: {base}")
                continue

            if wanted.endswith(".parquet"):
                df = read_parquet_from_zip(zf, member)

                # Fill NaNs for inactive types where appropriate
                if base in ("advanced_daily.parquet", "daily_summary.parquet"):
                    df = fill_inactive_type_nans(df)

                extracted_dfs[wanted] = df
                write_df(df, out_run_dir / wanted)
                if DEBUG:
                    print(f"    extracted {base}: shape={df.shape}")

            elif wanted.endswith(".txt"):
                txt = read_text_from_zip(zf, member)
                (out_run_dir / wanted).write_text(txt)
                if DEBUG:
                    print(f"    extracted {base}: {len(txt)} chars")

    # 3) Features
    feat: Dict[str, float] = {"pop": pop, "seed": seed}

    for source, col in DAILY_FEATURE_SPECS:
        df = extracted_dfs.get(f"{source}.parquet")
        feat.update(summarize_series_by_phases(df, col, day_col="day", prefix=f"{source}__"))

    conc = extracted_dfs.get("daily_edge_concentration.parquet")
    if conc is not None and not conc.empty and "edge_type" in conc.columns:
        for metric in ["risk_mass_gini", "risk_mass_top1pct_share"]:
            if metric not in conc.columns:
                continue
            for edge_type in sorted(conc["edge_type"].dropna().unique()):
                dft = conc.loc[conc["edge_type"] == edge_type, ["day", metric]].copy()
                feat.update(summarize_series_by_phases(dft, metric, day_col="day", prefix=f"edge_conc__{edge_type}__"))
            pivot = conc.pivot_table(index="day", columns="edge_type", values=metric, aggfunc="mean")
            mean_types = pivot.mean(axis=1).reset_index().rename(columns={0: metric})
            feat.update(summarize_series_by_phases(mean_types, metric, day_col="day", prefix="edge_conc__MEAN_TYPES__"))

    core = extracted_dfs.get("advanced_hypercore_sample.parquet")
    if core is not None and not core.empty and "edge_type" in core.columns:
        for stat in ["core_mean", "core_p95", "core_max"]:
            if stat not in core.columns:
                continue
            for edge_type in sorted(core["edge_type"].dropna().unique()):
                dft = core.loc[core["edge_type"] == edge_type, ["day", stat]].copy()
                feat.update(summarize_series_by_phases(dft, stat, day_col="day", prefix=f"core__{edge_type}__"))

    motifs = extracted_dfs.get("advanced_motifs_sample.parquet")
    if motifs is not None and not motifs.empty and "edge_type" in motifs.columns:
        for stat in ["wedge_mean", "triangles_mean", "clustering_mean"]:
            if stat not in motifs.columns:
                continue
            for edge_type in sorted(motifs["edge_type"].dropna().unique()):
                dft = motifs.loc[motifs["edge_type"] == edge_type, ["day", stat]].copy()
                feat.update(summarize_series_by_phases(dft, stat, day_col="day", prefix=f"motif__{edge_type}__"))

    hp = extracted_dfs.get("hotspot_persistence.parquet")
    if hp is not None and not hp.empty:
        if "days_in_topK" in hp.columns:
            feat["hotspots__sum_days_in_topK"] = float(hp["days_in_topK"].sum())
            feat["hotspots__mean_days_in_topK"] = float(hp["days_in_topK"].mean())
            feat["hotspots__p95_days_in_topK"] = float(hp["days_in_topK"].quantile(0.95))
        if "max_consecutive_days" in hp.columns:
            feat["hotspots__max_streak_max"] = float(hp["max_consecutive_days"].max())
            feat["hotspots__mean_streak"] = float(hp["max_consecutive_days"].mean())
            feat["hotspots__p95_streak"] = float(hp["max_consecutive_days"].quantile(0.95))

    feat_df = pd.DataFrame([feat])
    write_df(feat_df, out_run_dir / "run_features.parquet")

    return {
        "pop": pop,
        "seed": seed,
        "zip_path": str(zip_path),
        "out_dir": str(out_run_dir),
        "run_outcomes": run_outcomes,
        "features_df_path": str(out_run_dir / "run_features.parquet"),
    }


def build_master_tables(run_records: List[Dict[str, object]]) -> None:
    idx_df = pd.DataFrame(
        [{
            "pop": r["pop"],
            "seed": r["seed"],
            "zip_path": r.get("zip_path"),
            "out_dir": r.get("out_dir"),
        } for r in run_records]
    )
    write_df(idx_df, OUT_MASTER / "run_index.parquet")

    out_rows = []
    for r in run_records:
        row = {"pop": r["pop"], "seed": r["seed"]}
        row.update(r.get("run_outcomes", {}))
        out_rows.append(row)
    write_df(pd.DataFrame(out_rows), OUT_MASTER / "run_outcomes.parquet")

    feat_paths = [Path(r["features_df_path"]) for r in run_records if r.get("features_df_path")]
    feats = [pd.read_parquet(p) for p in feat_paths if p.exists()]
    if feats:
        write_df(pd.concat(feats, ignore_index=True), OUT_MASTER / "run_features.parquet")


def main() -> None:
    ensure_dirs()

    if not SIM_RUNS_DIR.exists():
        raise FileNotFoundError(f"Could not find {SIM_RUNS_DIR}. Run this from the sensitivity folder.")

    run_records: List[Dict[str, object]] = []
    pop_dirs = [p for p in SIM_RUNS_DIR.iterdir() if p.is_dir()]

    total = sum(1 for pop in pop_dirs for seed in SEEDS if (pop / seed / "exp0001").exists())
    done = 0

    for pop_dir in sorted(pop_dirs):
        for seed in SEEDS:
            seed_dir = pop_dir / seed
            exp_dir = seed_dir / "exp0001"
            if not exp_dir.exists():
                continue

            done += 1
            print(f"[{done:02d}/{total:02d}] {pop_dir.name}/{seed}")
            rec = extract_run(pop_dir, seed_dir)
            if rec is not None:
                run_records.append(rec)

    print(f"Extraction complete for {len(run_records)} runs.")
    print("Building master tables ...")
    build_master_tables(run_records)
    print(f"Done. Output under: {OUT_DIR}")


if __name__ == "__main__":
    main()