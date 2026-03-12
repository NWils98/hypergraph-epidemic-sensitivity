"""
run_sensitivity_all.py

Master script used to run the structural sensitivity experiments described in the paper:

    "Explaining Epidemic Sensitivity to Population Structure using Hypergraph Diagnostics"

The script performs the full experiment pipeline:

1. Iterates over population variants and stochastic seeds.
2. Updates the STRIDE configuration file for each run.
3. Executes the STRIDE simulator.
4. Collects the generated daily agent-state files.
5. Computes hypergraph-based structural diagnostics.
6. Generates run-level plots and metrics.
7. Stores processed results as parquet files.
8. Compresses results into a zip archive for storage.

The script assumes the STRIDE simulator is available in:

    ../bin/stride

and that the STRIDE configuration file is located at:

    ../config/run_default.xml

Each run produces a compressed archive containing:
    - daily hypergraph diagnostics
    - node-level exposure metrics
    - structural summaries
    - plots used for analysis

These outputs are later aggregated for the paper's sensitivity and explainability analyses.
"""


from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import extra_metrics as xm
import extra_plots
import metrics
import plots


# =============================================================================
# SETTINGS (stride + analysis loop)
# =============================================================================


RUN_PLAN = {
"pop_belgium600k_c500_teachers_censushh.csv": list(range(4, 9)),
    "P1_hh_highvar_same_mean.csv": list(range(4, 9)),
    "P2_hh_lowvar_same_mean.csv": list(range(4, 9)),
    "P3_weekend_stable.csv": list(range(4, 9)),
    "P4_weekend_mixed.csv": list(range(4, 9)),
    "P5_weekend_rewired.csv": list(range(4, 9)),
    "P6_age_younger.csv": list(range(4, 9)),
    "P7_age_older.csv": list(range(4, 9)),
    "P8_schools_local.csv": list(range(4, 9)),
    "P9_schools_mixed.csv": list(range(4, 9)),
    "P10_work_local.csv": list(range(4, 9)),
    "P11_work_mixed.csv": list(range(4, 9)),

}
# Output base for per-run folders (relative to repo root)
OUTPUT_BASE = Path("sim_output") / "sensitivity_runs"

# If stride needs extra args, add them here
STRIDE_EXTRA_ARGS: List[str] = []

# STRIDE writes day files in repo root for you, so we search both repo root and out_dir.
DAY_PATTERN = re.compile(r"day(\d+)_person_status\.csv$", re.IGNORECASE)

# Wait until day files stop changing (flush safety)
STABLE_SECONDS_REQUIRED = 3.0
MAX_WAIT_SECONDS = 30 * 60
POLL_INTERVAL = 1.0


# =============================================================================
# ANALYSIS SETTINGS (your hypergraph pipeline)
# =============================================================================

TOP_EDGES_PER_DAY = 2000
TOP_SUSCEPTIBLE_NODES_PER_DAY = 1000

# Keep parquet variants: do NOT delete results_work.
CLEANUP_AFTER_ZIP = True

PATTERN = DAY_PATTERN  # reuse


# =============================================================================
# Shared IO helpers
# =============================================================================

def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)

def zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src_dir))

def find_input_folder() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here.parent, here, Path.cwd()]
    for c in candidates:
        if c.exists() and any(PATTERN.search(p.name) for p in c.iterdir() if p.is_file()):
            return c
    raise FileNotFoundError("Could not find any dayX_person_status.csv in parent/script/CWD folders.")

def list_day_csvs(input_dir: Path) -> List[Tuple[int, Path]]:
    out = []
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        m = PATTERN.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    if not out:
        raise FileNotFoundError(f"No day CSVs found in {input_dir}")
    out.sort(key=lambda x: x[0])
    return out

def list_day_csvs_maybe(input_dir: Path) -> List[Tuple[int, Path]]:
    """Non-throwing variant used during waiting/detection."""
    out = []
    if not input_dir.exists():
        return out
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        m = PATTERN.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out

def snapshot_sizes(files: List[Path]) -> Dict[Path, int]:
    sizes: Dict[Path, int] = {}
    for f in files:
        try:
            sizes[f] = f.stat().st_size
        except FileNotFoundError:
            sizes[f] = -1
    return sizes


# =============================================================================
# Basic plots
# =============================================================================

def plot_time_series(summary: pd.DataFrame, outdir: Path) -> None:
    summary = summary.sort_values("day")

    plt.figure()
    plt.plot(summary["day"], summary["n_infected"])
    plt.xlabel("Day")
    plt.ylabel("Infected")
    plt.title("Infected over time")
    plt.tight_layout()
    plt.savefig(outdir / "fig_infected_over_time.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(summary["day"], summary["mean_ies_noninfected"], label="mean IES (susceptible)")
    plt.plot(summary["day"], summary["p95_ies_noninfected"], label="p95 IES (susceptible)")
    plt.xlabel("Day")
    plt.ylabel("IES")
    plt.title("Exposure over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_exposure_over_time.png", dpi=200)
    plt.close()

    cols = [c for c in summary.columns if c.endswith("__total_risk_mass")]
    if cols:
        plt.figure()
        for c in sorted(cols):
            plt.plot(summary["day"], summary[c], label=c.replace("__total_risk_mass", ""))
        plt.xlabel("Day")
        plt.ylabel("Total risk mass (active)")
        plt.title("Risk mass by type over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "fig_risk_mass_by_type.png", dpi=200)
        plt.close()

    cols2 = [c for c in summary.columns if c.endswith("__total_inf_sus_pairs")]
    if cols2:
        plt.figure()
        for c in sorted(cols2):
            plt.plot(summary["day"], summary[c], label=c.replace("__total_inf_sus_pairs", ""))
        plt.xlabel("Day")
        plt.ylabel("Infected×Susceptible pairs (active)")
        plt.title("Transmission opportunity proxy by type")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "fig_inf_sus_pairs_by_type.png", dpi=200)
        plt.close()


# =============================================================================
# helpers
# =============================================================================

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def slugify(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def set_xml_tag_text(xml_path: Path, tag: str, value: str) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    el = root.find(tag)
    if el is None:
        raise ValueError(f"Could not find <{tag}> in {xml_path}")
    el.text = value
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

def run_blocking(cmd: List[str], cwd: Path, env: Optional[dict] = None) -> None:
    print(f"\n[CMD] cwd={cwd}\n  " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd), env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")

def wait_for_outputs_stable(search_dirs: List[Path]) -> Path:
    t0 = time.time()
    last_change = time.time()
    prev_sizes: Optional[Dict[Path, int]] = None

    while True:
        if (time.time() - t0) > MAX_WAIT_SECONDS:
            raise TimeoutError(f"Timed out waiting for outputs to stabilize in: {search_dirs}")

        found_dir: Optional[Path] = None
        day_files: List[Path] = []

        for d in search_dirs:
            days = list_day_csvs_maybe(d)
            if days:
                found_dir = d
                day_files = [p for _, p in days]
                break

        if not day_files or found_dir is None:
            time.sleep(POLL_INTERVAL)
            continue

        sizes = snapshot_sizes(day_files)

        if prev_sizes is None or sizes != prev_sizes:
            prev_sizes = sizes
            last_change = time.time()
        else:
            if (time.time() - last_change) >= STABLE_SECONDS_REQUIRED:
                print(f"[OK] Outputs stable in {found_dir} (unchanged for {STABLE_SECONDS_REQUIRED:.1f}s)")
                return found_dir

        time.sleep(POLL_INTERVAL)

def move_day_files(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for _, p in list_day_csvs_maybe(src_dir):
        dest = dst_dir / p.name
        if dest.exists():
            dest.unlink()
        p.replace(dest)
        moved += 1
    return moved

def fmt_dt(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def fmt_dur(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_current_folder(zip_name: str) -> Path:
    input_dir = find_input_folder()
    days = list_day_csvs(input_dir)
    day0, day0_path = days[0]

    run_dir = Path.cwd()
    workdir = run_dir / "results_work"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # folders
    state_dir = workdir / "state"
    edges_dir = workdir / "edges_top"
    edges_full_dir = workdir / "edges_full"
    nodes_dir = workdir / "nodes_top"
    nodes_full_dir = workdir / "nodes_full"  # Option A
    hubs_dir = workdir / "hubs_top"
    super_dir = workdir / "superspreaders_top"
    coupling_dir = workdir / "coupling"
    blame_dir = workdir / "blame"

    state_dir.mkdir()
    edges_dir.mkdir()
    edges_full_dir.mkdir()
    nodes_dir.mkdir()
    nodes_full_dir.mkdir()
    hubs_dir.mkdir()
    super_dir.mkdir()
    coupling_dir.mkdir()
    blame_dir.mkdir()

    df0 = pd.read_csv(day0_path)

    need_static = ["id", "age", "telework"] + list(xm.EDGE_COLS.keys())
    need_state = ["id"] + list(xm.PRESENCE_COLS.keys()) + ["IsInfected", "IsSusceptible"]
    missing = (set(need_static) | set(need_state)) - set(df0.columns)
    if missing:
        raise ValueError(f"{day0_path.name} missing columns: {sorted(missing)}")

    static_df = df0[need_static].copy()
    static_df["id"] = static_df["id"].astype(int)
    for c in need_static:
        if c != "id":
            static_df[c] = static_df[c].astype(int)
    if not static_df["id"].is_unique:
        raise ValueError("Day0 has duplicate ids.")

    safe_write_parquet(static_df, workdir / "static_membership.parquet")

    state0 = df0[need_state].copy()
    state0["id"] = state0["id"].astype(int)
    for c in xm.PRESENCE_COLS.keys():
        state0[c] = state0[c].astype(np.int8)
    state0["IsInfected"] = state0["IsInfected"].astype(np.int8)
    state0["IsSusceptible"] = state0["IsSusceptible"].astype(np.int8)
    safe_write_parquet(state0, state_dir / f"state_day{day0:03d}.parquet")

    struct_df, struct_txt = xm.structure_stats(static_df)
    safe_write_parquet(struct_df, workdir / "hypergraph_struct.parquet")
    (workdir / "hypergraph_struct.txt").write_text(struct_txt, encoding="utf-8")

    hotspot = xm.HotspotState()

    daily_summary_rows = []
    daily_conc_rows = []
    daily_blame_rows = []

    for day, csv_path in days:
        print(f"[Day {day}] {csv_path.name if csv_path.exists() else '(csv already removed)'}")

        if day == day0:
            state_df = state0
        else:
            df = pd.read_csv(csv_path, usecols=need_state)
            state_df = df[need_state].copy()
            state_df["id"] = state_df["id"].astype(int)
            for c in xm.PRESENCE_COLS.keys():
                state_df[c] = state_df[c].astype(np.int8)
            state_df["IsInfected"] = state_df["IsInfected"].astype(np.int8)
            state_df["IsSusceptible"] = state_df["IsSusceptible"].astype(np.int8)

            safe_write_parquet(state_df, state_dir / f"state_day{day:03d}.parquet")

        edges_full, type_agg = xm.effective_edge_table(day, static_df, state_df)
        safe_write_parquet(edges_full, edges_full_dir / f"edges_full_day{day:03d}.parquet")

        if not edges_full.empty:
            edges_top = edges_full.sort_values(
                ["risk_mass_active", "infected_active", "prevalence_active", "active_size"],
                ascending=[False, False, False, False],
            ).head(TOP_EDGES_PER_DAY).reset_index(drop=True)
        else:
            edges_top = edges_full

        safe_write_parquet(edges_top, edges_dir / f"edges_top_day{day:03d}.parquet")

        conc = xm.edge_concentration_metrics(day, edges_full)
        daily_conc_rows.append(conc)

        nodes_full, hubs_top, super_top, blame_summary = xm.node_metrics(day, static_df, state_df, edges_full)

        safe_write_parquet(nodes_full, nodes_full_dir / f"nodes_full_day{day:03d}.parquet")

        nodes_top = nodes_full[nodes_full["IsInfected"] == 0].sort_values(
            ["ies_pure", "ies_max", "degree_active"],
            ascending=[False, False, False],
        ).head(TOP_SUSCEPTIBLE_NODES_PER_DAY).reset_index(drop=True)

        safe_write_parquet(nodes_top, nodes_dir / f"nodes_top_day{day:03d}.parquet")
        safe_write_parquet(hubs_top, hubs_dir / f"hubs_top_day{day:03d}.parquet")
        safe_write_parquet(super_top, super_dir / f"superspreaders_top_day{day:03d}.parquet")
        daily_blame_rows.append(blame_summary)

        coup_all, coup_inf = xm.coupling_matrices(day, state_df)
        safe_write_parquet(coup_all, coupling_dir / f"coupling_all_day{day:03d}.parquet")
        safe_write_parquet(coup_inf, coupling_dir / f"coupling_infected_day{day:03d}.parquet")

        if not edges_top.empty:
            hotspot.update(edges_top[["edge_type", "group_id"]])

        n_people = int(state_df.shape[0])
        n_inf = int(state_df["IsInfected"].sum())

        ies_non = nodes_full.loc[nodes_full["IsInfected"] == 0, "ies_pure"].to_numpy(dtype=float)
        mean_ies = float(ies_non.mean()) if ies_non.size else 0.0
        p95_ies = float(np.quantile(ies_non, 0.95)) if ies_non.size else 0.0
        max_ies = float(ies_non.max()) if ies_non.size else 0.0

        row = {
            "day": day,
            "n_people": n_people,
            "n_infected": n_inf,
            "infected_frac": (n_inf / n_people) if n_people else 0.0,
            "mean_ies_noninfected": mean_ies,
            "p95_ies_noninfected": p95_ies,
            "max_ies_noninfected": max_ies,
        }
        for _, r in type_agg.iterrows():
            et = str(r["edge_type"])
            row[f"{et}__n_edges_active"] = int(r["n_edges_active"])
            row[f"{et}__total_active_pairs"] = float(r["total_active_pairs"])
            row[f"{et}__total_inf_sus_pairs"] = float(r["total_inf_sus_pairs"])
            row[f"{et}__total_risk_mass"] = float(r["total_risk_mass"])
            row[f"{et}__total_infected_active"] = int(r["total_infected_active"])
            row[f"{et}__total_active_size"] = int(r["total_active_size"])
        daily_summary_rows.append(row)

        print(f"  infected={n_inf}/{n_people}  meanIES={mean_ies:.3f}  p95IES={p95_ies:.3f}")

    summary_df = pd.DataFrame(daily_summary_rows).sort_values("day").reset_index(drop=True)
    safe_write_parquet(summary_df, workdir / "daily_summary.parquet")

    conc_df = pd.concat(daily_conc_rows, ignore_index=True) if daily_conc_rows else pd.DataFrame()
    safe_write_parquet(conc_df, workdir / "daily_edge_concentration.parquet")

    blame_df = pd.concat(daily_blame_rows, ignore_index=True) if daily_blame_rows else pd.DataFrame()
    safe_write_parquet(blame_df, workdir / "daily_blame_summary.parquet")

    hotspot_df = hotspot.to_dataframe()
    safe_write_parquet(hotspot_df, workdir / "hotspot_persistence.parquet")

    plot_time_series(summary_df, workdir)

    extra_plots.main_for_run_all(workdir)
    metrics.main_for_run_all(workdir)
    plots.main_for_run_all(workdir)

    zip_path = run_dir / zip_name
    if zip_path.exists():
        zip_path.unlink()
    zip_dir(workdir, zip_path)
    print(f"[Zip] wrote {zip_path}")

    if CLEANUP_AFTER_ZIP:
        shutil.rmtree(workdir)
        print("[Cleanup] removed workdir, kept zip only.")

    return zip_path


# =============================================================================
# orchestration
# =============================================================================

def master() -> None:
    root = repo_root()
    config_xml = root / "config" / "run_default.xml"
    stride_bin = root / "bin" / "stride"

    if not config_xml.exists():
        raise FileNotFoundError(f"Missing config: {config_xml}")
    if not stride_bin.exists():
        raise FileNotFoundError(f"Missing stride bin: {stride_bin}")

    backup = config_xml.with_suffix(".xml.bak_master")
    if not backup.exists():
        shutil.copy2(config_xml, backup)
        print(f"[Backup] {backup}")

    master_start = time.time()
    print(f"[MASTER] start: {fmt_dt(master_start)}")
    # print(f"[MASTER] populations={len(POP_FILES)} seeds={len(SEEDS)} total_runs={len(POP_FILES)*len(SEEDS)}")
    print(f"[MASTER] populations={len(RUN_PLAN)} total_runs={sum(len(seeds) for seeds in RUN_PLAN.values())}")

    try:
        total = sum(len(seeds) for seeds in RUN_PLAN.values())
        run_idx = 0

        # for pop in POP_FILES:
        for pop, seeds in RUN_PLAN.items():
            for seed in seeds:
                pop_stem = slugify(Path(pop).stem)

                # for seed in SEEDS:
                run_idx += 1
                run_start = time.time()

                print("\n" + "=" * 90)
                print(f"[MASTER {run_idx}/{total}] start: {fmt_dt(run_start)}")
                print(f"[MASTER {run_idx}/{total}] pop={pop} seed={seed}")
                print("=" * 90)

                output_prefix = str(OUTPUT_BASE / pop_stem / f"seed{seed:02d}" / "exp0001")
                out_dir = (root / output_prefix).resolve()
                out_dir.mkdir(parents=True, exist_ok=True)

                # Step 1
                set_xml_tag_text(config_xml, "population_file", pop)
                set_xml_tag_text(config_xml, "rng_seed", str(seed))
                set_xml_tag_text(config_xml, "output_prefix", output_prefix)
                print("[Step 1/3] XML updated.")

                # Step 2
                print("[Step 2/3] Running STRIDE (blocking)...")
                stride_start = time.time()
                run_blocking([str(stride_bin)] + STRIDE_EXTRA_ARGS, cwd=root)
                stride_end = time.time()
                print(f"[Step 2/3] STRIDE done in {fmt_dur(stride_end - stride_start)}. Waiting for flush...")

                found_dir = wait_for_outputs_stable([out_dir, root])

                if found_dir.resolve() != out_dir.resolve():
                    n = move_day_files(found_dir, out_dir)
                    print(f"[Move] moved {n} day files from {found_dir} -> {out_dir}")
                else:
                    print("[Move] day files already in run folder")

                if not list_day_csvs_maybe(out_dir):
                    print(f"[WARN] No day files in {out_dir}. Skipping analysis.")
                    continue

                # Step 3
                zip_name = f"results__{pop_stem}__seed{seed:02d}.zip"
                print(f"[Step 3/3] Running analysis -> {zip_name}")

                old = Path.cwd()
                try:
                    os.chdir(out_dir)
                    analysis_start = time.time()
                    produced_zip = analyze_current_folder(zip_name=zip_name)
                    analysis_end = time.time()
                finally:
                    os.chdir(old)

                # Delete raw CSVs
                for _, p in list_day_csvs_maybe(out_dir):
                    p.unlink()
                print("[Cleanup] Removed raw day CSV files, kept parquet + zip.")

                run_end = time.time()
                print(f"[MASTER {run_idx}/{total}] finished: {fmt_dt(run_end)}")
                print(f"[MASTER {run_idx}/{total}] out_dir: {out_dir}")
                print(f"[MASTER {run_idx}/{total}] zip: {produced_zip.name}")
                print(f"[MASTER {run_idx}/{total}] stride: {fmt_dur(stride_end - stride_start)}  analysis: {fmt_dur(analysis_end - analysis_start)}  total: {fmt_dur(run_end - run_start)}")
                print(f"[MASTER] elapsed overall: {fmt_dur(run_end - master_start)}")
                print("[MASTER] Next STRIDE run can start now.")

    finally:
        master_end = time.time()
        print(f"\n[Restore] restored {config_xml} from {backup}")
        print(f"[MASTER] end: {fmt_dt(master_end)}  total elapsed: {fmt_dur(master_end - master_start)}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    master()


if __name__ == "__main__":
    main()