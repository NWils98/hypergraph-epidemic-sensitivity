from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import shutil


# =========================
# SETTINGS
# =========================

CLEAN_FIRST = False   # set True if you want to delete old results automatically

EXPECTED_FILES = {
    "extract_structural_sa.py": [
        "extracted_structural_sa/extracted"
    ],
    "build_panels_and_summaries.py": [
        "derived_structural_sa/run_level.parquet",
    ],
    "aggregate_seed_stats.py": [
        "derived_structural_sa/seed_stats.parquet"
    ],
    "compute_deltas_vs_baseline.py": [
        "derived_structural_sa/deltas_vs_baseline.parquet"
    ],
    "make_ranking_plots.py": [
        "derived_structural_sa/plots_rankings"
    ],
    "structure_outcome_linking.py": [
        "derived_structural_sa/linking"
    ],
    "time_analysis.py": [
        "derived_structural_sa/time_analysis"
    ],
}

PIPELINE_ORDER = [
    "extract_structural_sa.py",
    "build_panels_and_summaries.py",
    "aggregate_seed_stats.py",
    "compute_deltas_vs_baseline.py",
    "make_ranking_plots.py",
    "structure_outcome_linking.py",
]


# =========================
# Helpers
# =========================

def run_script(script: str):
    print("\n" + "="*60)
    print(f"RUNNING: {script}")
    print("="*60)

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script} FAILED")


def check_outputs(script: str):
    files = EXPECTED_FILES.get(script, [])
    for f in files:
        p = Path(f)
        if not p.exists():
            raise RuntimeError(f"{script} did not produce expected output: {p}")
        else:
            print(f"✓ Found: {p}")


def clean_previous_results():
    paths = [
        Path("extracted_structural_sa"),
        Path("derived_structural_sa"),
    ]

    for p in paths:
        if p.exists():
            print(f"Removing old directory: {p}")
            shutil.rmtree(p)


# =========================
# MAIN
# =========================

def main():

    print("\nSTRUCTURAL SENSITIVITY PIPELINE")
    print("="*60)

    if CLEAN_FIRST:
        clean_previous_results()

    for script in PIPELINE_ORDER:
        run_script(script)
        # check_outputs(script)

    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)

    print("\nKey outputs:")

    print("Run-level data:")
    print("  derived_structural_sa/run_level.parquet")

    print("\nSeed statistics:")
    print("  derived_structural_sa/seed_stats.parquet")

    print("\nSensitivity deltas:")
    print("  derived_structural_sa/deltas_vs_baseline.parquet")

    print("\nRanking plots:")
    print("  derived_structural_sa/plots_rankings/")

    print("\nStructure-outcome linking:")
    print("  derived_structural_sa/linking/")


if __name__ == "__main__":
    main()