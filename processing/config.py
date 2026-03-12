from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict

# Matches dayX_person_status.csv
PATTERN = re.compile(r"day(\d+)_person_status\.csv$", re.IGNORECASE)

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

STATIC_COLS = ["id", "age", "telework"] + list(EDGE_COLS.keys())
STATE_COLS = ["id"] + list(PRESENCE_COLS.keys()) + ["IsInfected", "IsSusceptible"]

SKIP_GID = 0

TOP_EDGES_PER_DAY = 2000
TOP_NODES_PER_DAY = 500

# Sensitivity / variants
MIN_ACTIVE_EDGE_SIZE_DEFAULT = 2  # ignore active edges of size < 2 when computing exposure
DROPSMALL_ACTIVE_EDGE_SIZE_VARIANT = 5  # example "drop small edges < 5"

# Decile curve settings
N_BINS_EXPOSURE = 10

@dataclass(frozen=True)
class Paths:
    workdir_name: str = "effective_hypergraph_work"
    results_zip_name: str = "effective_hypergraph_results.zip"