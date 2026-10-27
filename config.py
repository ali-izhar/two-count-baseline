"""Paths and dataset constants shared by the library code and repro/."""

from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"        # downloaded data (data/downloaders, data/loaders)
CACHE_DIR = ROOT / "data" / "cache"    # prepared bundles <name>.pkl (data/registry.py)

# One-sided tolerance in snapshots: an alarm d matches an event e if e <= d <= e + tolerance.
TOLERANCES = {"mit_reality": 5, "enron": 4, "askubuntu": 2}

# Enron events 159 and 160 are one week apart and are merged into one.
ENRON_CPS_MERGED = [124, 150, 160, 165]

# The fixed settings of steps 3, 8 and 9 were selected with these MIT event indices; three of them differ from the
# calendar dates of data/downloaders/mit_reality.py (19 -> 20, 64 -> 65, 168 -> 176). `--corrected` in steps 3, 8, 9
# reselects with the calendar dates (results/<step>_corrected.json).
SELECTION_CPS = {"mit_reality": [19, 64, 90, 96, 168, 229]}


def selection_events(name: str, cps, corrected: bool = False) -> list[int]:
    """The events of the setting selection: SELECTION_CPS, or the stored (corrected) events."""
    return list(cps) if corrected else SELECTION_CPS.get(name, list(cps))

CHEBYSHEV_K = 50   # Chebyshev moments per snapshot
CHEBYSHEV_R = 30   # Hutchinson probe vectors
LAD_MAX_EIGEN = 20 # top eigenvalues of D - A stored for LAD
