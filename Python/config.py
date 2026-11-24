"""Shared constants for the Bilbao data pipeline.

Every script in Python/ imports its settings from here, so a value
(city, period, world size, ...) is defined in one place only.

Run this file directly to print the settings and check that the
expected folders exist:

    python Python/config.py
"""

from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Built from this file's location, so scripts work no matter which folder
# they are started from.
REPO_ROOT = Path(__file__).resolve().parent.parent

FULL_DIR = REPO_ROOT / "Data" / "full"            # raw dataset (large)
SAMPLE_DIR = REPO_ROOT / "Data" / "sample"        # same schema, 1000 rows
PROCESSED_DIR = REPO_ROOT / "Data" / "processed"  # Bilbao-only parquet files
NETLOGO_DATA_DIR = REPO_ROOT / "Netlogo" / "data" # small CSVs read by NetLogo

# ---------------------------------------------------------------------------
# City
# ---------------------------------------------------------------------------
CITY_ID = 532

# "ESTACIÓN PARA PRUEBAS": a test station, excluded everywhere.
EXCLUDED_STATION_IDS = {40202557}

# cities.csv says Europe/Berlin, which has the same UTC offset as Madrid
# all year. Europe/Madrid is the correct zone for Bilbao.
TIMEZONE = "Europe/Madrid"

# City center from cities.csv (id 532). Used as the origin of the map
# projection and to split stations into "center" and "periphery".
CENTER_LAT = 43.2623
CENTER_LON = -2.93747

# ---------------------------------------------------------------------------
# Study period
# ---------------------------------------------------------------------------
# Candidate days (local time, inclusive). Weekends, public holidays and
# days with data gaps are removed later by days.py.
PERIOD_START = date(2023, 5, 1)
PERIOD_END = date(2023, 6, 30)

# ---------------------------------------------------------------------------
# NetLogo world (must match the Settings in the Interface tab)
# ---------------------------------------------------------------------------
WORLD_MIN_PXCOR, WORLD_MAX_PXCOR = -60, 60
WORLD_MIN_PYCOR, WORLD_MAX_PYCOR = -42, 42

# Patches kept free between the outermost station and the world edge.
WORLD_MARGIN_PATCHES = 3

# ---------------------------------------------------------------------------
# Model parameters written to params.csv
# ---------------------------------------------------------------------------
# Real street distance is about this many times the straight-line distance.
DETOUR_FACTOR = 1.3


if __name__ == "__main__":
    print(f"Repo root:      {REPO_ROOT}")
    for name, path in [
        ("Full data", FULL_DIR),
        ("Sample data", SAMPLE_DIR),
        ("Processed", PROCESSED_DIR),
        ("NetLogo data", NETLOGO_DATA_DIR),
    ]:
        status = "ok" if path.is_dir() else "missing (created later)"
        print(f"{name + ':':<15} {path.relative_to(REPO_ROOT)}  [{status}]")
    print(f"City:           {CITY_ID}, center ({CENTER_LAT}, {CENTER_LON}), {TIMEZONE}")
    print(f"Excluded:       {sorted(EXCLUDED_STATION_IDS)}")
    print(f"Period:         {PERIOD_START} to {PERIOD_END}")
    print(
        f"World:          x {WORLD_MIN_PXCOR}..{WORLD_MAX_PXCOR}, "
        f"y {WORLD_MIN_PYCOR}..{WORLD_MAX_PYCOR}, margin {WORLD_MARGIN_PATCHES}"
    )
    print(f"Detour factor:  {DETOUR_FACTOR}")
