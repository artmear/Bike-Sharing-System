"""Extract the Bilbao subset of the raw dataset into Data/processed/.

The raw CSVs are large (trips 2.5 GB, station_status 1.6 GB), so this
script filters them once and saves small parquet files. Every later step
reads those parquet files instead of the raw data.

Usage (from the repo root):

    python Python/extract_bilbao.py                   # all steps
    python Python/extract_bilbao.py --steps stations  # only some steps
    python Python/extract_bilbao.py --data-dir Data/sample --out-dir /tmp/x
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from config import CITY_ID, EXCLUDED_STATION_IDS, FULL_DIR, PROCESSED_DIR

STEPS = ["stations", "trips"]

# Rows read per chunk. About 2M rows x 7 columns stays well under 200 MB.
CHUNK_ROWS = 2_000_000

# Columns kept from trips.csv. Coordinates and battery are left out:
# Bilbao trips are station-based and its bikes are not electric.
TRIP_COLUMNS = {
    "bike_id": "int64",
    "city_id": "int64",
    "time_start": "float64",       # epoch seconds UTC, written as "1661625901.000000"
    "station_id_start": "float64", # float because empty values become NaN
    "station_id_end": "float64",
    "duration": "float64",         # seconds
    "distance": "float64",         # meters, straight line
}


def extract_stations(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Keep Bilbao's stations, drop the test station, compute capacity.

    stations.csv is small (about 1 MB), so it is read in one go.
    """
    stations = pd.read_csv(
        data_dir / "stations.csv",
        usecols=["id", "city_id", "name", "bike_racks", "special_racks", "lon", "lat"],
    )
    n_city = (stations["city_id"] == CITY_ID).sum()

    stations = stations[
        (stations["city_id"] == CITY_ID)
        & ~stations["id"].isin(EXCLUDED_STATION_IDS)
    ]

    stations = (
        stations.rename(columns={"id": "station_id"})
        .drop(columns="city_id")
        .assign(capacity=lambda d: d["bike_racks"] + d["special_racks"])
        .sort_values("name")
        .reset_index(drop=True)
    )

    out_path = out_dir / "stations.parquet"
    stations.to_parquet(out_path, index=False)

    print(f"[stations] {n_city} rows for city {CITY_ID}, "
          f"{n_city - len(stations)} excluded, {len(stations)} kept")
    print(f"[stations] capacity: min {stations['capacity'].min()}, "
          f"median {stations['capacity'].median():g}, "
          f"max {stations['capacity'].max()}, "
          f"total {stations['capacity'].sum()}")
    print(f"[stations] saved {out_path}")
    return stations


def extract_trips(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Keep Bilbao's trips, reading trips.csv in chunks.

    trips.csv has about 25M rows (2.5 GB), too much to load at once.
    pandas reads CHUNK_ROWS rows at a time; each chunk is filtered to
    Bilbao and only the matching rows are kept in memory.

    All dates are kept here; days.py picks the study days later.
    Trips that start or end at an excluded station are dropped.
    """
    reader = pd.read_csv(
        data_dir / "trips.csv",
        usecols=list(TRIP_COLUMNS),
        dtype=TRIP_COLUMNS,
        chunksize=CHUNK_ROWS,
    )

    parts = []
    n_read = 0
    t0 = time.time()
    for chunk in reader:
        n_read += len(chunk)
        parts.append(chunk[chunk["city_id"] == CITY_ID])
        print(f"\r[trips] read {n_read:>12,} rows, "
              f"Bilbao so far {sum(len(p) for p in parts):>10,} "
              f"({time.time() - t0:.0f} s)", end="", flush=True)
    print()

    trips = pd.concat(parts, ignore_index=True).drop(columns="city_id")

    # Clean types: epoch seconds as integers, station ids as nullable
    # integers (Int64 keeps missing ids as <NA> instead of NaN floats).
    trips["time_start"] = trips["time_start"].round().astype("int64")
    for col in ["station_id_start", "station_id_end"]:
        trips[col] = trips[col].astype("Int64")

    n_missing = trips[["station_id_start", "station_id_end"]].isna().any(axis=1).sum()
    excluded = (trips["station_id_start"].isin(EXCLUDED_STATION_IDS)
                | trips["station_id_end"].isin(EXCLUDED_STATION_IDS))
    trips = (trips[~excluded.fillna(False)]
             .sort_values("time_start")
             .reset_index(drop=True))

    out_path = out_dir / "trips.parquet"
    trips.to_parquet(out_path, index=False)

    print(f"[trips] {len(trips):,} Bilbao trips kept, "
          f"{excluded.sum()} dropped (test station), "
          f"{n_missing} with a missing station id")
    if len(trips):
        first, last = pd.to_datetime(trips["time_start"].agg(["min", "max"]),
                                     unit="s", utc=True)
        print(f"[trips] from {first:%Y-%m-%d} to {last:%Y-%m-%d} (UTC)")
    print(f"[trips] saved {out_path}")
    return trips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=FULL_DIR,
                        help="folder with the raw CSVs (default: Data/full)")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR,
                        help="output folder (default: Data/processed)")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS,
                        help="which extractions to run (default: all)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "stations" in args.steps:
        extract_stations(args.data_dir, args.out_dir)
    if "trips" in args.steps:
        extract_trips(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
