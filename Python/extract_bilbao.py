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
from pathlib import Path

import pandas as pd

from config import CITY_ID, EXCLUDED_STATION_IDS, FULL_DIR, PROCESSED_DIR

STEPS = ["stations"]


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


if __name__ == "__main__":
    main()
