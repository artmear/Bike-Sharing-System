"""Sanity checks for the CSV files in Netlogo/data/.

Run after build_inputs.py, before using the files in NetLogo:

    python Python/check_inputs.py
    python Python/check_inputs.py --data-dir some/other/folder

Prints PASS or FAIL for each check and exits with code 1 if any fails,
so it can also be used in scripts.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (EXCLUDED_STATION_IDS, NETLOGO_DATA_DIR, PROCESSED_DIR,
                    WORLD_MAX_PXCOR, WORLD_MAX_PYCOR, WORLD_MIN_PXCOR,
                    WORLD_MIN_PYCOR)

EARTH_RADIUS_M = 6_371_000
PARAM_NAMES = {"meters_per_patch", "ride_speed_mps", "detour_factor"}
# Plausible range for total requests per day (observed: ~6700 trips/day).
REQUESTS_PER_DAY_RANGE = (5_000, 10_000)


class Checker:
    """Collects the result of each check and prints it."""

    def __init__(self) -> None:
        self.failed = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        self.failed += not ok
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))


def great_circle_m(lon1, lat1, lon2, lat2):
    """Haversine distance in meters."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def check_stations(c: Checker, st: pd.DataFrame) -> None:
    print("stations.csv")
    c.check("43 stations with unique ids",
            len(st) == 43 and st["station_id"].is_unique, f"{len(st)} rows")
    c.check("test station excluded",
            not st["station_id"].isin(EXCLUDED_STATION_IDS).any())
    c.check("no missing values", not st.isna().any().any())
    inside = (st["xcor"].between(WORLD_MIN_PXCOR, WORLD_MAX_PXCOR)
              & st["ycor"].between(WORLD_MIN_PYCOR, WORLD_MAX_PYCOR))
    c.check("coordinates inside the world", inside.all(),
            f"x {st['xcor'].min()}..{st['xcor'].max()}, y {st['ycor'].min()}..{st['ycor'].max()}")
    c.check("0 < capacity <= max_bikes",
            ((st["capacity"] > 0) & (st["capacity"] <= st["max_bikes"])).all())
    c.check("0 <= initial_bikes <= max_bikes",
            st["initial_bikes"].between(0, st["max_bikes"]).all(),
            f"total {st['initial_bikes'].sum()}")
    c.check("zone is center or periphery",
            st["zone"].isin(["center", "periphery"]).all(),
            st["zone"].value_counts().to_dict().__str__())


def check_params(c: Checker, params: dict, st: pd.DataFrame) -> None:
    print("params.csv")
    c.check("all parameters present", PARAM_NAMES <= set(params),
            f"missing {PARAM_NAMES - set(params)}" if not PARAM_NAMES <= set(params) else "")
    c.check("all values positive", all(v > 0 for v in params.values()))
    c.check("ride speed plausible (1.5 to 6 m/s)",
            1.5 <= params.get("ride_speed_mps", 0) <= 6,
            f"{params.get('ride_speed_mps')} m/s")

    # Distances in the NetLogo world (patches x meters_per_patch) must match
    # real distances between the stations' lon/lat.
    geo = pd.read_parquet(PROCESSED_DIR / "stations.parquet").set_index("station_id")
    st = st.set_index("station_id").join(geo[["lon", "lat"]])
    i, j = np.triu_indices(len(st), k=1)
    real = great_circle_m(st["lon"].values[i], st["lat"].values[i],
                          st["lon"].values[j], st["lat"].values[j])
    world = np.hypot(st["xcor"].values[i] - st["xcor"].values[j],
                     st["ycor"].values[i] - st["ycor"].values[j]) * params["meters_per_patch"]
    err = np.abs(world - real) / real
    c.check("world distances match real distances (< 1% error)", err.max() < 0.01,
            f"max error {err.max():.2%} over {len(err)} station pairs")


def check_speeds(c: Checker, speeds: pd.DataFrame) -> None:
    print("speeds.csv")
    per_band = speeds.groupby("band_max_m").size()
    c.check("distance bands with the same number of speeds",
            len(per_band) >= 1 and per_band.nunique() == 1,
            f"{len(per_band)} bands x {per_band.iloc[0]} speeds")
    c.check("last band is open (very large limit)", speeds["band_max_m"].max() >= 100_000)
    c.check("speeds are positive and below 10 m/s",
            speeds["speed_mps"].between(0, 10, inclusive="right").all(),
            f"{speeds['speed_mps'].min()}..{speeds['speed_mps'].max()} m/s")


def check_demand(c: Checker, rates: pd.DataFrame, ids: set) -> None:
    print("demand_rates.csv")
    c.check("one row per station and hour",
            len(rates) == len(ids) * 24
            and set(rates["station_id"]) == ids
            and set(rates["hour"]) == set(range(24))
            and not rates.duplicated(["station_id", "hour"]).any(),
            f"{len(rates)} rows")
    c.check("rates are >= 0", (rates["rate_per_hour"] >= 0).all())
    per_day = rates["rate_per_hour"].sum()
    lo, hi = REQUESTS_PER_DAY_RANGE
    c.check(f"requests per day between {lo} and {hi}", lo <= per_day <= hi, f"{per_day:.0f}")


def check_destinations(c: Checker, dest: pd.DataFrame, bands: pd.DataFrame, ids: set) -> None:
    print("bands.csv")
    c.check("24 hours, each with one band",
            sorted(bands["hour"]) == list(range(24)) and bands["band"].notna().all(),
            f"bands {sorted(bands['band'].unique())}")

    print("destinations.csv")
    c.check("origins and destinations are known stations",
            set(dest["origin_id"]) <= ids and set(dest["dest_id"]) <= ids)
    c.check("every station is an origin", set(dest["origin_id"]) == ids)
    c.check("no trips to the same station", (dest["origin_id"] != dest["dest_id"]).all())
    c.check("bands match bands.csv", set(dest["band"]) == set(bands["band"]))
    c.check("probabilities in (0, 1]", dest["prob"].between(0, 1, inclusive="right").all())
    sums = dest.groupby(["origin_id", "band"])["prob"].sum()
    c.check("probabilities sum to 1 for each origin and band",
            np.allclose(sums, 1, atol=1e-4) and len(sums) == len(ids) * bands["band"].nunique(),
            f"{sums.min():.5f}..{sums.max():.5f}")


def check_validation(c: Checker, val: pd.DataFrame, st: pd.DataFrame) -> None:
    print("validation.csv")
    ids = set(st["station_id"])
    c.check("one row per station and 10-minute bin",
            len(val) == len(ids) * 144
            and set(val["station_id"]) == ids
            and set(val["minute_of_day"]) == set(range(0, 1440, 10)),
            f"{len(val)} rows")
    limit = val["station_id"].map(st.set_index("station_id")["max_bikes"])
    c.check("0 <= mean_bikes <= max_bikes", val["mean_bikes"].between(0, limit).all())
    shares = val[["share_empty", "share_docks_full", "share_full"]]
    c.check("shares between 0 and 1", ((shares >= 0) & (shares <= 1)).all().all())
    c.check("full implies all docks taken",
            (val["share_full"] <= val["share_docks_full"] + 1e-9).all())
    at_00 = val[val["minute_of_day"] == 0].set_index("station_id")["mean_bikes"]
    diff = (at_00 - st.set_index("station_id")["initial_bikes"]).abs()
    c.check("00:00 mean matches initial_bikes (within 1 bike)", diff.max() <= 1,
            f"max difference {diff.max():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=NETLOGO_DATA_DIR,
                        help="folder with the NetLogo CSVs (default: Netlogo/data)")
    args = parser.parse_args()
    d = args.data_dir

    st = pd.read_csv(d / "stations.csv")
    params = pd.read_csv(d / "params.csv").set_index("name")["value"].to_dict()
    ids = set(st["station_id"])

    c = Checker()
    check_stations(c, st)
    check_params(c, params, st)
    check_speeds(c, pd.read_csv(d / "speeds.csv"))
    check_demand(c, pd.read_csv(d / "demand_rates.csv"), ids)
    check_destinations(c, pd.read_csv(d / "destinations.csv"), pd.read_csv(d / "bands.csv"), ids)
    check_validation(c, pd.read_csv(d / "validation.csv"), st)

    print(f"\n{'All checks passed' if c.failed == 0 else f'{c.failed} check(s) FAILED'}")
    sys.exit(1 if c.failed else 0)


if __name__ == "__main__":
    main()
