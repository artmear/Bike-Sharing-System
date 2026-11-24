"""Build the small CSV files that the NetLogo model reads (Netlogo/data/).

Inputs are the parquet files from extract_bilbao.py, restricted to the
days chosen by days.py and cleaned by clean.py.

Outputs so far:
- stations.csv:     station_id, name, xcor, ycor, capacity, initial_bikes, zone
- params.csv:       name, value (one parameter per row)
- demand_rates.csv: station_id, hour, rate_per_hour

Usage (from the repo root):

    python Python/build_inputs.py
    python Python/build_inputs.py --steps stations
"""

import argparse
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from clean import load_study_trips
from config import (CENTER_LAT, CENTER_LON, DETOUR_FACTOR, NETLOGO_DATA_DIR,
                    PROCESSED_DIR, TIMEZONE, WORLD_MARGIN_PATCHES,
                    WORLD_MAX_PXCOR, WORLD_MAX_PYCOR, WORLD_MIN_PXCOR,
                    WORLD_MIN_PYCOR)
from days import selected_days

STEPS = ["stations", "demand"]

# Demand correction: minimum hours with a bike available (summed over all
# selected days) for a station-hour to get the corrected rate.
MIN_AVAILABLE_H = 2.0

# Meters per degree (equirectangular approximation, fine at city scale).
M_PER_DEG_LAT = 110_574
M_PER_DEG_LON_EQUATOR = 111_320


def project_to_meters(lon: pd.Series, lat: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Longitude/latitude to meters east/north of the city center."""
    x_m = (lon - CENTER_LON) * M_PER_DEG_LON_EQUATOR * np.cos(np.radians(CENTER_LAT))
    y_m = (lat - CENTER_LAT) * M_PER_DEG_LAT
    return x_m, y_m


def fit_to_world(x_m: pd.Series, y_m: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """Scale meters to patch coordinates, keeping a margin at the edges.

    The same scale is used on both axes, so the map is not distorted.
    The stations' bounding box is centered in the world.
    Returns (xcor, ycor, meters_per_patch).
    """
    usable_w = (WORLD_MAX_PXCOR - WORLD_MIN_PXCOR) - 2 * WORLD_MARGIN_PATCHES
    usable_h = (WORLD_MAX_PYCOR - WORLD_MIN_PYCOR) - 2 * WORLD_MARGIN_PATCHES
    meters_per_patch = max((x_m.max() - x_m.min()) / usable_w,
                           (y_m.max() - y_m.min()) / usable_h)

    world_mid_x = (WORLD_MAX_PXCOR + WORLD_MIN_PXCOR) / 2
    world_mid_y = (WORLD_MAX_PYCOR + WORLD_MIN_PYCOR) / 2
    xcor = world_mid_x + (x_m - (x_m.max() + x_m.min()) / 2) / meters_per_patch
    ycor = world_mid_y + (y_m - (y_m.max() + y_m.min()) / 2) / meters_per_patch
    return xcor, ycor, meters_per_patch


@cache
def bikes_per_minute() -> pd.DataFrame:
    """Docked bikes at every minute of every selected day, per station.

    Columns: station_id, day, minute (0..1439), hour (0..23), bikes.
    About 43 stations x 16 days x 1440 minutes = 1M rows.

    Snapshots are only written when something changes, so the count at a
    given minute is the last snapshot at or before it (merge_asof
    "backward"). Computed once and reused (@cache).
    """
    status = (pd.read_parquet(PROCESSED_DIR / "station_status.parquet",
                              columns=["station_id", "time", "bikes"])
              .sort_values("time"))
    station_ids = pd.read_parquet(PROCESSED_DIR / "stations.parquet")["station_id"]
    days = selected_days()

    grid = pd.DataFrame({
        "day": np.repeat(days, 1440),
        "minute": np.tile(np.arange(1440), len(days)),
        "time": np.concatenate([
            int(pd.Timestamp(d, tz=TIMEZONE).timestamp()) + 60 * np.arange(1440)
            for d in days
        ]),
    })
    grid["hour"] = grid["minute"] // 60

    parts = []
    for sid in station_ids:
        snaps = status.loc[status["station_id"] == sid, ["time", "bikes"]]
        part = pd.merge_asof(grid, snaps, on="time", direction="backward")
        part["station_id"] = sid
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["bikes"] = out["bikes"].astype(float)
    return out[["station_id", "day", "minute", "hour", "bikes"]]


def bikes_at_midnight() -> pd.Series:
    """Mean docked bikes at 00:00 over the selected days, per station."""
    grid = bikes_per_minute()
    return grid[grid["minute"] == 0].groupby("station_id")["bikes"].mean()


def build_stations(out_dir: Path) -> tuple[pd.DataFrame, float]:
    stations = pd.read_parquet(PROCESSED_DIR / "stations.parquet")

    x_m, y_m = project_to_meters(stations["lon"], stations["lat"])
    xcor, ycor, meters_per_patch = fit_to_world(x_m, y_m)

    dist_center = np.hypot(x_m, y_m)
    zone = np.where(dist_center <= dist_center.median(), "center", "periphery")

    mean_00 = bikes_at_midnight().loc[stations["station_id"]].to_numpy()
    initial = np.minimum(np.round(mean_00), stations["capacity"]).astype(int)

    out = pd.DataFrame({
        "station_id": stations["station_id"],
        "name": stations["name"],
        "xcor": xcor.round(2),
        "ycor": ycor.round(2),
        "capacity": stations["capacity"],
        "initial_bikes": initial,
        "zone": zone,
    })
    out_path = out_dir / "stations.csv"
    out.to_csv(out_path, index=False)

    print(f"[stations] {len(out)} stations, xcor {out['xcor'].min()}..{out['xcor'].max()}, "
          f"ycor {out['ycor'].min()}..{out['ycor'].max()}")
    print(f"[stations] {meters_per_patch:.1f} m per patch, "
          f"map {(x_m.max() - x_m.min()) / 1000:.2f} x {(y_m.max() - y_m.min()) / 1000:.2f} km")
    print(f"[stations] initial bikes {initial.sum()} of {out['capacity'].sum()} docks, "
          f"{(np.round(mean_00) > stations['capacity']).sum()} station(s) capped")
    print(f"[stations] zones: {(zone == 'center').sum()} center, "
          f"{(zone == 'periphery').sum()} periphery "
          f"(split at {dist_center.median():.0f} m from the center)")
    print(f"[stations] saved {out_path}")
    return out, meters_per_patch


def build_params(out_dir: Path, meters_per_patch: float) -> pd.DataFrame:
    """Global parameters, one per row.

    ride_speed_mps is measured along the street, not in a straight line:
    the model computes ride time as (straight distance x detour factor) /
    speed, so the speed must include the same detour factor. Otherwise
    simulated rides would last 30% longer than the real ones.
    """
    trips = load_study_trips()
    speed = trips["distance"] * DETOUR_FACTOR / trips["duration"]

    params = pd.DataFrame({
        "name": ["meters_per_patch", "ride_speed_mps", "detour_factor"],
        "value": [round(meters_per_patch, 2), round(speed.median(), 3), DETOUR_FACTOR],
    })
    out_path = out_dir / "params.csv"
    params.to_csv(out_path, index=False)

    print(f"[params] ride speed: median {speed.median():.2f} m/s "
          f"({speed.median() * 3.6:.1f} km/h along streets, "
          f"{speed.median() * 3.6 / DETOUR_FACTOR:.1f} km/h straight line), "
          f"from {len(trips):,} trips")
    print(f"[params] saved {out_path}")
    return params


def build_demand_rates(out_dir: Path) -> pd.DataFrame:
    """Rental requests per hour, per station and hour of day.

    Correction for censored demand: trips.csv only has rentals that
    happened. When a station is empty, people who wanted a bike there
    leave no trace. So instead of
        rate = trips started / number of days
    we use
        rate = trips started / hours the station had at least one bike
    which assumes that, within the same hour of day, demand while the
    station is empty is the same as while it has bikes.

    Cells with less than MIN_AVAILABLE_H of availability (summed over all
    days) would give unstable rates, so they keep the uncorrected rate.

    Also saves Data/processed/demand_rates_detail.csv (trips, available
    hours, raw and corrected rate) for the report.
    """
    grid = bikes_per_minute()
    n_days = grid["day"].nunique()
    avail_h = ((grid["bikes"] > 0)
               .groupby([grid["station_id"], grid["hour"]]).sum() / 60
               ).rename("available_h")

    trips = load_study_trips()
    n_trips = (trips.groupby([trips["station_id_start"].astype("int64"),
                              trips["start"].dt.hour]).size()
               .rename_axis(["station_id", "hour"]).rename("trips"))

    detail = pd.concat([avail_h, n_trips], axis=1).fillna({"trips": 0}).reset_index()
    detail["raw_rate"] = detail["trips"] / n_days
    detail["corrected"] = detail["available_h"] >= MIN_AVAILABLE_H
    detail["rate_per_hour"] = np.where(
        detail["corrected"],
        detail["trips"] / detail["available_h"].clip(lower=MIN_AVAILABLE_H),
        detail["raw_rate"],
    )
    detail.round(4).to_csv(PROCESSED_DIR / "demand_rates_detail.csv", index=False)

    out = detail[["station_id", "hour", "rate_per_hour"]].round(4)
    out_path = out_dir / "demand_rates.csv"
    out.to_csv(out_path, index=False)

    raw_day, corr_day = detail["raw_rate"].sum(), detail["rate_per_hour"].sum()
    print(f"[demand] {len(out)} rows ({out['station_id'].nunique()} stations x 24 hours)")
    print(f"[demand] stations have a bike {detail['available_h'].sum() / (n_days * len(out)):.1%} "
          f"of the time; {(~detail['corrected']).sum()} cells under {MIN_AVAILABLE_H} h kept raw")
    print(f"[demand] requests per day: raw {raw_day:.0f}, corrected {corr_day:.0f} "
          f"(+{corr_day / raw_day - 1:.1%})")
    print(f"[demand] saved {out_path}")
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=NETLOGO_DATA_DIR,
                        help="output folder (default: Netlogo/data)")
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS,
                        help="which files to build (default: all)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if "stations" in args.steps:
        _, meters_per_patch = build_stations(args.out_dir)
        build_params(args.out_dir, meters_per_patch)
    if "demand" in args.steps:
        build_demand_rates(args.out_dir)


if __name__ == "__main__":
    main()
