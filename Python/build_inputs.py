"""Build the small CSV files that the NetLogo model reads (Netlogo/data/).

Inputs are the parquet files from extract_bilbao.py, restricted to the
days chosen by days.py and cleaned by clean.py.

Outputs so far:
- stations.csv: station_id, name, xcor, ycor, capacity, initial_bikes, zone
- params.csv:   name, value (one parameter per row)

Usage (from the repo root):

    python Python/build_inputs.py
    python Python/build_inputs.py --steps stations
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clean import load_study_trips
from config import (CENTER_LAT, CENTER_LON, DETOUR_FACTOR, NETLOGO_DATA_DIR,
                    PROCESSED_DIR, TIMEZONE, WORLD_MARGIN_PATCHES,
                    WORLD_MAX_PXCOR, WORLD_MAX_PYCOR, WORLD_MIN_PXCOR,
                    WORLD_MIN_PYCOR)
from days import selected_days

STEPS = ["stations"]

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


def bikes_at_midnight(station_ids: pd.Series) -> pd.Series:
    """Mean docked bikes at 00:00 over the selected days, per station.

    Snapshots are only written when something changes, so the count at
    00:00 is the last snapshot at or before midnight.
    """
    status = (pd.read_parquet(PROCESSED_DIR / "station_status.parquet",
                              columns=["station_id", "time", "bikes"])
              .sort_values("time"))
    midnights = pd.DataFrame({"time": [
        int(pd.Timestamp(d, tz=TIMEZONE).timestamp()) for d in selected_days()
    ]})

    means = {}
    for sid in station_ids:
        snaps = status[status["station_id"] == sid][["time", "bikes"]]
        at_00 = pd.merge_asof(midnights, snaps, on="time", direction="backward")
        means[sid] = at_00["bikes"].astype(float).mean()
    return pd.Series(means)


def build_stations(out_dir: Path) -> tuple[pd.DataFrame, float]:
    stations = pd.read_parquet(PROCESSED_DIR / "stations.parquet")

    x_m, y_m = project_to_meters(stations["lon"], stations["lat"])
    xcor, ycor, meters_per_patch = fit_to_world(x_m, y_m)

    dist_center = np.hypot(x_m, y_m)
    zone = np.where(dist_center <= dist_center.median(), "center", "periphery")

    mean_00 = bikes_at_midnight(stations["station_id"]).loc[stations["station_id"]].to_numpy()
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


if __name__ == "__main__":
    main()
