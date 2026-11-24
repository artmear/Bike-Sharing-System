"""Build the small CSV files that the NetLogo model reads (Netlogo/data/).

Inputs are the parquet files from extract_bilbao.py, restricted to the
days chosen by days.py and cleaned by clean.py.

Outputs so far:
- stations.csv:     station_id, name, xcor, ycor, capacity, max_bikes,
                    initial_bikes, zone
- params.csv:       name, value (one parameter per row)
- speeds.csv:       band_max_m, speed_mps (ride speeds to draw from, by distance)
- demand_rates.csv: station_id, hour, rate_per_hour
- destinations.csv: origin_id, dest_id, band, prob
- bands.csv:        hour, band (which time band each hour belongs to)
- validation.csv:   station_id, minute_of_day, mean_bikes, mean_bikes_raw,
                    share_empty, share_docks_full, share_full
                    (observed, for model validation)

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

STEPS = ["stations", "demand", "destinations", "validation"]

# Width of the time bins in validation.csv, in minutes.
VALIDATION_BIN_MIN = 10

# max_bikes (docks + overflow) = this quantile of the observed bike count.
MAX_BIKES_QUANTILE = 0.99

# speeds.csv: distance bands (same number of trips each) and speeds per band.
SPEED_DISTANCE_BANDS = 5
SPEED_PERCENTILES = 100
OPEN_BAND_MAX_M = 1_000_000  # "no upper limit" for the last band

# Demand correction: minimum hours with a bike available (summed over all
# selected days) for a station-hour to get the corrected rate.
MIN_AVAILABLE_H = 2.0

# Time bands for the destination choice: hour of day -> band name.
BANDS = {
    "morning": range(6, 10),
    "midday": range(10, 16),
    "evening": range(16, 20),
    "night": [*range(20, 24), *range(0, 6)],
}
HOUR_TO_BAND = {h: band for band, hours in BANDS.items() for h in hours}

# Smoothing weight, in trips, given to the all-day destination
# distribution when estimating each band's distribution.
SMOOTHING_TRIPS = 20

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


@cache
def max_bikes_per_station() -> pd.Series:
    """Most bikes a station can hold in the model, docks plus overflow.

    In Bilbao people keep leaving bikes at a station after all docks are
    taken (32 of 43 stations are above capacity more than 1% of the
    time). The limit is the MAX_BIKES_QUANTILE of the observed count over
    the selected days, rounded up and never below the number of docks.
    The top 1% is ignored so that rare spikes do not set the limit.
    """
    grid = bikes_per_minute()
    capacity = (pd.read_parquet(PROCESSED_DIR / "stations.parquet")
                .set_index("station_id")["capacity"])
    observed = grid.groupby("station_id")["bikes"].quantile(MAX_BIKES_QUANTILE)
    return np.maximum(np.ceil(observed), capacity.loc[observed.index]).astype(int)


def bikes_at_midnight() -> pd.Series:
    """Mean docked bikes at 00:00 over the selected days, per station.

    Each day's count is capped at max_bikes before averaging, the same
    way as mean_bikes in validation.csv, so both start from the same value.
    """
    grid = bikes_per_minute()
    at_00 = grid[grid["minute"] == 0]
    capped = at_00["bikes"].clip(upper=at_00["station_id"].map(max_bikes_per_station()))
    return capped.groupby(at_00["station_id"]).mean()


def build_stations(out_dir: Path) -> tuple[pd.DataFrame, float]:
    stations = pd.read_parquet(PROCESSED_DIR / "stations.parquet")

    x_m, y_m = project_to_meters(stations["lon"], stations["lat"])
    xcor, ycor, meters_per_patch = fit_to_world(x_m, y_m)

    dist_center = np.hypot(x_m, y_m)
    zone = np.where(dist_center <= dist_center.median(), "center", "periphery")

    mean_00 = bikes_at_midnight().loc[stations["station_id"]].to_numpy()
    initial = np.round(mean_00).astype(int)
    max_bikes = max_bikes_per_station().loc[stations["station_id"]].to_numpy()

    out = pd.DataFrame({
        "station_id": stations["station_id"],
        "name": stations["name"],
        "xcor": xcor.round(2),
        "ycor": ycor.round(2),
        "capacity": stations["capacity"],
        "max_bikes": max_bikes,
        "initial_bikes": initial,
        "zone": zone,
    })
    out_path = out_dir / "stations.csv"
    out.to_csv(out_path, index=False)

    print(f"[stations] {len(out)} stations, xcor {out['xcor'].min()}..{out['xcor'].max()}, "
          f"ycor {out['ycor'].min()}..{out['ycor'].max()}")
    print(f"[stations] {meters_per_patch:.1f} m per patch, "
          f"map {(x_m.max() - x_m.min()) / 1000:.2f} x {(y_m.max() - y_m.min()) / 1000:.2f} km")
    extra = max_bikes - stations["capacity"].to_numpy()
    print(f"[stations] {out['capacity'].sum()} docks + {extra.sum()} overflow places "
          f"(p{MAX_BIKES_QUANTILE * 100:g} of observed bikes) at {(extra > 0).sum()} stations")
    print(f"[stations] initial bikes {initial.sum()}, "
          f"{(initial > out['capacity']).sum()} station(s) start above their docks")
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

    build_speeds(out_dir, trips)
    return params


def build_speeds(out_dir: Path, trips: pd.DataFrame) -> pd.DataFrame:
    """Distribution of ride speeds, by distance band, for drawing ride times.

    A single speed (the median) makes every ride as fast as a typical one,
    but some riders stop or take detours: real rides last 15.9 min on
    average against 11.9 min with the median speed, and the model would
    have ~40% fewer bikes on the road at peak hours. So each simulated
    ride draws its own speed.

    Short trips are slower (2.1 m/s in the shortest quarter, 3.4 m/s in
    the longest), so speeds are drawn within the trip's distance band:
    SPEED_DISTANCE_BANDS bands with the same number of trips, each with
    SPEED_PERCENTILES evenly spaced percentiles of its speeds.

    speeds.csv: band_max_m (upper street distance of the band, the last
    one is open), speed_mps. NetLogo picks the first band with
    band_max_m >= distance and draws one of its speeds at random.
    """
    street_m = trips["distance"] * DETOUR_FACTOR
    speed = street_m / trips["duration"]

    edges = np.quantile(street_m, np.linspace(0, 1, SPEED_DISTANCE_BANDS + 1))
    band_max = [int(round(e)) for e in edges[1:-1]] + [OPEN_BAND_MAX_M]
    band = np.searchsorted(edges[1:-1], street_m, side="right")

    levels = (np.arange(SPEED_PERCENTILES) + 0.5) * 100 / SPEED_PERCENTILES
    rows = []
    for b, top in enumerate(band_max):
        for v in np.percentile(speed[band == b], levels):
            rows.append({"band_max_m": top, "speed_mps": round(v, 3)})
    out = pd.DataFrame(rows)
    out_path = out_dir / "speeds.csv"
    out.to_csv(out_path, index=False)

    medians = out.groupby("band_max_m")["speed_mps"].median()
    print(f"[speeds] {SPEED_DISTANCE_BANDS} distance bands (limits "
          f"{[int(m) for m in band_max[:-1]]} m) x {SPEED_PERCENTILES} speeds; "
          f"band medians {medians.round(2).tolist()} m/s")
    print(f"[speeds] saved {out_path}")
    return out


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


def build_destinations(out_dir: Path) -> pd.DataFrame:
    """Probability of each destination given the origin and the time band.

    Destinations change a lot during the day (e.g. morning trips go to
    work areas), so there is one distribution per band. Some origins have
    few trips in a band, so each band's counts are smoothed towards the
    all-day distribution:

        prob(d | o, band) = (n(o, d, band) + K * p_all(d | o)) / (n(o, band) + K)

    with K = SMOOTHING_TRIPS. With many trips the band's own data
    dominates; with few, the all-day pattern fills the gaps.

    Also writes bands.csv (hour, band) so NetLogo knows which band
    applies at each hour.
    """
    trips = load_study_trips()
    od = pd.DataFrame({
        "origin_id": trips["station_id_start"].astype("int64"),
        "dest_id": trips["station_id_end"].astype("int64"),
        "band": trips["start"].dt.hour.map(HOUR_TO_BAND),
    })

    n_all = od.groupby(["origin_id", "dest_id"]).size().rename("n_all").reset_index()
    n_all["p_all"] = n_all["n_all"] / n_all.groupby("origin_id")["n_all"].transform("sum")

    # Every (origin, destination) pair seen during the day, in every band.
    grid = n_all.merge(pd.DataFrame({"band": list(BANDS)}), how="cross")
    n_band = od.groupby(["origin_id", "dest_id", "band"]).size().rename("n").reset_index()
    grid = grid.merge(n_band, on=["origin_id", "dest_id", "band"], how="left").fillna({"n": 0})
    n_origin = grid.groupby(["origin_id", "band"])["n"].transform("sum")

    grid["prob"] = (grid["n"] + SMOOTHING_TRIPS * grid["p_all"]) / (n_origin + SMOOTHING_TRIPS)

    out = (grid[["origin_id", "dest_id", "band", "prob"]]
           .sort_values(["origin_id", "band", "prob"], ascending=[True, True, False])
           .round({"prob": 6}))
    out_path = out_dir / "destinations.csv"
    out.to_csv(out_path, index=False)

    bands = pd.DataFrame({"hour": range(24), "band": [HOUR_TO_BAND[h] for h in range(24)]})
    bands.to_csv(out_dir / "bands.csv", index=False)

    sums = out.groupby(["origin_id", "band"])["prob"].sum()
    trips_per_cell = od.groupby(["origin_id", "band"]).size()
    print(f"[destinations] {len(out)} rows: {out['origin_id'].nunique()} origins x "
          f"{len(BANDS)} bands, {len(n_all)} of {43 * 42} possible pairs")
    print(f"[destinations] trips per origin and band: min {trips_per_cell.min()}, "
          f"median {trips_per_cell.median():.0f}; probabilities sum to "
          f"{sums.min():.4f}..{sums.max():.4f}")
    print(f"[destinations] saved {out_path} and bands.csv")
    return out


def build_validation(out_dir: Path) -> pd.DataFrame:
    """Observed bikes per station over the day, to compare with the model.

    One row per station and 10-minute bin (minute_of_day = 0, 10, ..., 1430),
    averaged over all minutes of the bin and all selected days:
    - mean_bikes:       bikes at the station, capped at max_bikes (the model
                        can never exceed it, so this is the comparable value)
    - mean_bikes_raw:   bikes as reported, without the cap
    - share_empty:      fraction of the time with 0 bikes
    - share_docks_full: fraction of the time with all docks taken
                        (bikes >= capacity; more bikes can still be left)
    - share_full:       fraction of the time with bikes >= max_bikes
                        (no room left at all: returns are redirected)
    """
    grid = bikes_per_minute()
    capacity = (pd.read_parquet(PROCESSED_DIR / "stations.parquet")
                .set_index("station_id")["capacity"])
    cap = grid["station_id"].map(capacity)
    limit = grid["station_id"].map(max_bikes_per_station())

    grid = grid.assign(
        minute_of_day=grid["minute"] // VALIDATION_BIN_MIN * VALIDATION_BIN_MIN,
        capped=grid["bikes"].clip(upper=limit),
        empty=grid["bikes"] == 0,
        docks_full=grid["bikes"] >= cap,
        full=grid["bikes"] >= limit,
    )
    out = (grid.groupby(["station_id", "minute_of_day"])
           .agg(mean_bikes=("capped", "mean"),
                mean_bikes_raw=("bikes", "mean"),
                share_empty=("empty", "mean"),
                share_docks_full=("docks_full", "mean"),
                share_full=("full", "mean"))
           .round(3)
           .reset_index())
    out_path = out_dir / "validation.csv"
    out.to_csv(out_path, index=False)

    total = out.groupby("minute_of_day")["mean_bikes"].sum()
    print(f"[validation] {len(out)} rows ({out['station_id'].nunique()} stations x "
          f"{out['minute_of_day'].nunique()} bins of {VALIDATION_BIN_MIN} min)")
    print(f"[validation] docked bikes in the system: max {total.max():.0f} "
          f"at {total.idxmax() // 60:02d}:{total.idxmax() % 60:02d}, "
          f"min {total.min():.0f} at {total.idxmin() // 60:02d}:{total.idxmin() % 60:02d}")
    print(f"[validation] stations empty {grid['empty'].mean():.1%}, "
          f"all docks taken {grid['docks_full'].mean():.1%}, "
          f"full up to max_bikes {grid['full'].mean():.1%} of the time")
    print(f"[validation] saved {out_path}")
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
    if "destinations" in args.steps:
        build_destinations(args.out_dir)
    if "validation" in args.steps:
        build_validation(args.out_dir)


if __name__ == "__main__":
    main()
