"""Trip cleaning rules, applied before any input is built.

What the raw data already looks like (checked on all Bilbao trips):
the dataset authors pre-filtered it. There are no trips that start and
end at the same station, no trips shorter than 121 s or 120 m, and no
straight-line speed above ~15 km/h. So the rules below remove very
little; they are kept so the cleaning is explicit and documented.

Rules (thresholds go in the report):
1. missing_station: start or end station id is empty.
2. false_start: same start and end station and duration < FALSE_START_MAX_S.
3. too_long: duration > MAX_DURATION_S.

Usage (from the repo root):

    python Python/clean.py    # report on the selected days and on the whole year

Other scripts call load_study_trips() to get the clean trips of the
selected days.
"""

import pandas as pd

from config import PROCESSED_DIR
from days import local_time, selected_days

FALSE_START_MAX_S = 60
MAX_DURATION_S = 3 * 3600


def cleaning_masks(trips: pd.DataFrame) -> dict[str, pd.Series]:
    """One boolean mask per rule, True where the trip is removed."""
    same_station = trips["station_id_start"] == trips["station_id_end"]
    return {
        "missing_station": trips[["station_id_start", "station_id_end"]].isna().any(axis=1),
        "false_start": (same_station & (trips["duration"] < FALSE_START_MAX_S)).fillna(False),
        "too_long": trips["duration"] > MAX_DURATION_S,
    }


def clean_trips(trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean trips, report).

    The report has one row per rule with the number of trips it matches.
    A trip can match more than one rule, so the rows may add up to more
    than the total removed.
    """
    masks = cleaning_masks(trips)
    removed = pd.concat(masks, axis=1).any(axis=1)

    report = pd.DataFrame({
        "rule": list(masks) + ["total removed"],
        "trips": [int(m.sum()) for m in masks.values()] + [int(removed.sum())],
    })
    report["share_pct"] = (100 * report["trips"] / len(trips)).round(3)
    return trips[~removed].reset_index(drop=True), report


def load_study_trips() -> pd.DataFrame:
    """Clean trips of the selected days, with a local start time column."""
    trips = pd.read_parquet(PROCESSED_DIR / "trips.parquet")
    trips["start"] = local_time(trips["time_start"])
    trips = trips[trips["start"].dt.date.isin(selected_days())]
    clean, _ = clean_trips(trips)
    return clean


def main() -> None:
    trips = pd.read_parquet(PROCESSED_DIR / "trips.parquet")
    on_days = local_time(trips["time_start"]).dt.date.isin(selected_days())

    for label, subset in [("selected days", trips[on_days]), ("whole year", trips)]:
        clean, report = clean_trips(subset)
        print(f"--- {label}: {len(subset):,} trips, {len(clean):,} kept ---")
        print(report.to_string(index=False))
        print()

    print(f"Thresholds: false start < {FALSE_START_MAX_S} s at the same station, "
          f"too long > {MAX_DURATION_S / 3600:g} h")


if __name__ == "__main__":
    main()
