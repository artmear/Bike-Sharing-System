"""Select the "typical spring weekdays" used to build the model inputs.

A candidate day (PERIOD_START..PERIOD_END in config.py, local time) is
kept only if it passes every check below, in this order:

1. weekday: Monday to Friday.
2. holiday: not a public holiday in Bilbao (list below).
3. data: there are trips and station snapshots on that day.
4. gaps: between DAY_START_HOUR and DAY_END_HOUR, no stretch longer than
   MAX_GAP_MIN without any trip start, nor without any station snapshot.
   Nights are not checked: snapshots are only written when something
   changes, so quiet nights naturally have gaps of 1-2 hours.
5. stations: every station has at least one snapshot that day. A station
   with no snapshots for a whole day had no trips either, so it was
   closed; the model always has all stations open.

Usage (from the repo root):

    python Python/days.py     # print the report, save Data/processed/days.csv

Other scripts call selected_days() to get the list of accepted dates.
"""

from datetime import date

import pandas as pd

from config import PERIOD_END, PERIOD_START, PROCESSED_DIR, TIMEZONE

# Public holidays in Bilbao in 2023 (all of them, not only May and June).
# Sources:
# - Basque Country: Decreto 56/2022 (BOPV no. 92, 13 May 2022), which also
#   includes the national holidays.
# - Bizkaia and Bilbao local holidays: Resolución de 15 de septiembre de
#   2022 de la Delegada Territorial de Trabajo de Bizkaia (BOB 27 Sep 2022).
HOLIDAYS = {
    date(2023, 1, 6): "Epiphany",
    date(2023, 4, 6): "Holy Thursday",
    date(2023, 4, 7): "Good Friday",
    date(2023, 4, 10): "Easter Monday",
    date(2023, 5, 1): "Labour Day",
    date(2023, 7, 25): "Santiago Apostol",
    date(2023, 7, 31): "San Ignacio (Bizkaia)",
    date(2023, 8, 15): "Assumption",
    date(2023, 8, 25): "Aste Nagusia Friday (Bilbao)",
    date(2023, 10, 12): "National Day",
    date(2023, 11, 1): "All Saints",
    date(2023, 12, 6): "Constitution Day",
    date(2023, 12, 8): "Immaculate Conception",
    date(2023, 12, 25): "Christmas",
}

# Gap check (rule 4).
DAY_START_HOUR = 7
DAY_END_HOUR = 23
MAX_GAP_MIN = 30

DAYS_FILE = PROCESSED_DIR / "days.csv"


def local_time(epoch_seconds: pd.Series) -> pd.Series:
    """Convert Unix epoch seconds (UTC) to local datetimes in TIMEZONE."""
    return pd.to_datetime(epoch_seconds, unit="s", utc=True).dt.tz_convert(TIMEZONE)


def max_gap_minutes(times: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Longest stretch in [start, end) with no event, in minutes.

    The edges count too: if the first event is at 09:00 and start is
    07:00, there is a 120 minute gap.
    """
    inside = times[(times >= start) & (times < end)].sort_values()
    edges = pd.concat([pd.Series([start]), inside, pd.Series([end])], ignore_index=True)
    return edges.diff().max().total_seconds() / 60


def build_day_table(trips: pd.DataFrame, status: pd.DataFrame, n_stations: int) -> pd.DataFrame:
    """One row per candidate day with the checks and the final decision."""
    trip_t = local_time(trips["time_start"])
    status_t = local_time(status["time"])
    trip_day = trip_t.dt.date
    status_day = status_t.dt.date

    trips_per_day = trip_day.value_counts()
    stations_per_day = status["station_id"].groupby(status_day).nunique()

    rows = []
    for day in pd.date_range(PERIOD_START, PERIOD_END, freq="D"):
        d = day.date()
        start = pd.Timestamp(d, tz=TIMEZONE) + pd.Timedelta(hours=DAY_START_HOUR)
        end = pd.Timestamp(d, tz=TIMEZONE) + pd.Timedelta(hours=DAY_END_HOUR)

        n_trips = int(trips_per_day.get(d, 0))
        n_st = int(stations_per_day.get(d, 0))
        trip_gap = status_gap = None
        if n_trips and n_st:
            trip_gap = max_gap_minutes(trip_t[trip_day == d], start, end)
            status_gap = max_gap_minutes(status_t[status_day == d], start, end)

        if day.weekday() >= 5:
            reason = "weekend"
        elif d in HOLIDAYS:
            reason = f"holiday: {HOLIDAYS[d]}"
        elif not (n_trips and n_st):
            reason = "no data"
        elif max(trip_gap, status_gap) > MAX_GAP_MIN:
            reason = f"gap of {max(trip_gap, status_gap):.0f} min"
        elif n_st < n_stations:
            reason = f"{n_stations - n_st} station(s) closed"
        else:
            reason = ""

        rows.append({
            "date": d,
            "weekday": day.strftime("%a"),
            "trips": n_trips,
            "stations": n_st,
            "max_trip_gap_min": trip_gap,
            "max_status_gap_min": status_gap,
            "selected": reason == "",
            "reason": reason,
        })
    return pd.DataFrame(rows)


def selected_days() -> list[date]:
    """Accepted dates, read from the file saved by running this script."""
    days = pd.read_csv(DAYS_FILE, parse_dates=["date"])
    return [d.date() for d in days.loc[days["selected"], "date"]]


def main() -> None:
    trips = pd.read_parquet(PROCESSED_DIR / "trips.parquet", columns=["time_start"])
    status = pd.read_parquet(PROCESSED_DIR / "station_status.parquet",
                             columns=["station_id", "time"])
    n_stations = len(pd.read_parquet(PROCESSED_DIR / "stations.parquet"))

    table = build_day_table(trips, status, n_stations)
    table.to_csv(DAYS_FILE, index=False, float_format="%.0f")

    weekdays = table[table["reason"] != "weekend"]
    with pd.option_context("display.width", 200):
        print(weekdays.to_string(index=False, na_rep="-", float_format="%.0f"))

    sel = table[table["selected"]]
    print(f"\n{len(sel)} days selected out of {len(table)} "
          f"({len(weekdays)} weekdays). Excluded weekdays by reason:")
    print(weekdays.loc[~weekdays["selected"], "reason"]
          .str.replace(r"\d+ min", "N min", regex=True)
          .str.replace(r"^\d+ station", "N station", regex=True)
          .value_counts().to_string())
    print(f"Trips per selected day: mean {sel['trips'].mean():.0f}, "
          f"min {sel['trips'].min()}, max {sel['trips'].max()}")
    print(f"Saved {DAYS_FILE}")


if __name__ == "__main__":
    main()
