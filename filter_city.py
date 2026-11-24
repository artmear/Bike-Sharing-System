"""
Filter CSV files to include only data from a specific city.
"""

import pandas as pd
from pathlib import Path
import sys

if len(sys.argv) < 2:
    print("Usage: python filter_city.py <city_id> [--sample]")
    sys.exit(1)

city_id = int(sys.argv[1])
use_sample = "--sample" in sys.argv

data_source = "sample" if use_sample else "full"
data_dir = Path("Data") / data_source

# Get city name
cities_df = pd.read_csv(data_dir / "cities.csv")
city_row = cities_df[cities_df["id"] == city_id]
if city_row.empty:
    print(f"Error: City ID {city_id} not found")
    sys.exit(1)
city_name = city_row.iloc[0]["name"]

output_dir = Path("Data/filtered") / city_name
output_dir.mkdir(parents=True, exist_ok=True)

files_to_filter = {
    "cities.csv": "id",
    "stations.csv": "city_id",
    "trips.csv": "city_id",
    "city_areas.csv": "city_id",
}

reference_files = ["bikes.csv", "bike_types.csv"]

print(f"Filtering data for city: {city_name} (ID: {city_id})")
print(f"Source: {data_source}")
print(f"Output: {output_dir}\n")

# First, filter and save files with city_id
station_ids = None
for filename in files_to_filter.keys():
    input_path = data_dir / filename
    output_path = output_dir / filename
    
    df = pd.read_csv(input_path)

    print(f"Processing {filename} (Rows: {len(df):,})...")
    if filename == "cities.csv":
        df_filtered = df[df["id"] == city_id]
    else:
        df_filtered = df[df["city_id"] == city_id]
    
    # Keep track of station IDs for filtering station_status
    if filename == "stations.csv":
        station_ids = set(df_filtered["id"].values)
    
    df_filtered.to_csv(output_path, index=False)
    print(f"[OK] {filename}: {len(df_filtered):,} rows")

# Copy reference files
for filename in reference_files:
    input_path = data_dir / filename
    output_path = output_dir / filename
    
    df = pd.read_csv(input_path)
    df.to_csv(output_path, index=False)
    print(f"[OK] {filename}: {len(df):,} rows")

# Filter station_status by station_ids in the city
input_path = data_dir / "station_status.csv"
output_path = output_dir / "station_status.csv"
df = pd.read_csv(input_path)
df_filtered = df[df["station_id"].isin(station_ids)]
df_filtered.to_csv(output_path, index=False)
print(f"[OK] station_status.csv: {len(df_filtered):,} rows")

print(f"\nDone!")
