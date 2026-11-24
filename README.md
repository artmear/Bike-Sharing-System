# Bike-Sharing System

## Installation
 
### Prerequisites
 
- **Git**
- **Git LFS**, because the full dataset (`Data/full/dataset.zip`, about 880 MB) is stored with Git LFS
- **Python 3**
- About **6 GB of free disk space** (the zip plus the extracted CSV files)

Install Git LFS:
 
| System | Command |
|---|---|
| Ubuntu / Debian | `sudo apt install git-lfs` |
| macOS | `brew install git-lfs` |
| Windows | Already included in [Git for Windows](https://git-scm.com/download/win) |

Once installed, enable it (once per computer):
 
```bash
git lfs install
```
 
> This step must be done **before** cloning. If you clone without Git LFS enabled, you get a small text file of a few bytes instead of the zip.
 
### 1. Clone the repository
 
```bash
git clone https://github.com/artmear/Bike-Sharing-System.git
cd Bike-Sharing-System
```
 
Check that the zip was downloaded correctly:
 
```bash
git lfs ls-files
```
 
`Data/full/dataset.zip` should be listed, and the file should be about 880 MB.
 
### 2. Set up the Python environment
 
```bash
python3 -m venv .venv
```
 
Activate the environment:
 
```bash
# Linux / macOS
source .venv/bin/activate
 
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```
 
Install the dependencies:
 
```bash
pip install -r requirements.txt
```
 
### 3. Extract the dataset
 
This command works on any operating system:
 
```bash
python -m zipfile -e Data/full/dataset.zip Data/full/
```
 
When it finishes, the `Data/` folder should look like this:
 
```
Data/
├── LICENSE
├── full/
│   ├── dataset.zip
│   ├── bikes.csv
│   ├── bike_types.csv
│   ├── cities.csv
│   ├── city_areas.csv
│   ├── stations.csv
│   ├── station_status.csv   (about 38 M rows, 1.7 GiB)
│   └── trips.csv            (about 25 M rows, 2.7 GiB)
└── sample/                  (first 1,000 rows of each file, for quick tests)
```
 
> The CSV files in `Data/full/` are listed in `.gitignore` and **must not be added to Git**: they are too large for GitHub. To test code quickly, use the files in `Data/sample/`.

## Data model & file schemas

All timestamps are **Unix epoch seconds in UTC**; join via `cities.timezone` to convert to local times. All coordinates are WGS‑84 (EPSG:4326). 
<details>
<summary><code>trips.csv</code> (25 210 627 rows)</summary>

| column                                     | unit | description                                   |
| ------------------------------------------ | ---- | --------------------------------------------- |
| bike\_id                                   | –    | Bike identifier; `bikes.id`                   |
| city\_id                                   | –    | Rental city; `cities.id`                      |
| time\_start                                | s    | Rental start (UTC)                            |
| lon\_start, lat\_start, lon\_end, lat\_end | °    | Start & end coordinates WGS-84                |
| station\_id\_start, station\_id\_end       | –    | Station IDs, `NULL` if free‑floating          |
| battery\_start, battery\_end               | %    | State of charge (e‑bikes only)                |
| duration                                   | s    | Trip duration                                 |
| distance                                   | m    | Great‑circle distance (PostGIS `ST_Distance`) |

</details>

<details>
<summary><code>cities.csv</code> (267 rows)</summary>

| column                     | unit | description                                         |
| -------------------------- | ---- | --------------------------------------------------- |
| id                         | –    | Primary key                                         |
| name                       | –    | Name of the bike‑sharing scheme                     |
| lat, lon                   | °    | Approximate city center                             |
| timezone                   | –    | IANA timezone string (e.g. `Europe/Paris`)          |
| country                    | –    | ISO‑3166 alpha‑3 country code                       |
| return\_to\_official\_only | bool | `true` if bikes must be left at a (virtual) station |

</details>

<details>
<summary><code>city_areas.csv</code> (267 rows)</summary>

| column        | description                     |
| ------------- | ------------------------------- |
| city\_id      | `cities.id`                     |
| geom\_ewkb    | Operational area estimated via DBSCAN in **EWKB** |
| geom\_ewkt    | … in **EWKT**                   |
| geom\_geojson | … in **GeoJSON**                |

</details>

<details>
<summary><code>bike_types.csv</code> (121 rows)</summary>

| column            | unit | description                 |
| ----------------- | ---- | --------------------------- |
| id                | –    | Technical bike type         |
| vehicle\_image    | –    | URL of representative image |
| name              | –    | Commercial name             |
| description       | –    | Free‑text description       |
| form\_factor      | –    | `regular` / `cargo`         |
| rider\_capacity   | –    | Typical seats (1, 2, …)     |
| propulsion\_type  | –    | `human` / `electric`        |
| max\_range        | m    | Nominal electric range      |
| battery\_capacity | Wh   | Battery energy              |

</details>

<details>
<summary><code>bikes.csv</code> (88 444 rows)</summary>

| column         | unit | description                     |
| -------------- | ---- | ------------------------------- |
| id             | –    | Bike identifier                 |
| bike\_type\_id | –    | Technical type; `bike_types.id` |
| computer\_id   | –    | On‑board computer identifier    |

</details>

<details>
<summary><code>stations.csv</code> (13 192 rows)</summary>

| column         | unit | description                     |
| -------------- | ---- | ------------------------------- |
| id             | –    | Station identifier              |
| city\_id       | –    | `cities.id`                     |
| name           | –    | Human‑readable label            |
| app\_number    | –    | Number shown to users           |
| terminal\_type | –    | Hardware generation (12 values) |
| place\_type    | –    | Unknown (23 observed values)    |
| bike\_racks    | –    | Regular parking positions       |
| special\_racks | –    | Charging racks                  |
| lon, lat       | °    | Location                        |

</details>

<details>
<summary><code>station_status.csv</code> (38 279 885 rows)</summary>

| column                     | unit | description                  |
| -------------------------- | ---- | ---------------------------- |
| station\_id                | –    | `stations.id`                |
| time                       | s    | Snapshot timestamp (UTC)     |
| bikes                      | –    | Total bikes currently docked |
| booked\_bikes              | –    | Bikes reserved by users      |
| bikes\_available\_to\_rent | –    | Bikes ready for rental       |
| free\_racks                | –    | Empty regular docks          |
| free\_special\_racks       | –    | Empty charging docks         |
| maintenance                | bool | `true` = offline             |

</details>