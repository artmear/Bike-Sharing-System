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

