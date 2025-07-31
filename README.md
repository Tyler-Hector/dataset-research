# Files and Folders Excluded by .gitignore
- All virtual environments: venv/, .env/
- Python cache: __pycache__/
- Large data files: *.csv, *.zip, *.parquet
- Data folders: raw_data/, processed_data/, validated_data/, temp_data/, plots/
- System files: .DS_Store

# SCAT Flight Trajectory Analysis Pipeline

## Dataset Overview
The SCAT dataset contains historical flight trajectory data and related information used for air traffic management and research.

Contents:
- Flight Plans (FPL): Departure and arrival airports, routes, altitudes.
- Clearances: ATC instructions like climb and descent.
- Surveillance Data (Plots): Position reports (latitude, longitude, altitude) recorded every few seconds.
- Predicted Trajectories: Estimated flight paths used by ATC.
- Weather & Airspace Data: Forecast data and sector information.

Organization:
SCAT dataset.zip
    ├── scat20161015_20161021.zip
    ├── scat20161112_20161118.zip
    ├── ...
    └── scat20170916_20170922.zip

Each inner ZIP contains thousands of JSON files, each representing one flight.

Size:
- ~5.5 GB compressed
- >100k JSON files across 13 weeks of data.

Source: https://data.mendeley.com/datasets/8yn985bwz5/1

---

## Problem Statement
Develop a trajectory analysis pipeline that predicts the future positions of aircraft to improve defense system situational awareness and support early threat detection.

---

## Steps Completed

### 1. Merge Raw Files
- Combined all weekly flight JSON/CSV files into a single merged file.

### 2. Clean & Validate Flights
- Removed rows with:
  - NaN values (lat, lon, alt, time)
  - Invalid coordinates (lat outside -90 to 90, lon outside -180 to 180)
  - Negative altitude
- Dropped duplicate entries.
- Removed flights with fewer than 10 data points.

### 3. Train/Val/Test Split
- Generated `train_ids.txt`, `val_ids.txt`, and `test_ids.txt` for reproducible splits.
- Old approach of creating 160k CSVs was abandoned in favor of a single Parquet file.

### 4. Switch to Single File
- Combined all cleaned flights into `validated_data/validated_cleaned.parquet`.
- Organized splits via text files for easy filtering.

### 5. Load Utility
- Built `load_dataset.py`:
  - Function: `load_splits()`
  - Loads the main Parquet file and returns train, validation, and test splits.

### 6. Visualization Tools
- Built `plot_trajectory.py`:
  - Supports 2D (lat vs lon) and 3D (lat, lon, alt) plots.
  - Added color gradient for time progression, start/end markers.
  - Option to save plots in `plots/`.

### 6.5 Visualization Test Script
- Created `test_plot.py` to:
  - Plot a specific flight (2D & 3D).
  - Plot a random flight.
  - Generate a multi-flight comparison (2D).
  - Save all plots in `plots/`.

### 7 Visualization Test Script
- Built an interactive tool to label flight trajectories as Valid or Invalid:
  -Loads validated_cleaned.parquet.
  -3D visualization with Plotly (Lat, Lon, Alt).
  -Keyboard shortcuts: V (Valid), I (Invalid), S (Skip).
  -Saves annotations to annotations.csv every 10 labels.
  -Tracks progress in the sidebar

---

### **8. Weather Data Integration**
- Enhanced preprocessing to include weather features from `grib_meteo.json`:
  - `temp`: Temperature at the aircraft's position.
  - `wind_spd`: Wind speed.
  - `wind_dir`: Wind direction.
- Matching strategy:
  - Used `cKDTree` for fast nearest-neighbor search between flight coordinates and weather grid points.
  - Added weather columns to every trajectory point in the merged file.
- Updated cleaning script:
  - Validates presence of weather columns before creating `validated_cleaned.parquet`.
  - Splits remain consistent (IDs unchanged).
- Final validated dataset includes: ['flight_id', 'time', 'lat', 'lon', 'alt', 'temp', 'wind_spd', 'wind_dir']


## Setup
Clone the repo and create a virtual environment:

```bash
git clone <your-repo-url>
cd dataset-research-SCAT

python -m venv venv
# Activate:
& venv\Scripts\Activate.ps1   (Windows)
source venv/bin/activate      (macOS/Linux)

pip install -r requirements.txt
