# SCAT Flight Trajectory Prediction Pipeline

This repository implements an end-to-end machine learning pipeline to process, analyze, and forecast aircraft flight trajectories using the SCAT dataset. It includes preprocessing, visualization, annotation tools, weather integration, and a GRU-based deep learning model.

## Folder Structure

* `raw_data/` - Original SCAT JSON/CSV files (ignored by git)
* `validated_data/` - Cleaned + weather-integrated dataset (.parquet format)
* `models/` - Contains trained model weights (e.g., `best.pt`)
* `plots/` - Output from trajectory visualization tools
* `error_gallery/` - Samples of failure cases from validation
* `data/` - Placeholder for GitHub structure (contains only `.gitkeep`)

Note: Data folders are visible in the repo structure, but actual datasets are excluded using `.gitignore`.

## Dataset Overview

The SCAT dataset contains:

* Flight plans, clearances, and surveillance data (lat, lon, alt, time)
* Predicted trajectories from ATC systems
* Weather and airspace sector data

Source: [https://data.mendeley.com/datasets/8yn985bwz5/1](https://data.mendeley.com/datasets/8yn985bwz5/1)
Size: \~5.5 GB compressed, >100,000 JSON files

## Problem Statement

Predict future flight positions using current trajectory and weather features. This supports situational awareness in defense and air traffic systems.

## Pipeline Overview

### 1. Merge Raw Files

* Combined all SCAT ZIP archives into one large file.

### 2. Clean and Validate Flights

* Removed rows with NaNs, invalid coordinates, negative altitude
* Dropped duplicates and flights with <10 data points

### 3. Train/Val/Test Split

* Generated `train_ids.txt`, `val_ids.txt`, `test_ids.txt`
* Used consistent split across experiments

### 4. Switch to Single File

* Combined cleaned flights into `validated_cleaned.parquet`
* Retained split files for filtering

### 5. Data Loading Utility

* `load_dataset.py` with `load_splits()` function
* Loads train, val, test splits from the Parquet file

### 6. Trajectory Visualization

* `plot_trajectory.py` supports 2D and 3D plots
* Color gradient by time, start/end markers
* Saved to `plots/`

### 6.5. Visualization Test Script

* `test_plot.py` for quick plotting sanity checks
* Random samples, multi-flight comparisons

### 7. Annotation Tool

* Streamlit-based labeling tool with Plotly 3D
* Keyboard shortcuts: V (valid), I (invalid), S (skip)
* Saves labels to `annotations.csv`

### 8. Weather Integration

* Merged `grib_meteo.json` data using KDTree
* Added `temp`, `wind_spd`, `wind_dir` to each point
* Updates reflected in the validated dataset

### 9. EDA and Weather Visuals

* Quiver plots for wind
* Temperature color maps
* 3D weather-aware trajectory visualization
* Missing data statistics

### 10. Feature Engineering

* Computed velocity and bearing
* Normalized lat, lon, alt, velocity, bearing

### 11. Sequence Dataset Creation

* Custom PyTorch dataset
* Inputs: 20-step sequence, Outputs: next 5 steps
* Configurable sequence length, stride, etc.

### 12. GRU Forecast Model

* Sequence-to-sequence GRU network
* Configurable: hidden size, layers, learning rate
* Trained using MSE loss and Adam optimizer

### 13. Training Loop

* Logs training and validation loss
* Runs on GPU if available
* Best model saved to `models/best.pt`

### 14. Evaluation and Error Analysis

* Denormalized predictions for interpretability
* Plotted predicted vs actual trajectories
* Failure cases saved to `error_gallery/`

## Setup Instructions

```bash
git clone https://github.com/<your-org>/dataset-research-SCAT.git
cd dataset-research-SCAT

python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Common Commands

```bash
# Visualize a random trajectory
python test_plot.py --mode=random

# Launch the annotation tool
streamlit run annotation_tool.py

# Train the model
python train_forecast_model.py --epochs=20

# Evaluate model performance
python eval_model.py
```

## Notes

* All large datasets are excluded by `.gitignore`
* Folder structure is retained using `.gitkeep` files to ensure reproducibility
* Do not place real data inside version-controlled folders unless explicitly allowed
