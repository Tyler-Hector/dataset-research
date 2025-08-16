# SCAT Flight Trajectory Prediction Pipeline

This repository implements an end-to-end pipeline for analyzing and forecasting aircraft trajectories using the SCAT dataset. It includes preprocessing, weather integration, visualization, annotation, and a GRU-based deep learning model.

## Folder Structure
- `raw_data/` – Original SCAT JSON/CSV (ignored by git)  
- `validated_data/` – Cleaned + weather-integrated dataset (.parquet)  
- `models/` – Trained model weights (e.g. `best.pt`)  
- `plots/` – Trajectory visualizations  
- `error_gallery/` – Failure case samples  
- `data/` – Placeholder folders (`.gitkeep` only)  

## Dataset
- Source: [Mendeley SCAT Dataset](https://data.mendeley.com/datasets/8yn985bwz5/1)  
- ~5.5 GB compressed, >100k flights  
- Includes flight plans, surveillance, predicted trajectories, and weather/airspace data  

## Pipeline
1. **Preprocessing** – Clean, validate, and merge raw flight data  
2. **Splits** – Train/Val/Test IDs stored as `.txt` files  
3. **Visualization** – 2D & 3D plots with weather features  
4. **Annotation Tool** – Streamlit app with keyboard shortcuts (V/I/S)  
5. **Weather Integration** – Adds temperature, wind speed, and direction  
6. **Feature Engineering** – Velocity, bearing, normalization  
7. **GRU Forecast Model** – Sequence-to-sequence network for predicting future positions  
8. **Evaluation** – Predicted vs actual plots, failure cases in `error_gallery/`  

## Usage
```bash
# Clone & setup
git clone https://github.com/<your-org>/dataset-research-SCAT.git
cd dataset-research-SCAT
python -m venv venv
venv\Scripts\activate   # (Windows) 
# or: source venv/bin/activate (macOS/Linux)
pip install -r requirements.txt

# Visualize a random trajectory
python test_plot.py

# Launch the annotation tool
streamlit run annotation_tool.py

# Train the GRU model
python train_forecast_model.py --epochs 20

* All large datasets are excluded by `.gitignore`
* Folder structure is retained using `.gitkeep` files to ensure reproducibility
* Do not place real data inside version-controlled folders unless explicitly allowed
