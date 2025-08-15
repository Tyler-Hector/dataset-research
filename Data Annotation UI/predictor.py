import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# ==== CONFIG ====
MODEL_PATH = r"C:\Users\debas\AIRTRAJ\dataset-research\Data Annotation UI\coord_predictor.keras"
SCALER_PATH = r"C:\Users\debas\AIRTRAJ\dataset-research\Data Annotation UI\coord_scaler.pkl"
PLOT_FOLDER = "static/plots"
SEQ_LENGTH = 10
os.makedirs(PLOT_FOLDER, exist_ok=True)

# ==== LOAD MODEL & SCALER ====
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# ==== UTILITIES ====
def create_sequences(data, seq_length=SEQ_LENGTH):
    """Create overlapping sequences from data."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def enforce_strict_increments(times):
    """Ensure time is strictly monotonic with smooth increments."""
    times = times.copy()
    for i in range(1, len(times)):
        if times[i] < times[i - 1]:
            increment = times[i - 1] - times[i - 2] if i > 1 else 0
            times[i] = times[i - 1] + max(increment, 0)
    return times

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance in meters."""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ==== PREDICTION PIPELINE ====
def predict_from_csv(csv_path):
    # --- Load CSV ---
    df = pd.read_csv(csv_path)
    required_cols = ['lat', 'lon', 'alt', 'time']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # --- Scale data ---
    data = df[required_cols].values
    data_scaled = scaler.transform(data)

    # --- Create sequences ---
    X, y_true_scaled = create_sequences(data_scaled)

    # --- Model prediction ---
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_true_scaled)

    # --- Save raw predicted times for debugging ---
    raw_pred_time = y_pred[:, 3].copy()

    # --- Correct times ---
    y_pred[:, 3] = enforce_strict_increments(y_pred[:, 3])

    # --- Error calculations ---
    horizontal_error = haversine(y_true[:, 0], y_true[:, 1],
                                 y_pred[:, 0], y_pred[:, 1])
    alt_error = np.abs(y_true[:, 2] - y_pred[:, 2])
    error_3d = np.sqrt(horizontal_error**2 + alt_error**2)

    segment_distances = haversine(y_true[:-1, 0], y_true[:-1, 1],
                                  y_true[1:, 0], y_true[1:, 1])
    total_flight_length_m = np.sum(segment_distances)
    avg_3d_error_m = np.mean(error_3d)
    percentage_error = (avg_3d_error_m / total_flight_length_m) * 100 if total_flight_length_m > 0 else 0

    metrics = {
        "Total Flight Path Length (km)": round(total_flight_length_m / 1000, 3),
        "Average Horizontal Error (m)": round(np.mean(horizontal_error), 3),
        "Average Altitude Error (m)": round(np.mean(alt_error), 3),
        "Average 3D Error (m)": round(avg_3d_error_m, 3),
        "Prediction Error (% of Path)": round(percentage_error, 4),
    }

    # --- Plot results ---
    fig = plt.figure(figsize=(12, 6))

    # 3D Trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(y_true[:, 0], y_true[:, 1], y_true[:, 2], label='Actual', alpha=0.7)
    ax1.plot(y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], label='Predicted', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Latitude')
    ax1.set_ylabel('Longitude')
    ax1.set_zlabel('Altitude')
    ax1.set_title('3D Trajectory: Actual vs Predicted')
    ax1.legend()

    # Time Series
    ax2 = fig.add_subplot(122)
    ax2.plot(y_true[:, 3], label='Actual Time')
    ax2.plot(y_pred[:, 3], label='Predicted Time (Corrected)', linestyle='--')
    ax2.set_xlabel('Sequence Step')
    ax2.set_ylabel('Time')
    ax2.set_title('Time Prediction')
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(PLOT_FOLDER, "prediction_plot.png")
    plt.savefig(plot_path)
    plt.close(fig)

    # --- Prepare sample predictions ---
    feature_names = ['lat', 'lon', 'alt', 'time']
    sample_predictions = {
        'true': {feat: y_true[:5, i].round(6).tolist() for i, feat in enumerate(feature_names)},
        'predicted': {feat: y_pred[:5, i].round(6).tolist() for i, feat in enumerate(feature_names)},
        'raw_predicted_time': raw_pred_time[:5].round(6).tolist()
    }

    return sample_predictions, metrics

# ==== CLI ENTRY ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict trajectory from CSV using trained Keras model.")
    parser.add_argument("--csv", required=True, help="Path to input CSV file containing lat, lon, alt, time columns.")
    args = parser.parse_args()

    samples, metrics = predict_from_csv(args.csv)

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== Sample Predictions ===")
    print(samples)
