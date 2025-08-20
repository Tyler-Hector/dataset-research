import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

# ==== CONFIG ====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "coord_predictor_delta.keras")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "coord_scaler_delta.pkl")
PLOT_FOLDER = "static/plots"
SEQ_LENGTH = 10
os.makedirs(PLOT_FOLDER, exist_ok=True)

# ==== LOAD MODEL & SCALER ====
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# ==== UTILITIES ====
def create_sequences(data, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

# ==== PREDICTION PIPELINE ====
def predict_from_csv(csv_path, max_plot_steps=600):
    df = pd.read_csv(csv_path)
    required_cols = ['lat', 'lon', 'alt', 'time']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Compute time delta
    df['time_delta'] = df['time'].diff().fillna(0)
    features = ['lat', 'lon', 'alt', 'time_delta']
    data = df[features].values
    data_scaled = scaler.transform(data)

    # Sequences
    X, y_true_scaled = create_sequences(data_scaled)
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_true_scaled)

    # Reconstruct absolute time from deltas
    y_true[:, 3] = np.cumsum(y_true[:, 3]) + df['time'].iloc[0]

    # Linear interpolation for predicted time
    n_steps = len(y_pred)
    y_pred[:, 3] = np.linspace(y_true[0, 3], y_true[-1, 3], n_steps)

    # Error metrics
    horizontal_error = haversine(y_true[:, 0], y_true[:, 1], y_pred[:, 0], y_pred[:, 1])
    alt_error = np.abs(y_true[:, 2] - y_pred[:, 2])
    error_3d = np.sqrt(horizontal_error ** 2 + alt_error ** 2)
    total_flight_length_m = np.sum(haversine(y_true[:-1,0], y_true[:-1,1], y_true[1:,0], y_true[1:,1]))
    avg_3d_error_m = np.mean(error_3d)
    percentage_error = (avg_3d_error_m / total_flight_length_m) * 100 if total_flight_length_m>0 else 0

    metrics = {
        "Total Flight Path Length (km)": round(total_flight_length_m / 1000,3),
        "Average Horizontal Error (m)": round(np.mean(horizontal_error),3),
        "Average Altitude Error (m)": round(np.mean(alt_error),3),
        "Average 3D Error (m)": round(avg_3d_error_m,3),
        "Prediction Error (% of Path)": round(percentage_error,4)
    }

    # Clip sequences for plotting
    plot_steps = min(len(y_true), max_plot_steps)
    y_true_plot = y_true[:plot_steps]
    y_pred_plot = y_pred[:plot_steps]

    # Plot
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(y_true_plot[:,0], y_true_plot[:,1], y_true_plot[:,2], label='Actual')
    ax1.plot(y_pred_plot[:,0], y_pred_plot[:,1], y_pred_plot[:,2], linestyle='--', label='Predicted')
    ax1.set_xlabel('Latitude'); ax1.set_ylabel('Longitude'); ax1.set_zlabel('Altitude')
    ax1.legend(); ax1.set_title("3D Trajectory")

    ax2 = fig.add_subplot(122)
    ax2.plot(y_true_plot[:,3], label='Actual Time')
    ax2.plot(y_pred_plot[:,3], linestyle='--', label='Predicted Time')
    ax2.set_xlabel("Sequence Step"); ax2.set_ylabel("Time"); ax2.legend(); ax2.set_title("Time Prediction")

    plt.tight_layout()
    PLOT_FOLDER = r"C:\Users\debas\traj\trajectory-prediction-research\Data Annotation UI\static\plots"
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    plot_path = os.path.join(PLOT_FOLDER, "prediction_plot.png")

    plt.savefig(plot_path)
    plt.close(fig)

    # Sample predictions
    feature_names = ['lat','lon','alt','time']
    sample_predictions = {
        'true': {feat: y_true[:5,i].round(6).tolist() for i,feat in enumerate(feature_names)},
        'predicted': {feat: y_pred[:5,i].round(6).tolist() for i,feat in enumerate(feature_names)}
    }

    return sample_predictions, metrics, plot_path
