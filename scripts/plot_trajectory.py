import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import random
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_flight(
    parquet_path: str = "data/validated_data/validated_cleaned.parquet",  # ← Fixed path
    flight_id: Optional[str] = None,
    mode: str = "2D",
    save: bool = False,
    output_dir: str = "plots",
    sample_points: Optional[int] = None,
    exaggerate_altitude: bool = True,
    both: bool = False,
    color_by: str = "time",
    show_wind: bool = False
):
    """Plot flight trajectory with weather features."""
    # Data loading with validation
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    cols = ["flight_id", "lat", "lon", "alt", "time", "temp", "wind_spd", "wind_dir"]
    df = pd.read_parquet(parquet_path, columns=cols)
    df["flight_id"] = df["flight_id"].astype(str)

    # Select flight
    if flight_id is None:
        flight_id = random.choice(df["flight_id"].unique())
        logger.info(f"Randomly selected flight_id: {flight_id}")

    flight_df = df[df["flight_id"] == str(flight_id)].copy()
    if flight_df.empty:
        raise ValueError(f"No data for flight_id {flight_id}")

    flight_df = flight_df.sort_values("time").drop_duplicates(subset=["time"])

    # Downsample
    if sample_points and len(flight_df) > sample_points:
        flight_df = flight_df.sample(n=sample_points).sort_values("time")

    # Color mapping
    color_data = {
        "time": (flight_df["time"].rank(method="first") / len(flight_df), "viridis"),
        "temp": (flight_df["temp"], "coolwarm"),
        "wind_spd": (flight_df["wind_spd"], "plasma")
    }.get(color_by, (None, None))

    colors, cmap = color_data

    # Plotting
    def plot_2d():
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(flight_df["lon"], flight_df["lat"], c=colors, cmap=cmap, s=20)
        plt.colorbar(label=color_by)
        
        if show_wind:
            bbox_width = flight_df["lon"].max() - flight_df["lon"].min()
            scale = 0.05 * bbox_width / flight_df["wind_spd"].max()
            for _, row in flight_df.iloc[::max(1, len(flight_df)//20)].iterrows():
                u = row["wind_spd"] * np.cos(np.radians(row["wind_dir"]))
                v = row["wind_spd"] * np.sin(np.radians(row["wind_dir"]))
                plt.arrow(row["lon"], row["lat"], u*scale, v*scale, color="blue", width=0.0001)

        plt.title(f"Flight {flight_id} Trajectory (2D)")
        if save:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/flight_{flight_id}_2D.png")
        else:
            plt.show()

    def plot_3d():
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(flight_df["lon"], flight_df["lat"], flight_df["alt"], c=colors, cmap=cmap, s=10)
        fig.colorbar(sc, label=color_by)
        
        if exaggerate_altitude:
            z_pad = 0.1 * (flight_df["alt"].max() - flight_df["alt"].min())
            ax.set_zlim(flight_df["alt"].min() - z_pad, flight_df["alt"].max() + z_pad)

        ax.set_title(f"Flight {flight_id} Trajectory (3D)")
        if save:
            plt.savefig(f"{output_dir}/flight_{flight_id}_3D.png")
        else:
            plt.show()

    if both:
        plot_2d()
        plot_3d()
    else:
        {"2D": plot_2d, "3D": plot_3d}.get(mode, lambda: None)()