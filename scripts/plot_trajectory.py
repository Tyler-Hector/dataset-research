import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import random

def plot_flight(
    parquet_path="validated_data/validated_cleaned.parquet",
    flight_id=None,
    mode="2D",
    save=False,
    output_dir="plots",
    sample_points=None,
    exaggerate_altitude=True,
    both=False,
    color_by="time",  # options: "time", "temp", "wind_spd"
    show_wind=False   # overlay wind arrows in 2D
):
    """
    Plot a single flight trajectory with weather features.

    Parameters:
    ----------
    parquet_path : str
        Path to the validated Parquet dataset.
    flight_id : str or None
        ID of the flight to plot. If None, picks a random flight.
    mode : str
        "2D" or "3D". Ignored if both=True.
    save : bool
        Save the plot if True.
    output_dir : str
        Directory for saving plots.
    sample_points : int or None
        Randomly sample N points from the flight for plotting.
    exaggerate_altitude : bool
        Add extra z-axis padding for 3D plot.
    both : bool
        Generate both 2D and 3D plots.
    color_by : str
        "time", "temp", or "wind_spd".
    show_wind : bool
        If True, overlays wind direction arrows in 2D.
    """

    # Load data
    df = pd.read_parquet(parquet_path, columns=["flight_id", "lat", "lon", "alt", "time", "temp", "wind_spd", "wind_dir"])
    df["flight_id"] = df["flight_id"].astype(str)

    # Pick random flight if not provided
    if flight_id is None:
        flight_id = random.choice(df["flight_id"].unique())
        print(f"Randomly selected flight_id: {flight_id}")

    # Filter for the flight
    flight_df = df[df["flight_id"] == str(flight_id)].copy()
    if flight_df.empty:
        print(f"No data found for flight_id {flight_id}")
        return

    flight_df = flight_df.sort_values("time")

    # Downsample if needed
    if sample_points and len(flight_df) > sample_points:
        flight_df = flight_df.sample(n=sample_points).sort_values("time")

    # Prepare color mapping
    colors, cmap = None, None
    if color_by == "temp":
        colors = flight_df["temp"]
        cmap = "coolwarm"
    elif color_by == "wind_spd":
        colors = flight_df["wind_spd"]
        cmap = "plasma"
    elif color_by == "time":
        t = np.linspace(0, 1, len(flight_df))
        colors = t
        cmap = "viridis"

    if save:
        os.makedirs(output_dir, exist_ok=True)

    # Helper: Plot 2D
    def plot_2d():
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(flight_df["lon"], flight_df["lat"], c=colors, cmap=cmap, s=10)
        plt.scatter(flight_df["lon"].iloc[0], flight_df["lat"].iloc[0], c="green", s=50, label="Start")
        plt.scatter(flight_df["lon"].iloc[-1], flight_df["lat"].iloc[-1], c="red", s=50, label="End")

        if show_wind:
            step = max(1, len(flight_df)//50)  # avoid clutter
            for i in range(0, len(flight_df), step):
                u = flight_df["wind_spd"].iloc[i] * np.cos(np.radians(flight_df["wind_dir"].iloc[i]))
                v = flight_df["wind_spd"].iloc[i] * np.sin(np.radians(flight_df["wind_dir"].iloc[i]))
                plt.arrow(flight_df["lon"].iloc[i], flight_df["lat"].iloc[i], u*0.01, v*0.01, head_width=0.05, color="blue")

        plt.colorbar(sc, label=color_by)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"Flight {flight_id} Trajectory (2D)")
        plt.legend()
        if save:
            filename = os.path.join(output_dir, f"flight_{flight_id}_2D.png")
            plt.savefig(filename)
            print(f"Saved 2D plot to {filename}")
        else:
            plt.show()

    # Helper: Plot 3D
    def plot_3d():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(flight_df["lon"], flight_df["lat"], flight_df["alt"], c=colors, cmap=cmap, s=8)
        ax.scatter(flight_df["lon"].iloc[0], flight_df["lat"].iloc[0], flight_df["alt"].iloc[0], c="green", s=50)
        ax.scatter(flight_df["lon"].iloc[-1], flight_df["lat"].iloc[-1], flight_df["alt"].iloc[-1], c="red", s=50)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude")
        ax.set_title(f"Flight {flight_id} Trajectory (3D)")
        if exaggerate_altitude:
            ax.set_zlim(flight_df["alt"].min() - 50, flight_df["alt"].max() + 50)
        fig.colorbar(sc, ax=ax, label=color_by)
        if save:
            filename = os.path.join(output_dir, f"flight_{flight_id}_3D.png")
            plt.savefig(filename)
            print(f"Saved 3D plot to {filename}")
        else:
            plt.show()

    # Generate plots
    if both:
        plot_2d()
        plot_3d()
    else:
        if mode == "2D":
            plot_2d()
        elif mode == "3D":
            plot_3d()
        else:
            raise ValueError("mode must be '2D' or '3D'")


# Multi-flight comparison with weather info (2D only)
def plot_multiple_flights(
    parquet_path="validated_data/validated_cleaned.parquet",
    flight_ids=None,
    count=3,
    save=False,
    output_dir="plots"
):
    df = pd.read_parquet(parquet_path, columns=["flight_id", "lat", "lon"])
    df["flight_id"] = df["flight_id"].astype(str)

    if flight_ids is None:
        flight_ids = random.sample(list(df["flight_id"].unique()), count)

    plt.figure(figsize=(8, 6))
    for fid in flight_ids:
        flight_df = df[df["flight_id"] == fid]
        plt.plot(flight_df["lon"], flight_df["lat"], label=f"Flight {fid}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Multiple Flight Trajectories (2D)")
    plt.legend()
    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "multi_flight_2D.png")
        plt.savefig(filename)
        print(f"Saved multi-flight plot to {filename}")
    else:
        plt.show()