import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import random
import numpy as np

def plot_flight(
    parquet_path="validated_data/validated_cleaned.parquet",
    flight_id=None,
    mode="2D",
    save=False,
    output_dir="plots",
    sample_points=None,
    exaggerate_altitude=True,
    both=False,
    color_by_time=True
):
    """
    Plot a single flight trajectory from the Parquet dataset with enhancements.

    Parameters:
    ----------
    parquet_path : str
        Path to the Parquet dataset.
    flight_id : str or int or None
        ID of the flight to plot. If None, selects a random flight.
    mode : str
        "2D" for lat-lon plot, "3D" for lat-lon-alt plot. Ignored if both=True.
    save : bool
        If True, saves the plot as PNG in the specified directory.
    output_dir : str
        Directory to save plots if save=True.
    sample_points : int or None
        If provided, randomly samples this many points for plotting.
    exaggerate_altitude : bool
        If True, adds padding to z-axis for better visibility.
    both : bool
        If True, generates both 2D and 3D plots in one call.
    color_by_time : bool
        If True, adds a color gradient by time to show flight progression.
    """
    # Load data
    df = pd.read_parquet(parquet_path, columns=["flight_id", "lat", "lon", "alt", "time"])
    df["flight_id"] = df["flight_id"].astype(str)

    # Pick random flight if none provided
    if flight_id is None:
        flight_id = random.choice(df["flight_id"].unique())
        print(f"Randomly selected flight_id: {flight_id}")

    # Filter and sort
    flight_df = df[df["flight_id"] == str(flight_id)].copy()
    if flight_df.empty:
        print(f"No data found for flight_id {flight_id}.")
        return

    flight_df = flight_df.sort_values("time")

    # Downsample if needed
    if sample_points and len(flight_df) > sample_points:
        flight_df = flight_df.sample(n=sample_points).sort_values("time")

    # Prepare color gradient if enabled
    colors = None
    if color_by_time:
        t = np.linspace(0, 1, len(flight_df))
        colors = plt.cm.viridis(t)  # Use a nice color map

    if save:
        os.makedirs(output_dir, exist_ok=True)

    # Helper: Plot 2D
    def plot_2d():
        plt.figure(figsize=(8, 6))
        if color_by_time:
            plt.scatter(flight_df["lon"], flight_df["lat"], c=t, cmap="viridis", s=5)
        else:
            plt.plot(flight_df["lon"], flight_df["lat"], marker="o", markersize=1, linewidth=0.5)
        # Start and end markers
        plt.scatter(flight_df["lon"].iloc[0], flight_df["lat"].iloc[0], c="green", s=40, label="Start")
        plt.scatter(flight_df["lon"].iloc[-1], flight_df["lat"].iloc[-1], c="red", s=40, label="End")
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
        if color_by_time:
            ax.scatter(flight_df["lon"], flight_df["lat"], flight_df["alt"], c=t, cmap="viridis", s=5)
        else:
            ax.scatter(flight_df["lon"], flight_df["lat"], flight_df["alt"], s=3)
        ax.scatter(flight_df["lon"].iloc[0], flight_df["lat"].iloc[0], flight_df["alt"].iloc[0], c="green", s=50)
        ax.scatter(flight_df["lon"].iloc[-1], flight_df["lat"].iloc[-1], flight_df["alt"].iloc[-1], c="red", s=50)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude")
        ax.set_title(f"Flight {flight_id} Trajectory (3D)")
        if exaggerate_altitude:
            ax.set_zlim(flight_df["alt"].min() - 50, flight_df["alt"].max() + 50)
        if save:
            filename = os.path.join(output_dir, f"flight_{flight_id}_3D.png")
            plt.savefig(filename)
            print(f"Saved 3D plot to {filename}")
        else:
            plt.show()

    # Plot based on mode
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

# New: Multi-flight comparison (2D only)
def plot_multiple_flights(
    parquet_path="validated_data/validated_cleaned.parquet",
    flight_ids=None,
    count=3,
    save=False,
    output_dir="plots"
):
    """
    Plot multiple flights on one 2D map for comparison.

    Parameters:
    ----------
    flight_ids : list of str or None
        Specific flight IDs. If None, randomly picks `count` flights.
    count : int
        Number of random flights if flight_ids=None.
    """
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