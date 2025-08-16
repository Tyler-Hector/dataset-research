import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Single flight plotting
def plot_trajectory(
    parquet_path,
    flight_id=None,
    mode="2D",
    color_by="time",
    show_wind=False,
    sample_points=None,
    save=False,
    output_dir="plots"
):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing dataset: {parquet_path}")

    cols = ["flight_id", "lat", "lon", "alt", "time", "temp", "wind_spd", "wind_dir"]
    df = pd.read_parquet(parquet_path, columns=cols)
    df["flight_id"] = df["flight_id"].astype(str)
    df["time"] = pd.to_datetime(df["time"])

    # pick random flight if not given
    if flight_id is None:
        flight_id = random.choice(df["flight_id"].unique())
        print(f"[info] randomly selected flight {flight_id}")

    fdf = df[df["flight_id"] == str(flight_id)].copy()
    fdf = fdf.sort_values("time").drop_duplicates("time")

    if sample_points and len(fdf) > sample_points:
        fdf = fdf.iloc[:: len(fdf) // sample_points]

    # pick colormap
    if color_by == "time":
        colors = fdf["time"].rank(method="first") / len(fdf)
        cmap = "viridis"
    elif color_by == "temp":
        colors = fdf["temp"]
        cmap = "coolwarm"
    elif color_by == "wind_spd":
        colors = fdf["wind_spd"]
        cmap = "plasma"
    else:
        colors, cmap = None, None

    # 2D plot 
    def plot2d():
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(fdf["lon"], fdf["lat"], c=colors, cmap=cmap, s=15)
        if cmap:
            plt.colorbar(sc, label=color_by)
        plt.scatter(fdf["lon"].iloc[0], fdf["lat"].iloc[0], c="green", s=50, marker="o", label="start")
        plt.scatter(fdf["lon"].iloc[-1], fdf["lat"].iloc[-1], c="red", s=50, marker="X", label="end")

        if show_wind:
            step = max(1, len(fdf) // 25)
            plt.quiver(
                fdf["lon"].iloc[::step],
                fdf["lat"].iloc[::step],
                np.cos(np.radians(fdf["wind_dir"].iloc[::step])) * fdf["wind_spd"].iloc[::step],
                np.sin(np.radians(fdf["wind_dir"].iloc[::step])) * fdf["wind_spd"].iloc[::step],
                color="blue",
                scale=200
            )

        plt.title(f"Flight {flight_id} trajectory (2D)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()

        if save:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"flight_{flight_id}_2D.png"))
            plt.close()
        else:
            plt.show()

    # 3D plot 
    def plot3d():
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(fdf["lon"], fdf["lat"], fdf["alt"], c=colors, cmap=cmap, s=10)
        if cmap:
            fig.colorbar(sc, label=color_by)

        ax.scatter(fdf["lon"].iloc[0], fdf["lat"].iloc[0], fdf["alt"].iloc[0], c="green", s=50, marker="o")
        ax.scatter(fdf["lon"].iloc[-1], fdf["lat"].iloc[-1], fdf["alt"].iloc[-1], c="red", s=50, marker="X")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_zlabel("Altitude")
        ax.set_title(f"Flight {flight_id} trajectory (3D)")

        if save:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"flight_{flight_id}_3D.png"))
            plt.close()
        else:
            plt.show()

    if mode == "2D":
        plot2d()
    elif mode == "3D":
        plot3d()
    else:
        raise ValueError("mode must be '2D' or '3D'")

    return flight_id


# Multi-flight comparison
def compare_flights_2d(parquet_path, n_random=4, sample_points=None, save=False, output_dir="plots"):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing dataset: {parquet_path}")

    df = pd.read_parquet(parquet_path, columns=["flight_id", "lat", "lon", "time"])
    df["flight_id"] = df["flight_id"].astype(str)
    flights = random.sample(list(df["flight_id"].unique()), n_random)

    plt.figure(figsize=(10, 8))
    for fid in flights:
        fdf = df[df["flight_id"] == fid].sort_values("time")
        if sample_points and len(fdf) > sample_points:
            fdf = fdf.iloc[:: len(fdf) // sample_points]
        plt.plot(fdf["lon"], fdf["lat"], label=f"flight {fid}")

    plt.title(f"{n_random} random flights (2D)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"multi_flights_{n_random}_2D.png"))
        plt.close()
    else:
        plt.show()

# Script entry point
if __name__ == "__main__":
    parquet_path = r"C:\Users\flyin\OneDrive\Documents\GitHub\dataset-research-SCAT\data\validated\validated_cleaned_FINAL.parquet"

    fid = plot_trajectory(parquet_path, mode="2D", color_by="time", show_wind=True, sample_points=600, save=True)
    plot_trajectory(parquet_path, flight_id=fid, mode="3D", color_by="temp", sample_points=600, save=True)
    compare_flights_2d(parquet_path, n_random=3, sample_points=500, save=True)

    print("Plots saved in ./plots/")