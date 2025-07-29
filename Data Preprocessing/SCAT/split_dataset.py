import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Config
merged_file = r"C:\Users\flyin\OneDrive\Documents\GitHub\dataset-research\processed_data\SCAT_cleaned_full.csv"
output_dir = r"C:\Users\flyin\OneDrive\Documents\GitHub\dataset-research\validated_data"
output_parquet = os.path.join(output_dir, "validated_cleaned.parquet")
output_csv = os.path.join(output_dir, "validated_cleaned.csv")  # optional
min_points = 10
random_seed = 42
save_csv = True  # Set to True if you want CSV in addition to Parquet

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

print(f"Loading merged file from {merged_file}...")
df = pd.read_csv(merged_file)

required_cols = {"flight_id", "time", "lat", "lon", "alt"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing required columns. Found: {df.columns}")

print(f"Loaded {len(df)} rows. Cleaning and combining into one file...")

cleaned_flights = []
valid_ids = []
invalid_ids = []

# Clean by grouping per flight_id
for flight_id, flight_df in df.groupby("flight_id"):
    flight_df = flight_df.sort_values("time")
    flight_df = flight_df.dropna(subset=["lat", "lon", "alt", "time"])
    flight_df = flight_df.drop_duplicates()
    flight_df = flight_df[
        (flight_df["lat"].between(-90, 90)) &
        (flight_df["lon"].between(-180, 180)) &
        (flight_df["alt"] >= 0)
    ]
    if len(flight_df) < min_points:
        invalid_ids.append(flight_id)
        continue
    cleaned_flights.append(flight_df)
    valid_ids.append(flight_id)

# Combine into one big DataFrame
combined_df = pd.concat(cleaned_flights, ignore_index=True)
print(f"Combined {len(valid_ids)} flights into {len(combined_df)} rows.")

# Save as Parquet (fast and compressed)
combined_df.to_parquet(output_parquet, index=False)
print(f"Saved cleaned data to {output_parquet}")

# Save as CSV
if save_csv:
    combined_df.to_csv(output_csv, index=False)
    print(f"Saved CSV copy to {output_csv}")

# Train/Val/Test split by flight_id
train_ids, temp_ids = train_test_split(valid_ids, test_size=0.3, random_state=random_seed)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=random_seed)

# Save ID files
with open(os.path.join(output_dir, "train_ids.txt"), "w") as f:
    f.write("\n".join(map(str, train_ids)))
with open(os.path.join(output_dir, "val_ids.txt"), "w") as f:
    f.write("\n".join(map(str, val_ids)))
with open(os.path.join(output_dir, "test_ids.txt"), "w") as f:
    f.write("\n".join(map(str, test_ids)))

print("Train/Val/Test split complete:")
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
print(f"Skipped {len(invalid_ids)} flights (too few points).")