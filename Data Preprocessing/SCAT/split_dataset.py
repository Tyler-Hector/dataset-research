import pandas as pd
import os
from sklearn.model_selection import train_test_split

# config
merged_file = r"C:\Users\flyin\OneDrive\Documents\GitHub\dataset-research-SCAT\processed_data\SCAT_cleaned_full.csv"
output_dir = r"C:\Users\flyin\OneDrive\Documents\GitHub\dataset-research-SCAT\validated_data"
output_parquet = os.path.join(output_dir, "validated_cleaned.parquet")
output_csv = os.path.join(output_dir, "validated_cleaned.csv")
min_points = 10
random_seed = 42
save_csv = True

# make sure output folder exists
os.makedirs(output_dir, exist_ok=True)

# load merged file
print(f"Loading merged dataset from {merged_file}...")
df = pd.read_csv(merged_file)

print(f"Loaded {len(df):,} rows.")
print("Columns:", list(df.columns))

# verify required columns
required_cols = {"flight_id", "time", "lat", "lon", "alt"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing required columns. Found: {df.columns}")

# fail if weather columns missing
if not {"temp", "wind_spd", "wind_dir"}.issubset(df.columns):
    raise ValueError("Weather columns missing in merged file! Aborting to prevent overwriting.")
else:
    print("Weather columns found.")

# clean & validate flights
print("\nCleaning flights...")
cleaned_flights = []
valid_ids = []
invalid_ids = []

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

# combine everything
combined_df = pd.concat(cleaned_flights, ignore_index=True)
print(f"Combined {len(valid_ids):,} valid flights into {len(combined_df):,} rows.")
print(f"Skipped {len(invalid_ids):,} flights (too few points).")

# save validated dataset
combined_df.to_parquet(output_parquet, index=False)
print(f"Saved validated data to {output_parquet}")

if save_csv:
    combined_df.to_csv(output_csv, index=False)
    print(f"Saved CSV copy to {output_csv}")

# split by flight_id
print("\nSplitting flights into Train/Val/Test...")
train_ids, temp_ids = train_test_split(valid_ids, test_size=0.3, random_state=random_seed)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=random_seed)

# save ID files
with open(os.path.join(output_dir, "train_ids.txt"), "w") as f:
    f.write("\n".join(map(str, train_ids)))
with open(os.path.join(output_dir, "val_ids.txt"), "w") as f:
    f.write("\n".join(map(str, val_ids)))
with open(os.path.join(output_dir, "test_ids.txt"), "w") as f:
    f.write("\n".join(map(str, test_ids)))

print("\nSplit complete:")
print(f"Train: {len(train_ids):,}, Val: {len(val_ids):,}, Test: {len(test_ids):,}")
print(f"Total validated flights: {len(valid_ids):,}")
print(f"Files saved in: {output_dir}")