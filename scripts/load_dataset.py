import pandas as pd
import os

def load_splits(parquet_path, split_dir, only=None, columns=None):
    """
    Load train, val, and test splits from the validated dataset.

    Args:
        parquet_path (str): Path to the validated_cleaned.parquet file
        split_dir (str): Directory containing train_ids.txt, val_ids.txt, test_ids.txt
        only (str, optional): Return only one split ('train', 'val', 'test')
        columns (list, optional): List of columns to load

    Returns:
        train_df, val_df, test_df OR single DataFrame (if only is set)
    """
    print(f"Loading dataset from {parquet_path}...")

    df = pd.read_parquet(parquet_path, engine="fastparquet")

    # Optional: select specific columns
    if columns:
        df = df[columns]

    # Load split IDs
    def read_ids(filename):
        path = os.path.join(split_dir, filename)
        with open(path, "r") as f:
            return set(line.strip() for line in f)

    train_ids = read_ids("train_ids.txt")
    val_ids = read_ids("val_ids.txt")
    test_ids = read_ids("test_ids.txt")

    # Create splits (convert flight_id to string for consistency
    df["flight_id"] = df["flight_id"].astype(str)
    train_df = df[df["flight_id"].isin(train_ids)]
    val_df = df[df["flight_id"].isin(val_ids)]
    test_df = df[df["flight_id"].isin(test_ids)]

    print(f"✔ Loaded: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")

    if only == "train":
        return train_df
    elif only == "val":
        return val_df
    elif only == "test":
        return test_df
    else:
        return train_df, val_df, test_df