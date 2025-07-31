import pandas as pd
import os

def load_splits(
    parquet_path="validated_data/validated_cleaned.parquet",
    split_dir="validated_data",
    only=None,
    columns=None
):
    """
    Load train, val, and test splits from the validated dataset.

    Parameters
    ----------
    parquet_path : str
        Path to the validated dataset (.parquet file).
    split_dir : str
        Directory containing train_ids.txt, val_ids.txt, test_ids.txt.
    only : str or None
        If "train", "val", or "test", returns only that subset.
        If None, returns all three splits as (train_df, val_df, test_df).
    columns : list or None
        If provided, selects only these columns from the DataFrame.

    Returns
    -------
    - If only=None: (train_df, val_df, test_df)
    - If only="train"/"val"/"test": single DataFrame
    """
    print(f"Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # select specific columns
    if columns:
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in dataset: {missing_cols}")
        df = df[columns]

    # load split IDs
    def load_ids(file_name):
        path = os.path.join(split_dir, file_name)
        with open(path, "r") as f:
            return set(line.strip() for line in f if line.strip())

    train_ids = load_ids("train_ids.txt")
    val_ids = load_ids("val_ids.txt")
    test_ids = load_ids("test_ids.txt")

    df["flight_id"] = df["flight_id"].astype(str)

    if only == "train":
        return df[df["flight_id"].isin(train_ids)]
    elif only == "val":
        return df[df["flight_id"].isin(val_ids)]
    elif only == "test":
        return df[df["flight_id"].isin(test_ids)]
    else:
        train_df = df[df["flight_id"].isin(train_ids)]
        val_df = df[df["flight_id"].isin(val_ids)]
        test_df = df[df["flight_id"].isin(test_ids)]
        return train_df, val_df, test_df