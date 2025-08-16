import pandas as pd
import os
from typing import Optional, Union, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_splits(
    parquet_path: str = "data/validated_data/validated_cleaned.parquet",  
    split_dir: str = "data/validated_data", 
    only: Optional[str] = None,
    columns: Optional[List[str]] = None
) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """
    Load train/val/test splits from a Parquet file with predefined flight IDs.
    
    Args:
        parquet_path: Path to the Parquet file.
        split_dir: Directory containing {train/val/test}_ids.txt.
        only: If "train", "val", or "test", returns only that split.
        columns: Optional list of columns to load (saves memory).
    
    Returns:
        DataFrames for requested splits.
    
    Example:
        >>> train_df, val_df, test_df = load_splits("data.parquet", "splits")
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # Load data with fallback engine support
    try:
        df = pd.read_parquet(parquet_path, columns=columns, engine="fastparquet")
    except ImportError:
        logger.warning("fastparquet not found, falling back to pyarrow")
        df = pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")

    # Load split IDs
    def read_ids(filename: str) -> set:
        path = os.path.join(split_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file not found: {path}")
        with open(path, "r") as f:
            return {line.strip() for line in f}

    train_ids = read_ids("train_ids.txt")
    val_ids = read_ids("val_ids.txt")
    test_ids = read_ids("test_ids.txt")

    # Filter splits (convert IDs to strings for consistency)
    df["flight_id"] = df["flight_id"].astype(str)
    train_df = df[df["flight_id"].isin(train_ids)]
    val_df = df[df["flight_id"].isin(val_ids)]
    test_df = df[df["flight_id"].isin(test_ids)]

    logger.info(f"Loaded splits - Train: {len(train_df):,} rows | Val: {len(val_df):,} | Test: {len(test_df):,}")

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }.get(only, (train_df, val_df, test_df))