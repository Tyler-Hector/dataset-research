import pandas as pd
import os
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_flight_data(
    input_path: str = "data/validated_data/validated_cleaned.csv", 
    output_parquet: Optional[str] = "data/validated_data/validated_cleaned.parquet",
    output_csv: Optional[str] = None,
    time_col: str = "time",
    id_col: str = "flight_id"
) -> pd.DataFrame:
    """
    Clean flight trajectory data by:
    - Removing duplicate timestamps
    - Ensuring numeric types
    - Sorting by time
    
    Args:
        input_path: Path to input CSV/Parquet.
        output_parquet: Path to save cleaned Parquet.
        output_csv: Path to save cleaned CSV.
    
    Returns:
        Cleaned DataFrame.
    """
    # Read input
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    
    # Clean data
    logger.info(f"Initial rows: {len(df):,}")
    
    # Convert time to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Remove duplicates
    df = df.sort_values([id_col, time_col])
    df = df.drop_duplicates(subset=[id_col, time_col])
    logger.info(f"After deduplication: {len(df):,}")
    
    # Ensure numeric cols
    numeric_cols = ["lat", "lon", "alt", "temp", "wind_spd", "wind_dir"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Save outputs
    if output_parquet:
        df.to_parquet(output_parquet, engine="pyarrow")
        logger.info(f"Saved cleaned data to {output_parquet}")
    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved cleaned data to {output_csv}")
    
    return df

# Example usage:
if __name__ == "__main__":
    print("=== Script started ===")
    df = clean_flight_data(
    input_path="data/validated_data/validated_cleaned.csv",  
    output_parquet="data/validated_data/validated_cleaned_FINAL.parquet",
    output_csv="data/validated_data/validated_cleaned_FINAL.csv"
    )
    print(f"=== Cleaned {len(df)} rows ===")