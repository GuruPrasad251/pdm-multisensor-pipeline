import pandas as pd
import os

def ingest_data(input_path: str, output_path: str):
    """
    Ingest raw CSV data and save a snapshot.
    """

    # Load raw data
    df = pd.read_csv(input_path)

    # Basic profiling (for logs / reproducibility)
    print("Dataset shape:", df.shape)
    print("Column names:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save raw snapshot
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    ingest_data(
        input_path="data/raw/ai4i2020.csv",
        output_path="data/raw/ai4i2020_snapshot.csv"
    )
