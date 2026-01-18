import pandas as pd
import os

def clean_data(input_path: str, output_path: str):
    """
    Clean and preprocess predictive maintenance data.
    """

    df = pd.read_csv(input_path)

    # Trim column names
    df.columns = df.columns.str.strip()

    # Handle missing values (median imputation)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Convert categorical column
    df["Type"] = df["Type"].astype("category")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    # Outlier removal using IQR
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    df = df[
        ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
          (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    ]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned data
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    clean_data(
        input_path="data/raw/ai4i2020_snapshot.csv",
        output_path="data/cleaned/ai4i2020_cleaned.csv"
    )
