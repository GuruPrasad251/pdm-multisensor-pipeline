import pandas as pd
import os
from scipy.stats import zscore

def build_features(input_path: str, output_path: str):
    print("Loading cleaned data from:", input_path)

    df = pd.read_csv(input_path)
    print("Data loaded. Shape:", df.shape)

    df = df.sort_values(by="UDI").reset_index(drop=True)

    df["Torque_roll_mean"] = df["Torque [Nm]"].rolling(window=5).mean()
    df["Speed_roll_mean"] = df["Rotational speed [rpm]"].rolling(window=5).mean()
    df["Temp_roll_mean"] = df["Process temperature [K]"].rolling(window=5).mean()

    df["Torque_z"] = zscore(df["Torque [Nm]"])
    df["Speed_z"] = zscore(df["Rotational speed [rpm]"])
    df["Temp_z"] = zscore(df["Process temperature [K]"])

    df = df.dropna().reset_index(drop=True)

    abt_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Torque_roll_mean",
        "Speed_roll_mean",
        "Temp_roll_mean",
        "Torque_z",
        "Speed_z",
        "Temp_z",
        "Machine failure"
    ]

    abt = df[abt_columns]

    print("Saving ABT to:", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    abt.to_csv(output_path, index=False)

    print("ABT saved successfully. Shape:", abt.shape)
    return abt


if __name__ == "__main__":
    build_features(
        input_path="data/cleaned/ai4i2020_cleaned.csv",
        output_path="data/processed/abt.csv"
    )
