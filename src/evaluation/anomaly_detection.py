import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


def run_anomaly_detection(input_path: str):
    print("Loading ABT for anomaly detection:", input_path)

    df = pd.read_csv(input_path)

    # Separate features only (no target)
    X = df.drop("Machine failure", axis=1)

    # Train Isolation Forest
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    df["anomaly_score"] = iso.fit_predict(X)

    # Convert output: -1 = anomaly, 1 = normal
    df["anomaly_label"] = df["anomaly_score"].map({1: "Normal", -1: "Anomaly"})

    # Save table
    os.makedirs("tables", exist_ok=True)
    anomaly_table = df["anomaly_label"].value_counts().reset_index()
    anomaly_table.columns = ["Label", "Count"]
    anomaly_table.to_excel("tables/RQ4_Table1.xlsx", index=False)

    # Plot anomaly distribution
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6, 4))
    anomaly_table.set_index("Label")["Count"].plot(kind="bar")
    plt.title("Anomaly Detection Results")
    plt.ylabel("Number of Observations")
    plt.tight_layout()
    plt.savefig("figures/RQ4_Fig1.pdf")
    plt.close()

    print("Anomaly detection completed. Outputs saved.")


if __name__ == "__main__":
    run_anomaly_detection("data/processed/abt.csv")
