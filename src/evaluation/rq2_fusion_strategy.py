import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def run_rq2_experiment(input_path: str):
    print("Running RQ2: Fusion strategy comparison")

    df = pd.read_csv(input_path)

    y = df["Machine failure"]

    # -------------------------
    # Case 1: Raw sensor features only
    # -------------------------
    raw_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X_raw = df[raw_features]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    model_raw = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model_raw.fit(Xr_train, yr_train)
    yr_pred = model_raw.predict(Xr_test)

    acc_raw = accuracy_score(yr_test, yr_pred)
    f1_raw = f1_score(yr_test, yr_pred, zero_division=0)

    # -------------------------
    # Case 2: Engineered (fused) features
    # -------------------------
    X_fused = df.drop("Machine failure", axis=1)

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        X_fused, y, test_size=0.2, random_state=42, stratify=y
    )

    model_fused = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model_fused.fit(Xf_train, yf_train)
    yf_pred = model_fused.predict(Xf_test)

    acc_fused = accuracy_score(yf_test, yf_pred)
    f1_fused = f1_score(yf_test, yf_pred, zero_division=0)

    # -------------------------
    # Save table
    # -------------------------
    os.makedirs("tables", exist_ok=True)

    rq2_table = pd.DataFrame({
        "Fusion Strategy": ["Raw Features Only", "Temporal + Feature-Level Fusion"],
        "Accuracy": [acc_raw, acc_fused],
        "F1-score": [f1_raw, f1_fused]
    })

    rq2_table.to_excel("tables/RQ2_Table1.xlsx", index=False)

    # -------------------------
    # Save figure
    # -------------------------
    os.makedirs("figures", exist_ok=True)

    rq2_table.set_index("Fusion Strategy")[["Accuracy", "F1-score"]].plot(
        kind="bar",
        figsize=(7, 5)
    )

    plt.title("RQ2: Sensor Fusion Strategy Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figures/RQ2_Fig1.pdf")
    plt.close()

    print("RQ2 figure and table saved successfully.")
    print(rq2_table)


if __name__ == "__main__":
    run_rq2_experiment("data/processed/abt.csv")
