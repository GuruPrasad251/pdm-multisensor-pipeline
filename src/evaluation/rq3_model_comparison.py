import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def run_rq3_experiment(input_path: str):
    print("Running RQ3: Model capacity comparison")

    df = pd.read_csv(input_path)

    y = df["Machine failure"]

    # -------------------------
    # Model 1: Random Forest (Reduced feature set)
    # -------------------------
    reduced_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    X_reduced = df[reduced_features]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_reduced = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    rf_reduced.fit(Xr_train, yr_train)
    yr_pred = rf_reduced.predict(Xr_test)

    acc_reduced = accuracy_score(yr_test, yr_pred)
    f1_reduced = f1_score(yr_test, yr_pred, zero_division=0)

    # -------------------------
    # Model 2: Random Forest (Full feature set)
    # -------------------------
    X_full = df.drop("Machine failure", axis=1)

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_full = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    rf_full.fit(Xf_train, yf_train)
    yf_pred = rf_full.predict(Xf_test)

    acc_full = accuracy_score(yf_test, yf_pred)
    f1_full = f1_score(yf_test, yf_pred, zero_division=0)

    # -------------------------
    # Create figure
    # -------------------------
    os.makedirs("figures", exist_ok=True)

    comparison_df = pd.DataFrame({
        "Model": ["RF (Reduced Features)", "RF (Full Feature Set)"],
        "Accuracy": [acc_reduced, acc_full],
        "F1-score": [f1_reduced, f1_full]
    })

    comparison_df.set_index("Model").plot(
        kind="bar",
        figsize=(7, 5)
    )

    plt.title("RQ3: Effect of Model Capacity on Performance")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figures/RQ3_Fig1.pdf")
    plt.close()

    print("RQ3 figure saved as figures/RQ3_Fig1.pdf")
    print(comparison_df)


if __name__ == "__main__":
    run_rq3_experiment("data/processed/abt.csv")
