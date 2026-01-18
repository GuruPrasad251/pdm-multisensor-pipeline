import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def run_rq1_experiment(input_path: str):
    print("Running RQ1: Single-sensor vs Fused-sensor comparison (RAW data)")

    # -------------------------
    # Load RAW operational data
    # -------------------------
    df = pd.read_csv(input_path)

    # -------------------------
    # Drop identifier columns
    # -------------------------
    df = df.drop(columns=["UDI", "Product ID", "Type"], errors="ignore")

    y = df["Machine failure"]

    # -------------------------
    # Model 1: Single Sensor (Torque only)
    # -------------------------
    X_single = df[["Torque [Nm]"]]

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_single, y, test_size=0.2, random_state=42, stratify=y
    )

    model_single = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model_single.fit(Xs_train, ys_train)
    ys_pred = model_single.predict(Xs_test)

    acc_single = accuracy_score(ys_test, ys_pred)
    f1_single = f1_score(ys_test, ys_pred, zero_division=0)

    # -------------------------
    # Model 2: Fused Sensors (All numeric sensors)
    # -------------------------
    X_fused = df.drop("Machine failure", axis=1)

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        X_fused, y, test_size=0.2, random_state=42, stratify=y
    )

    model_fused = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model_fused.fit(Xf_train, yf_train)
    yf_pred = model_fused.predict(Xf_test)

    acc_fused = accuracy_score(yf_test, yf_pred)
    f1_fused = f1_score(yf_test, yf_pred, zero_division=0)

    # -------------------------
    # Save RQ1 table
    # -------------------------
    os.makedirs("tables", exist_ok=True)

    rq1_table = pd.DataFrame({
        "Model": ["Single Sensor (Torque)", "Fused Sensors (All Sensors)"],
        "Accuracy": [acc_single, acc_fused],
        "F1-score": [f1_single, f1_fused]
    })

    rq1_table.to_excel("tables/RQ1_Table1.xlsx", index=False)

    print("RQ1 table saved as tables/RQ1_Table1.xlsx")
    print(rq1_table)

    # -------------------------
    # Feature Importance (VALID NOW)
    # -------------------------
    importances = model_fused.feature_importances_

    feature_names = X_fused.columns
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    # -------------------------
    # Save RQ1 figure
    # -------------------------
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )
    plt.gca().invert_yaxis()
    plt.title("RQ1: Top 10 Feature Importances (RAW Sensor Fusion)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("figures/RQ1_Fig1.pdf")
    plt.close()

    print("RQ1 figure saved as figures/RQ1_Fig1.pdf")
    print(importance_df)


if __name__ == "__main__":
    run_rq1_experiment("data/raw/ai4i2020_snapshot.csv")
