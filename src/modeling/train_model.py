import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


def train_model(input_path: str):
    print("Loading ABT from:", input_path)

    df = pd.read_csv(input_path)
    print("ABT shape:", df.shape)

    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("F1-score:", f1)

    # AUC only if both classes exist
    if len(model.classes_) > 1:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print("AUC:", auc)
    else:
        auc = "Not defined"
        print("AUC: Not defined (single-class prediction)")

    # Save metrics table
    os.makedirs("tables", exist_ok=True)
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "F1-score", "AUC"],
        "Value": [accuracy, f1, auc]
    })
    metrics_df.to_excel("tables/RQ3_Table1.xlsx", index=False)

    # Feature importance figure
    os.makedirs("figures", exist_ok=True)
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    importances.head(10).plot(kind="bar")
    plt.title("Top 10 Feature Importances")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("figures/RQ1_Fig1.pdf")
    plt.close()

    # Save trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_model.pkl")

    print("Model, figures, and tables saved successfully.")


if __name__ == "__main__":
    train_model("data/processed/abt.csv")
