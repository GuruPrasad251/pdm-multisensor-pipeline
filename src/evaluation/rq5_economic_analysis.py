import pandas as pd
import os
import matplotlib.pyplot as plt


def run_rq5_analysis(input_path: str):
    print("Running RQ5: Economic and reliability analysis")

    df = pd.read_csv(input_path)

    # -------------------------
    # Basic assumptions
    # -------------------------
    cost_per_failure = 1000  # Euros
    detection_rate = 0.3     # Assume 30% of failures are avoided

    # -------------------------
    # Failure statistics
    # -------------------------
    total_failures = df["Machine failure"].sum()

    failures_avoided = int(total_failures * detection_rate)
    failures_remaining = total_failures - failures_avoided

    cost_before = total_failures * cost_per_failure
    cost_after = failures_remaining * cost_per_failure
    cost_saved = cost_before - cost_after

    # -------------------------
    # Save table
    # -------------------------
    os.makedirs("tables", exist_ok=True)

    rq5_table = pd.DataFrame({
        "Metric": [
            "Total Failures",
            "Failures Avoided",
            "Failures Remaining",
            "Cost Before PdM (€)",
            "Cost After PdM (€)",
            "Estimated Cost Savings (€)"
        ],
        "Value": [
            total_failures,
            failures_avoided,
            failures_remaining,
            cost_before,
            cost_after,
            cost_saved
        ]
    })

    rq5_table.to_excel("tables/RQ5_Table1.xlsx", index=False)

    # -------------------------
    # Save figure
    # -------------------------
    os.makedirs("figures", exist_ok=True)

    cost_df = pd.DataFrame({
        "Scenario": ["Before PdM", "After PdM"],
        "Cost (€)": [cost_before, cost_after]
    })

    cost_df.set_index("Scenario").plot(
        kind="bar",
        figsize=(6, 4),
        legend=False
    )

    plt.title("RQ5: Estimated Maintenance Cost Reduction")
    plt.ylabel("Cost (€)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("figures/RQ5_Fig1.pdf")
    plt.close()

    print("RQ5 figure and table saved successfully.")
    print(rq5_table)


if __name__ == "__main__":
    run_rq5_analysis("data/raw/ai4i2020_snapshot.csv")