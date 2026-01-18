from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime


# -------------------------
# Default DAG arguments
# -------------------------
default_args = {
    "owner": "student",
    "depends_on_past": False,
    "retries": 0
}


# -------------------------
# Define DAG
# -------------------------
with DAG(
    dag_id="pdm_multisensor_pipeline",
    default_args=default_args,
    description="Predictive Maintenance Data Engineering Pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    # -------------------------
    # Data ingestion
    # -------------------------
    ingest_data = BashOperator(
        task_id="ingest_data",
        bash_command="echo 'Raw data already ingested from Kaggle'"
    )

    # -------------------------
    # Data cleaning
    # -------------------------
    clean_data = BashOperator(
        task_id="clean_data",
        bash_command="python src/data_cleaning/clean_data.py"
    )

    # -------------------------
    # Feature engineering
    # -------------------------
    build_features = BashOperator(
        task_id="build_features",
        bash_command="python src/feature_engineering/build_features.py"
    )

    # -------------------------
    # Model training
    # -------------------------
    train_model = BashOperator(
        task_id="train_model",
        bash_command="python src/modeling/train_model.py"
    )

    # -------------------------
    # RQ1 analysis
    # -------------------------
    rq1_analysis = BashOperator(
        task_id="rq1_single_vs_fused",
        bash_command="python src/evaluation/rq1_single_vs_fused.py"
    )

    # -------------------------
    # RQ2 analysis
    # -------------------------
    rq2_analysis = BashOperator(
        task_id="rq2_fusion_strategy",
        bash_command="python src/evaluation/rq2_fusion_strategy.py"
    )

    # -------------------------
    # RQ3 analysis
    # -------------------------
    rq3_analysis = BashOperator(
        task_id="rq3_model_comparison",
        bash_command="python src/evaluation/rq3_model_comparison.py"
    )

    # -------------------------
    # RQ4 anomaly detection
    # -------------------------
    rq4_analysis = BashOperator(
        task_id="rq4_anomaly_detection",
        bash_command="python src/evaluation/anomaly_detection.py"
    )

    # -------------------------
    # RQ5 economic analysis
    # -------------------------
    rq5_analysis = BashOperator(
        task_id="rq5_economic_analysis",
        bash_command="python src/evaluation/rq5_economic_analysis.py"
    )

    # -------------------------
    # Task dependencies
    # -------------------------
    ingest_data >> clean_data >> build_features >> train_model
    train_model >> rq1_analysis >> rq2_analysis >> rq3_analysis
    rq3_analysis >> rq4_analysis >> rq5_analysis
