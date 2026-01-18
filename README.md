# Predictive Maintenance – Multi-Sensor Data Engineering Pipeline

## Project Overview
This project implements an end-to-end data engineering pipeline for predictive
maintenance (PdM) using multi-sensor machine data. The goal is to demonstrate
how sensor fusion, feature engineering, and machine learning can improve failure
prediction, anomaly detection, and economic decision-making.

The project was developed as part of a Data Engineering course and focuses on
pipeline design, reproducibility, and clear technical implementation rather than
production deployment.

---

## Dataset
- **Name:** AI4I 2020 Predictive Maintenance Dataset
- **Source:** Kaggle  
- **Raw file:** `ai4i2020.csv`

The dataset contains sensor readings such as temperature, torque, rotational
speed, tool wear, and machine failure labels.

---

## Research Questions
**RQ1:** How accurately can machine failure be predicted using fused sensor data compared to single-sensor analysis?  
**RQ2:** What is the optimal sensor fusion strategy for enhancing prediction lead time?  
**RQ3:** To what extent do advanced models outperform simpler baseline approaches in failure prediction?  
**RQ4:** How effective are anomaly detection techniques for identifying precursors to degradation?  
**RQ5:** What is the economic benefit and reliability improvement achieved by deploying the predictive maintenance pipeline?

Each research question is supported by at least one auto-generated figure and one
auto-generated table.

---
##Project Structure

## How to Run the Pipeline (Without Airflow)

From the project root directory, run the following scripts in order:

```bash
python src/data_cleaning/clean_data.py
python src/feature_engineering/build_features.py
python src/modeling/train_model.py
python src/evaluation/rq1_single_vs_fused.py
python src/evaluation/rq2_fusion_strategy.py
python src/evaluation/rq3_model_comparison.py
python src/evaluation/anomaly_detection.py
python src/evaluation/rq5_economic_analysis.py

## Airflow DAG

The project includes an Apache Airflow DAG (`pdm_pipeline_dag.py`) that orchestrates
the complete data engineering workflow, including data cleaning, feature
engineering, model training, and evaluation for all research questions.

The DAG is designed for conceptual execution and reflects real-world pipeline
sequencing using clearly defined task dependencies.

## Reproducibility

- All figures and tables are generated directly from code
- No manual editing of results is performed
- A consistent folder structure is maintained
- Fixed random seeds are used where applicable

## Notes

Economic analysis (RQ5) is performed using the raw operational dataset to preserve
true failure frequencies, while machine learning models are trained on processed
and engineered data.

## Project Structure
