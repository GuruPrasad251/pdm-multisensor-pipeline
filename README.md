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
```
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

1.Results of RQ1

RQ1 compares machine failure prediction performance using single-sensor inputs
versus fused multi-sensor data. The results show that overall accuracy remains
similar between the two approaches, while the F1-score is higher for the fused
sensor model. This behavior is primarily due to the strong class imbalance in the
dataset, where normal operating conditions significantly outnumber failure events.

Accuracy is dominated by the correct classification of normal samples, which both
single-sensor and fused-sensor models can predict reliably. As a result, accuracy
does not change substantially when additional sensor information is introduced.
However, the fused-sensor model benefits from complementary information across
multiple sensors, improving its ability to detect rare failure events. This leads to
higher recall and precision for the failure class, which is reflected in an improved
F1-score.

These results indicate that sensor fusion enhances the practical usefulness of the
model by improving failure detection capability, even when overall accuracy appears
unchanged. This aligns with real-world predictive maintenance objectives, where
detecting rare failures is more important than maximizing accuracy.


2. Explanation for Similar Results in RQ2 and RQ3

The results for RQ2 and RQ3 appear similar because both experiments rely on the
same underlying supervised learning model (Random Forest) and the same processed
tabular dataset. In RQ2, the comparison focuses on raw sensor features versus
temporally engineered and feature-level fused sensor representations. In RQ3, the
comparison evaluates reduced feature sets versus full feature sets using the same
model architecture.

Since the dataset is highly imbalanced and dominated by normal operating samples,
overall accuracy remains relatively stable across both experiments. Improvements
from temporal fusion (RQ2) and increased feature richness (RQ3) primarily enhance
the detection of rare failure events rather than overall correctness, which explains
why F1-score improves while accuracy remains similar. As a result, the numerical
patterns observed in RQ2 and RQ3 are comparable, reflecting the same underlying
data characteristics and evaluation metrics rather than experimental redundancy.

Additionally, deep learning architectures were not implemented due to the absence
of explicit Remaining Useful Life (RUL) labels and the tabular nature of the dataset.
Therefore, RQ3 evaluates model capacity and feature richness as a proxy for model
expressiveness, which naturally produces trends similar to feature-level fusion in
RQ2.

3. Explanation for High Number of Anomalies in RQ4

The anomaly detection results in RQ4 show a relatively large number of detected
anomalies because the Isolation Forest algorithm is an unsupervised method that
identifies deviations from the dominant normal operating patterns, not confirmed
failures. In predictive maintenance, many anomalous observations represent early
or mild deviations, noise, or transient operating changes rather than immediate
machine failures.

This behavior is expected and desirable in early-warning systems, where sensitivity
to deviations is prioritized over precision. The detected anomalies should be
interpreted as potential precursors to degradation rather than direct failure events.
Only a subset of these anomalies eventually lead to actual machine failure, which is
why anomaly counts are higher than failure counts.

4. Overall Interpretation

The observed results across RQ2, RQ3, and RQ4 are consistent with real-world
predictive maintenance systems. Temporal and feature-level fusion improve failure
detection capability without significantly altering overall accuracy, while anomaly
detection intentionally flags a broad range of abnormal behaviors to support early
intervention. These findings validate the robustness and practical relevance of the
proposed predictive maintenance pipeline.


5. Explanation ofResults for RQ5

RQ5 evaluates the economic benefit and reliability improvement achieved by deploying
the predictive maintenance pipeline. The analysis shows a reduction in the number of
failures and a corresponding decrease in estimated maintenance costs after the
introduction of the PdM system. This improvement is quantified using a rule-based
economic model applied to the raw operational dataset, which preserves true failure
frequencies.

The economic analysis is based on conservative assumptions, including a fixed cost
per failure event and a partial early-detection rate. Rather than assuming perfect
failure prevention, the model estimates that only a subset of failures can be avoided
through early intervention. This results in realistic and defensible cost savings
rather than inflated estimates.

The reduction in failure count represents an improvement in system reliability,
while the decrease in total maintenance cost demonstrates the economic value of the
pipeline. Although simplified, this analysis effectively illustrates how data-driven
predictive maintenance can support decision-making by reducing unplanned downtime
and maintenance expenses in industrial environments.
