# AI-Powered Intrusion Detection Solution

## Project Overview

This project presents an AI-powered intrusion detection solution using machine learning. The objective of this project is to classify network traffic as either normal or malicious. The project is developed as a proof-of-concept Network Intrusion Detection System using the UNSW-NB15 dataset.

The solution follows a complete machine learning workflow, including dataset loading, preprocessing, model training, model evaluation, visualization, and dashboard-based demonstration.

## Course Information

- Course: Information Security
- CLO: CLO 4 - Create solutions to real-life scenarios using different security related tools
- Project Type: Individual Assignment
- Submission Type: GitHub Repository and MS Word Report

## Real-World Scenario

Organizations face continuous cyber threats such as denial-of-service attacks, reconnaissance attempts, exploitation attempts, unauthorized access, and abnormal network behavior. Traditional security tools like firewalls are important, but they may not be sufficient to detect all suspicious activities.

In this project, machine learning is used to support a Network Intrusion Detection System by automatically classifying network traffic as normal or malicious. This helps security analysts identify possible threats more efficiently.

## Dataset

The UNSW-NB15 dataset was used in this project because it contains realistic normal and malicious network traffic records. It is suitable for developing and evaluating intrusion detection models.

Dataset source:

Kaggle Dataset: dhoogla/unswnb15

The dataset contains the following files:

- UNSW_NB15_training-set.parquet
- UNSW_NB15_testing-set.parquet

## Tools and Technologies Used

- Python
- Google Colab
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Streamlit
- UNSW-NB15 Dataset

## Machine Learning Model

The Random Forest Classifier was used in this project. Random Forest was selected because it performs well on structured classification problems and can handle complex relationships between network traffic features.

The target variable used in this project was:

- label = 0: Normal traffic
- label = 1: Attack or malicious traffic

The column `attack_cat` was removed from the input features to avoid data leakage because it directly represents the attack category.

## Data Preprocessing Steps

The following preprocessing steps were performed:

1. Loaded the training and testing datasets.
2. Checked dataset shape, columns, data types, and class distribution.
3. Removed target and leakage columns from input features.
4. Identified numerical and categorical columns.
5. Applied median imputation and standard scaling to numerical features.
6. Applied most frequent imputation and one-hot encoding to categorical features.
7. Trained a Random Forest classifier using the processed training data.
8. Evaluated the model using the unseen testing dataset.

## Model Results

The model achieved the following performance on the testing dataset:

| Metric | Result |
|---|---:|
| Accuracy | 87.13% |
| Precision | 82.72% |
| Recall | 96.87% |
| F1 Score | 89.23% |

## Confusion Matrix

The confusion matrix results were:

| Category | Count |
|---|---:|
| True Normal | 27,825 |
| Normal classified as Attack | 9,175 |
| Attack classified as Normal | 1,420 |
| True Attack | 43,912 |

## Security Analysis

The model achieved a high recall score of 96.87%, which is very important in intrusion detection. Recall shows how many actual attacks were correctly detected by the model. A low recall would be dangerous because it means real malicious traffic may pass through the system undetected.

The model also produced some false positives, where normal traffic was classified as attack traffic. This may increase the workload of security analysts, but it is generally less dangerous than missing real attacks. In cybersecurity, false negatives are more serious than false positives because missed attacks can result in unauthorized access, data loss, or system compromise.

Overall, the model is effective as a proof-of-concept machine learning based intrusion detection solution.

## Dashboard

A Streamlit dashboard was also developed to demonstrate the project in an interactive way.

The dashboard includes:

- Dataset overview
- Data explorer
- Model training
- Accuracy, precision, recall, and F1-score
- Confusion matrix
- ROC curve
- Feature importance
- Batch prediction
- Simulated live detection

The simulated live detection module automatically passes test traffic records into the trained model one by one and displays whether each record is normal or malicious.

## How to Run the Notebook

Open the Jupyter Notebook or Google Colab notebook and run all cells step by step.

Dataset can be downloaded from Kaggle using:

```python
!pip install kaggle
!kaggle datasets download -d dhoogla/unswnb15 -p /content/dataset
```

After downloading, unzip the dataset and load the parquet files:

```python
import pandas as pd

train_df = pd.read_parquet("/content/dataset/extracted/UNSW_NB15_training-set.parquet")
test_df = pd.read_parquet("/content/dataset/extracted/UNSW_NB15_testing-set.parquet")
```

## How to Run the Dashboard

First install the required packages:

```bash
pip install -r requirements.txt
```

Then run the Streamlit dashboard:

```bash
streamlit run app.py
```

## Repository Contents

This repository contains:

```text
IDS_ML_Solution.ipynb
app.py
README.md
requirements.txt
model_results_summary.csv
classification_report.txt
security_analysis.txt
confusion_matrix.png
feature_importance.png
feature_importance.csv
```

## Future Enhancements

This project can be improved in the following ways:

1. Testing additional machine learning algorithms such as XGBoost, SVM, or Neural Networks.
2. Improving feature engineering to reduce false positives.
3. Tuning hyperparameters for better model performance.
4. Integrating live packet capture tools such as Zeek, Suricata, Wireshark, or CICFlowMeter.
5. Deploying the model as a real-time intrusion detection service.

## Conclusion

This project demonstrates that machine learning can support Network Intrusion Detection Systems by automatically identifying malicious network traffic. The high recall score shows that the model is effective in detecting most attacks. Therefore, the model can be used as a security decision-support tool and can be further improved for real-time threat detection.