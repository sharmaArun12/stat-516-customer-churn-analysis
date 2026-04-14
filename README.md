# Customer Churn Prediction: Model Comparison

## Group Members

* Arun Sharma
* Umar Mohammed Yousuf
* Alex Vukovic

## Overview

This project looks at customer churn prediction using three different modeling approaches: Logistic Regression, XGBoost, and an FT-Transformer. The idea is to compare how these models behave on the same dataset and to see whether more complex models actually provide better results.

Since the data is structured (tabular), it also gives a good setting to compare traditional methods with newer deep learning approaches.

## Dataset

The analysis is based on the Telco Customer Churn dataset. It includes information about customer demographics, services used, billing details, and whether the customer has churned.

The dataset is not included in this repository. To run the code, download it separately and place it in the project directory with the name:

WA_Fn-UseC_-Telco-Customer-Churn.csv

## Project Structure

```
stat-516-customer-churn-analysis/
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── python/
│   │   ├── ft_transformer.py
│   │   └── xgboost_model.py
│   └── r/
│       └── logistic_regression.R
├── report/
│   └── final_report.pdf
```

## Models Used

**Logistic Regression (R)**
Used as a baseline model. It’s simple, interpretable, and gives a reference point for comparison.

**XGBoost (Python)**
A tree-based ensemble model that performs well on tabular data and captures non-linear relationships.

**FT-Transformer (Python)**
A deep learning model designed for tabular data. It uses attention mechanisms to model interactions between features.

## How to Run

### Python Models

Install dependencies:

```
pip install -r requirements.txt
```

Run FT-Transformer:

```
python3 src/python/ft_transformer.py
```

Run XGBoost:

```
python3 src/python/xgboost_model.py
```

### R Model

Install required packages:

```
install.packages(c("tidyverse", "caret", "glmnet", "pROC", "DescTools"))
```

Run the script:

```
source("src/r/logistic_regression.R")
```

## Evaluation

The models are compared using multiple metrics instead of relying on a single measure. These include:

* ROC-AUC
* PR-AUC
* Accuracy
* Precision, Recall, and F1 Score
* Brier Score
* Log Loss

Using multiple metrics gives a better understanding of how each model performs under different aspects.

## Results

The detailed results and comparisons are included in the final report. Overall, the project focuses on understanding whether increased model complexity leads to meaningful improvements, or if simpler models remain competitive on this type of data.

## Contribution

This project was completed as part of a group assignment. Different models and parts of the analysis were handled across team members, with all results brought together in the final report.
