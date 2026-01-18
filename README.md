# Heart Disease Binary Classification

## Problem Statement
The objective of this project is to build and evaluate multiple machine learning
classification models to predict the presence of heart disease in a patient
based on clinical attributes. This is a binary classification problem where
the target variable indicates whether heart disease is present or not.

---

## Dataset Description
The dataset used in this project contains 1025 instances and 14 features.
It includes patient medical attributes such as age, cholesterol level,
blood pressure, and other heart-related measurements.
The target variable is binary:
- 0 indicates absence of heart disease
- 1 indicates the presence of heart disease

---

## Models Implemented
The following six classification models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbours (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

---

## Model Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.81 | 0.93 | 0.76 | 0.91 | 0.83 | 0.63 |
| Decision Tree | 0.99 | 0.99 | 1.00 | 0.97 | 0.99 | 0.97 |
| KNN | 0.86 | 0.96 | 0.87 | 0.86 | 0.87 | 0.73 |
| Naive Bayes | 0.83 | 0.90 | 0.81 | 0.88 | 0.84 | 0.66 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| XGBoost | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

---

## Observations
- Logistic Regression showed moderate performance due to its linear decision boundary.
- Naive Bayes achieved reasonable results but was limited by its assumption of feature independence.
- KNN performance was affected by sensitivity to distance and data distribution.
- Decision Tree performed very well but showed minor misclassifications.
- Random Forest achieved excellent performance by combining multiple decision trees.
- XGBoost achieved the best performance with perfect scores across all evaluation metrics and
  was selected as the final model due to its boosted ensemble structure and regularization.

---

## Conclusion
Ensemble-based models significantly outperformed individual classifiers in this
task. XGBoost emerged as the most reliable model, achieving the highest Accuracy,
AUC and MCC, making it suitable for heart disease prediction.
