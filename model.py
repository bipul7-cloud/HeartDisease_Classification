# ==========================================
# STEP 2: TRAIN & SAVE MODELS
# File: model/train_models.py
# ==========================================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --------------------------
# Create directory to save models
# --------------------------
os.makedirs("model/saved_models", exist_ok=True)

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# --------------------------
# Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# Scaling (for Logistic & KNN)
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/saved_models/scaler.pkl")

# --------------------------
# Helper function
# --------------------------
def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("AUC:", roc_auc_score(y_true, y_prob))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))
    print("MCC:", matthews_corrcoef(y_true, y_pred))

# --------------------------
# Logistic Regression
# --------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
joblib.dump(lr, "model/saved_models/logistic.pkl")

evaluate(
    "Logistic Regression",
    y_test,
    lr.predict(X_test_scaled),
    lr.predict_proba(X_test_scaled)[:, 1]
)

# --------------------------
# Decision Tree
# --------------------------
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
joblib.dump(dt, "model/saved_models/decision_tree.pkl")

evaluate(
    "Decision Tree",
    y_test,
    dt.predict(X_test),
    dt.predict_proba(X_test)[:, 1]
)

# --------------------------
# KNN
# --------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
joblib.dump(knn, "model/saved_models/knn.pkl")

evaluate(
    "KNN",
    y_test,
    knn.predict(X_test_scaled),
    knn.predict_proba(X_test_scaled)[:, 1]
)

# --------------------------
# Naive Bayes
# --------------------------
nb = GaussianNB()
nb.fit(X_train, y_train)
joblib.dump(nb, "model/saved_models/naive_bayes.pkl")

evaluate(
    "Naive Bayes",
    y_test,
    nb.predict(X_test),
    nb.predict_proba(X_test)[:, 1]
)

# --------------------------
# Random Forest
# --------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "model/saved_models/random_forest.pkl")

evaluate(
    "Random Forest",
    y_test,
    rf.predict(X_test),
    rf.predict_proba(X_test)[:, 1]
)

# --------------------------
# XGBoost
# --------------------------
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)
joblib.dump(xgb, "model/saved_models/xgboost.pkl")

evaluate(
    "XGBoost",
    y_test,
    xgb.predict(X_test),
    xgb.predict_proba(X_test)[:, 1]
)

print("\n✅ All models trained and saved successfully.")
