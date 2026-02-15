import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

# -------------------------------------------------
# PAGE CONFIG (IMPORTANT – removes blank feel)
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="❤️",
    layout="centered"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("❤️ Heart Disease Detection System")
st.markdown(
    "This application allows you to **upload test data**, "
    "select a **trained machine learning model**, and "
    "view **performance metrics and confusion matrix**."
)

st.markdown("---")

# -------------------------------------------------
# SIDEBAR (INTERACTIVE NAVIGATION)
# -------------------------------------------------
st.sidebar.header("🔧 Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# -------------------------------------------------
# HANDLE NO FILE UPLOADED (NO BLACK SCREEN)
# -------------------------------------------------
if uploaded_file is None:
    st.info("⬅️ Please upload a CSV file from the sidebar to begin.")
    st.stop()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df = pd.read_csv(uploaded_file)

if "target" not in df.columns:
    st.error("❌ Uploaded file must contain a 'target' column.")
    st.stop()

X_test = df.drop("target", axis=1)
y_test = df["target"]

st.success("✅ Test dataset loaded successfully!")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
scaler = joblib.load("model/saved_models/scaler.pkl")

models = {
    "Logistic Regression": joblib.load("model/saved_models/logistic.pkl"),
    "Decision Tree": joblib.load("model/saved_models/decision_tree.pkl"),
    "KNN": joblib.load("model/saved_models/knn.pkl"),
    "Naive Bayes": joblib.load("model/saved_models/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/saved_models/random_forest.pkl"),
    "XGBoost": joblib.load("model/saved_models/xgboost.pkl"),
}

model = models[model_name]

# -------------------------------------------------
# PREPROCESS
# -------------------------------------------------
if model_name in ["Logistic Regression", "KNN"]:
    X_test_used = scaler.transform(X_test)
else:
    X_test_used = X_test

# -------------------------------------------------
# PREDICTIONS
# -------------------------------------------------
y_pred = model.predict(X_test_used)
y_prob = model.predict_proba(X_test_used)[:, 1]

# -------------------------------------------------
# METRICS DISPLAY (CARD-LIKE)
# -------------------------------------------------
st.markdown("## 📊 Model Performance Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric("AUC", f"{roc_auc_score(y_test, y_prob):.3f}")
col3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")

col4, col5, col6 = st.columns(3)

col4.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
col5.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
col6.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
st.markdown("## 🔍 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
st.write(cm)

st.markdown("---")
st.caption("Models are trained offline and loaded for evaluation.")
