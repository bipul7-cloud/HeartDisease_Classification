import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="❤️",
    layout="centered"
)

# ==================================================
# HEADER
# ==================================================
st.title("❤️ Heart Disease Detection System")
st.markdown(
    """
    Upload a **test CSV file**, select a **trained machine learning model**,  
    and view performance metrics along with the confusion matrix.
    """
)

st.markdown("---")

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
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

# ==================================================
# HANDLE NO FILE
# ==================================================
if uploaded_file is None:
    st.info("⬅️ Please upload a CSV file from the sidebar to begin.")
    st.stop()

# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_csv(uploaded_file)

if "target" not in df.columns:
    st.error("Uploaded CSV must contain a 'target' column.")
    st.stop()

X_test = df.drop("target", axis=1)
y_test = df["target"]

st.success("✅ Test dataset loaded successfully!")

# ==================================================
# LOAD TRAINED MODELS (ROOT DIRECTORY)
# ==================================================
scaler = joblib.load("scaler.pkl")

models = {
    "Logistic Regression": joblib.load("logistic.pkl"),
    "Decision Tree": joblib.load("decision_tree.pkl"),
    "KNN": joblib.load("knn.pkl"),
    "Naive Bayes": joblib.load("naive_bayes.pkl"),
    "Random Forest": joblib.load("random_forest.pkl"),
    "XGBoost": joblib.load("xgboost.pkl"),
}

model = models[model_name]

# ==================================================
# PREPROCESSING
# ==================================================
if model_name in ["Logistic Regression", "KNN"]:
    X_used = scaler.transform(X_test)
else:
    X_used = X_test

# ==================================================
# PREDICTIONS
# ==================================================
y_pred = model.predict(X_used)
y_prob = model.predict_proba(X_used)[:, 1]

# ==================================================
# METRICS
# ==================================================
st.markdown("## 📊 Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric("AUC", f"{roc_auc_score(y_test, y_prob):.3f}")
col3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")

col4, col5, col6 = st.columns(3)
col4.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
col5.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
col6.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")

# ==================================================
# CONFUSION MATRIX
# ==================================================
st.markdown("## 🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

st.markdown("---")
st.caption("Models are trained offline and loaded for evaluation.")
