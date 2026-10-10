import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from src import charts
from src.data import load_data

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

ROOT = Path(__file__).parent
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "churn_model.pkl")


@st.cache_data
def load_df():
    return load_data(ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")


@st.cache_data
def load_metrics():
    with open(ROOT / "models" / "metrics.json") as f:
        return json.load(f)


model = load_model()
df = load_df()
metrics = load_metrics()
cat_cols = df.drop(columns="Churn").select_dtypes(exclude="number").columns.tolist()

st.title("Customer Churn Prediction")
tab_predict, tab_eda, tab_model = st.tabs(["Prediction", "EDA", "Model"])

# --- Prediction ---
with tab_predict:
    inputs = {}
    cols = st.columns(3)
    for i, col in enumerate(cat_cols):
        inputs[col] = cols[i % 3].selectbox(col, sorted(df[col].unique()))

    c1, c2 = st.columns(2)
    inputs["tenure"] = c1.number_input("tenure (months)", 0, 72, 12)
    inputs["MonthlyCharges"] = c2.number_input("MonthlyCharges", 18.0, 120.0, 70.0, step=0.5)
    inputs["TotalCharges"] = inputs["tenure"] * inputs["MonthlyCharges"]

    if st.button("Predict", width="stretch"):
        proba = model.predict_proba(pd.DataFrame([inputs]))[0, 1]
        st.metric("Churn probability", f"{proba:.0%}")
        (st.error if proba >= 0.5 else st.success)(
            "Likely to churn" if proba >= 0.5 else "Likely to stay")

# --- EDA ---
with tab_eda:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['Churn'].mean():.1%}")
    c3.metric("Median tenure", f"{df['tenure'].median():.0f} months")
    c4.metric("Mean monthly charges", f"${df['MonthlyCharges'].mean():.2f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Numerical features")
        num_col = st.selectbox("Feature", NUM_COLS)
        st.altair_chart(charts.churn_boxplot(df, num_col), width="stretch")
    with c2:
        st.subheader("Churn rate by category")
        cat_col = st.selectbox("Feature", cat_cols, index=cat_cols.index("Contract"))
        st.altair_chart(charts.churn_rate_bar(df, cat_col), width="stretch")

    st.subheader("Correlation")
    st.altair_chart(charts.correlation_heatmap(df, NUM_COLS + ["Churn"]), width="stretch")

    with st.expander("Raw data"):
        st.dataframe(df, width="stretch", hide_index=True)

# --- Model ---
with tab_model:
    test = metrics["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall (churn)", f"{test['recall']:.3f}")
    c2.metric("Precision (churn)", f"{test['precision']:.3f}")
    c3.metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")
    c4.metric("CV ROC-AUC", f"{metrics['cv_roc_auc']:.3f}")

    st.subheader("Comparison (test set)")
    comparison = pd.DataFrame(metrics["comparison"]).T.rename(columns={
        "recall": "Recall", "precision": "Precision", "f1": "F1", "roc_auc": "ROC-AUC"})
    st.table(comparison.style.format("{:.3f}"))

    X = df.drop(columns="Churn")
    _, X_test, _, y_test = train_test_split(
        X, df["Churn"], test_size=0.2, stratify=df["Churn"], random_state=42)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion matrix")
        st.altair_chart(charts.confusion_heatmap(metrics["confusion_matrix"]), width="stretch")
    with c2:
        st.subheader("ROC curve")
        st.altair_chart(charts.roc_curve_chart(fpr, tpr), width="stretch")

    st.subheader("Top coefficients")
    names = [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]
    coefs = pd.DataFrame({"feature": names, "coef": model.named_steps["model"].coef_[0]})
    coefs = coefs.sort_values("coef")
    st.altair_chart(charts.coefficient_bar(pd.concat([coefs.head(10), coefs.tail(10)])),
                    width="stretch")
