import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

from src import charts
from src.features import add_features

st.set_page_config(page_title="Medical Cost Prediction", layout="wide")

RISK_ORDER = ["nonsmoker_normal", "nonsmoker_obese", "smoker_normal", "smoker_obese"]
BMI_ORDER = ["underweight", "normal", "overweight", "obese"]
ROOT = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "decision_tree.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "insurance.csv").drop_duplicates().reset_index(drop=True)


@st.cache_data
def load_metrics():
    with open(ROOT / "models" / "metrics.json") as f:
        return json.load(f)


model = load_model()
df = load_data()
metrics = load_metrics()

st.title("Medical Cost Prediction")
tab_predict, tab_eda, tab_model = st.tabs(["Prediction", "EDA", "Model"])

# --- Prediction ---
with tab_predict:
    col1, col2 = st.columns(2)
    age = col1.number_input("Age", 18, 64, 30)
    bmi = col2.number_input("BMI", 15.0, 55.0, 25.0, step=0.1)
    children = col1.number_input("Children", 0, 5, 0)
    sex = col2.selectbox("Sex", ["male", "female"])
    smoker = col1.selectbox("Smoker", ["no", "yes"])
    region = col2.selectbox("Region", ["southeast", "southwest", "northwest", "northeast"])

    if st.button("Predict", width="stretch"):
        X = pd.DataFrame([{
            "age": age, "sex": sex, "bmi": bmi,
            "children": children, "smoker": smoker, "region": region,
        }])
        st.metric("Estimated charges", f"${model.predict(X)[0]:,.2f}")

# --- EDA ---
with tab_eda:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Median charges", f"${df['charges'].median():,.0f}")
    c3.metric("Smokers", f"{(df['smoker'] == 'yes').mean():.0%}")
    c4.metric("Mean BMI", f"{df['bmi'].mean():.1f}")

    st.subheader("Distributions")
    c1, c2 = st.columns(2)
    num_col = c1.selectbox("Numerical feature", ["charges", "age", "bmi", "children"])
    c1.altair_chart(charts.histogram(df, num_col), width="stretch")
    cat_col = c2.selectbox("Categorical feature vs charges", ["smoker", "sex", "region"])
    c2.altair_chart(charts.boxplot(df, cat_col), width="stretch")

    st.subheader("Charges by age and BMI")
    c1, c2 = st.columns(2)
    c1.altair_chart(charts.scatter(df, "age"), width="stretch")
    c2.altair_chart(charts.scatter(df, "bmi"), width="stretch")
    st.caption("Smoking splits charges into separate bands. For smokers, BMI ≥ 30 raises them further.")

    st.subheader("Engineered features")
    eda = add_features(df)
    c1, c2 = st.columns(2)
    c1.altair_chart(charts.median_bar(eda, "risk_group", RISK_ORDER), width="stretch")
    c2.altair_chart(charts.median_bar(eda, "bmi_category", BMI_ORDER), width="stretch")

    with st.expander("Raw data"):
        st.dataframe(df, width="stretch", hide_index=True)

# --- Model ---
with tab_model:
    test = metrics["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test R²", f"{test['r2']:.3f}")
    c2.metric("Test RMSE", f"${test['rmse']:,.0f}")
    c3.metric("Test MAE", f"${test['mae']:,.0f}")
    c4.metric("CV R²", f"{metrics['cv_r2']:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Configuration")
        params = metrics["best_params"] | {"depth": metrics["depth"], "leaves": metrics["leaves"]}
        st.table(
            pd.DataFrame(params.items(), columns=["Parameter", "Value"]).astype(str),
            hide_index=True,
        )
    with c2:
        st.subheader("Comparison (test set)")
        comparison = pd.DataFrame(metrics["comparison"]).T.rename(
            columns={"r2": "R²", "rmse": "RMSE", "mae": "MAE"})
        st.table(
            comparison.style.format({"R²": "{:.3f}", "RMSE": "${:,.0f}", "MAE": "${:,.0f}"}),
        )

    X = df.drop("charges", axis=1)
    _, X_test, _, y_test = train_test_split(X, df["charges"], test_size=0.2, random_state=42)
    tree = model.named_steps["model"]
    names = [n.split("__", 1)[1] for n in model.named_steps["prep"].get_feature_names_out()]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Actual vs predicted")
        st.altair_chart(charts.actual_vs_predicted(y_test, model.predict(X_test)),
                        width="stretch")
    with c2:
        st.subheader("Feature importance")
        importance = pd.DataFrame({"feature": names, "importance": tree.feature_importances_})
        st.altair_chart(charts.importance_bar(importance[importance["importance"] > 0]),
                        width="stretch")

    st.subheader("Tree structure")
    depth = st.slider("Levels shown", 1, tree.get_depth(), min(3, tree.get_depth()))
    dot = export_graphviz(tree, max_depth=depth, feature_names=names, filled=True,
                          rounded=True, impurity=False, precision=0)
    st.graphviz_chart(dot, width="stretch")
