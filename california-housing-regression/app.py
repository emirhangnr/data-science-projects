import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src import charts

st.set_page_config(page_title="California Housing", layout="wide")

ROOT = Path(__file__).parent
TARGET = "MedHouseVal"


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "hgb_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "california_housing.csv")


@st.cache_data
def load_metrics():
    with open(ROOT / "models" / "metrics.json") as f:
        return json.load(f)


model = load_model()
df = load_data()
metrics = load_metrics()
# Scatter charts use a sample to keep the page light
sample = df.sample(5000, random_state=42)

st.title("California Housing Prices")
tab_predict, tab_eda, tab_model = st.tabs(["Prediction", "EDA", "Model"])

# --- Prediction ---
with tab_predict:
    med = df.median()
    col1, col2 = st.columns(2)
    med_inc = col1.number_input("Median income ($10k)", 0.5, 15.0, round(med["MedInc"], 1), step=0.1)
    house_age = col2.number_input("House age (years)", 1, 52, int(med["HouseAge"]))
    ave_rooms = col1.number_input("Average rooms", 1.0, 15.0, round(med["AveRooms"], 1), step=0.1)
    ave_bedrms = col2.number_input("Average bedrooms", 0.5, 5.0, round(med["AveBedrms"], 1), step=0.1)
    population = col1.number_input("Population", 3, 36000, int(med["Population"]), step=50)
    ave_occup = col2.number_input("Average occupancy", 0.5, 10.0, round(med["AveOccup"], 1), step=0.1)
    latitude = col1.number_input("Latitude", 32.5, 42.0, 34.05, step=0.01)
    longitude = col2.number_input("Longitude", -124.4, -114.3, -118.25, step=0.01)

    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=5, height=250)

    if st.button("Predict", width="stretch"):
        X = pd.DataFrame([{
            "MedInc": med_inc, "HouseAge": house_age, "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms, "Population": population, "AveOccup": ave_occup,
            "Latitude": latitude, "Longitude": longitude,
        }])
        st.metric("Estimated median house value", f"${model.predict(X)[0] * 100_000:,.0f}")
        st.caption("House values in the training data are capped at $500,000.")

# --- EDA ---
with tab_eda:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts", f"{len(df):,}")
    c2.metric("Median house value", f"${df[TARGET].median() * 100_000:,.0f}")
    c3.metric("Median income", f"${df['MedInc'].median() * 10_000:,.0f}")
    c4.metric("Capped values", f"{(df[TARGET] >= 5).mean():.1%}")

    st.subheader("Distributions")
    c1, c2 = st.columns(2)
    col = c1.selectbox("Feature", df.columns, index=list(df.columns).index(TARGET))
    c1.altair_chart(charts.histogram(df, col), width="stretch")
    x_col = c2.selectbox("Feature vs house value", df.columns.drop(TARGET))
    c2.altair_chart(charts.scatter(sample, x_col), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Location")
        st.altair_chart(charts.geo_map(sample), width="stretch")
        st.caption("Coastal areas around San Francisco and Los Angeles have the highest values.")
    with c2:
        st.subheader("Correlation")
        st.altair_chart(charts.correlation(df), width="stretch")
        st.caption("Median income has the strongest correlation with house value.")

    with st.expander("Raw data"):
        st.dataframe(df, width="stretch", hide_index=True)

# --- Model ---
with tab_model:
    test = metrics["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test R²", f"{test['r2']:.3f}")
    c2.metric("Test RMSE", f"${test['rmse'] * 100_000:,.0f}")
    c3.metric("Test MAE", f"${test['mae'] * 100_000:,.0f}")
    c4.metric("CV RMSE", f"${metrics['cv_rmse'] * 100_000:,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Configuration")
        st.table(
            pd.DataFrame(metrics["best_params"].items(), columns=["Parameter", "Value"]).astype(str),
            hide_index=True,
        )
    with c2:
        st.subheader("Comparison (test set)")
        comparison = pd.DataFrame(metrics["comparison"]).T.rename(
            columns={"r2": "R²", "rmse": "RMSE", "mae": "MAE"})
        st.table(comparison.style.format("{:.3f}"))
        st.caption("RMSE and MAE are in $100k.")

    # Same split as train.py, so these are unseen test rows
    X = df.drop(TARGET, axis=1)
    _, X_test, _, y_test = train_test_split(X, df[TARGET], test_size=0.2, random_state=42)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Actual vs predicted")
        st.altair_chart(charts.actual_vs_predicted(y_test, model.predict(X_test)), width="stretch")
    with c2:
        st.subheader("Feature importance")
        importance = pd.DataFrame(metrics["importance"].items(), columns=["feature", "importance"])
        st.altair_chart(charts.importance_bar(importance), width="stretch")
        st.caption("Permutation importance on the test set.")
