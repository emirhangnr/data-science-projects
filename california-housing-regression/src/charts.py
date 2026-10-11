import altair as alt
import pandas as pd

BLUE = "#2a78d6"
TARGET = "MedHouseVal"
TARGET_TITLE = "House value ($100k)"

# Allow charts with more than 5,000 rows (full dataset histograms)
alt.data_transformers.disable_max_rows()


def histogram(df, col):
    return alt.Chart(df).mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        alt.X(f"{col}:Q", bin=alt.Bin(maxbins=50), title=col),
        alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip(f"{col}:Q", bin=True, title=col), alt.Tooltip("count():Q", title="Count")],
    )


def geo_map(df):
    # Each point is a district, colored by house value
    return alt.Chart(df).mark_circle(size=12, opacity=0.6).encode(
        alt.X("Longitude:Q", scale=alt.Scale(zero=False)),
        alt.Y("Latitude:Q", scale=alt.Scale(zero=False)),
        alt.Color(f"{TARGET}:Q", title=TARGET_TITLE, scale=alt.Scale(scheme="viridis")),
        tooltip=["Latitude", "Longitude", alt.Tooltip(f"{TARGET}:Q", format=".2f", title=TARGET_TITLE)],
    ).properties(height=450)


def scatter(df, col):
    return alt.Chart(df).mark_circle(size=20, opacity=0.3, color=BLUE).encode(
        alt.X(f"{col}:Q", title=col, scale=alt.Scale(zero=False)),
        alt.Y(f"{TARGET}:Q", title=TARGET_TITLE),
        tooltip=[col, alt.Tooltip(f"{TARGET}:Q", format=".2f", title=TARGET_TITLE)],
    )


def correlation(df):
    corr = df.corr().reset_index().melt("index", var_name="feature", value_name="corr")
    base = alt.Chart(corr).encode(
        alt.X("feature:N", title=None, sort=None),
        alt.Y("index:N", title=None, sort=None),
    )
    cells = base.mark_rect().encode(
        alt.Color("corr:Q", title="Correlation",
                  scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True)),
        tooltip=["index", "feature", alt.Tooltip("corr:Q", format=".2f")],
    )
    text = base.mark_text(fontSize=11).encode(alt.Text("corr:Q", format=".2f"))
    return (cells + text).properties(height=400)


def actual_vs_predicted(y_true, y_pred):
    data = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    line = alt.Chart(pd.DataFrame({"x": [0, 5]})).mark_line(
        color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="x:Q")
    points = alt.Chart(data).mark_circle(size=20, opacity=0.3, color=BLUE).encode(
        alt.X("Actual:Q", title=f"Actual {TARGET_TITLE}"),
        alt.Y("Predicted:Q", title=f"Predicted {TARGET_TITLE}"),
        tooltip=[alt.Tooltip("Actual:Q", format=".2f"), alt.Tooltip("Predicted:Q", format=".2f")],
    )
    return line + points


def importance_bar(data):
    return alt.Chart(data).mark_bar(color=BLUE, cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
        alt.X("importance:Q", title="Drop in R² when shuffled"),
        alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".3f", title="Importance")],
    )
