import altair as alt
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
CURRENCY = "$,.0f"
SMOKER_COLOR = alt.Color(
    "smoker:N", title="Smoker",
    scale=alt.Scale(domain=["no", "yes"], range=[BLUE, ORANGE]),
)


def histogram(df, col):
    return alt.Chart(df).mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        alt.X(f"{col}:Q", bin=alt.Bin(maxbins=30), title=col),
        alt.Y("count():Q", title="Count"),
        tooltip=[alt.Tooltip(f"{col}:Q", bin=True, title=col), alt.Tooltip("count():Q", title="Count")],
    )


def boxplot(df, col):
    return alt.Chart(df).mark_boxplot(size=40, color=BLUE).encode(
        alt.X(f"{col}:N", title=col, axis=alt.Axis(labelAngle=0)),
        alt.Y("charges:Q", title="Charges", axis=alt.Axis(format=CURRENCY)),
    )


def scatter(df, col):
    return alt.Chart(df).mark_circle(size=50, opacity=0.6).encode(
        alt.X(f"{col}:Q", title=col, scale=alt.Scale(zero=False)),
        alt.Y("charges:Q", title="Charges", axis=alt.Axis(format=CURRENCY)),
        SMOKER_COLOR,
        tooltip=[col, "smoker", alt.Tooltip("charges:Q", format=CURRENCY, title="Charges")],
    )


def median_bar(df, col, order):
    data = df.groupby(col, observed=True)["charges"].median().reset_index()
    return alt.Chart(data).mark_bar(color=BLUE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        alt.X(f"{col}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        alt.Y("charges:Q", title="Median charges", axis=alt.Axis(format=CURRENCY)),
        tooltip=[col, alt.Tooltip("charges:Q", format=CURRENCY, title="Median")],
    )


def actual_vs_predicted(y_true, y_pred):
    data = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    lo, hi = data.min().min(), data.max().max()
    line = alt.Chart(pd.DataFrame({"x": [lo, hi]})).mark_line(
        color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="x:Q")
    points = alt.Chart(data).mark_circle(size=50, opacity=0.6, color=BLUE).encode(
        alt.X("Actual:Q", axis=alt.Axis(format=CURRENCY)),
        alt.Y("Predicted:Q", axis=alt.Axis(format=CURRENCY)),
        tooltip=[alt.Tooltip("Actual:Q", format=CURRENCY), alt.Tooltip("Predicted:Q", format=CURRENCY)],
    )
    return line + points


def importance_bar(data):
    return alt.Chart(data).mark_bar(color=BLUE, cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
        alt.X("importance:Q", title="Importance"),
        alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".3f", title="Importance")],
    )
