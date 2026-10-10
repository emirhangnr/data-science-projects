import altair as alt
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
CHURN_COLOR = alt.Color(
    "Churn:N", title="Churn",
    scale=alt.Scale(domain=["No", "Yes"], range=[BLUE, ORANGE]),
)


def churn_boxplot(df, col):
    data = df.assign(Churn=df["Churn"].map({0: "No", 1: "Yes"}))
    return alt.Chart(data).mark_boxplot(size=50).encode(
        alt.X("Churn:N", axis=alt.Axis(labelAngle=0)),
        alt.Y(f"{col}:Q", title=col),
        CHURN_COLOR,
    )


def churn_rate_bar(df, col):
    data = df.groupby(col)["Churn"].mean().reset_index()
    return alt.Chart(data).mark_bar(color=ORANGE, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        alt.X(f"{col}:N", sort="y", title=None, axis=alt.Axis(labelAngle=0)),
        alt.Y("Churn:Q", title="Churn rate", axis=alt.Axis(format="%")),
        tooltip=[col, alt.Tooltip("Churn:Q", format=".1%", title="Churn rate")],
    )


def correlation_heatmap(df, cols):
    corr = df[cols].corr().reset_index().melt("index", var_name="col", value_name="corr")
    base = alt.Chart(corr).encode(alt.X("col:N", title=None), alt.Y("index:N", title=None))
    heat = base.mark_rect().encode(
        alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
                  legend=None),
    )
    text = base.mark_text().encode(text=alt.Text("corr:Q", format=".2f"))
    return heat + text


def confusion_heatmap(cm):
    labels = ["No", "Yes"]
    data = pd.DataFrame([
        {"Actual": labels[i], "Predicted": labels[j], "Count": cm[i][j]}
        for i in range(2) for j in range(2)
    ])
    base = alt.Chart(data).encode(
        alt.X("Predicted:N", axis=alt.Axis(labelAngle=0)),
        alt.Y("Actual:N"),
    )
    heat = base.mark_rect().encode(alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None))
    text = base.mark_text(fontSize=16).encode(text="Count:Q")
    return heat + text


def roc_curve_chart(fpr, tpr):
    data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1]})).mark_line(
        color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="x:Q")
    curve = alt.Chart(data).mark_line(color=BLUE).encode(
        alt.X("FPR:Q", title="False positive rate"),
        alt.Y("TPR:Q", title="True positive rate"),
    )
    return diagonal + curve


def coefficient_bar(data):
    return alt.Chart(data).mark_bar().encode(
        alt.X("coef:Q", title="Coefficient"),
        alt.Y("feature:N", sort="-x", title=None),
        color=alt.condition(alt.datum.coef > 0, alt.value(ORANGE), alt.value(BLUE)),
        tooltip=["feature", alt.Tooltip("coef:Q", format=".3f", title="Coefficient")],
    )
