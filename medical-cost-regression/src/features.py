import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
    ).astype(str)

    smoker = df["smoker"] == "yes"
    obese = df["bmi"] >= 30
    df["risk_group"] = np.select(
        [smoker & obese, smoker & ~obese, ~smoker & obese, ~smoker & ~obese],
        ["smoker_obese", "smoker_normal", "nonsmoker_obese", "nonsmoker_normal"],
        default="unknown",
    )

    df["has_children"] = (df["children"] > 0).astype(int)
    return df
