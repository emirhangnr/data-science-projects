import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from src.features import add_features

CATEGORICAL = ["sex", "smoker", "region", "bmi_category", "risk_group"]
NUMERICAL = ["age", "bmi", "children", "has_children"]


def make_pipeline(model):
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERICAL),
        ])),
        ("model", model),
    ])


def scores(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


df = pd.read_csv("data/insurance.csv").drop_duplicates().reset_index(drop=True)
X = df.drop("charges", axis=1)
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tuned decision tree (the model used by the app)
param_grid = {
    "model__max_depth": [3, 5, 7, 10, 15, None],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 5, 10, 20],
    "model__criterion": ["squared_error"],
}
grid = GridSearchCV(make_pipeline(DecisionTreeRegressor(random_state=42)),
                    param_grid, cv=5, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# Baseline (metrics only, not saved as a model)
baseline = make_pipeline(DecisionTreeRegressor(random_state=42)).fit(X_train, y_train)

tree = model.named_steps["model"]
metrics = {
    "best_params": {k.replace("model__", ""): v for k, v in grid.best_params_.items()},
    "cv_r2": float(grid.best_score_),
    "depth": int(tree.get_depth()),
    "leaves": int(tree.get_n_leaves()),
    "test": scores(y_test, model.predict(X_test)),
    "comparison": {
        "Decision Tree (tuned)": scores(y_test, model.predict(X_test)),
        "Decision Tree (baseline)": scores(y_test, baseline.predict(X_test)),
    },
}

joblib.dump(model, "models/decision_tree.pkl")
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Best params:", metrics["best_params"])
print(f"Test R2: {metrics['test']['r2']:.4f}")
