import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def scores(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


df = pd.read_csv("data/california_housing.csv")
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Tuned HistGradientBoosting (the model used by the app)
param_dist = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_iter": [200, 500, 1000],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 50],
    "l2_regularization": [0.0, 1.0],
}
search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    param_dist, n_iter=15, cv=3,
    scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE,
)
search.fit(X_train, y_train)
model = search.best_estimator_

# Other models (metrics only, not saved)
others = {
    "HistGradientBoosting (baseline)": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
}
comparison = {"HistGradientBoosting (tuned)": scores(y_test, model.predict(X_test))}
for name, m in others.items():
    comparison[name] = scores(y_test, m.fit(X_train, y_train).predict(X_test))

# HistGradientBoosting has no feature_importances_, so use permutation importance
perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE)

metrics = {
    "best_params": search.best_params_,
    "cv_rmse": float(-search.best_score_),
    "test": comparison["HistGradientBoosting (tuned)"],
    "comparison": comparison,
    "importance": dict(zip(X.columns, perm.importances_mean.round(4).tolist())),
}

joblib.dump(model, "models/hgb_model.pkl")
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Best params:", metrics["best_params"])
print(f"Test RMSE: {metrics['test']['rmse']:.4f}  R2: {metrics['test']['r2']:.4f}")
