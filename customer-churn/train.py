import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import load_data


def make_pipeline(model, num_cols, cat_cols):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ])),
        ("model", model),
    ])


def scores(model, X, y):
    y_pred = model.predict(X)
    return {
        "recall": float(recall_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, model.predict_proba(X)[:, 1])),
    }


df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
X = df.drop(columns="Churn")
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}
fitted = {name: make_pipeline(m, num_cols, cat_cols).fit(X_train, y_train)
          for name, m in models.items()}

# Logistic Regression is used by the app: highest recall on the churn class
model = fitted["Logistic Regression"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

metrics = {
    "test": scores(model, X_test, y_test),
    "cv_roc_auc": float(cv_auc.mean()),
    "confusion_matrix": confusion_matrix(y_test, model.predict(X_test)).tolist(),
    "comparison": {name: scores(m, X_test, y_test) for name, m in fitted.items()},
}

joblib.dump(model, "models/churn_model.pkl")
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Test recall: {metrics['test']['recall']:.3f}")
print(f"CV ROC-AUC: {metrics['cv_roc_auc']:.3f}")
