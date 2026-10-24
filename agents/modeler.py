"""AJAN 4: Veri Sızıntısı & Modelleme. Split -> Pipeline (imputer/scaler/encoder yalnızca train'e fit)."""
import inspect
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score, mean_squared_error)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   PowerTransformer, RobustScaler, StandardScaler)

SCALERS = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler,
           "RobustScaler": RobustScaler, "PowerTransformer": PowerTransformer}


def detect_task(y: pd.Series) -> str:
    return "classification" if (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10) else "regression"


def build_pipeline(X: pd.DataFrame, model, scaler_name: str) -> Pipeline:
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scaler_name in SCALERS:
        num_steps.append(("scale", SCALERS[scaler_name]()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def run(df: pd.DataFrame, target: str, scaler_name: str, test_size: float = 0.2, seed: int = 42):
    df = df.dropna(subset=[target]).copy()
    X, y = df.drop(columns=target), df[target]
    X = X.loc[:, [c for c in X.columns if X[c].nunique() < len(X)]]  # ID benzeri kolonları at
    task = detect_task(y)
    if task == "classification":
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
        models = {"LogisticRegression": LogisticRegression(max_iter=1000),
                  "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)}
        cv_metric = "f1_weighted"
    else:
        models = {"Ridge": Ridge(),
                  "RandomForest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)}
        cv_metric = "r2"

    # 1) ÖNCE böl  2) tüm dönüşümler pipeline içinde, yalnızca train'e fit
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if task == "classification" else None)

    rows = []
    for name, m in models.items():
        pipe = build_pipeline(X_tr, m, scaler_name)
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring=cv_metric)  # fold başına yeniden fit
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        if task == "classification":
            r = {"Model": name, f"CV {cv_metric}": round(cv.mean(), 4),
                 "Test Accuracy": round(accuracy_score(y_te, pred), 4),
                 "Test F1": round(f1_score(y_te, pred, average="weighted"), 4)}
            if y.nunique() == 2:
                r["Test ROC-AUC"] = round(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1]), 4)
        else:
            r = {"Model": name, "CV r2": round(cv.mean(), 4),
                 "Test R2": round(r2_score(y_te, pred), 4),
                 "Test MAE": round(mean_absolute_error(y_te, pred), 4),
                 "Test RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4)}
        rows.append(r)
    return task, pd.DataFrame(rows)


def source_code() -> str:
    return inspect.getsource(build_pipeline)
