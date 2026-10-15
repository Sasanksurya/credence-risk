"""Shared fixtures: a tiny synthetic model with the same structure as the real pipeline."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.model_loader import load_artifacts
from app.services.risk_scoring import ENGINEERED_FEATURES, engineer_features

RAW_NUMERIC = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "EXT_SOURCE_1"]
CATEGORICAL = ["CODE_GENDER", "NAME_CONTRACT_TYPE"]
FEATURES = RAW_NUMERIC + CATEGORICAL + ENGINEERED_FEATURES
REQUIRED_INPUTS = RAW_NUMERIC + CATEGORICAL


def make_applicants(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.uniform(40_000, 300_000, n)
    credit = income * rng.uniform(1, 6, n)
    ext1 = rng.uniform(0, 1, n)
    ext1[rng.random(n) < 0.1] = np.nan
    return pd.DataFrame({
        "SK_ID_CURR": 100_000 + np.arange(n),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": credit * rng.uniform(0.03, 0.08, n),
        "AMT_GOODS_PRICE": credit * rng.uniform(0.8, 1.0, n),
        "EXT_SOURCE_1": ext1,
        "CODE_GENDER": rng.choice(["F", "M"], n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
    })


def build_model() -> Pipeline:
    df = make_applicants(600, seed=1)
    rng = np.random.default_rng(2)
    ratio = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    z = 0.5 * ratio - 2 * df["EXT_SOURCE_1"].fillna(0.5) + 0.3 * (df["CODE_GENDER"] == "M")
    y = (rng.random(len(df)) < 1 / (1 + np.exp(-z))).astype(int)
    y[0], y[1] = 1, 0  # guarantee both classes exist

    X = engineer_features(df)[FEATURES]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), RAW_NUMERIC + ENGINEERED_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    model = Pipeline([("preprocessor", pre),
                      ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    return model.fit(X, y)


def write_artifacts(folder, model=None, features=None) -> None:
    model = model or build_model()
    joblib.dump(model, folder / "credence_risk_model.joblib")
    (folder / "feature_columns.json").write_text(json.dumps(features or FEATURES))
    (folder / "model_metrics.json").write_text(json.dumps({
        "model": "Synthetic test model", "roc_auc": 0.5, "average_precision": 0.1,
        "validation_rows": 100, "random_state": 42}))


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("artifacts")
    write_artifacts(folder)
    return folder


@pytest.fixture(scope="session")
def art(artifact_dir):
    return load_artifacts(str(artifact_dir))


@pytest.fixture()
def applicants():
    return make_applicants(50, seed=3)