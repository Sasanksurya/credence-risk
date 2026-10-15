"""Exact linear SHAP contributions for the saved Logistic Regression pipeline.

For a linear model with independent features, the SHAP value of feature j is
coef_j * (x_j - mean_j), measured in log-odds of TARGET = 1.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from app.services.risk_scoring import prepare_features


class ExplainError(RuntimeError):
    """Raised when explanations cannot be produced consistently with the saved model."""


@dataclass
class Context:
    coef: np.ndarray
    mean: np.ndarray
    base_value: float
    group_index: dict


def build_context(art, report: dict) -> Context:
    pre = art.model.named_steps["preprocessor"]
    clf = art.model.named_steps["classifier"]
    names = [str(n) for n in pre.get_feature_names_out()]
    if names != report["transformed_names"]:
        raise ExplainError("The explainability report does not match the saved model. "
                           "Rerun scripts\\build_explainability_artifacts.py.")
    coef = np.asarray(clf.coef_, dtype=float).ravel()
    mean = np.asarray(report["mean"], dtype=float)
    if not (len(coef) == len(mean) == len(names)):
        raise ExplainError("Explainability report dimensions do not match the model.")
    groups = {}
    for i, feature in enumerate(report["original"]):
        groups.setdefault(feature, []).append(i)
    base = float(clf.intercept_[0] + coef @ mean)
    return Context(coef, mean, base, {k: np.array(v) for k, v in groups.items()})


def explain_row(art, ctx: Context, record: pd.DataFrame) -> dict:
    X, warnings = prepare_features(record, art)
    xt = art.model.named_steps["preprocessor"].transform(X)
    x = (xt.toarray() if sparse.issparse(xt) else np.asarray(xt)).ravel()
    contrib = ctx.coef * (x - ctx.mean)
    logit = ctx.base_value + float(contrib.sum())

    pos = list(art.model.classes_).index(1)
    prob = float(art.model.predict_proba(X)[0, pos])
    if abs(1 / (1 + np.exp(-logit)) - prob) > 1e-4:
        raise ExplainError("Contributions do not reconstruct the model's prediction.")

    return {
        "base_value": ctx.base_value, "logit": logit, "probability": prob,
        "contributions": {f: float(contrib[idx].sum()) for f, idx in ctx.group_index.items()},
        "values": X.iloc[0].to_dict(), "warnings": warnings,
    }