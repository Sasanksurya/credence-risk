"""Reusable scoring service: schema validation, feature engineering, prediction, risk bands."""
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from app.config import RISK_BAND_LABELS, RISK_BAND_THRESHOLDS
from app.services.model_loader import Artifacts

ENGINEERED_FEATURES = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_GOODS_RATIO"]
ID_COLUMN = "SK_ID_CURR"
IGNORED_COLUMNS = {"TARGET", ID_COLUMN}  # identifiers / label, never model inputs
SCORE_COLUMN = "Predicted_Probability"
BAND_COLUMN = "Risk_Band"


class SchemaError(ValueError):
    """Raised when input data cannot be scored."""


@dataclass
class ScoringResult:
    results: pd.DataFrame
    warnings: list = field(default_factory=list)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same formulas as the training notebook. Zero denominators become missing."""
    out = df.copy()
    income = out["AMT_INCOME_TOTAL"].replace(0, np.nan)
    goods = out["AMT_GOODS_PRICE"].replace(0, np.nan)
    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / income
    out["ANNUITY_INCOME_RATIO"] = (out["AMT_ANNUITY"] * 12) / income
    out["CREDIT_GOODS_RATIO"] = out["AMT_CREDIT"] / goods
    for c in ENGINEERED_FEATURES:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
    return out


def required_input_columns(feature_columns: Sequence[str]) -> list:
    return [c for c in feature_columns if c not in ENGINEERED_FEATURES]


def prepare_features(df: pd.DataFrame, art: Artifacts):
    """Validate, coerce and order inputs exactly as the saved model expects."""
    if df is None or df.empty:
        raise SchemaError("The input contains no rows.")

    required = required_input_columns(art.feature_columns)
    missing = [c for c in required if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:15]) + (" ..." if len(missing) > 15 else "")
        raise SchemaError(f"{len(missing)} required column(s) are missing: {shown}")

    warnings = []
    work = df[required].copy()

    for c in art.numeric_features:
        if c in ENGINEERED_FEATURES:
            continue
        before = int(work[c].notna().sum())
        work[c] = pd.to_numeric(work[c], errors="coerce")
        invalid = before - int(work[c].notna().sum())
        if invalid:
            warnings.append(f"{c}: {invalid} non-numeric value(s) were treated as missing.")

    for c in art.categorical_features:
        col = work[c].astype("object")
        col = col.where(col.notna(), np.nan).replace("", np.nan)
        work[c] = col
        unseen = sorted(set(col.dropna().astype(str)) - set(art.known_categories.get(c, [])))
        if unseen:
            warnings.append(f"{c}: unseen categories {unseen[:5]} were ignored by the encoder.")

    work = engineer_features(work)
    return work[art.feature_columns], warnings


def assign_risk_bands(proba, thresholds=RISK_BAND_THRESHOLDS, labels=RISK_BAND_LABELS):
    """Low < t1 <= Moderate < t2 <= Elevated < t3 <= High. Illustrative bands only."""
    idx = np.digitize(np.asarray(proba, dtype=float), np.asarray(thresholds))
    return np.asarray(labels, dtype=object)[idx]


def score_dataframe(df: pd.DataFrame, art: Artifacts,
                    thresholds: Optional[Sequence[float]] = None,
                    labels: Optional[Sequence[str]] = None) -> ScoringResult:
    """Return probability of TARGET = 1 and an illustrative band for each row, in input order."""
    X, warnings = prepare_features(df, art)
    pos = list(art.model.classes_).index(1)
    proba = art.model.predict_proba(X)[:, pos]

    results = pd.DataFrame(index=df.index)
    if ID_COLUMN in df.columns:
        results[ID_COLUMN] = df[ID_COLUMN].values
    results[SCORE_COLUMN] = np.round(proba, 4)
    results[BAND_COLUMN] = assign_risk_bands(
        proba, thresholds or RISK_BAND_THRESHOLDS, labels or RISK_BAND_LABELS)
    return ScoringResult(results, warnings)