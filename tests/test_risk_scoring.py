"""Tests for feature engineering, schema validation, scoring and risk bands."""
import numpy as np
import pandas as pd
import pytest

from app import config
from app.services.model_loader import load_artifacts
from app.services.risk_scoring import (BAND_COLUMN, ENGINEERED_FEATURES, SCORE_COLUMN, SchemaError,
                                       assign_risk_bands, engineer_features, prepare_features,
                                       required_input_columns, score_dataframe)
from tests.conftest import FEATURES, REQUIRED_INPUTS, make_applicants

BANDS = {"Low", "Moderate", "Elevated", "High"}


# ---------- feature engineering ----------
def test_engineered_ratio_formulas():
    df = pd.DataFrame({"AMT_INCOME_TOTAL": [12000.0], "AMT_CREDIT": [24000.0],
                       "AMT_ANNUITY": [1000.0], "AMT_GOODS_PRICE": [20000.0]})
    out = engineer_features(df)
    assert out["CREDIT_INCOME_RATIO"].iloc[0] == pytest.approx(2.0)
    assert out["ANNUITY_INCOME_RATIO"].iloc[0] == pytest.approx(1.0)  # annuity x 12 / income
    assert out["CREDIT_GOODS_RATIO"].iloc[0] == pytest.approx(1.2)


def test_zero_denominators_become_missing_not_infinite():
    df = pd.DataFrame({"AMT_INCOME_TOTAL": [0.0], "AMT_CREDIT": [5000.0],
                       "AMT_ANNUITY": [100.0], "AMT_GOODS_PRICE": [0.0]})
    out = engineer_features(df)
    for col in ENGINEERED_FEATURES:
        assert np.isnan(out[col].iloc[0])
        assert not np.isinf(out[col].iloc[0])


def test_engineer_features_does_not_modify_input():
    df = pd.DataFrame({"AMT_INCOME_TOTAL": [1.0], "AMT_CREDIT": [2.0],
                       "AMT_ANNUITY": [3.0], "AMT_GOODS_PRICE": [4.0]})
    engineer_features(df)
    assert list(df.columns) == ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]


# ---------- schema ----------
def test_required_columns_exclude_engineered_features():
    assert required_input_columns(FEATURES) == REQUIRED_INPUTS


def test_valid_schema_is_ordered_like_the_model(art, applicants):
    shuffled = applicants[list(reversed(applicants.columns))]
    X, warnings = prepare_features(shuffled, art)
    assert list(X.columns) == art.feature_columns
    assert warnings == []


def test_missing_required_column_is_reported(art, applicants):
    with pytest.raises(SchemaError, match="AMT_CREDIT"):
        score_dataframe(applicants.drop(columns=["AMT_CREDIT"]), art)


def test_empty_input_is_rejected(art):
    with pytest.raises(SchemaError):
        score_dataframe(pd.DataFrame(), art)


def test_identifier_and_target_columns_are_ignored(art, applicants):
    with_target = applicants.assign(TARGET=1)
    a = score_dataframe(applicants, art).results[SCORE_COLUMN].to_numpy()
    b = score_dataframe(with_target, art).results[SCORE_COLUMN].to_numpy()
    assert np.allclose(a, b)


# ---------- robustness ----------
def test_unexpected_category_warns_and_still_scores(art, applicants):
    bad = applicants.copy()
    bad.loc[bad.index[0], "CODE_GENDER"] = "Z"
    result = score_dataframe(bad, art)
    assert any("CODE_GENDER" in w and "unseen" in w for w in result.warnings)
    assert result.results[SCORE_COLUMN].between(0, 1).all()


def test_missing_values_are_supported(art, applicants):
    gappy = applicants.copy()
    gappy.loc[gappy.index[:5], ["EXT_SOURCE_1", "AMT_GOODS_PRICE"]] = np.nan
    gappy.loc[gappy.index[:5], "CODE_GENDER"] = None
    result = score_dataframe(gappy, art)
    assert result.results[SCORE_COLUMN].notna().all()
    assert result.results[SCORE_COLUMN].between(0, 1).all()


def test_non_numeric_values_warn_and_are_treated_as_missing(art, applicants):
    bad = applicants.copy()
    bad["AMT_CREDIT"] = bad["AMT_CREDIT"].astype(object)
    bad.loc[bad.index[0], "AMT_CREDIT"] = "abc"
    result = score_dataframe(bad, art)
    assert any("AMT_CREDIT" in w and "non-numeric" in w for w in result.warnings)
    assert result.results[SCORE_COLUMN].notna().all()


def test_zero_income_and_goods_price_still_score(art, applicants):
    zero = applicants.copy()
    zero.loc[zero.index[0], ["AMT_INCOME_TOTAL", "AMT_GOODS_PRICE"]] = 0
    result = score_dataframe(zero, art)
    assert 0 <= result.results[SCORE_COLUMN].iloc[0] <= 1


# ---------- scoring ----------
def test_single_applicant_scoring(art, applicants):
    result = score_dataframe(applicants.iloc[[0]], art)
    assert len(result.results) == 1
    assert 0 <= result.results[SCORE_COLUMN].iloc[0] <= 1
    assert result.results[BAND_COLUMN].iloc[0] in BANDS


def test_batch_scoring_preserves_rows_order_and_ids(art, applicants):
    result = score_dataframe(applicants, art).results
    assert len(result) == len(applicants)
    assert result["SK_ID_CURR"].tolist() == applicants["SK_ID_CURR"].tolist()
    assert result.index.tolist() == applicants.index.tolist()
    assert set(result[BAND_COLUMN]) <= BANDS


def test_large_batch_probabilities_stay_in_range(art):
    big = make_applicants(2000, seed=9)
    proba = score_dataframe(big, art).results[SCORE_COLUMN]
    assert len(proba) == 2000
    assert proba.between(0, 1).all()


def test_single_and_batch_scores_agree(art, applicants):
    batch = score_dataframe(applicants, art).results[SCORE_COLUMN].to_numpy()
    for i in (0, 7, 31):
        single = score_dataframe(applicants.iloc[[i]], art).results[SCORE_COLUMN].iloc[0]
        assert single == pytest.approx(batch[i], abs=1e-4)


# ---------- risk bands ----------
def test_band_boundaries_use_documented_cutoffs():
    probs = [0.0, 0.0999, 0.10, 0.2999, 0.30, 0.4999, 0.50, 1.0]
    expected = ["Low", "Low", "Moderate", "Moderate", "Elevated", "Elevated", "High", "High"]
    assert list(assign_risk_bands(probs)) == expected


def test_band_thresholds_are_configurable():
    bands = assign_risk_bands([0.15, 0.25, 0.45, 0.70], thresholds=(0.2, 0.4, 0.6))
    assert list(bands) == ["Low", "Moderate", "Elevated", "High"]


def test_threshold_setting_is_validated(monkeypatch):
    monkeypatch.setenv("RISK_BAND_THRESHOLDS", "0.2,0.4,0.6")
    assert config._thresholds() == (0.2, 0.4, 0.6)
    for bad in ("0.5,0.3,0.1", "0.1,0.2", "abc", "0.1,0.5,1.5"):
        monkeypatch.setenv("RISK_BAND_THRESHOLDS", bad)
        with pytest.raises(ValueError):
            config._thresholds()


# ---------- real saved model (skipped when files are absent) ----------
REAL_MODEL = config.MODELS_DIR / "credence_risk_model.joblib"
REAL_DATA = config.ROOT / "datasets" / "application_train.csv"


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason="real model artifact not present")
def test_real_artifacts_load_with_consistent_schema():
    real = load_artifacts()
    assert len(real.feature_columns) == 123
    assert real.feature_columns == list(real.model.feature_names_in_)
    assert real.metrics["validation_rows"] == 61503


@pytest.mark.skipif(not (REAL_MODEL.is_file() and REAL_DATA.is_file()),
                    reason="real model or dataset not present")
def test_real_model_scores_first_rows_as_in_smoke_test():
    real = load_artifacts()
    df = pd.read_csv(REAL_DATA, nrows=5)
    result = score_dataframe(df, real).results
    assert result[SCORE_COLUMN].between(0, 1).all()
    assert result["SK_ID_CURR"].iloc[0] == 100002
    assert result[SCORE_COLUMN].iloc[0] == pytest.approx(0.9523, abs=1e-3)